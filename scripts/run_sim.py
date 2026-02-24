#!/usr/bin/env python
"""Run a pretrained LeRobot policy in MuJoCo simulation and save video output.

Loads an ACT policy trained on the Aloha bimanual transfer cube task,
runs it in the gym-aloha MuJoCo environment, and saves episode videos.

Usage:
    python scripts/run_sim.py
    python scripts/run_sim.py --n-episodes 5 --device cpu
    python scripts/run_sim.py --policy lerobot/act_aloha_sim_insertion_human --task AlohaInsertion-v0
"""

import argparse
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from tqdm import trange

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.configs import AlohaEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import write_video

# Map of known pretrained models to their training datasets
KNOWN_DATASETS = {
    "lerobot/act_aloha_sim_transfer_cube_human": "lerobot/aloha_sim_transfer_cube_human",
    "lerobot/act_aloha_sim_insertion_human": "lerobot/aloha_sim_insertion_human",
}


def run_episode(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seed: int = 42,
) -> tuple[list[np.ndarray], dict]:
    """Run a single episode and collect frames + metrics."""
    policy.reset()
    observation, info = env.reset(seed=[seed])

    frames = []
    # Capture initial frame
    if isinstance(env, gym.vector.SyncVectorEnv):
        frames.append(env.envs[0].render())

    max_steps = env.call("_max_episode_steps")[0]
    done = np.array([False] * env.num_envs)
    total_reward = 0.0
    success = False
    steps = 0

    for step in trange(max_steps, desc="  Steps", leave=False):
        observation = preprocess_observation(observation)
        observation = add_envs_task(env, observation)
        observation = env_preprocessor(observation)
        observation = preprocessor(observation)

        with torch.inference_mode():
            action = policy.select_action(observation)
        action = postprocessor(action)

        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action_np = action_transition[ACTION].to("cpu").numpy()

        observation, reward, terminated, truncated, info = env.step(action_np)
        total_reward += float(reward[0])

        # Capture frame
        if isinstance(env, gym.vector.SyncVectorEnv):
            frames.append(env.envs[0].render())

        if "final_info" in info:
            final_info = info["final_info"]
            if isinstance(final_info, dict) and "is_success" in final_info:
                success = bool(final_info["is_success"][0])

        done = terminated | truncated | done
        steps = step + 1
        if np.all(done):
            break

    metrics = {
        "steps": steps,
        "total_reward": total_reward,
        "success": success,
        "seed": seed,
    }
    return frames, metrics


def main():
    parser = argparse.ArgumentParser(description="Run pretrained LeRobot policy in MuJoCo simulation")
    parser.add_argument(
        "--policy", type=str, default="lerobot/act_aloha_sim_transfer_cube_human",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--task", type=str, default="AlohaTransferCube-v0",
        help="Gymnasium environment task ID",
    )
    parser.add_argument("--n-episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, cuda, or mps")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/videos",
        help="Directory to save video files",
    )
    parser.add_argument("--seed", type=int, default=1000, help="Starting random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Policy:  {args.policy}")
    print(f"Task:    {args.task}")
    print(f"Device:  {args.device}")
    print(f"Episodes: {args.n_episodes}")
    print()

    # 1. Configure environment
    env_cfg = AlohaEnv(
        task=args.task,
        fps=50,
        episode_length=400,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )

    # 2. Create vectorized environment (single env for simplicity)
    print("Creating environment...")
    envs = make_env(cfg=env_cfg, n_envs=1, use_async_envs=False)
    vec_env = envs["aloha"][0]

    # 3. Load pretrained policy
    print(f"Loading policy from {args.policy}...")
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy)
    policy_cfg.pretrained_path = args.policy
    policy_cfg.device = args.device

    # Load dataset metadata for normalization stats
    dataset_id = KNOWN_DATASETS.get(args.policy)
    ds_meta = None
    if dataset_id:
        print(f"Loading dataset stats from {dataset_id}...")
        ds_meta = LeRobotDatasetMetadata(dataset_id)

    if ds_meta:
        policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    else:
        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    print(f"Policy loaded: {policy_cfg.type} ({sum(p.numel() for p in policy.parameters()):,} params)")

    # 4. Create pre/post processors
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg,
    )
    # Try loading from pretrained path first; fall back to creating from scratch
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg, pretrained_path=args.policy,
        )
    except FileNotFoundError:
        logging.info("No processor configs in pretrained model, creating from dataset stats")
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            dataset_stats=ds_meta.stats if ds_meta else None,
        )

    # 5. Run episodes
    all_metrics = []
    video_threads = []

    for ep in range(args.n_episodes):
        seed = args.seed + ep
        print(f"\nEpisode {ep + 1}/{args.n_episodes} (seed={seed})")

        start = time.time()
        frames, metrics = run_episode(
            env=vec_env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seed=seed,
        )
        elapsed = time.time() - start
        metrics["elapsed_s"] = round(elapsed, 1)
        all_metrics.append(metrics)

        status = "SUCCESS" if metrics["success"] else "FAIL"
        print(f"  {status} | reward={metrics['total_reward']:.1f} | "
              f"steps={metrics['steps']} | {elapsed:.1f}s")

        # Save video in background thread
        if frames:
            stacked = np.stack(frames, axis=0)
            video_path = output_dir / f"episode_{ep:03d}.mp4"
            thread = threading.Thread(
                target=write_video,
                args=(str(video_path), stacked, env_cfg.fps),
            )
            thread.start()
            video_threads.append((thread, video_path))

    # Wait for all videos to finish writing
    for thread, path in video_threads:
        thread.join()
        print(f"  Saved: {path}")

    # Close environment
    vec_env.close()

    # Print summary
    n_success = sum(1 for m in all_metrics if m["success"])
    avg_reward = np.mean([m["total_reward"] for m in all_metrics])
    print(f"\n{'='*50}")
    print(f"Results: {n_success}/{args.n_episodes} successful ({n_success/args.n_episodes*100:.0f}%)")
    print(f"Avg reward: {avg_reward:.2f}")
    print(f"Videos saved to: {output_dir.resolve()}")

    # Save metrics
    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"episodes": all_metrics, "summary": {
            "success_rate": n_success / args.n_episodes,
            "avg_reward": float(avg_reward),
            "policy": args.policy,
            "task": args.task,
            "n_episodes": args.n_episodes,
        }}, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
