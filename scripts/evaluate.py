#!/usr/bin/env python
"""Evaluate a trained ACT policy checkpoint in the ALOHA sim environment.

Loads a locally-trained checkpoint (or pretrained HuggingFace model) and runs
it in the gym-aloha MuJoCo environment, saving video + metrics.

Usage:
    # Evaluate a trained checkpoint
    python scripts/evaluate.py --checkpoint outputs/train/act_transfer_cube/last

    # Evaluate pretrained baseline for comparison
    python scripts/evaluate.py --checkpoint lerobot/act_aloha_sim_transfer_cube_human

    # Customize evaluation
    python scripts/evaluate.py --checkpoint outputs/train/act_transfer_cube/last \
        --n-episodes 10 --device mps
"""

import argparse
import json
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

# Map of HuggingFace pretrained models to their training datasets
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


def is_local_checkpoint(path: str) -> bool:
    """Check if path points to a local directory (vs HuggingFace model ID)."""
    return Path(path).exists()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained policy in ALOHA sim")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to local checkpoint dir or HuggingFace model ID",
    )
    parser.add_argument(
        "--task", type=str, default="AlohaTransferCube-v0",
        help="Gymnasium environment task ID",
    )
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, cuda, or mps")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save videos/metrics (default: outputs/eval/<checkpoint_name>)",
    )
    parser.add_argument("--seed", type=int, default=1000, help="Starting random seed")
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        checkpoint_name = Path(args.checkpoint).name if is_local_checkpoint(args.checkpoint) \
            else args.checkpoint.replace("/", "_")
        output_dir = Path("outputs/eval") / checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)

    is_local = is_local_checkpoint(args.checkpoint)
    print(f"Evaluating {'local checkpoint' if is_local else 'pretrained model'}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Task:       {args.task}")
    print(f"  Device:     {args.device}")
    print(f"  Episodes:   {args.n_episodes}")
    print(f"  Output:     {output_dir}")
    print()

    # 1. Configure environment
    env_cfg = AlohaEnv(
        task=args.task,
        fps=50,
        episode_length=400,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )

    # 2. Create vectorized environment
    print("Creating environment...")
    envs = make_env(cfg=env_cfg, n_envs=1, use_async_envs=False)
    vec_env = envs["aloha"][0]

    # 3. Load policy
    print(f"Loading policy from {args.checkpoint}...")
    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    policy_cfg.pretrained_path = args.checkpoint
    policy_cfg.device = args.device

    # Load dataset metadata for normalization stats (if known pretrained model)
    dataset_id = KNOWN_DATASETS.get(args.checkpoint)
    ds_meta = None
    if dataset_id:
        print(f"Loading dataset stats from {dataset_id}...")
        ds_meta = LeRobotDatasetMetadata(dataset_id)

    if ds_meta:
        policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    else:
        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy loaded: {policy_cfg.type} ({n_params:,} params)")

    # 4. Create pre/post processors
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg,
    )
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg, pretrained_path=args.checkpoint,
        )
    except FileNotFoundError:
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

        # Save video in background
        if frames:
            stacked = np.stack(frames, axis=0)
            video_path = output_dir / f"episode_{ep:03d}.mp4"
            thread = threading.Thread(
                target=write_video,
                args=(str(video_path), stacked, env_cfg.fps),
            )
            thread.start()
            video_threads.append((thread, video_path))

    # Wait for videos
    for thread, path in video_threads:
        thread.join()
        print(f"  Saved: {path}")

    vec_env.close()

    # 6. Print summary
    n_success = sum(1 for m in all_metrics if m["success"])
    avg_reward = np.mean([m["total_reward"] for m in all_metrics])
    print(f"\n{'='*60}")
    print(f"Results: {n_success}/{args.n_episodes} successful "
          f"({n_success/args.n_episodes*100:.0f}%)")
    print(f"Avg reward: {avg_reward:.2f}")
    print(f"Videos saved to: {output_dir.resolve()}")

    # 7. Save metrics
    summary = {
        "checkpoint": args.checkpoint,
        "is_local_checkpoint": is_local,
        "task": args.task,
        "n_episodes": args.n_episodes,
        "success_rate": n_success / args.n_episodes,
        "avg_reward": float(avg_reward),
        "device": args.device,
    }
    results = {"summary": summary, "episodes": all_metrics}
    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
