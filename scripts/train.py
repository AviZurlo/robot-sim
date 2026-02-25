#!/usr/bin/env python
"""Train a policy from scratch on robot simulation tasks.

Supports two tasks:
  - transfer_cube: ACT policy on ALOHA bimanual cube transfer (51M params, vision+state, slow)
  - pusht: Diffusion Policy on PushT 2D pushing task (4.4M params, state-only, fast)

Usage:
    # Fast PushT training (state-only, ~3 min on CPU for 1000 steps)
    python scripts/train.py --task pusht --steps 1000

    # ALOHA transfer cube (vision, slow)
    python scripts/train.py --task transfer_cube --steps 5000 --device mps

    # Resume from checkpoint
    python scripts/train.py --task pusht --steps 2000 --resume
"""

import argparse
import json
import time
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors


# Task configurations
TASKS = {
    "transfer_cube": {
        "dataset": "lerobot/aloha_sim_transfer_cube_human",
        "policy_type": "act",
        "output_dir": "outputs/train/act_transfer_cube",
        "batch_size": 8,
        "lr": 1e-5,
    },
    "pusht": {
        "dataset": "lerobot/pusht",
        "policy_type": "diffusion",
        "output_dir": "outputs/train/diffusion_pusht",
        "batch_size": 64,
        "lr": 1e-4,
    },
}


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def setup_act(ds_meta, features, args):
    """Set up ACT policy for transfer_cube task."""
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    cfg.optimizer_lr = args.lr
    cfg.optimizer_lr_backbone = args.lr

    # Delta timestamps for ACT
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, ds_meta.fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, ds_meta.fps)
        for k in cfg.image_features
    }

    return cfg, ACTPolicy, delta_timestamps


def setup_diffusion_pusht(ds_meta, features, args):
    """Set up Diffusion Policy for PushT task (state-only, no images)."""
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}

    # State-only: use observation.state as STATE, duplicate as ENV to satisfy
    # DiffusionConfig's requirement for at least one image or environment state.
    state_shape = features["observation.state"].shape
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=state_shape),
        "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=state_shape),
    }

    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        # Small network for fast CPU training
        down_dims=(64, 128, 256),
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        num_inference_steps=10,
        device=args.device,
    )
    cfg.optimizer_lr = args.lr

    # Delta timestamps for Diffusion Policy (n_obs_steps=2 at 10fps)
    obs_ts = [i / ds_meta.fps for i in range(-(cfg.n_obs_steps - 1), 1)]
    action_ts = [i / ds_meta.fps for i in range(-(cfg.n_obs_steps - 1), cfg.horizon - cfg.n_obs_steps + 1)]

    delta_timestamps = {
        "observation.state": obs_ts,
        "action": action_ts,
    }

    return cfg, DiffusionPolicy, delta_timestamps


def main():
    parser = argparse.ArgumentParser(description="Train a policy on robot simulation tasks")
    parser.add_argument("--task", type=str, default="transfer_cube",
                        choices=list(TASKS.keys()),
                        help="Task: 'transfer_cube' (ACT, slow) or 'pusht' (Diffusion, fast)")
    parser.add_argument("--steps", type=int, default=None, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, mps, or cuda")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--log-freq", type=int, default=50, help="Log every N steps")
    parser.add_argument("--save-freq", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--use-cache", action="store_true",
                        help="Use lerobot-cache for faster frame loading (pre-decodes video to safetensors)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Cache directory (default: ~/.cache/lerobot-cache/<dataset>)")
    args = parser.parse_args()

    # Apply task defaults for unset args
    task_cfg = TASKS[args.task]
    if args.steps is None:
        args.steps = 1000 if args.task == "pusht" else 5000
    if args.batch_size is None:
        args.batch_size = task_cfg["batch_size"]
    if args.lr is None:
        args.lr = task_cfg["lr"]
    if args.output_dir is None:
        args.output_dir = task_cfg["output_dir"]
    dataset_id = task_cfg["dataset"]

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {task_cfg['policy_type'].upper()} policy — task: {args.task}")
    print(f"  Dataset:    {dataset_id}")
    print(f"  Steps:      {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Device:     {device}")
    print(f"  Output:     {output_dir}")
    print()

    # 1. Load dataset metadata
    print("Loading dataset metadata...")
    ds_meta = LeRobotDatasetMetadata(dataset_id)
    features = dataset_to_policy_features(ds_meta.features)

    print(f"  Episodes: {ds_meta.total_episodes}")
    print(f"  Frames:   {ds_meta.total_frames}")
    print(f"  FPS:      {ds_meta.fps}")
    print()

    # 2. Create policy config based on task
    if args.task == "pusht":
        cfg, PolicyClass, delta_timestamps = setup_diffusion_pusht(ds_meta, features, args)
    else:
        cfg, PolicyClass, delta_timestamps = setup_act(ds_meta, features, args)

    print(f"  Input features:  {list(cfg.input_features.keys())}")
    print(f"  Output features: {list(cfg.output_features.keys())}")
    print()

    # 3. Try to resume from checkpoint
    start_step = 0
    last_checkpoint = output_dir / "last" / "model.safetensors"
    if args.resume and last_checkpoint.exists():
        print(f"Resuming from {output_dir / 'last'}...")
        policy = PolicyClass.from_pretrained(output_dir / "last", config=cfg)
        state_path = output_dir / "last" / "training_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            start_step = state.get("step", 0)
            print(f"  Resuming from step {start_step}")
    else:
        print(f"Creating {task_cfg['policy_type'].upper()} policy from scratch...")
        policy = PolicyClass(cfg)

    policy.train()
    policy.to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} ({n_trainable:,} trainable)")
    print()

    # 4. Create pre/post processors (normalization)
    # For pusht state-only, build stats that include the duplicated env_state key
    dataset_stats = ds_meta.stats
    if args.task == "pusht":
        dataset_stats = dict(dataset_stats)
        dataset_stats["observation.environment_state"] = dataset_stats["observation.state"]

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_stats)

    # 5. Load dataset
    print("Loading dataset...")
    if args.use_cache:
        from lerobot_cache import CachedDataset
        dataset = CachedDataset(
            dataset_id,
            cache_dir=args.cache_dir,
            auto_cache=True,
            delta_timestamps=delta_timestamps,
        )
        info = dataset.cache_info()
        print(f"  Cache: {info['cache_dir']}")
        print(f"  Cached frames: {info['cached_frames']}")
        print(f"  Disk: {info.get('disk_size_gb', 0):.2f} GB" if 'disk_size_gb' in info else "")
    else:
        dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)
    print(f"  Loaded {len(dataset)} samples")
    print()

    # 6. Create optimizer
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    if args.resume and (output_dir / "last" / "optimizer.pt").exists():
        opt_state = torch.load(output_dir / "last" / "optimizer.pt", map_location=device,
                               weights_only=True)
        optimizer.load_state_dict(opt_state)
        print("  Resumed optimizer state")

    # 7. Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # 8. Training loop
    is_pusht = args.task == "pusht"
    print(f"Starting training from step {start_step}...")
    print("=" * 60)

    loss_history = []
    step = start_step
    t_start = time.time()
    done = False

    while not done:
        for batch in dataloader:
            if step >= args.steps:
                done = True
                break

            # For pusht: duplicate observation.state as observation.environment_state
            if is_pusht:
                batch["observation.environment_state"] = batch["observation.state"].clone()

            batch = preprocessor(batch)
            loss, output_dict = policy.forward(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()

            loss_val = loss.item()
            loss_history.append({"step": step, "loss": loss_val})

            if output_dict:
                loss_history[-1].update({
                    k: v for k, v in output_dict.items() if isinstance(v, (int, float))
                })

            if step % args.log_freq == 0:
                elapsed = time.time() - t_start
                steps_done = step - start_step + 1
                steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
                eta_s = (args.steps - step) / steps_per_sec if steps_per_sec > 0 else 0

                extra = ""
                if output_dict:
                    parts = []
                    for k, v in output_dict.items():
                        if isinstance(v, (int, float)):
                            parts.append(f"{k}={v:.4f}")
                    if parts:
                        extra = " | " + " | ".join(parts)

                print(f"  step {step:>6d}/{args.steps} | loss={loss_val:.4f}{extra} | "
                      f"{steps_per_sec:.1f} steps/s | ETA {eta_s/60:.1f}m")

            # Save checkpoint
            if step > 0 and step % args.save_freq == 0:
                _save_checkpoint(output_dir, step, policy, preprocessor, postprocessor,
                                 optimizer, loss_history)

            step += 1

    # 9. Save final checkpoint
    elapsed_total = time.time() - t_start
    _save_checkpoint(output_dir, step, policy, preprocessor, postprocessor,
                     optimizer, loss_history)

    # 10. Print summary
    print("=" * 60)
    print(f"Training complete!")
    print(f"  Steps:        {step}")
    print(f"  Time:         {elapsed_total/60:.1f} min")
    if loss_history:
        first_losses = [h["loss"] for h in loss_history[:10]]
        last_losses = [h["loss"] for h in loss_history[-10:]]
        print(f"  Initial loss: {sum(first_losses)/len(first_losses):.4f}")
        print(f"  Final loss:   {sum(last_losses)/len(last_losses):.4f}")
    print(f"  Checkpoints:  {output_dir}")
    print(f"  Loss log:     {output_dir / 'loss_history.json'}")
    print()
    print(f"To evaluate: python scripts/evaluate.py --checkpoint {output_dir / 'last'} --task {args.task}")


def _save_checkpoint(
    output_dir: Path,
    step: int,
    policy,
    preprocessor,
    postprocessor,
    optimizer,
    loss_history: list,
) -> None:
    """Save policy checkpoint and training state."""
    step_dir = output_dir / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(step_dir)
    preprocessor.save_pretrained(step_dir)
    postprocessor.save_pretrained(step_dir)

    training_state = {"step": step}
    (step_dir / "training_state.json").write_text(json.dumps(training_state))
    torch.save(optimizer.state_dict(), step_dir / "optimizer.pt")

    # Update "last" symlink
    last_dir = output_dir / "last"
    if last_dir.is_symlink():
        last_dir.unlink()
    elif last_dir.exists():
        import shutil
        shutil.rmtree(last_dir)
    last_dir.symlink_to(step_dir.name)

    # Save loss history
    with open(output_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f)

    print(f"  [checkpoint] Saved step {step} -> {step_dir}")


if __name__ == "__main__":
    main()
