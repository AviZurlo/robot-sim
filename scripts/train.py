#!/usr/bin/env python
"""Train an ACT policy from scratch on the ALOHA sim transfer cube task.

Downloads human demonstration data from HuggingFace Hub and trains an ACT
(Action Chunking with Transformers) policy using LeRobot's training pipeline.

Usage:
    # Quick test (100 steps, ~2 min on MPS)
    python scripts/train.py --steps 100 --device mps

    # Short training run (~10 min on MPS)
    python scripts/train.py --steps 1000 --device mps

    # Full training run (~30 min on MPS, enough to see learning)
    python scripts/train.py --steps 5000 --device mps

    # Resume from checkpoint
    python scripts/train.py --steps 5000 --device mps --resume
"""

import argparse
import json
import time
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def main():
    parser = argparse.ArgumentParser(description="Train ACT policy on ALOHA sim transfer cube")
    parser.add_argument("--dataset", type=str, default=DATASET_ID, help="HuggingFace dataset ID")
    parser.add_argument("--steps", type=int, default=5000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, mps, or cuda")
    parser.add_argument("--output-dir", type=str, default="outputs/train/act_transfer_cube",
                        help="Directory to save checkpoints")
    parser.add_argument("--log-freq", type=int, default=50, help="Log every N steps")
    parser.add_argument("--save-freq", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training ACT policy from scratch")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Steps:      {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Device:     {device}")
    print(f"  Output:     {output_dir}")
    print()

    # 1. Load dataset metadata (features, stats, fps)
    print("Loading dataset metadata...")
    ds_meta = LeRobotDatasetMetadata(args.dataset)
    features = dataset_to_policy_features(ds_meta.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    print(f"  Episodes: {ds_meta.total_episodes}")
    print(f"  Frames:   {ds_meta.total_frames}")
    print(f"  FPS:      {ds_meta.fps}")
    print(f"  Input features:  {list(input_features.keys())}")
    print(f"  Output features: {list(output_features.keys())}")
    print()

    # 2. Create policy config and model
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
    )
    # Override LR if specified
    cfg.optimizer_lr = args.lr
    cfg.optimizer_lr_backbone = args.lr

    # Try to resume from checkpoint
    start_step = 0
    last_checkpoint = output_dir / "last" / "model.safetensors"
    if args.resume and last_checkpoint.exists():
        print(f"Resuming from {output_dir / 'last'}...")
        policy = ACTPolicy.from_pretrained(output_dir / "last", config=cfg)
        # Load training state
        state_path = output_dir / "last" / "training_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            start_step = state.get("step", 0)
            print(f"  Resuming from step {start_step}")
    else:
        print("Creating ACT policy from scratch...")
        policy = ACTPolicy(cfg)

    policy.train()
    policy.to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} ({n_trainable:,} trainable)")
    print()

    # 3. Create pre/post processors (normalization)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=ds_meta.stats)

    # 4. Set up delta timestamps for ACT
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, ds_meta.fps),
    }
    # Add image features
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, ds_meta.fps)
        for k in cfg.image_features
    }

    # 5. Load dataset
    print("Loading dataset...")
    dataset = LeRobotDataset(args.dataset, delta_timestamps=delta_timestamps)
    print(f"  Loaded {len(dataset)} samples")
    print()

    # 6. Create optimizer using policy preset
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    # Resume optimizer state if available
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

            batch = preprocessor(batch)
            loss, output_dict = policy.forward(batch)
            loss.backward()
            # Gradient clipping (ACT preset uses 10.0)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()

            loss_val = loss.item()
            loss_history.append({"step": step, "loss": loss_val})

            # Add component losses if available
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
    print(f"To evaluate: python scripts/evaluate.py --checkpoint {output_dir / 'last'}")


def _save_checkpoint(
    output_dir: Path,
    step: int,
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    optimizer,
    loss_history: list,
) -> None:
    """Save policy checkpoint and training state."""
    # Save step-specific checkpoint
    step_dir = output_dir / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(step_dir)
    preprocessor.save_pretrained(step_dir)
    postprocessor.save_pretrained(step_dir)

    # Save training state
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
