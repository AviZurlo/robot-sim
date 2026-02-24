#!/usr/bin/env python
"""Run a full experiment cycle: train → evaluate → generate plots.

Chains together train.py, evaluate.py, and visualize.py into a single command.
Optionally evaluates the pretrained baseline for comparison.

Usage:
    # Quick experiment (500 steps, 5 eval episodes)
    python scripts/run_experiment.py --steps 500 --n-episodes 5 --device mps

    # Full experiment (5000 steps, 10 eval episodes, with baseline comparison)
    python scripts/run_experiment.py --steps 5000 --n-episodes 10 --device mps --baseline

    # Resume training from checkpoint, then evaluate and plot
    python scripts/run_experiment.py --steps 5000 --device mps --resume
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: list[str], label: str) -> int:
    """Run a command, streaming output. Returns exit code."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run full experiment: train → evaluate → visualize")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--n-episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, mps, or cuda")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="outputs/train/act_transfer_cube",
                        help="Training output directory")
    parser.add_argument("--baseline", action="store_true",
                        help="Also evaluate the pretrained baseline for comparison")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only evaluate + plot")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation, only plot")
    args = parser.parse_args()

    t_total = time.time()
    python = sys.executable
    train_dir = args.output_dir
    checkpoint = f"{train_dir}/last"

    # Resolve eval output dirs
    eval_dir = "outputs/eval/last"
    baseline_dir = "outputs/eval/lerobot_act_aloha_sim_transfer_cube_human"

    # Step 1: Train
    if not args.skip_train:
        cmd = [
            python, "scripts/train.py",
            "--steps", str(args.steps),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--device", args.device,
            "--output-dir", train_dir,
        ]
        if args.resume:
            cmd.append("--resume")
        rc = run_cmd(cmd, f"STEP 1/3: Training ACT policy ({args.steps} steps)")
        if rc != 0:
            print(f"\nTraining failed (exit code {rc}). Aborting.")
            sys.exit(rc)

    # Step 2: Evaluate
    if not args.skip_eval:
        # Evaluate trained model
        cmd = [
            python, "scripts/evaluate.py",
            "--checkpoint", checkpoint,
            "--n-episodes", str(args.n_episodes),
            "--device", args.device,
            "--output-dir", eval_dir,
        ]
        rc = run_cmd(cmd, f"STEP 2/3: Evaluating trained model ({args.n_episodes} episodes)")
        if rc != 0:
            print(f"\nEvaluation failed (exit code {rc}). Continuing to plots...")

        # Evaluate pretrained baseline
        if args.baseline:
            cmd = [
                python, "scripts/evaluate.py",
                "--checkpoint", "lerobot/act_aloha_sim_transfer_cube_human",
                "--n-episodes", str(args.n_episodes),
                "--device", args.device,
                "--output-dir", baseline_dir,
            ]
            rc = run_cmd(cmd, "STEP 2b/3: Evaluating pretrained baseline")
            if rc != 0:
                print(f"\nBaseline evaluation failed (exit code {rc}). Continuing to plots...")

    # Step 3: Generate plots
    cmd = [
        python, "scripts/visualize.py",
        "--train-log", f"{train_dir}/loss_history.json",
        "--eval-metrics", f"{eval_dir}/eval_metrics.json",
        "--baseline-metrics", f"{baseline_dir}/eval_metrics.json",
        "--output-dir", "outputs/plots",
    ]
    rc = run_cmd(cmd, "STEP 3/3: Generating visualization plots")
    if rc != 0:
        print(f"\nPlot generation failed (exit code {rc}).")
        sys.exit(rc)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT COMPLETE ({elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"  Training:    {train_dir}")
    print(f"  Evaluation:  {eval_dir}")
    print(f"  Plots:       outputs/plots/")
    print()


if __name__ == "__main__":
    main()
