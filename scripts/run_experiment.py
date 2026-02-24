#!/usr/bin/env python
"""Run a full experiment cycle: train → evaluate → generate plots.

Chains together train.py, evaluate.py, and visualize.py into a single command.
Optionally evaluates the pretrained baseline for comparison.

Usage:
    # Quick PushT experiment (fast, state-only)
    python scripts/run_experiment.py --task pusht --steps 1000 --n-episodes 5

    # ALOHA experiment (slow, vision)
    python scripts/run_experiment.py --task transfer_cube --steps 5000 --n-episodes 10 --device mps

    # With pretrained baseline comparison
    python scripts/run_experiment.py --task transfer_cube --steps 5000 --device mps --baseline
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# Task defaults
TASK_DEFAULTS = {
    "transfer_cube": {
        "output_dir": "outputs/train/act_transfer_cube",
        "steps": 5000,
    },
    "pusht": {
        "output_dir": "outputs/train/diffusion_pusht",
        "steps": 1000,
    },
}


def run_cmd(cmd: list[str], label: str) -> int:
    """Run a command, streaming output. Returns exit code."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run full experiment: train → evaluate → visualize")
    parser.add_argument("--task", type=str, default="transfer_cube",
                        choices=list(TASK_DEFAULTS.keys()),
                        help="Task: 'transfer_cube' or 'pusht'")
    parser.add_argument("--steps", type=int, default=None, help="Training steps")
    parser.add_argument("--n-episodes", type=int, default=10, help="Evaluation episodes")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu, mps, or cuda")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default=None, help="Training output directory")
    parser.add_argument("--baseline", action="store_true",
                        help="Also evaluate the pretrained baseline for comparison")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, only evaluate + plot")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation, only plot")
    args = parser.parse_args()

    task_cfg = TASK_DEFAULTS[args.task]
    if args.steps is None:
        args.steps = task_cfg["steps"]
    if args.output_dir is None:
        args.output_dir = task_cfg["output_dir"]

    t_total = time.time()
    python = sys.executable
    train_dir = args.output_dir
    checkpoint = f"{train_dir}/last"

    # Resolve eval output dirs
    eval_dir = f"outputs/eval/{Path(train_dir).name}"
    baseline_dir = "outputs/eval/lerobot_act_aloha_sim_transfer_cube_human"

    # Step 1: Train
    if not args.skip_train:
        cmd = [
            python, "scripts/train.py",
            "--task", args.task,
            "--steps", str(args.steps),
            "--device", args.device,
            "--output-dir", train_dir,
        ]
        if args.batch_size is not None:
            cmd += ["--batch-size", str(args.batch_size)]
        if args.lr is not None:
            cmd += ["--lr", str(args.lr)]
        if args.resume:
            cmd.append("--resume")
        rc = run_cmd(cmd, f"STEP 1/3: Training ({args.task}, {args.steps} steps)")
        if rc != 0:
            print(f"\nTraining failed (exit code {rc}). Aborting.")
            sys.exit(rc)

    # Step 2: Evaluate
    if not args.skip_eval:
        cmd = [
            python, "scripts/evaluate.py",
            "--checkpoint", checkpoint,
            "--task", args.task,
            "--n-episodes", str(args.n_episodes),
            "--device", args.device,
            "--output-dir", eval_dir,
        ]
        rc = run_cmd(cmd, f"STEP 2/3: Evaluating trained model ({args.n_episodes} episodes)")
        if rc != 0:
            print(f"\nEvaluation failed (exit code {rc}). Continuing to plots...")

        # Evaluate pretrained baseline (only for transfer_cube)
        if args.baseline and args.task == "transfer_cube":
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
