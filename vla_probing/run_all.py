"""Run all VLA probes and aggregate results.

Usage:
    python -m vla_probing.run_all --model xvla --device mps
    python -m vla_probing.run_all --model xvla --wandb  # with W&B logging
"""

import argparse
import json
import time
from pathlib import Path

from vla_probing.probes.base import common_args, make_adapter
from vla_probing.probes.baseline import BaselineProbe
from vla_probing.probes.spatial_symmetry import SpatialSymmetryProbe
from vla_probing.probes.camera_sensitivity import CameraSensitivityProbe
from vla_probing.probes.view_ablation import ViewAblationProbe
from vla_probing.probes.counterfactual import CounterfactualProbe
from vla_probing.probes.null_action import NullActionProbe
from vla_probing.probes.attention import AttentionProbe
from vla_probing.probes.perturbation import PerturbationProbe
from vla_probing.scene import WidowXScene
from vla_probing.tracking import ExperimentTracker


ALL_PROBES = [
    BaselineProbe,
    SpatialSymmetryProbe,
    CameraSensitivityProbe,
    ViewAblationProbe,
    CounterfactualProbe,
    NullActionProbe,
    AttentionProbe,
    PerturbationProbe,
]


def main() -> None:
    parser = common_args("Run all VLA probes")
    parser.add_argument(
        "--probes",
        nargs="*",
        help="Specific probes to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/probes",
        help="Directory for output artifacts",
    )
    args = parser.parse_args()

    # Set up
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model} on {args.device}...")
    adapter = make_adapter(args.model, args.device)
    scene = WidowXScene()
    tracker = ExperimentTracker(enabled=args.wandb)

    if args.wandb:
        tracker.init_run(
            name=f"all_probes_{args.model}",
            config=vars(args),
            tags=[args.model, "full_suite"],
        )

    # Filter probes if specified
    probes_to_run = ALL_PROBES
    if args.probes:
        probe_names = set(args.probes)
        probes_to_run = [p for p in ALL_PROBES if p.name in probe_names]
        if not probes_to_run:
            print(f"No matching probes. Available: {[p.name for p in ALL_PROBES]}")
            return

    # Run probes
    all_results = {}
    for probe_cls in probes_to_run:
        probe = probe_cls(adapter, scene, tracker)
        print(f"\n{'='*60}")
        print(f"Running probe: {probe.name} — {probe.description}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            result = probe.run(seed=args.seed, n_seeds=args.n_seeds)
            elapsed = time.time() - t0

            all_results[probe.name] = {
                "metrics": result.metrics,
                "elapsed_s": elapsed,
                "variant": result.probe_variant,
            }

            print(f"\nResults ({elapsed:.1f}s):")
            for k, v in result.metrics.items():
                print(f"  {k}: {v:.6f}")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"\nFAILED ({elapsed:.1f}s): {e}")
            all_results[probe.name] = {"error": str(e), "elapsed_s": elapsed}
            import traceback
            traceback.print_exc()

    # Save results summary
    summary_path = output_dir / f"probe_results_{args.model}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nResults saved to {summary_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY — {args.model}")
    print(f"{'='*60}")
    for name, data in all_results.items():
        status = "OK" if "metrics" in data else "FAILED"
        elapsed = data.get("elapsed_s", 0)
        print(f"  {name:25s} [{status}] ({elapsed:.1f}s)")

    tracker.finish()
    scene.close()


if __name__ == "__main__":
    main()
