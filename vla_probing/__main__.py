"""Entry point for running individual probes or the full suite.

Usage:
    python -m vla_probing                          # run all probes
    python -m vla_probing baseline                 # run a specific probe
    python -m vla_probing baseline --wandb         # with W&B logging
    python -m vla_probing --model xvla --device mps
"""

import sys


def main() -> None:
    # If a probe name is given as first arg, dispatch to that probe
    probe_modules = {
        "baseline": "vla_probing.probes.baseline",
        "spatial_symmetry": "vla_probing.probes.spatial_symmetry",
        "camera_sensitivity": "vla_probing.probes.camera_sensitivity",
        "view_ablation": "vla_probing.probes.view_ablation",
        "counterfactual": "vla_probing.probes.counterfactual",
        "null_action": "vla_probing.probes.null_action",
        "attention": "vla_probing.probes.attention",
        "perturbation": "vla_probing.probes.perturbation",
        "vlm_query": "vla_probing.probes.vlm_query",
    }

    if len(sys.argv) > 1 and sys.argv[1] in probe_modules:
        probe_name = sys.argv.pop(1)
        import importlib

        mod = importlib.import_module(probe_modules[probe_name])
        mod.main()
    else:
        from vla_probing.run_all import main as run_all

        run_all()


if __name__ == "__main__":
    main()
