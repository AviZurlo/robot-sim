"""Benchmark training throughput: cached vs uncached, with detailed timing."""

import argparse
import json
import time
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

sys.path.insert(0, str(Path(__file__).parent.parent))
from lerobot_cache import CachedDataset


def make_delta_timestamps(delta_indices, fps):
    if delta_indices is None:
        return [0.0]
    return [i / fps for i in delta_indices]


def benchmark_training(dataset, policy, optimizer, device, num_steps, label, batch_size=8):
    """Run training steps and return detailed timing."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    loader_iter = iter(loader)
    
    step_times = []
    data_times = []
    forward_times = []
    backward_times = []
    losses = []
    
    for step in range(num_steps):
        # Data loading
        t0 = time.time()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        t_data = time.time()
        
        # Forward
        loss, _ = policy.forward(batch)
        t_forward = time.time()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t_backward = time.time()
        
        data_times.append(t_data - t0)
        forward_times.append(t_forward - t_data)
        backward_times.append(t_backward - t_forward)
        step_times.append(t_backward - t0)
        losses.append(loss.item())
        
        if (step + 1) % 10 == 0 or step == 0:
            avg_step = sum(step_times[-10:]) / len(step_times[-10:])
            avg_data = sum(data_times[-10:]) / len(data_times[-10:])
            print(f"  [{label}] step {step+1}/{num_steps} | loss={loss.item():.4f} | "
                  f"step={avg_step:.2f}s (data={avg_data:.3f}s) | "
                  f"{1/avg_step:.2f} steps/s")
    
    return {
        "label": label,
        "num_steps": num_steps,
        "step_times": step_times,
        "data_times": data_times,
        "forward_times": forward_times,
        "backward_times": backward_times,
        "losses": losses,
        "avg_step_time": sum(step_times) / len(step_times),
        "avg_data_time": sum(data_times) / len(data_times),
        "avg_forward_time": sum(forward_times) / len(forward_times),
        "avg_backward_time": sum(backward_times) / len(backward_times),
        "steps_per_sec": len(step_times) / sum(step_times),
        "total_time": sum(step_times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lerobot/aloha_sim_transfer_cube_human")
    parser.add_argument("--cache-dir", default="outputs/cache/aloha")
    parser.add_argument("--cached-steps", type=int, default=500)
    parser.add_argument("--uncached-steps", type=int, default=20, help="Fewer steps for uncached (it's slow)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="outputs/benchmark/training_benchmark.json")
    args = parser.parse_args()
    
    device = torch.device("cpu")
    repo_id = args.dataset
    
    # Load metadata
    ds_meta = LeRobotDatasetMetadata(repo_id)
    features = dataset_to_policy_features(ds_meta.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}
    
    # Delta timestamps
    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    cfg.device = "cpu"
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, ds_meta.fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, ds_meta.fps)
        for k in cfg.image_features
    }
    
    print("=" * 60)
    print("TRAINING THROUGHPUT BENCHMARK")
    print(f"Dataset: {repo_id}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    results = {}
    
    # --- Benchmark 1: UNCACHED (video decode on the fly) ---
    print(f"\n--- Uncached (video decode) — {args.uncached_steps} steps ---")
    ds_uncached = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
    policy1 = ACTPolicy(cfg)
    policy1.train()
    policy1.to(device)
    optimizer1 = torch.optim.AdamW(policy1.parameters(), lr=1e-5)
    
    results["uncached"] = benchmark_training(
        ds_uncached, policy1, optimizer1, device,
        args.uncached_steps, "UNCACHED", args.batch_size
    )
    
    # --- Benchmark 2: CACHED (safetensors) ---
    print(f"\n--- Cached (safetensors) — {args.cached_steps} steps ---")
    ds_cached = CachedDataset(repo_id, cache_dir=args.cache_dir, auto_cache=False,
                               delta_timestamps=delta_timestamps)
    policy2 = ACTPolicy(cfg)
    policy2.train()
    policy2.to(device)
    optimizer2 = torch.optim.AdamW(policy2.parameters(), lr=1e-5)
    
    results["cached"] = benchmark_training(
        ds_cached, policy2, optimizer2, device,
        args.cached_steps, "CACHED", args.batch_size
    )
    
    # --- Summary ---
    speedup = results["uncached"]["avg_step_time"] / results["cached"]["avg_step_time"]
    data_speedup = results["uncached"]["avg_data_time"] / results["cached"]["avg_data_time"]
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'Uncached':>12} {'Cached':>12} {'Speedup':>10}")
    print("-" * 64)
    print(f"{'Avg step time':<30} {results['uncached']['avg_step_time']:>11.2f}s {results['cached']['avg_step_time']:>11.2f}s {speedup:>9.1f}x")
    print(f"{'Avg data load time':<30} {results['uncached']['avg_data_time']:>11.3f}s {results['cached']['avg_data_time']:>11.3f}s {data_speedup:>9.1f}x")
    print(f"{'Avg forward time':<30} {results['uncached']['avg_forward_time']:>11.3f}s {results['cached']['avg_forward_time']:>11.3f}s {'—':>10}")
    print(f"{'Avg backward time':<30} {results['uncached']['avg_backward_time']:>11.3f}s {results['cached']['avg_backward_time']:>11.3f}s {'—':>10}")
    print(f"{'Steps/sec':<30} {results['uncached']['steps_per_sec']:>12.3f} {results['cached']['steps_per_sec']:>12.3f} {speedup:>9.1f}x")
    print(f"{'Total time':<30} {results['uncached']['total_time']:>11.1f}s {results['cached']['total_time']:>11.1f}s {'—':>10}")
    
    # Data loading as % of step time
    uncached_data_pct = results["uncached"]["avg_data_time"] / results["uncached"]["avg_step_time"] * 100
    cached_data_pct = results["cached"]["avg_data_time"] / results["cached"]["avg_step_time"] * 100
    print(f"\n{'Data loading % of step time':<30} {uncached_data_pct:>11.1f}% {cached_data_pct:>11.1f}%")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable
    save_data = {
        "metadata": {
            "dataset": repo_id,
            "device": str(device),
            "batch_size": args.batch_size,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "uncached": {k: v for k, v in results["uncached"].items()},
        "cached": {k: v for k, v in results["cached"].items()},
        "speedup": {
            "overall": speedup,
            "data_loading": data_speedup,
        }
    }
    output_path.write_text(json.dumps(save_data, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
