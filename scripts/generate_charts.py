"""Generate comparison charts for lerobot-cache performance."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.facecolor'] = 'white'


def load_benchmark_data(benchmark_path, training_path=None):
    """Load data loading benchmark and optional training benchmark."""
    data = {}
    
    if benchmark_path and Path(benchmark_path).exists():
        data["loading"] = json.loads(Path(benchmark_path).read_text())
    
    if training_path and Path(training_path).exists():
        data["training"] = json.loads(Path(training_path).read_text())
    
    return data


def chart_data_loading_speedup(output_dir):
    """Bar chart: data loading FPS by backend."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Data from our benchmark
    backends = ['Video\n(sequential)', 'Video\n(random)', 'Safetensors\n(sequential)', 'Safetensors\n(random)', 'NumPy mmap\n(sequential)', 'NumPy mmap\n(random)']
    fps = [859, 631, 15279, 16429, 2046, 2215]
    ms_per_frame = [1.16, 1.58, 0.07, 0.06, 0.49, 0.45]
    colors = ['#e74c3c', '#c0392b', '#2ecc71', '#27ae60', '#3498db', '#2980b9']
    
    # Chart 1: FPS comparison (log scale)
    bars1 = ax1.bar(backends, fps, color=colors, edgecolor='white', linewidth=1.5)
    ax1.set_ylabel('Frames per Second (FPS)', fontweight='bold')
    ax1.set_title('Data Loading Speed by Backend', fontweight='bold', fontsize=14)
    ax1.set_yscale('log')
    ax1.set_ylim(100, 30000)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, fps):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.1,
                f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Chart 2: Speedup vs video baseline
    speedups = [fps_val / 631 for fps_val in fps]  # vs video random baseline
    bars2 = ax2.bar(backends, speedups, color=colors, edgecolor='white', linewidth=1.5)
    ax2.set_ylabel('Speedup vs Video (random)', fontweight='bold')
    ax2.set_title('Speedup Factor', fontweight='bold', fontsize=14)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline (video random)')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    # Add value labels
    for bar, val in zip(bars2, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val:.1f}x', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    path = Path(output_dir) / 'data_loading_speedup.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def chart_training_comparison(training_path, output_dir):
    """Charts from training benchmark data."""
    data = json.loads(Path(training_path).read_text())
    
    cached = data["cached"]
    uncached = data["uncached"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Chart 1: Loss curve (cached only — has enough steps)
    ax = axes[0, 0]
    steps = list(range(1, len(cached["losses"]) + 1))
    ax.plot(steps, cached["losses"], color='#2ecc71', linewidth=2, label='Cached (safetensors)')
    if len(uncached["losses"]) > 1:
        uncached_steps = list(range(1, len(uncached["losses"]) + 1))
        ax.plot(uncached_steps, uncached["losses"], color='#e74c3c', linewidth=2, linestyle='--', label='Uncached (video)')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curve (ACT, ALOHA)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    # Chart 2: Step time breakdown (stacked bar)
    ax = axes[0, 1]
    labels = ['Uncached\n(video decode)', 'Cached\n(safetensors)']
    data_times = [uncached["avg_data_time"], cached["avg_data_time"]]
    forward_times = [uncached["avg_forward_time"], cached["avg_forward_time"]]
    backward_times = [uncached["avg_backward_time"], cached["avg_backward_time"]]
    
    x = np.arange(len(labels))
    width = 0.5
    
    ax.bar(x, data_times, width, label='Data Loading', color='#e74c3c')
    ax.bar(x, forward_times, width, bottom=data_times, label='Forward Pass', color='#3498db')
    ax.bar(x, backward_times, width, 
           bottom=[d+f for d, f in zip(data_times, forward_times)], 
           label='Backward Pass', color='#f39c12')
    
    ax.set_ylabel('Time per Step (seconds)')
    ax.set_title('Step Time Breakdown', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Chart 3: Data loading time comparison (the key insight)
    ax = axes[1, 0]
    data_load_comparison = {
        'Video decode\n(uncached)': uncached["avg_data_time"] * 1000,
        'Safetensors\n(cached)': cached["avg_data_time"] * 1000,
    }
    colors_dl = ['#e74c3c', '#2ecc71']
    bars = ax.bar(data_load_comparison.keys(), data_load_comparison.values(), 
                  color=colors_dl, edgecolor='white', linewidth=1.5, width=0.5)
    ax.set_ylabel('Data Loading Time (ms per step)')
    ax.set_title('Data Loading Time: Cached vs Uncached', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    speedup = uncached["avg_data_time"] / cached["avg_data_time"]
    ax.text(0.5, 0.85, f'{speedup:.1f}x faster data loading', 
            transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold',
            color='#27ae60',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3', edgecolor='#27ae60'))
    
    for bar, val in zip(bars, data_load_comparison.values()):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f} ms', ha='center', va='bottom', fontweight='bold')
    
    # Chart 4: Throughput comparison
    ax = axes[1, 1]
    throughputs = {
        'Uncached': uncached["steps_per_sec"],
        'Cached': cached["steps_per_sec"],
    }
    colors_tp = ['#e74c3c', '#2ecc71']
    bars = ax.bar(throughputs.keys(), throughputs.values(), 
                  color=colors_tp, edgecolor='white', linewidth=1.5, width=0.5)
    ax.set_ylabel('Steps per Second')
    ax.set_title('Training Throughput', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, throughputs.values()):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('lerobot-cache: ALOHA ACT Training Performance\n(Mac mini M-series, 32GB RAM, CPU)',
                 fontweight='bold', fontsize=15, y=1.02)
    plt.tight_layout()
    path = Path(output_dir) / 'training_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def chart_headline(output_dir):
    """Single headline chart: the big speedup number for data loading."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Frame\nLoading', 'Full Dataset\nPre-decode', 'Training\nThroughput']
    uncached = [631, 0, 0]  # FPS, N/A, steps/s
    cached = [16429, 107, 0]  # FPS, frames/s, steps/s
    
    # Simple comparison: data loading FPS
    labels = ['Video Decode\n(default)', 'Safetensors Cache\n(lerobot-cache)']
    values = [631, 16429]
    colors = ['#e74c3c', '#2ecc71']
    
    bars = ax.barh(labels, values, color=colors, height=0.5, edgecolor='white', linewidth=2)
    ax.set_xlabel('Frames per Second (FPS)', fontweight='bold', fontsize=13)
    ax.set_title('lerobot-cache: Data Loading Performance\nALOHA Dataset (20K frames, 480×640, float32)', 
                 fontweight='bold', fontsize=15)
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val * 1.15, bar.get_y() + bar.get_height()/2.,
                f'{val:,} FPS', ha='left', va='center', fontweight='bold', fontsize=14)
    
    # Add speedup annotation
    ax.text(0.65, 0.5, '26x\nfaster', transform=ax.transAxes,
            fontsize=36, fontweight='bold', color='#27ae60',
            ha='center', va='center', alpha=0.8)
    
    plt.tight_layout()
    path = Path(output_dir) / 'headline_speedup.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


def main():
    output_dir = Path("outputs/benchmark/charts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_path = "outputs/benchmark/training_benchmark.json"
    
    # Always generate the data loading chart (hardcoded from benchmark results)
    print("Generating data loading speedup chart...")
    chart_data_loading_speedup(output_dir)
    
    print("Generating headline chart...")
    chart_headline(output_dir)
    
    # Generate training charts if data exists
    if Path(training_path).exists():
        print("Generating training comparison charts...")
        chart_training_comparison(training_path, output_dir)
    else:
        print(f"Training benchmark not found at {training_path} — skipping training charts")
    
    print(f"\nAll charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
