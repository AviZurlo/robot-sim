"""Quantitative metrics for VLA probing experiments.

Includes L2 action error, trajectory DTW, smoothness (jerk),
attention IoU, and perturbation sensitivity.
"""

import numpy as np


def l2_action_error(
    predicted: np.ndarray, ground_truth: np.ndarray
) -> float:
    """Euclidean distance between predicted and ground-truth actions.

    Args:
        predicted: (action_dim,) or (T, action_dim) predicted actions.
        ground_truth: Same shape as predicted.

    Returns:
        Mean L2 error across timesteps.
    """
    predicted = np.atleast_2d(predicted)
    ground_truth = np.atleast_2d(ground_truth)
    return float(np.mean(np.linalg.norm(predicted - ground_truth, axis=-1)))


def trajectory_dtw(
    traj_a: np.ndarray, traj_b: np.ndarray
) -> float:
    """Dynamic Time Warping distance between two trajectories.

    Args:
        traj_a: (T1, D) trajectory.
        traj_b: (T2, D) trajectory.

    Returns:
        DTW distance (float).
    """
    from dtw import dtw as dtw_func

    alignment = dtw_func(traj_a, traj_b, dist_method="euclidean")
    return float(alignment.distance)


def trajectory_jerk(traj: np.ndarray, dt: float = 1.0) -> float:
    """Mean absolute jerk (3rd derivative) as smoothness metric.

    Args:
        traj: (T, D) trajectory positions.
        dt: Time step between samples.

    Returns:
        Mean absolute jerk across all dimensions and timesteps.
    """
    if len(traj) < 4:
        return 0.0
    vel = np.diff(traj, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return float(np.mean(np.abs(jerk)))


def trajectory_spread(trajectories: list[np.ndarray]) -> float:
    """Action variance across multiple random seeds.

    Measures stochasticity of flow matching predictions.

    Args:
        trajectories: List of (T, D) trajectory arrays from different seeds.

    Returns:
        Mean per-timestep standard deviation across seeds.
    """
    if len(trajectories) < 2:
        return 0.0
    # Stack to (n_seeds, T, D) — truncate to shortest
    min_len = min(t.shape[0] for t in trajectories)
    stacked = np.stack([t[:min_len] for t in trajectories])
    return float(np.mean(np.std(stacked, axis=0)))


def attention_iou(
    attention_map: np.ndarray,
    ground_truth_mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Intersection over Union between attention map and object region.

    Args:
        attention_map: (H, W) attention heatmap, values in [0, 1].
        ground_truth_mask: (H, W) binary mask of the ground-truth object region.
        threshold: Threshold to binarize the attention map.

    Returns:
        IoU score in [0, 1].
    """
    # Normalize attention map to [0, 1]
    attn_min, attn_max = attention_map.min(), attention_map.max()
    if attn_max - attn_min > 1e-8:
        attn_norm = (attention_map - attn_min) / (attn_max - attn_min)
    else:
        attn_norm = np.zeros_like(attention_map)

    attn_binary = (attn_norm >= threshold).astype(bool)
    gt_binary = ground_truth_mask.astype(bool)

    intersection = np.logical_and(attn_binary, gt_binary).sum()
    union = np.logical_or(attn_binary, gt_binary).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


def perturbation_sensitivity(
    baseline_actions: np.ndarray,
    perturbed_actions: np.ndarray,
) -> float:
    """L2 delta in predicted actions due to a perturbation.

    Args:
        baseline_actions: (T, D) or (D,) actions from unperturbed input.
        perturbed_actions: Same shape, from perturbed input.

    Returns:
        Mean L2 distance between baseline and perturbed actions.
    """
    return l2_action_error(perturbed_actions, baseline_actions)


def compute_all_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray | None = None,
    reference_traj: np.ndarray | None = None,
    attention_map: np.ndarray | None = None,
    object_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all available metrics for a probe result.

    Returns a dict with available metric values. Skips metrics
    where required inputs are not provided.
    """
    result: dict[str, float] = {}

    if ground_truth is not None:
        result["l2_error"] = l2_action_error(predicted, ground_truth)

    # Trajectory metrics on XYZ portion (first 3 dims)
    pred_xyz = np.atleast_2d(predicted)[:, :3]
    result["trajectory_jerk"] = trajectory_jerk(pred_xyz)

    if reference_traj is not None:
        ref_xyz = np.atleast_2d(reference_traj)[:, :3]
        result["trajectory_dtw"] = trajectory_dtw(pred_xyz, ref_xyz)

    if attention_map is not None and object_mask is not None:
        result["attention_iou"] = attention_iou(attention_map, object_mask)

    return result
