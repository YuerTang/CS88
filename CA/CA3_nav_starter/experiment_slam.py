#!/usr/bin/env python3
"""Part A: EKF-SLAM experiments.

A1 - Two trajectory designs (greedy forward vs looping revisit).
A2 - Noise sensitivity analysis on process and measurement noise.

Usage:
    python experiment_slam.py
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse

from environment import Environment, RobotSimulator
from slam import EKFSLAM, get_covariance_ellipse

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "partA")
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_STEPS = 500
SEED = 42


# ---------------------------------------------------------------------------
# Control functions
# ---------------------------------------------------------------------------


def control_greedy_forward(t: int) -> tuple[float, float]:
    """Path A - L-shaped sweep, each landmark seen roughly once."""
    if t < 150:
        return (1.0, 0.0)  # straight east
    elif t < 170:
        return (0.3, 0.78)  # turn left ~90 deg
    elif t < 320:
        return (1.0, 0.0)  # straight north
    elif t < 340:
        return (0.3, 0.78)  # turn left ~90 deg
    else:
        return (0.8, 0.0)  # straight west


def control_looping_revisit(t: int) -> tuple[float, float]:
    """Path B - Two orbits so landmarks are observed multiple times."""
    if t < 50:
        return (1.0, 0.0)  # straight NE toward cluster
    elif t < 100:
        return (1.0, 0.15)  # gentle curve
    elif t < 350:
        return (1.0, 0.25)  # tight orbit, full circle(s)
    elif t < 380:
        return (1.0, 0.0)  # translate orbit center
    else:
        return (1.0, 0.25)  # second orbit


# ---------------------------------------------------------------------------
# Shared SLAM runner
# ---------------------------------------------------------------------------


def run_slam_experiment(
    control_fn: Any,
    init_pose: list[float],
    R: np.ndarray,
    Q: np.ndarray,
    num_steps: int = NUM_STEPS,
    seed: int = SEED,
    snapshot_steps: list[int] | None = None,
) -> dict[str, Any]:
    """Run a full SLAM experiment and return collected data.

    Args:
        control_fn: Callable(t) -> (v, omega).
        init_pose: Initial [x, y, theta].
        R: 3x3 process noise covariance.
        Q: 2x2 measurement noise covariance.
        num_steps: Simulation length.
        seed: Random seed.
        snapshot_steps: List of timesteps at which to save a snapshot
            of landmark estimates, covariances, path, and robot pose.

    Returns:
        Dictionary with true_path, est_path, landmarks, and metric arrays.
    """
    np.random.seed(seed)
    env = Environment(seed=seed)
    sim = RobotSimulator(env, init_pose, R, Q)
    ekf = EKFSLAM(init_pose=init_pose, R=R, Q=Q)

    pose_errors: list[float] = []
    pose_uncertainties: list[float] = []
    avg_landmark_uncertainties: list[float] = []
    snapshots: list[dict[str, Any]] = []
    snap_set = set(snapshot_steps) if snapshot_steps else set()

    for t in range(num_steps):
        control = control_fn(t)
        measurements = sim.step(control)
        ekf.step(control, measurements, dt=sim.dt)

        # --- metrics at this step ---
        true_xy = sim.true_pose[:2]
        est_xy = ekf.get_current_pose()[:2]
        pose_errors.append(float(np.linalg.norm(true_xy - est_xy)))

        pose_cov = ekf.get_pose_covariance()
        pose_uncertainties.append(float(np.sqrt(np.trace(pose_cov))))

        lm_covs = ekf.get_landmark_covariances()
        if lm_covs:
            avg_landmark_uncertainties.append(
                float(np.mean([np.trace(c) for c in lm_covs.values()]))
            )
        else:
            avg_landmark_uncertainties.append(0.0)

        # --- snapshot ---
        if t in snap_set:
            snapshots.append(
                {
                    "step": t,
                    "est_landmarks": ekf.get_estimated_landmarks(),
                    "landmark_covs": ekf.get_landmark_covariances(),
                    "true_path": sim.get_true_path().copy(),
                    "est_path": ekf.get_estimated_path().copy(),
                    "robot_pose": ekf.get_current_pose(),
                    "true_pose": sim.true_pose.copy(),
                }
            )

    # --- Final landmark position error (map accuracy) ---
    est_lms = ekf.get_estimated_landmarks()
    lm_errors: list[float] = []
    for lm_id, est_pos in est_lms.items():
        true_pos = env.landmarks[lm_id]
        lm_errors.append(float(np.linalg.norm(est_pos - true_pos)))
    avg_landmark_error = float(np.mean(lm_errors)) if lm_errors else 0.0

    return {
        "env": env,
        "ekf": ekf,
        "true_path": sim.get_true_path(),
        "est_path": ekf.get_estimated_path(),
        "est_landmarks": est_lms,
        "landmark_covs": ekf.get_landmark_covariances(),
        "pose_errors": np.array(pose_errors),
        "pose_uncertainties": np.array(pose_uncertainties),
        "avg_landmark_uncertainties": np.array(avg_landmark_uncertainties),
        "avg_landmark_error": avg_landmark_error,
        "snapshots": snapshots,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_trajectory(
    ax: plt.Axes,
    data: dict[str, Any],
    title: str,
) -> None:
    """Plot true vs estimated path with landmark uncertainty ellipses."""
    env = data["env"]
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlim(env.x_min, env.x_max)
    ax.set_ylim(env.y_min, env.y_max)
    ax.set_xlabel("x [m]", fontsize=13)
    ax.set_ylabel("y [m]", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # True obstacles
    for lm_id, lm_pos in env.landmarks.items():
        circle = Circle(
            (lm_pos[0], lm_pos[1]), env.obstacle_radius, color="gray", alpha=0.3
        )
        ax.add_patch(circle)
        ax.plot(lm_pos[0], lm_pos[1], "k+", markersize=8, alpha=0.4)
        ax.text(
            lm_pos[0] + 0.4, lm_pos[1] + 0.4, f"L{lm_id}",
            fontsize=8, color="dimgray",
        )

    # Estimated landmarks with uncertainty
    for lm_id, lm_pos in data["est_landmarks"].items():
        cov = data["landmark_covs"][lm_id]
        ax.plot(lm_pos[0], lm_pos[1], "ro", markersize=7, zorder=5)
        w, h, angle = get_covariance_ellipse(cov, n_std=2.0)
        ell = Ellipse(
            (lm_pos[0], lm_pos[1]),
            w,
            h,
            angle=angle,
            facecolor="red",
            edgecolor="red",
            linewidth=1.5,
            linestyle="--",
            alpha=0.15,
            zorder=4,
        )
        ax.add_patch(ell)

    tp = data["true_path"]
    ep = data["est_path"]
    ax.plot(tp[:, 0], tp[:, 1], "k-", linewidth=1.8, alpha=0.7, label="True path")
    ax.plot(
        ep[:, 0], ep[:, 1], "b--", linewidth=1.8, alpha=0.7, label="Estimated path"
    )

    # Mark start position
    ax.plot(tp[0, 0], tp[0, 1], "gs", markersize=12, zorder=6, label="Start")
    ax.plot(tp[-1, 0], tp[-1, 1], "r^", markersize=12, zorder=6, label="End")

    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)


def plot_metrics(
    ax_err: plt.Axes,
    ax_unc: plt.Axes,
    ax_lm: plt.Axes,
    data: dict[str, Any],
    label: str = "",
    color: str = "tab:blue",
) -> None:
    """Plot 3-panel metrics (pose error, pose uncertainty, avg LM uncertainty)."""
    steps = np.arange(len(data["pose_errors"]))

    ax_err.plot(steps, data["pose_errors"], color=color, linewidth=1.4, label=label)
    ax_err.set_ylabel("Pose error [m]", fontsize=13)
    ax_err.set_xlabel("Step", fontsize=13)
    ax_err.tick_params(labelsize=11)
    ax_err.grid(True, alpha=0.3)

    ax_unc.plot(
        steps, data["pose_uncertainties"], color=color, linewidth=1.4, label=label
    )
    ax_unc.set_ylabel("Pose uncertainty [m]", fontsize=13)
    ax_unc.set_xlabel("Step", fontsize=13)
    ax_unc.tick_params(labelsize=11)
    ax_unc.grid(True, alpha=0.3)

    ax_lm.plot(
        steps,
        data["avg_landmark_uncertainties"],
        color=color,
        linewidth=1.4,
        label=label,
    )
    ax_lm.set_ylabel("Avg LM uncertainty", fontsize=13)
    ax_lm.set_xlabel("Step", fontsize=13)
    ax_lm.tick_params(labelsize=11)
    ax_lm.grid(True, alpha=0.3)


def plot_snapshot_grid(
    snapshots: list[dict[str, Any]],
    env: Any,
    title: str,
    save_path: str,
) -> None:
    """Plot a row of map snapshots at different timesteps."""
    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, snap in zip(axes, snapshots):
        step = snap["step"]
        ax.set_title(f"Step {step}", fontsize=14, fontweight="bold")
        ax.set_xlim(env.x_min, env.x_max)
        ax.set_ylim(env.y_min, env.y_max)
        ax.set_xlabel("x [m]", fontsize=11)
        ax.set_ylabel("y [m]", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # True obstacles (faded)
        for lm_pos in env.landmarks.values():
            circle = Circle(
                (lm_pos[0], lm_pos[1]),
                env.obstacle_radius,
                color="gray",
                alpha=0.15,
            )
            ax.add_patch(circle)
            ax.plot(lm_pos[0], lm_pos[1], "k+", markersize=6, alpha=0.3)

        # Estimated landmarks + uncertainty ellipses
        for lm_id, lm_pos in snap["est_landmarks"].items():
            if lm_id not in snap["landmark_covs"]:
                continue
            cov = snap["landmark_covs"][lm_id]
            unc = np.trace(cov)

            # Color: blue (low unc) -> red (high unc)
            clipped = min(unc / 2.0, 1.0)  # normalise roughly
            color = plt.cm.RdYlBu_r(clipped)

            ax.plot(lm_pos[0], lm_pos[1], "o", color=color, markersize=7,
                    markeredgecolor="black", markeredgewidth=0.6, zorder=5)

            w, h, angle = get_covariance_ellipse(cov, n_std=2.0)
            ell = Ellipse(
                (lm_pos[0], lm_pos[1]), w, h, angle=angle,
                facecolor=color, edgecolor="black", linewidth=0.8,
                alpha=0.25, zorder=4,
            )
            ax.add_patch(ell)

        # Path up to this snapshot
        tp = snap["true_path"]
        ep = snap["est_path"]
        if len(tp) > 1:
            ax.plot(tp[:, 0], tp[:, 1], "k-", linewidth=1.2, alpha=0.5)
        if len(ep) > 1:
            ax.plot(ep[:, 0], ep[:, 1], "b--", linewidth=1.2, alpha=0.5)

        # Robot position
        rx, ry, rth = snap["true_pose"]
        ax.plot(rx, ry, "ko", markersize=8, zorder=6)
        ax.arrow(rx, ry, 0.6 * np.cos(rth), 0.6 * np.sin(rth),
                 head_width=0.25, length_includes_head=True,
                 color="black", linewidth=1.5, zorder=6)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# A1 - Path design experiment
# ---------------------------------------------------------------------------

SNAPSHOT_STEPS = [49, 149, 249, 349, 499]


def experiment_a1() -> None:
    """A1: Compare greedy-forward vs looping-revisit trajectories."""
    print("=" * 60)
    print("A1: Path Design Experiment")
    print("=" * 60)

    R = np.diag([0.1, 0.1, np.deg2rad(3.0)]) ** 2
    Q = np.diag([0.5, np.deg2rad(8.0)]) ** 2

    data_a = run_slam_experiment(
        control_greedy_forward, [0.0, 0.0, 0.0], R, Q,
        snapshot_steps=SNAPSHOT_STEPS,
    )
    data_b = run_slam_experiment(
        control_looping_revisit, [0.0, 0.0, np.pi / 6], R, Q,
        snapshot_steps=SNAPSHOT_STEPS,
    )

    # --- Snapshot grids (uncertainty evolution) ---
    for suffix, data, label in [
        ("a1_snapshots_path_a", data_a, "Path A \u2013 Greedy Forward: Uncertainty Evolution"),
        ("a1_snapshots_path_b", data_b, "Path B \u2013 Looping Revisit: Uncertainty Evolution"),
    ]:
        plot_snapshot_grid(
            data["snapshots"], data["env"], label,
            os.path.join(RESULTS_DIR, f"{suffix}.png"),
        )
        print(f"  Saved {suffix}.png")

    # --- Individual trajectory figures (large) ---
    for suffix, data, title in [
        ("a1_path_a", data_a, "Path A \u2013 Greedy Forward"),
        ("a1_path_b", data_b, "Path B \u2013 Looping Revisit"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_trajectory(ax, data, title)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"{suffix}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved {suffix}.png")

    # --- Side-by-side trajectory comparison (kept for overview) ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    plot_trajectory(axes[0], data_a, "Path A \u2013 Greedy Forward")
    plot_trajectory(axes[1], data_b, "Path B \u2013 Looping Revisit")
    fig.suptitle("A1: Trajectory Comparison", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "a1_trajectories.png"), dpi=150)
    plt.close(fig)
    print("  Saved a1_trajectories.png")

    # --- Metrics comparison figure ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for data, label, color in [
        (data_a, "Greedy Forward", "tab:blue"),
        (data_b, "Looping Revisit", "tab:orange"),
    ]:
        plot_metrics(axes[0], axes[1], axes[2], data, label=label, color=color)
    for ax in axes:
        ax.legend(fontsize=12)
    axes[0].set_title("A1: Metrics Over Time", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "a1_metrics.png"), dpi=150)
    plt.close(fig)
    print("  Saved a1_metrics.png")

    # --- Summary ---
    for name, data in [("Greedy Forward", data_a), ("Looping Revisit", data_b)]:
        print(f"\n  {name}:")
        print(f"    Final pose error:        {data['pose_errors'][-1]:.4f} m")
        print(f"    Mean pose error:         {np.mean(data['pose_errors']):.4f} m")
        print(f"    Final pose uncertainty:  {data['pose_uncertainties'][-1]:.4f} m")
        print(
            f"    Final avg LM unc:        {data['avg_landmark_uncertainties'][-1]:.4f}"
        )
        print(f"    Landmarks observed:      {len(data['est_landmarks'])}")


# ---------------------------------------------------------------------------
# A2 - Noise sensitivity experiment
# ---------------------------------------------------------------------------

R_LEVELS = {
    "Low": np.diag([0.05, 0.05, np.deg2rad(1.5)]) ** 2,
    "Medium": np.diag([0.1, 0.1, np.deg2rad(3.0)]) ** 2,
    "High": np.diag([0.3, 0.3, np.deg2rad(8.0)]) ** 2,
}

Q_LEVELS = {
    "Low": np.diag([0.2, np.deg2rad(3.0)]) ** 2,
    "Medium": np.diag([0.5, np.deg2rad(8.0)]) ** 2,
    "High": np.diag([1.5, np.deg2rad(15.0)]) ** 2,
}


def experiment_a2() -> None:
    """A2: Noise sensitivity analysis using Path B."""
    print("\n" + "=" * 60)
    print("A2: Noise Sensitivity Experiment")
    print("=" * 60)

    init_pose = [0.0, 0.0, np.pi / 6]
    Q_fixed = Q_LEVELS["Medium"]
    R_fixed = R_LEVELS["Medium"]

    colors = {"Low": "tab:green", "Medium": "tab:blue", "High": "tab:red"}

    # --- Process noise sweep (R varies, Q fixed) ---
    print("\n  Process noise sweep (Q fixed at Medium):")
    r_results: dict[str, dict] = {}
    for level, R in R_LEVELS.items():
        r_results[level] = run_slam_experiment(
            control_looping_revisit, init_pose, R, Q_fixed
        )
        d = r_results[level]
        print(
            f"    R={level}: pose_err={d['pose_errors'][-1]:.4f}  "
            f"lm_err={d['avg_landmark_error']:.4f}  "
            f"pose_unc={d['pose_uncertainties'][-1]:.4f}  "
            f"lm_unc={d['avg_landmark_uncertainties'][-1]:.4f}"
        )

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for level, data in r_results.items():
        plot_metrics(
            axes[0], axes[1], axes[2], data, label=f"R={level}", color=colors[level]
        )
    for ax in axes:
        ax.legend(fontsize=12)
    axes[0].set_title(
        "A2: Process Noise Sensitivity (Q fixed)", fontsize=16, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "a2_process_noise.png"), dpi=150)
    plt.close(fig)
    print("  Saved a2_process_noise.png")

    # --- Measurement noise sweep (Q varies, R fixed) ---
    print("\n  Measurement noise sweep (R fixed at Medium):")
    q_results: dict[str, dict] = {}
    for level, Q in Q_LEVELS.items():
        q_results[level] = run_slam_experiment(
            control_looping_revisit, init_pose, R_fixed, Q
        )
        d = q_results[level]
        print(
            f"    Q={level}: pose_err={d['pose_errors'][-1]:.4f}  "
            f"lm_err={d['avg_landmark_error']:.4f}  "
            f"pose_unc={d['pose_uncertainties'][-1]:.4f}  "
            f"lm_unc={d['avg_landmark_uncertainties'][-1]:.4f}"
        )

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for level, data in q_results.items():
        plot_metrics(
            axes[0], axes[1], axes[2], data, label=f"Q={level}", color=colors[level]
        )
    for ax in axes:
        ax.legend(fontsize=12)
    axes[0].set_title(
        "A2: Measurement Noise Sensitivity (R fixed)", fontsize=16, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "a2_measurement_noise.png"), dpi=150)
    plt.close(fig)
    print("  Saved a2_measurement_noise.png")

    # --- Summary table ---
    fig, ax_tab = plt.subplots(figsize=(12, 5))
    ax_tab.axis("off")
    rows = []
    for sweep_name, results in [
        ("Process noise sweep", r_results),
        ("Measurement noise sweep", q_results),
    ]:
        for level, data in results.items():
            rows.append(
                [
                    sweep_name,
                    level,
                    f"{data['pose_errors'][-1]:.4f}",
                    f"{data['avg_landmark_error']:.4f}",
                    f"{data['pose_uncertainties'][-1]:.4f}",
                    f"{data['avg_landmark_uncertainties'][-1]:.4f}",
                ]
            )
    col_labels = [
        "Sweep",
        "Level",
        "Final Pose Err",
        "Avg LM Pos Err",
        "Final Pose Unc",
        "Final Avg LM Unc",
    ]
    table = ax_tab.table(
        cellText=rows, colLabels=col_labels, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    ax_tab.set_title(
        "A2: Noise Sensitivity Summary", fontsize=16, fontweight="bold", pad=20
    )
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "a2_summary_table.png"), dpi=150)
    plt.close(fig)
    print("  Saved a2_summary_table.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plt.switch_backend("Agg")  # non-interactive for batch runs
    experiment_a1()
    experiment_a2()
    print("\nAll Part A results saved to", RESULTS_DIR)
