#!/usr/bin/env python3
"""Part C: Path shortcutting improvement experiments.

Applies shortcut_path to PRM and RRT results, measuring length and waypoint
reduction.

Usage:
    python experiment_improve.py
"""

from __future__ import annotations

import os
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from environment import Environment
from motion_planning import PRM, RRT
from path_utils import path_length, shortcut_path

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "partC")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42
N_TRIALS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_env(seed: int = SEED) -> Environment:
    """Create constrained environment (obstacle_radius=1.5)."""
    env = Environment(seed=seed)
    env.obstacle_radius = 1.0
    return env


def draw_env(ax: plt.Axes, env: Environment) -> None:
    """Draw environment obstacles."""
    ax.set_xlim(env.x_min, env.x_max)
    ax.set_ylim(env.y_min, env.y_max)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    for lm_pos in env.landmarks.values():
        circle = Circle(
            (lm_pos[0], lm_pos[1]), env.obstacle_radius, color="gray", alpha=0.4
        )
        ax.add_patch(circle)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def experiment_c() -> None:
    """Run PRM and RRT with shortcutting on the constrained scenario."""
    print("=" * 60)
    print(f"Part C: Path Shortcutting ({N_TRIALS} trials each)")
    print("=" * 60)

    env = make_env()
    start = np.array([1.0, 1.0])
    goal = np.array([17.0, 17.0])

    records: dict[str, list[dict[str, Any]]] = {"PRM": [], "RRT": []}
    n_success = {"PRM": 0, "RRT": 0}

    for i in range(N_TRIALS):
        seed_i = SEED + i
        np.random.seed(seed_i)

        # PRM
        prm = PRM(env, n_samples=500, k_neighbors=10)
        t0 = time.perf_counter()
        prm.build_roadmap()
        prm_path = prm.plan(start, goal)
        plan_time = time.perf_counter() - t0
        if prm_path is not None:
            n_success["PRM"] += 1
            rng = np.random.RandomState(seed_i)
            t1 = time.perf_counter()
            prm_short = shortcut_path(prm_path, env, max_iterations=200, rng=rng)
            short_time = time.perf_counter() - t1
            records["PRM"].append(
                {
                    "original": prm_path,
                    "shortened": prm_short,
                    "len_before": path_length(prm_path),
                    "len_after": path_length(prm_short),
                    "wp_before": len(prm_path),
                    "wp_after": len(prm_short),
                    "plan_time": plan_time,
                    "short_time": short_time,
                    "total_time": plan_time + short_time,
                    "nodes": len(prm.nodes),
                }
            )

        # RRT
        np.random.seed(seed_i)
        rrt = RRT(env, max_iter=5000, step_size=0.5, goal_sample_rate=0.1)
        t0 = time.perf_counter()
        rrt_path = rrt.plan(start, goal, goal_tolerance=0.5)
        plan_time = time.perf_counter() - t0
        if rrt_path is not None:
            n_success["RRT"] += 1
            rng = np.random.RandomState(seed_i)
            t1 = time.perf_counter()
            rrt_short = shortcut_path(rrt_path, env, max_iterations=200, rng=rng)
            short_time = time.perf_counter() - t1
            records["RRT"].append(
                {
                    "original": rrt_path,
                    "shortened": rrt_short,
                    "len_before": path_length(rrt_path),
                    "len_after": path_length(rrt_short),
                    "wp_before": len(rrt_path),
                    "wp_after": len(rrt_short),
                    "plan_time": plan_time,
                    "short_time": short_time,
                    "total_time": plan_time + short_time,
                    "nodes": len(rrt.nodes),
                }
            )

    # --- Before / after example figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for col, name in enumerate(["PRM", "RRT"]):
        if not records[name]:
            continue
        rec = records[name][0]  # first successful trial
        for row, (key, label, color) in enumerate(
            [
                ("original", "Before Shortcut", "tab:blue"),
                ("shortened", "After Shortcut", "tab:green"),
            ]
        ):
            ax = axes[row, col]
            draw_env(ax, env)
            p = rec[key]
            ax.plot(
                p[:, 0],
                p[:, 1],
                "-o",
                color=color,
                markersize=3,
                linewidth=1.5,
                label=label,
            )
            ax.plot(start[0], start[1], "go", markersize=12, zorder=6)
            ax.plot(goal[0], goal[1], "r*", markersize=16, zorder=6)
            ln = path_length(p)
            ax.set_title(
                f"{name} - {label}\nL={ln:.2f}m, {len(p)} waypoints",
                fontsize=10,
            )
            ax.legend(fontsize=8)

    fig.suptitle("Part C: Before vs After Shortcutting", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "c_before_after.png"), dpi=150)
    plt.close(fig)
    print("  Saved c_before_after.png")

    # --- Summary table ---
    rows = []
    for name in ["PRM", "RRT"]:
        recs = records[name]
        if not recs:
            continue
        sr = n_success[name] / N_TRIALS * 100
        lb = np.mean([r["len_before"] for r in recs])
        la = np.mean([r["len_after"] for r in recs])
        pct_len = (lb - la) / lb * 100 if lb > 0 else 0
        wb = np.mean([r["wp_before"] for r in recs])
        wa = np.mean([r["wp_after"] for r in recs])
        pct_wp = (wb - wa) / wb * 100 if wb > 0 else 0
        avg_plan = np.mean([r["plan_time"] for r in recs])
        avg_short = np.mean([r["short_time"] for r in recs])
        avg_total = np.mean([r["total_time"] for r in recs])
        avg_nodes = np.mean([r["nodes"] for r in recs])
        rows.append(
            [
                name,
                f"{sr:.0f}%",
                f"{lb:.2f}",
                f"{la:.2f}",
                f"{pct_len:.1f}%",
                f"{pct_wp:.0f}%",
                f"{avg_plan:.3f}",
                f"{avg_short:.4f}",
                f"{avg_total:.3f}",
                f"{avg_nodes:.0f}",
            ]
        )
        print(
            f"  {name}: success={sr:.0f}%  "
            f"len {lb:.2f}->{la:.2f} ({pct_len:.1f}%)  "
            f"wp {wb:.0f}->{wa:.0f} ({pct_wp:.0f}%)  "
            f"plan={avg_plan:.3f}s  short={avg_short:.4f}s  "
            f"total={avg_total:.3f}s  nodes={avg_nodes:.0f}"
        )

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")
    col_labels = [
        "Planner",
        "Success",
        "Len Before",
        "Len After",
        "Len %",
        "WP %",
        "Plan [s]",
        "Short [s]",
        "Total [s]",
        "Nodes",
    ]
    table = ax.table(
        cellText=rows, colLabels=col_labels, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    ax.set_title("Part C: Shortcutting Summary", fontsize=12, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "c_summary_table.png"), dpi=150)
    plt.close(fig)
    print("  Saved c_summary_table.png")

    # --- Box-plot of length reductions ---
    fig, axes_bp = plt.subplots(1, 2, figsize=(12, 5))
    for idx, name in enumerate(["PRM", "RRT"]):
        recs = records[name]
        if not recs:
            continue
        before = [r["len_before"] for r in recs]
        after = [r["len_after"] for r in recs]
        ax = axes_bp[idx]
        bp = ax.boxplot(
            [before, after],
            tick_labels=["Before", "After"],
            patch_artist=True,
        )
        bp["boxes"][0].set_facecolor("lightcoral")
        bp["boxes"][1].set_facecolor("lightgreen")
        ax.set_ylabel("Path length [m]")
        ax.set_title(f"{name}: Length Distribution")
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Part C: Path Length Before/After", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "c_boxplot.png"), dpi=150)
    plt.close(fig)
    print("  Saved c_boxplot.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plt.switch_backend("Agg")
    experiment_c()
    print("\nAll Part C results saved to", RESULTS_DIR)
