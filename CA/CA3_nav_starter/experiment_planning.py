#!/usr/bin/env python3
"""Part B: PRM vs RRT motion planning experiments.

B1 - Two scenarios (open vs constrained).
B2 - Metrics comparison over N=30 trials.
B3 - Parameter studies for PRM and RRT.

Usage:
    python experiment_planning.py
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
from path_utils import path_length

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "partB")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_env(seed: int = SEED, obstacle_radius: float = 0.8) -> Environment:
    """Create an Environment and optionally override obstacle radius."""
    env = Environment(seed=seed)
    env.obstacle_radius = obstacle_radius
    return env


def run_planner(
    planner_cls: type,
    env: Environment,
    start: np.ndarray,
    goal: np.ndarray,
    seed: int = SEED,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a single planner trial and collect metrics.

    Returns:
        dict with keys: path, success, length, time_s, nodes.
    """
    np.random.seed(seed)
    planner = planner_cls(environment=env, **kwargs)

    t0 = time.perf_counter()
    if isinstance(planner, PRM):
        planner.build_roadmap()
        path = planner.plan(start, goal)
        n_nodes = len(planner.nodes)
    else:
        path = planner.plan(start, goal, goal_tolerance=0.5)
        n_nodes = len(planner.nodes)
    elapsed = time.perf_counter() - t0

    success = path is not None
    length = path_length(path) if success else float("nan")

    return {
        "path": path,
        "success": success,
        "length": length,
        "time_s": elapsed,
        "nodes": n_nodes,
    }


def draw_env(ax: plt.Axes, env: Environment) -> None:
    """Draw environment obstacles on an Axes."""
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
# B1 - Two scenarios
# ---------------------------------------------------------------------------


def experiment_b1() -> None:
    """B1: Single run visualization on open and constrained scenarios."""
    print("=" * 60)
    print("B1: Scenario Visualization")
    print("=" * 60)

    scenarios = {
        "Open": {
            "env": make_env(obstacle_radius=0.8),
            "start": np.array([1.0, 1.0]),
            "goal": np.array([19.0, 1.0]),
        },
        "Constrained": {
            "env": make_env(obstacle_radius=1.0),
            "start": np.array([1.0, 1.0]),
            "goal": np.array([17.0, 17.0]),
        },
    }

    for scenario_name, cfg in scenarios.items():
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        env, start, goal = cfg["env"], cfg["start"], cfg["goal"]

        for idx, (planner_name, cls, kw) in enumerate(
            [
                ("PRM", PRM, {"n_samples": 500, "k_neighbors": 10}),
                ("RRT", RRT, {"max_iter": 5000, "step_size": 0.5}),
            ]
        ):
            ax = axes[idx]
            draw_env(ax, env)
            result = run_planner(cls, env, start, goal, seed=SEED, **kw)

            if result["success"]:
                p = result["path"]
                ax.plot(p[:, 0], p[:, 1], "g-", linewidth=2.5, label="Path", zorder=5)
            ax.plot(start[0], start[1], "go", markersize=12, label="Start", zorder=6)
            ax.plot(goal[0], goal[1], "r*", markersize=16, label="Goal", zorder=6)
            status = f"L={result['length']:.1f}m" if result["success"] else "No path"
            ax.set_title(f"{planner_name} - {scenario_name}\n{status}", fontsize=11)
            ax.legend(fontsize=8)

        fig.suptitle(f"B1: {scenario_name} Scenario", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fname = f"b1_{scenario_name.lower()}.png"
        fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# B2 - Metrics comparison (N=30)
# ---------------------------------------------------------------------------


def experiment_b2(n_trials: int = 30) -> None:
    """B2: Statistical comparison of PRM vs RRT on the constrained scenario."""
    print("\n" + "=" * 60)
    print(f"B2: Metrics Comparison ({n_trials} trials)")
    print("=" * 60)

    env = make_env(obstacle_radius=1.0)
    start = np.array([1.0, 1.0])
    goal = np.array([17.0, 17.0])

    results: dict[str, list[dict]] = {"PRM": [], "RRT": []}

    for i in range(n_trials):
        seed_i = SEED + i
        results["PRM"].append(
            run_planner(
                PRM, env, start, goal, seed=seed_i, n_samples=500, k_neighbors=10
            )
        )
        results["RRT"].append(
            run_planner(
                RRT, env, start, goal, seed=seed_i, max_iter=5000, step_size=0.5
            )
        )

    # Build summary rows
    rows = []
    for name, trials in results.items():
        successes = [t for t in trials if t["success"]]
        sr = len(successes) / len(trials) * 100
        avg_len = np.nanmean([t["length"] for t in successes]) if successes else 0
        avg_time = np.mean([t["time_s"] for t in trials])
        avg_nodes = np.mean([t["nodes"] for t in trials])
        rows.append(
            [
                name,
                f"{sr:.0f}%",
                f"{avg_len:.2f}",
                f"{avg_time:.3f}",
                f"{avg_nodes:.0f}",
            ]
        )
        print(
            f"  {name}: success={sr:.0f}%  avg_len={avg_len:.2f}  "
            f"avg_time={avg_time:.3f}s  avg_nodes={avg_nodes:.0f}"
        )

    # --- Table figure ---
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    col_labels = ["Planner", "Success %", "Avg Length", "Avg Time [s]", "Avg Nodes"]
    table = ax.table(
        cellText=rows, colLabels=col_labels, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    ax.set_title(
        f"B2: PRM vs RRT ({n_trials} trials, constrained)",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "b2_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved b2_comparison.png")


# ---------------------------------------------------------------------------
# B3 - Parameter studies
# ---------------------------------------------------------------------------


def _run_param_study(
    planner_cls: type,
    env: Environment,
    start: np.ndarray,
    goal: np.ndarray,
    param_name: str,
    param_values: list[Any],
    fixed_kwargs: dict[str, Any],
    n_trials: int = 20,
) -> dict[str, list[float]]:
    """Sweep one parameter and aggregate success rate + avg path length."""
    success_rates: list[float] = []
    avg_lengths: list[float] = []

    for val in param_values:
        kw = {**fixed_kwargs, param_name: val}
        successes = 0
        lengths: list[float] = []

        for i in range(n_trials):
            res = run_planner(planner_cls, env, start, goal, seed=SEED + i, **kw)
            if res["success"]:
                successes += 1
                lengths.append(res["length"])

        success_rates.append(successes / n_trials * 100)
        avg_lengths.append(np.mean(lengths) if lengths else float("nan"))

    return {"success_rates": success_rates, "avg_lengths": avg_lengths}


def experiment_b3(n_trials: int = 20) -> None:
    """B3: Parameter studies for PRM and RRT."""
    print("\n" + "=" * 60)
    print(f"B3: Parameter Studies ({n_trials} trials each)")
    print("=" * 60)

    env = make_env(obstacle_radius=1.0)
    start = np.array([1.0, 1.0])
    goal = np.array([17.0, 17.0])

    studies = [
        {
            "title": "PRM: n_samples",
            "cls": PRM,
            "param": "n_samples",
            "values": [100, 200, 400, 800],
            "fixed": {"k_neighbors": 10},
        },
        {
            "title": "RRT: step_size",
            "cls": RRT,
            "param": "step_size",
            "values": [0.2, 0.5, 1.0],
            "fixed": {"max_iter": 5000, "goal_sample_rate": 0.1},
        },
        {
            "title": "RRT: goal_sample_rate",
            "cls": RRT,
            "param": "goal_sample_rate",
            "values": [0.0, 0.1, 0.3, 0.7],
            "fixed": {"max_iter": 5000, "step_size": 0.5},
        },
    ]

    fig, axes = plt.subplots(len(studies), 2, figsize=(12, 4 * len(studies)))

    for row, study in enumerate(studies):
        print(f"\n  {study['title']} sweep: {study['values']}")
        out = _run_param_study(
            study["cls"],
            env,
            start,
            goal,
            study["param"],
            study["values"],
            study["fixed"],
            n_trials=n_trials,
        )

        x_vals = study["values"]

        ax_sr = axes[row, 0]
        ax_sr.bar(range(len(x_vals)), out["success_rates"], color="steelblue")
        ax_sr.set_xticks(range(len(x_vals)))
        ax_sr.set_xticklabels([str(v) for v in x_vals])
        ax_sr.set_ylabel("Success rate [%]")
        ax_sr.set_xlabel(study["param"])
        ax_sr.set_title(f"{study['title']} - Success Rate")
        ax_sr.set_ylim(0, 105)
        ax_sr.grid(True, alpha=0.3, axis="y")

        ax_len = axes[row, 1]
        valid = [(v, l) for v, l in zip(x_vals, out["avg_lengths"]) if not np.isnan(l)]
        if valid:
            vv, ll = zip(*valid)
            ax_len.bar(range(len(vv)), ll, color="coral")
            ax_len.set_xticks(range(len(vv)))
            ax_len.set_xticklabels([str(v) for v in vv])
        ax_len.set_ylabel("Avg path length [m]")
        ax_len.set_xlabel(study["param"])
        ax_len.set_title(f"{study['title']} - Avg Path Length")
        ax_len.grid(True, alpha=0.3, axis="y")

        for v, sr, al in zip(x_vals, out["success_rates"], out["avg_lengths"]):
            print(f"    {study['param']}={v}: success={sr:.0f}%  avg_len={al:.2f}")

    fig.suptitle("B3: Parameter Studies", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "b3_parameter_studies.png"), dpi=150)
    plt.close(fig)
    print("  Saved b3_parameter_studies.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plt.switch_backend("Agg")
    experiment_b1()
    experiment_b2()
    experiment_b3()
    print("\nAll Part B results saved to", RESULTS_DIR)
