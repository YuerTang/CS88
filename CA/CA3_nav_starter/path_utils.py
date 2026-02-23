"""Shared path utilities for motion planning experiments."""

from __future__ import annotations

from typing import Optional

import numpy as np

from environment import Environment


def shortcut_path(
    path: np.ndarray,
    env: Environment,
    max_iterations: int = 200,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Shorten a path by removing unnecessary intermediate waypoints.

    For each iteration, pick two random waypoints and check if the direct
    line between them is collision-free.  If so, remove all intermediate
    waypoints to produce a shorter path.

    Args:
        path: (N, 2) array of waypoints.
        env: Environment used for collision checking.
        max_iterations: Number of random shortcut attempts.
        rng: Optional random state for reproducibility.

    Returns:
        Shortened (M, 2) array of waypoints where M <= N.
    """
    if rng is None:
        rng = np.random.RandomState()

    path = list(path)  # work with a mutable list

    for _ in range(max_iterations):
        if len(path) <= 2:
            break

        i = rng.randint(0, len(path) - 2)
        j = rng.randint(i + 2, len(path))

        if env.is_path_collision_free(np.array(path[i]), np.array(path[j])):
            path = path[: i + 1] + path[j:]

    return np.array(path)


def path_length(path: np.ndarray) -> float:
    """Compute the total Euclidean length of a path.

    Args:
        path: (N, 2) array of waypoints.

    Returns:
        Total path length.
    """
    diffs = np.diff(path, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))
