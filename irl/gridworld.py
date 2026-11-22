"""
A small gridworld for testing MaxEnt IRL against a known reward.
"""

import numpy as np

from irl.maxent_irl import TabularMDP

# (dx, dy); y grows downward, as in the WPG map
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]
ACTION_NAMES = ['up', 'down', 'left', 'right', 'stay']

FEATURE_NAMES = ['at_goal', 'on_hazard', 'dist_to_goal']


def state_index(x: int, y: int, width: int) -> int:
    return y * width + x


def state_coords(s: int, width: int) -> tuple[int, int]:
    return s % width, s // width


def make_gridworld(width: int, height: int, slip: float = 0.0,
                   start: tuple[int, int] | None = None) -> TabularMDP:
    """
    Build a gridworld MDP.

    Args:
        width, height: grid size.
        slip: probability that a random action (uniform over all actions) is executed
            instead of the chosen one. Moves into a wall leave the agent in place.
        start: fixed start cell; if None, start uniformly at random.
    """
    n_states = width * height
    n_actions = len(ACTIONS)
    T = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        x, y = state_coords(s, width)
        for a in range(n_actions):
            for executed in range(n_actions):
                prob = (1 - slip) * (executed == a) + slip / n_actions
                dx, dy = ACTIONS[executed]
                nx = min(max(x + dx, 0), width - 1)
                ny = min(max(y + dy, 0), height - 1)
                T[s, a, state_index(nx, ny, width)] += prob

    if start is None:
        p0 = np.full(n_states, 1.0 / n_states)
    else:
        p0 = np.zeros(n_states)
        p0[state_index(*start, width)] = 1.0
    return TabularMDP(T=T, p0=p0)


def make_features(width: int, height: int, goal: tuple[int, int],
                  hazards: list[tuple[int, int]]) -> np.ndarray:
    """
    State features, shape (S, 3), in the order of FEATURE_NAMES.
    The distance feature is normalized to [0, 1].
    """
    phi = np.zeros((width * height, len(FEATURE_NAMES)))
    max_dist = (width - 1) + (height - 1)
    for s in range(width * height):
        x, y = state_coords(s, width)
        phi[s, 0] = float((x, y) == goal)
        phi[s, 1] = float((x, y) in hazards)
        phi[s, 2] = (abs(x - goal[0]) + abs(y - goal[1])) / max_dist
    return phi


def default_layout() -> dict:
    """An 8x8 grid with the goal in a corner and a wall of hazards in between."""
    return {
        'width': 8,
        'height': 8,
        'goal': (6, 6),
        'hazards': [(3, 2), (3, 3), (3, 4), (4, 4), (5, 4), (1, 5), (2, 5)],
    }
