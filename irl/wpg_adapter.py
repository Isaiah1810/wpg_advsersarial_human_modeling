"""
Convert WPG simulator outputs into the inputs of MaxEnt IRL for the poacher's reward:
a TabularMDP, state features, and demonstrations.

The state space is the poacher MDP's (see mdp/explicit.py):
    (distance, sees_drone, hiding, using_jammer)
and the features are those same four variables.

Each game run directory (poacher_data.csv, drone_data.csv, game_data.csv, map_data.csv)
becomes one episode of poacher states and abstract actions. Works for scripted runs
(`wpg live=false ...`) and for live human runs, as long as the CSV format is the same.

Run from the repository root:
    python -m irl.wpg_adapter <dir with game runs> [<dir> ...]
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from irl.maxent_irl import TabularMDP, Trajectory, maxent_irl
from wpg.constants import DRONE_DATA_CSV, GAME_DATA_CSV, MAP_DATA_CSV, POACHER_DATA_CSV
from wpg.game_map import CellType

# Same range as the distance variable in mdp/DronePoach.nm; larger distances are capped
MAX_DIST = 20

ACTION_NAMES = ['approach', 'retreat', 'sidestep', 'stay', 'hide', 'shoot', 'jam']
ACTION_INDEX = {name: i for i, name in enumerate(ACTION_NAMES)}
N_ACTIONS = len(ACTION_NAMES)

FEATURE_NAMES = ['distance', 'sees_drone', 'hiding', 'using_jammer']

# Sizes of the state variables, in the order of the PoacherState fields
_STATE_SIZES = (MAX_DIST + 1, 2, 2, 2)
N_LIVE_STATES = int(np.prod(_STATE_SIZES))
TERMINAL = N_LIVE_STATES  # absorbing game-over state, all features 0
N_STATES = N_LIVE_STATES + 1


@dataclass(frozen=True)
class PoacherState:
    """
    Poacher state at one time step.

    Attributes:
        distance: Euclidean distance to the drone, rounded up and capped at MAX_DIST.
        sees_drone: the poacher detects the drone.
        hiding: the poacher is on a tree cell.
        using_jammer: GPS is jammed this step.
    """
    distance: int
    sees_drone: bool
    hiding: bool
    using_jammer: bool


def encode_state(state: PoacherState) -> int:
    """Map a PoacherState to an index in [0, N_LIVE_STATES). Share this with the MDP code."""
    values = (state.distance, int(state.sees_drone), int(state.hiding), int(state.using_jammer))
    return int(np.ravel_multi_index(values, _STATE_SIZES))


def decode_state(s: int) -> PoacherState:
    """Inverse of encode_state for live states."""
    d, sees, hiding, jammer = np.unravel_index(s, _STATE_SIZES)
    return PoacherState(int(d), bool(sees), bool(hiding), bool(jammer))


def make_features() -> np.ndarray:
    """State features, shape (N_STATES, K), in the order of FEATURE_NAMES."""
    phi = np.zeros((N_STATES, len(FEATURE_NAMES)))
    for s in range(N_LIVE_STATES):
        st = decode_state(s)
        phi[s] = [st.distance / MAX_DIST, st.sees_drone, st.hiding, st.using_jammer]
    # TERMINAL keeps all-zero features
    return phi


@dataclass
class Episode:
    """
    One game converted to poacher states and actions.

    Attributes:
        states: state index per step, shape (n,).
        actions: action index per step, shape (n,); -1 where it cannot be inferred
            (the last step, unless the poacher shot or jammed).
        run_dir: where the game data came from.
    """
    states: np.ndarray
    actions: np.ndarray
    run_dir: Path


def _as_bool(column: pd.Series) -> np.ndarray:
    return column.astype(str).str.lower().eq('true').to_numpy()


def load_episode(run_dir: Path) -> Episode:
    """Read one game run directory and convert it to an Episode."""
    game = pd.read_csv(run_dir / GAME_DATA_CSV)
    poacher = pd.read_csv(run_dir / POACHER_DATA_CSV)
    drone = pd.read_csv(run_dir / DRONE_DATA_CSV)
    grid = np.loadtxt(run_dir / MAP_DATA_CSV, delimiter=',', dtype=int, ndmin=2)  # grid[y][x]
    n = len(game)
    if not (len(poacher) == len(drone) == n):
        raise ValueError(f"{run_dir}: CSVs have different numbers of rows")

    px, py = game['poacher.x'].to_numpy(), game['poacher.y'].to_numpy()
    dx, dy = game['drone.x'].to_numpy(), game['drone.y'].to_numpy()
    true_dist = np.hypot(px - dx, py - dy)

    height, width = grid.shape
    on_tree = np.array([0 <= x < width and 0 <= y < height and grid[y, x] == CellType.TREES
                        for x, y in zip(px, py)])
    sees_drone = _as_bool(poacher['drone_detected'])
    jamming = ~_as_bool(drone['gps_available'])
    # A shot at step t is seen by the drone at step t+1
    drone_saw_shot = drone['poacher_action'].astype(str).eq('fire').to_numpy()
    scripted_actions = poacher['action'].astype(str).to_numpy()

    states = np.array([
        encode_state(PoacherState(
            distance=min(MAX_DIST, math.ceil(true_dist[t])),
            sees_drone=bool(sees_drone[t]),
            hiding=bool(on_tree[t]),
            using_jammer=bool(jamming[t]),
        ))
        for t in range(n)
    ], dtype=int)

    # Actions. Scripted runs log the action names; human runs do not, so shooting and
    # jamming are also detected from the drone's view one step later.
    # Priority when several apply: shoot > jam > movement.
    actions = np.full(n, -1, dtype=int)
    for t in range(n):
        has_next = t + 1 < n
        if 'fire' in scripted_actions[t] or (has_next and drone_saw_shot[t + 1]):
            actions[t] = ACTION_INDEX['shoot']
        elif 'deny_gps' in scripted_actions[t] or (has_next and jamming[t + 1]):
            actions[t] = ACTION_INDEX['jam']
        elif not has_next:
            continue  # unknown: the game ended before the next position was recorded
        elif (px[t + 1], py[t + 1]) == (px[t], py[t]):
            actions[t] = ACTION_INDEX['stay']
        elif on_tree[t + 1]:
            actions[t] = ACTION_INDEX['hide']
        else:
            # compare against the drone's position when the move was chosen
            new_dist = math.hypot(px[t + 1] - dx[t], py[t + 1] - dy[t])
            if new_dist < true_dist[t]:
                actions[t] = ACTION_INDEX['approach']
            elif new_dist > true_dist[t]:
                actions[t] = ACTION_INDEX['retreat']
            else:
                actions[t] = ACTION_INDEX['sidestep']

    return Episode(states=states, actions=actions, run_dir=run_dir)


def find_runs(roots: list[Path]) -> list[Path]:
    """All directories under the roots that contain a complete set of game CSVs."""
    needed = [GAME_DATA_CSV, POACHER_DATA_CSV, DRONE_DATA_CSV, MAP_DATA_CSV]
    runs = {p.parent for root in roots for p in Path(root).rglob(POACHER_DATA_CSV)}
    return sorted(r for r in runs if all((r / f).exists() for f in needed))


def load_episodes(roots: list[Path]) -> list[Episode]:
    return [load_episode(run) for run in find_runs(roots)]


def build_demos(episodes: list[Episode], horizon: int) -> list[Trajectory]:
    """
    Pad each episode with TERMINAL to exactly `horizon` steps. The padding guarantees at
    least one TERMINAL step, so horizon must exceed the longest episode.
    Unknown actions are filled with 'stay'; they do not affect the IRL gradient, which
    only uses states.
    """
    demos = []
    for ep in episodes:
        n = len(ep.states)
        if n >= horizon:
            raise ValueError(f"{ep.run_dir}: episode has {n} steps, horizon must be > {n}")
        states = np.concatenate([ep.states, np.full(horizon - n, TERMINAL)])
        actions = np.concatenate([np.where(ep.actions < 0, ACTION_INDEX['stay'], ep.actions),
                                  np.full(horizon - n, ACTION_INDEX['stay'])])
        demos.append(Trajectory(states, actions))
    return demos


def estimate_mdp(episodes: list[Episode], alpha: float = 1.0) -> TabularMDP:
    """
    Estimate T and p0 from episodes by counting transitions.

    Smoothing: each (s, a) gets `alpha` pseudo-counts spread over the next states seen
    from s with any action, so rarely tried actions borrow from what usually happens in s
    instead of jumping to states never observed. States never visited go to TERMINAL.
    The final transition of each episode (to TERMINAL) is split evenly over all actions
    when the action is unknown, since the game ending usually does not depend on it.
    """
    counts = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    for ep in episodes:
        next_states = np.append(ep.states[1:], TERMINAL)
        for s, a, s_next in zip(ep.states, ep.actions, next_states):
            if a >= 0:
                counts[s, a, s_next] += 1
            else:
                counts[s, :, s_next] += 1.0 / N_ACTIONS

    pooled = counts.sum(axis=1)  # (S, S), next-state counts from s over all actions
    pooled_total = pooled.sum(axis=1, keepdims=True)
    prior = np.zeros((N_STATES, N_STATES))
    visited = pooled_total[:, 0] > 0
    prior[visited] = pooled[visited] / pooled_total[visited]
    prior[~visited, TERMINAL] = 1.0

    n_sa = counts.sum(axis=2, keepdims=True)
    T = (counts + alpha * prior[:, None, :]) / (n_sa + alpha)
    T[TERMINAL] = 0.0
    T[TERMINAL, :, TERMINAL] = 1.0

    p0 = np.bincount([ep.states[0] for ep in episodes], minlength=N_STATES).astype(float)
    return TabularMDP(T=T, p0=p0 / p0.sum())


def bc_policy(episodes: list[Episode], horizon: int, alpha: float = 1.0) -> np.ndarray:
    """Behavior cloning baseline: smoothed action frequencies per state, shape (H, S, A)."""
    counts = np.zeros((N_STATES, N_ACTIONS))
    for ep in episodes:
        known = ep.actions >= 0
        np.add.at(counts, (ep.states[known], ep.actions[known]), 1)
    policy = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * N_ACTIONS)
    return np.broadcast_to(policy, (horizon, N_STATES, N_ACTIONS))


def step_log_likelihood(policy: np.ndarray, episodes: list[Episode]) -> float:
    """Average log pi_t(a_t | s_t) per step, over real steps with a known action."""
    total, steps = 0.0, 0
    for ep in episodes:
        t = np.flatnonzero(ep.actions >= 0)
        total += np.sum(np.log(policy[t, ep.states[t], ep.actions[t]]))
        steps += len(t)
    return total / steps


def main():
    parser = argparse.ArgumentParser(description="Learn poacher reward weights from WPG game runs.")
    parser.add_argument('roots', nargs='+', type=Path, help="directories containing game run folders")
    parser.add_argument('--horizon', type=int, default=101, help="sim.timeout + 1")
    parser.add_argument('--test-frac', type=float, default=0.2, help="fraction of episodes held out")
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=1.0, help="smoothing for T and BC")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    episodes = load_episodes(args.roots)
    if not episodes:
        raise SystemExit(f"no game runs found under {[str(r) for r in args.roots]}")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(episodes))
    n_test = int(len(episodes) * args.test_frac)
    test = [episodes[i] for i in order[:n_test]]
    train = [episodes[i] for i in order[n_test:]]

    all_states = np.concatenate([ep.states for ep in train])
    all_actions = np.concatenate([ep.actions for ep in train])
    print(f"episodes: {len(train)} train, {len(test)} test; {len(all_states)} train steps; "
          f"{len(np.unique(all_states))} of {N_STATES} states visited")
    print("action counts: " + ", ".join(
        f"{name} {np.sum(all_actions == i)}" for i, name in enumerate(ACTION_NAMES)
    ) + f", unknown {np.sum(all_actions < 0)}")

    mdp = estimate_mdp(train, alpha=args.alpha)
    phi = make_features()
    result = maxent_irl(mdp, phi, build_demos(train, args.horizon), args.horizon,
                        n_iters=args.iters, lr=0.05, l2=args.l2, verbose=True)

    print(f"\n{'feature':<16}{'weight':>8}")
    for name, w in zip(FEATURE_NAMES, result.theta):
        print(f"{name:<16}{w:>8.3f}")

    if test:
        uniform = np.full((args.horizon, N_STATES, N_ACTIONS), 1.0 / N_ACTIONS)
        print("\nheld-out log-likelihood per step (higher is better):")
        print(f"  MaxEnt IRL  {step_log_likelihood(result.policy, test):.4f}")
        print(f"  BC          {step_log_likelihood(bc_policy(train, args.horizon, args.alpha), test):.4f}")
        print(f"  uniform     {step_log_likelihood(uniform, test):.4f}")


if __name__ == '__main__':
    main()
