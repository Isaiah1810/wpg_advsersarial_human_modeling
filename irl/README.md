# IRL
Maximum (causal) entropy inverse reinforcement learning (Ziebart 2008, 2010) for learning
the poacher's reward from gameplay. Given an MDP without a reward, state features, and
demonstrations, it learns linear reward weights `theta` such that `R(s) = theta . phi(s)`.

The algorithm (`maxent_irl.py`) only depends on numpy and does not depend on WPG, so any
MDP can be plugged in. `wpg_adapter.py` connects it to the WPG game data.

## Files
- `maxent_irl.py`: the algorithm (`maxent_irl`) and helpers
- `wpg_adapter.py`: converts WPG game runs into IRL inputs and learns the poacher's reward
- `gridworld.py`: a small gridworld with a known reward, used for testing
- `test_maxent_irl.py`: tests
- `demo_gridworld.py`: prints each stage of IRL on the gridworld

## Pipeline
```
game runs (CSVs) --load_episode--> states + actions --build_demos--> demos ---------+
                 --estimate_mdp--> T, p0 (temporary MDP) --------------------------+--> maxent_irl --> theta
make_features --> phi (the 4 poacher state variables) ----------------------------+
```
Until the poacher MDP in `mdp/` is ready, `T` and `p0` are estimated by counting
transitions in the game data. To use the team's MDP instead, replace the output of
`estimate_mdp` with a `TabularMDP` built from it. Everything else stays the same, as long
as it uses the same state numbering (`encode_state` / `decode_state`) and action set.

## Running on WPG Data
From the repository root, with the virtual environment active:

1. Generate games. `live=false` is required, otherwise the game waits for keyboard input.
   ```shell
   wpg --multirun 'run=range(200)' poacher.behavior=2 live=false sim.seed=200 hydra.sweep.dir=data/b2
   ```
2. Learn the reward:
   ```shell
   python -m irl.wpg_adapter data/b2
   ```
   Any directory containing game run folders works, including several at once
   (`python -m irl.wpg_adapter data/b1 data/b2`). Human runs work too if they use the same
   CSV format.

Options: `--iters` (default 500), `--l2` (default 0.01), `--test-frac` (default 0.2),
`--alpha` (smoothing, default 1.0), `--horizon` (must be `sim.timeout + 1`, default 101),
`--seed` (train/test split).

The output shows the number of episodes and visited states, the action counts, training
progress, the learned weights, and the held-out log-likelihood per step for MaxEnt IRL, a
behavior cloning (BC) baseline, and a uniform random policy.

### State Space
The state is the poacher MDP's state from `mdp/explicit.py`, and the features are the same
four variables:

| Variable | Values | Meaning |
|---|---|---|
| `distance` | 0 to 20 | Euclidean distance to the drone, rounded up, capped at 20 |
| `sees_drone` | True / False | The poacher detects the drone |
| `hiding` | True / False | The poacher is on a tree cell |
| `using_jammer` | True / False | GPS is jammed this step |

That gives 21 x 2 x 2 x 2 = 168 states, plus a `TERMINAL` state (index 168) for game over,
with all-zero features. The index is
`distance * 8 + sees_drone * 4 + hiding * 2 + using_jammer`.

### Actions
Actions are inferred from the logs, the same way for scripted and human games:
`approach`, `retreat`, `sidestep`, `stay`, `hide`, `shoot`, `jam`. Shooting and jamming are
also detected from the drone's log one step later, since human games do not record actions.
If several apply in one step, only one is kept (shoot > jam > movement). The last step of
each game has no known action and is left out of the evaluation.

### Example Results
200 scripted games per behavior (160 train, 40 test):

| Weight | Behavior 1 (ignores the drone) | Behavior 2 (hides when it sees the drone) |
|---|---|---|
| distance | 0.64 | 3.62 |
| sees_drone | 0.27 | 0.66 |
| hiding | 0.00 | 3.71 |
| using_jammer | 0.00 | 0.00 |

Hiding separates the two behaviors. `using_jammer` is 0 because these poachers never jam:
a weight of 0 for a feature that never appears means "no data", not "does not care".

## Running the Tests
```shell
python -m irl.test_maxent_irl
python -m irl.demo_gridworld
```
or, with pytest installed, `python -m pytest irl`. The tests check that the gradient
matches finite differences, and that IRL recovers a known reward on the gridworld.

## Using the Algorithm Directly
```python
from irl.maxent_irl import TabularMDP, Trajectory, maxent_irl

mdp = TabularMDP(T=T, p0=p0)    # T: (S, A, S) with T[s, a, s'] = P(s' | s, a); p0: (S,)
demos = [Trajectory(states, actions), ...]    # integer indices, each of length H
result = maxent_irl(mdp, phi, demos, horizon=H, l2=0.01)    # phi: (S, K)

result.theta     # learned weights, shape (K,)
result.reward    # reward per state, shape (S,)
result.policy    # time-dependent policy, shape (H, S, A): policy[t, s, a]
result.history   # per-iteration log_likelihood and grad_norm
```

Other helpers: `log_likelihood(policy, demos)` to score held-out data,
`expected_return(mdp, reward, policy)`, and `sample_trajectories(mdp, policy, n, rng)`.

## Notes
- **Convergence:** training stops when the gradient norm falls below `tol` (1e-4) or at
  the iteration limit. Check the last printed `|grad|`: if it is still clearly above 0.001,
  rerun with a larger `--iters`. Behavior 2 above needed more than 500 (`--iters 2000`).
- **Fixed length:** every demonstration must have exactly `horizon` states.
  `wpg_adapter` pads games that end early with `TERMINAL`.
- **Initial distribution:** `p0` should be the distribution of the first states in the data.
- **Feature scaling:** scale features to similar ranges (e.g. 0 to 1).
- **Constant feature:** a feature that is 1 in every state has no effect with fixed-length
  episodes. With an absorbing terminal state, it acts as a per-step "stay alive" reward.
- **Regularization:** use a small `l2` with human data so `theta` does not keep growing when
  demonstrations are very consistent.
- **Impossible transitions:** if a transition in the data has probability 0 under `T`, the
  demonstrations cannot be matched and `theta` will diverge. Check `T[s_t, a_t, s_t+1] > 0`
  when switching to the team's MDP.
- **Interpreting theta:** overlapping features (e.g. `sees_drone` and `hiding`) can trade
  off against each other, so the exact values are not reliable. Report signs, relative
  order, and held-out log-likelihood.
- **IRL vs. BC:** BC predicts the actions of the same poacher better, since it fits each
  state separately. IRL's advantage is a reward that still applies when the drone changes.
- **Memory:** `T` is dense, `S * A * S` floats. Fine for a few thousand states; beyond that,
  a sparse representation is needed.

## Known Limitations
- The game does not log poaching per step, so it cannot be an action or a feature yet.
- `distance` is rounded up from Euclidean distance. If the team's MDP defines it
  differently, `encode_state` must be changed to match.
- Results so far come from scripted poachers and a temporary MDP, not from human data.
