"""
Walk through MaxEnt IRL on the gridworld and print what happens at each stage.

Run from the repository root:
    python -m irl.demo_gridworld
"""

import numpy as np

from irl.gridworld import (ACTION_NAMES, FEATURE_NAMES, default_layout, make_features, make_gridworld,
                           state_coords, state_index)
from irl.maxent_irl import (expected_return, feature_expectations, log_likelihood, maxent_irl,
                            sample_trajectories, soft_value_iteration, state_visitation)

HORIZON = 20
TRUE_THETA = np.array([3.0, -2.0, -1.0])
ARROWS = {'up': '^', 'down': 'v', 'left': '<', 'right': '>', 'stay': '.'}


def policy_map(policy, layout):
    """Most likely action at t=0 in each cell, one string per row."""
    width, height = layout['width'], layout['height']
    rows = []
    for y in range(height):
        cells = []
        for x in range(width):
            if (x, y) == layout['goal']:
                cells.append('G')
            elif (x, y) in layout['hazards']:
                cells.append('X')
            else:
                cells.append(ARROWS[ACTION_NAMES[policy[0, state_index(x, y, width)].argmax()]])
        rows.append(' '.join(cells))
    return rows


def main():
    layout = default_layout()
    width, height = layout['width'], layout['height']
    mdp = make_gridworld(width, height, slip=0.1)
    phi = make_features(width, height, layout['goal'], layout['hazards'])
    rng = np.random.default_rng(0)

    print("== 1. Expert demonstrations from a known reward ==")
    print(f"true theta: {dict(zip(FEATURE_NAMES, TRUE_THETA))}")
    expert = soft_value_iteration(mdp, phi @ TRUE_THETA, HORIZON)
    demos = sample_trajectories(mdp, expert, 300, rng)
    held_out = sample_trajectories(mdp, expert, 100, rng)
    print(f"first demo (x, y): {[tuple(int(c) for c in state_coords(s, width)) for s in demos[0].states[:8]]} ...")
    print(f"feature expectations of demos: {np.round(feature_expectations(demos, phi), 2)}")

    print("\n== 2. Learn theta (log-likelihood should rise, |grad| should fall to ~0) ==")
    result = maxent_irl(mdp, phi, demos, HORIZON, n_iters=1000, lr=0.05, verbose=True)
    print(f"uniform-policy log-likelihood for reference: {(HORIZON - 1) * np.log(1 / mdp.n_actions):.3f}")

    print("\n== 3. Compare ==")
    print(f"{'feature':<14}{'true':>8}{'learned':>10}")
    for name, true, learned in zip(FEATURE_NAMES, TRUE_THETA, result.theta):
        print(f"{name:<14}{true:>8.2f}{learned:>10.2f}")

    mu_learned = state_visitation(mdp, result.policy) @ phi
    print(f"\nfeature expectations, demos:   {np.round(feature_expectations(demos, phi), 2)}")
    print(f"feature expectations, learned: {np.round(mu_learned, 2)}")

    true_reward = phi @ TRUE_THETA
    uniform = np.full_like(expert, 1.0 / mdp.n_actions)
    v_expert, v_learned, v_uniform = (expected_return(mdp, true_reward, p) for p in (expert, result.policy, uniform))
    print(f"\nreturn under TRUE reward: expert {v_expert:.2f}, learned {v_learned:.2f}, random {v_uniform:.2f}")
    print(f"held-out log-likelihood:  expert {log_likelihood(expert, held_out):.2f}, "
          f"learned {log_likelihood(result.policy, held_out):.2f}, random {log_likelihood(uniform, held_out):.2f}")

    print("\nmost likely action at t=0 (G = goal, X = hazard)")
    print(f"{'expert':<18}learned")
    for left, right in zip(policy_map(expert, layout), policy_map(result.policy, layout)):
        print(f"{left}   {right}")


if __name__ == '__main__':
    main()
