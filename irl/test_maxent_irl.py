"""
Tests for MaxEnt IRL on a gridworld with a known reward.

Run from the repository root with either:
    python -m pytest irl
    python -m irl.test_maxent_irl
"""

import numpy as np

from irl.gridworld import FEATURE_NAMES, default_layout, make_features, make_gridworld
from irl.maxent_irl import (
    TabularMDP,
    expected_return,
    feature_expectations,
    log_likelihood,
    maxent_gradient,
    maxent_irl,
    sample_trajectories,
    soft_value_iteration,
    state_visitation,
)

HORIZON = 20
TRUE_THETA = np.array([3.0, -2.0, -1.0])  # likes the goal, avoids hazards, prefers being close


def _setup(slip: float, start=None):
    layout = default_layout()
    mdp = make_gridworld(layout['width'], layout['height'], slip=slip, start=start)
    phi = make_features(layout['width'], layout['height'], layout['goal'], layout['hazards'])
    return mdp, phi


def test_mdp_rejects_bad_transitions():
    T = np.full((2, 1, 2), 0.4)  # rows sum to 0.8
    try:
        TabularMDP(T=T, p0=np.array([1.0, 0.0]))
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-normalized T")


def test_policy_and_visitation_are_normalized():
    mdp, phi = _setup(slip=0.1)
    policy = soft_value_iteration(mdp, phi @ TRUE_THETA, HORIZON)
    assert np.allclose(policy.sum(axis=2), 1.0)
    assert np.isclose(state_visitation(mdp, policy).sum(), HORIZON)


def test_gradient_matches_finite_differences():
    # With deterministic dynamics and a fixed start state, mu_demos - E[features]
    # is exactly the gradient of the demonstration log-likelihood.
    mdp, phi = _setup(slip=0.0, start=(0, 0))
    rng = np.random.default_rng(0)
    expert = soft_value_iteration(mdp, phi @ TRUE_THETA, HORIZON)
    demos = sample_trajectories(mdp, expert, 30, rng)
    mu_demos = feature_expectations(demos, phi)

    theta = rng.normal(size=len(TRUE_THETA))
    grad, _ = maxent_gradient(mdp, phi, mu_demos, theta, HORIZON)

    eps = 1e-5
    numeric = np.zeros_like(theta)
    for k in range(len(theta)):
        step = np.zeros_like(theta)
        step[k] = eps
        ll_plus = log_likelihood(soft_value_iteration(mdp, phi @ (theta + step), HORIZON), demos)
        ll_minus = log_likelihood(soft_value_iteration(mdp, phi @ (theta - step), HORIZON), demos)
        numeric[k] = (ll_plus - ll_minus) / (2 * eps)

    assert np.allclose(grad, numeric, rtol=1e-4, atol=1e-6), f"analytic {grad} vs numeric {numeric}"


def _recover(seed: int = 0, n_demos: int = 300):
    mdp, phi = _setup(slip=0.1)
    rng = np.random.default_rng(seed)
    expert = soft_value_iteration(mdp, phi @ TRUE_THETA, HORIZON)
    demos = sample_trajectories(mdp, expert, n_demos, rng)
    held_out = sample_trajectories(mdp, expert, 100, rng)
    result = maxent_irl(mdp, phi, demos, HORIZON, n_iters=1000, lr=0.05)
    return mdp, phi, expert, demos, held_out, result


def test_recovers_true_reward():
    mdp, phi, expert, demos, held_out, result = _recover()

    # 1. Learned weights have the right signs and roughly the right direction.
    #    The exact magnitudes are not identifiable from finite data (dist_to_goal and
    #    at_goal overlap), so a tight check on theta itself would be flaky.
    assert np.all(np.sign(result.theta) == np.sign(TRUE_THETA)), f"theta {result.theta}"
    cosine = result.theta @ TRUE_THETA / (np.linalg.norm(result.theta) * np.linalg.norm(TRUE_THETA))
    assert cosine > 0.7, f"cosine similarity {cosine:.3f}, theta {result.theta}"

    # 2. Feature expectations under the learned policy match the demonstrations
    mu_demos = feature_expectations(demos, phi)
    mu_learned = state_visitation(mdp, result.policy) @ phi
    assert np.allclose(mu_learned, mu_demos, rtol=0.05, atol=0.05), f"{mu_learned} vs {mu_demos}"

    # 3. Under the TRUE reward, the learned policy does almost as well as the expert
    #    (normalized so that a uniform random policy scores 0 and the expert scores 1)
    true_reward = phi @ TRUE_THETA
    uniform = np.full_like(expert, 1.0 / mdp.n_actions)
    v_expert = expected_return(mdp, true_reward, expert)
    v_learned = expected_return(mdp, true_reward, result.policy)
    v_uniform = expected_return(mdp, true_reward, uniform)
    normalized = (v_learned - v_uniform) / (v_expert - v_uniform)
    assert normalized > 0.95, f"normalized return {normalized:.3f}"

    # 4. Predicts unseen demonstrations better than a uniform policy
    assert log_likelihood(result.policy, held_out) > log_likelihood(uniform, held_out)


def main():
    tests = [
        test_mdp_rejects_bad_transitions,
        test_policy_and_visitation_are_normalized,
        test_gradient_matches_finite_differences,
        test_recovers_true_reward,
    ]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")

    _, _, _, _, held_out, result = _recover()
    print()
    print(f"{'feature':<14}{'true':>8}{'learned':>10}")
    for name, true, learned in zip(FEATURE_NAMES, TRUE_THETA, result.theta):
        print(f"{name:<14}{true:>8.2f}{learned:>10.2f}")
    print(f"\niterations: {len(result.history)}, final |grad|: {result.history[-1]['grad_norm']:.5f}")
    print(f"held-out log-likelihood per trajectory: {log_likelihood(result.policy, held_out):.2f}")


if __name__ == '__main__':
    main()
