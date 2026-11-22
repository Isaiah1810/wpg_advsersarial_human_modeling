"""
Maximum (causal) entropy inverse reinforcement learning for tabular MDPs.

The MDP is passed in as plain arrays, so any source can be plugged in without
changing the algorithm: a toy gridworld, transitions estimated from simulator
data, or a model exported from PRISM/Storm.

Model of the demonstrator (Ziebart 2008, 2010):
    R(s)      = theta . phi(s)
    Q_t(s, a) = R(s) + sum_s' T(s' | s, a) V_{t+1}(s')
    V_t(s)    = log sum_a exp Q_t(s, a)                     (V_H = 0)
    pi_t(a|s) = exp(Q_t(s, a) - V_t(s))

theta is fit by gradient ascent on the log-likelihood of the demonstrations.
The gradient is the difference between feature expectations:
    grad = mu_demos - sum_s D(s) phi(s)
where D(s) is the expected number of visits to s under the current policy.

Taking the expectation over s' inside Q makes this the maximum *causal*
entropy variant, which is the correct one for stochastic dynamics.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TabularMDP:
    """
    A finite MDP without a reward.

    Attributes:
        T: transition probabilities, shape (S, A, S), T[s, a, s'] = P(s' | s, a).
        p0: initial state distribution, shape (S,).
    """
    T: np.ndarray
    p0: np.ndarray

    def __post_init__(self):
        self.T = np.asarray(self.T, dtype=float)
        self.p0 = np.asarray(self.p0, dtype=float)
        if self.T.ndim != 3 or self.T.shape[0] != self.T.shape[2]:
            raise ValueError(f"T must have shape (S, A, S), got {self.T.shape}")
        if self.p0.shape != (self.T.shape[0],):
            raise ValueError(f"p0 must have shape ({self.T.shape[0]},), got {self.p0.shape}")
        if not np.allclose(self.T.sum(axis=2), 1.0):
            raise ValueError("each T[s, a, :] must sum to 1")
        if not np.isclose(self.p0.sum(), 1.0):
            raise ValueError("p0 must sum to 1")

    @property
    def n_states(self) -> int:
        return self.T.shape[0]

    @property
    def n_actions(self) -> int:
        return self.T.shape[1]


@dataclass
class Trajectory:
    """
    One demonstration of fixed length H.

    Attributes:
        states: state indices s_0 .. s_{H-1}, shape (H,).
        actions: action taken in each state, shape (H,). The last action has no
            effect within the horizon and is ignored by the likelihood.
    """
    states: np.ndarray
    actions: np.ndarray

    def __post_init__(self):
        self.states = np.asarray(self.states, dtype=int)
        self.actions = np.asarray(self.actions, dtype=int)
        if self.states.shape != self.actions.shape:
            raise ValueError("states and actions must have the same length")


@dataclass
class IRLResult:
    """
    Output of maxent_irl.

    Attributes:
        theta: learned feature weights, shape (K,).
        reward: learned reward per state, shape (S,).
        policy: soft-optimal policy for the learned reward, shape (H, S, A).
        history: per-iteration 'log_likelihood' and 'grad_norm'.
    """
    theta: np.ndarray
    reward: np.ndarray
    policy: np.ndarray
    history: list = field(default_factory=list)


def logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable log(sum(exp(x))) along an axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(x_max, axis=axis) + np.log(np.sum(np.exp(x - x_max), axis=axis))


def soft_value_iteration(mdp: TabularMDP, reward: np.ndarray, horizon: int) -> np.ndarray:
    """
    Backward pass: compute the soft-optimal, time-dependent policy.

    Returns:
        policy of shape (H, S, A), policy[t, s, a] = pi_t(a | s).
    """
    policy = np.empty((horizon, mdp.n_states, mdp.n_actions))
    V = np.zeros(mdp.n_states)  # V_H = 0
    for t in reversed(range(horizon)):
        Q = reward[:, None] + mdp.T @ V  # (S, A)
        V = logsumexp(Q, axis=1)
        policy[t] = np.exp(Q - V[:, None])
    return policy


def state_visitation(mdp: TabularMDP, policy: np.ndarray) -> np.ndarray:
    """
    Forward pass: expected number of visits to each state over the horizon.

    Returns:
        D of shape (S,), D[s] = sum_{t=0}^{H-1} P(s_t = s).
    """
    T_flat = mdp.T.reshape(-1, mdp.n_states)  # (S*A, S), so the step is one matrix product
    D_t = mdp.p0.copy()
    D = np.zeros(mdp.n_states)
    for t in range(policy.shape[0]):
        D += D_t
        # D_{t+1}(s') = sum_{s,a} D_t(s) pi_t(a|s) T(s'|s,a)
        D_t = (D_t[:, None] * policy[t]).reshape(-1) @ T_flat
    return D


def feature_expectations(demos: list[Trajectory], phi: np.ndarray) -> np.ndarray:
    """Average over demonstrations of the summed state features, shape (K,)."""
    return np.mean([phi[traj.states].sum(axis=0) for traj in demos], axis=0)


def log_likelihood(policy: np.ndarray, demos: list[Trajectory]) -> float:
    """Average over demonstrations of sum_t log pi_t(a_t | s_t), for t < H-1."""
    t = np.arange(policy.shape[0] - 1)
    return float(np.mean([
        np.sum(np.log(policy[t, traj.states[:-1], traj.actions[:-1]]))
        for traj in demos
    ]))


def maxent_gradient(mdp: TabularMDP, phi: np.ndarray, mu_demos: np.ndarray, theta: np.ndarray,
                    horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Gradient of the demonstration log-likelihood with respect to theta.

    Returns:
        (grad, policy), where policy is the soft-optimal policy for theta.
    """
    policy = soft_value_iteration(mdp, phi @ theta, horizon)
    D = state_visitation(mdp, policy)
    return mu_demos - D @ phi, policy


def maxent_irl(mdp: TabularMDP, phi: np.ndarray, demos: list[Trajectory], horizon: int,
               n_iters: int = 500, lr: float = 0.05, l2: float = 0.0, tol: float = 1e-4,
               theta_init: np.ndarray | None = None, verbose: bool = False) -> IRLResult:
    """
    Learn linear reward weights from demonstrations with maximum causal entropy IRL.

    Args:
        mdp: the MDP without a reward.
        phi: state features, shape (S, K). Scale features to similar ranges.
        demos: demonstrations, each exactly `horizon` steps long. For episodes that
            end early, pad them with an absorbing terminal state whose features are 0.
        horizon: number of states per trajectory (H).
        n_iters: maximum number of gradient steps.
        lr: Adam step size.
        l2: L2 regularization strength; keeps theta finite for near-deterministic demos.
        tol: stop when the gradient norm falls below this.
        theta_init: starting weights; zeros if None.
        verbose: print progress every 50 iterations.
    """
    phi = np.asarray(phi, dtype=float)
    if phi.shape[0] != mdp.n_states:
        raise ValueError(f"phi must have {mdp.n_states} rows, got {phi.shape[0]}")
    for traj in demos:
        if len(traj.states) != horizon:
            raise ValueError(f"all demonstrations must have length {horizon}, got {len(traj.states)}")

    mu_demos = feature_expectations(demos, phi)
    theta = np.zeros(phi.shape[1]) if theta_init is None else np.asarray(theta_init, dtype=float).copy()

    # Adam (gradient ascent)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    history = []
    for i in range(1, n_iters + 1):
        grad, policy = maxent_gradient(mdp, phi, mu_demos, theta, horizon)
        grad -= l2 * theta
        grad_norm = float(np.linalg.norm(grad))
        history.append({'log_likelihood': log_likelihood(policy, demos), 'grad_norm': grad_norm})

        if verbose and (i == 1 or i % 50 == 0):
            print(f"iter {i:4d}  log-lik {history[-1]['log_likelihood']:9.3f}  |grad| {grad_norm:.5f}")
        if grad_norm < tol:
            break

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        theta += lr * (m / (1 - beta1 ** i)) / (np.sqrt(v / (1 - beta2 ** i)) + eps)

    reward = phi @ theta
    return IRLResult(theta=theta, reward=reward, policy=soft_value_iteration(mdp, reward, horizon),
                     history=history)


def expected_return(mdp: TabularMDP, reward: np.ndarray, policy: np.ndarray) -> float:
    """Expected sum of rewards over the horizon when following policy from p0."""
    return float(state_visitation(mdp, policy) @ reward)


def sample_trajectories(mdp: TabularMDP, policy: np.ndarray, n: int,
                        rng: np.random.Generator) -> list[Trajectory]:
    """Sample n trajectories of length H = policy.shape[0]."""
    horizon = policy.shape[0]
    demos = []
    for _ in range(n):
        states = np.empty(horizon, dtype=int)
        actions = np.empty(horizon, dtype=int)
        s = rng.choice(mdp.n_states, p=mdp.p0)
        for t in range(horizon):
            a = rng.choice(mdp.n_actions, p=policy[t, s])
            states[t], actions[t] = s, a
            s = rng.choice(mdp.n_states, p=mdp.T[s, a])
        demos.append(Trajectory(states, actions))
    return demos
