"""Explicit tabular pair MDP and entropy-regularized Bellman solver.

Decision states are ordered pairs only. The last transition column represents
termination with value zero and no available actions (hence no terminal entropy).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .simulation import stimulus_pairs, utility_vector
from .trajectories import Trial

ACTIONS = ("left", "right")


@dataclass(frozen=True)
class TransitionSummary:
    observations: int
    sessions: int
    consecutive_edges: int
    session_endings: int
    gap_endings: int


@dataclass(frozen=True)
class PairMDP:
    """P has shape (n_states, 2, n_states + 1); last column is terminal.

    The Bellman solver also accepts action-dependent test kernels. The empirical
    builder always makes actions identical; v0 IRL rejects unequal kernels.
    """

    states: tuple[tuple[int, int], ...]
    transitions: np.ndarray
    summary: TransitionSummary | None = None

    def __post_init__(self) -> None:
        states = tuple(map(tuple, stimulus_pairs(self.states).tolist()))
        if len(set(states)) != len(states):
            raise ValueError("MDP states must be unique ordered pairs.")
        transitions = np.array(self.transitions, dtype=float, copy=True)
        n = len(states)
        if transitions.shape != (n, 2, n + 1):
            raise ValueError("Transitions must have shape (states, 2, states + terminal).")
        if not np.isfinite(transitions).all() or (transitions < 0).any():
            raise ValueError("Transition probabilities must be finite and nonnegative.")
        if not np.allclose(transitions.sum(axis=2), 1, atol=1e-12, rtol=0):
            raise ValueError("Every transition row must sum to one, including termination.")
        transitions.setflags(write=False)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "transitions", transitions)

    @property
    def features(self) -> np.ndarray:
        """phi[s, action, stimulus]: one-hot selected stimulus, with no truth."""
        return np.eye(8)[np.asarray(self.states) - 1]

    @property
    def action_independent(self) -> bool:
        return np.array_equal(self.transitions[:, 0], self.transitions[:, 1])

    def rewards(self, weights: Sequence[float]) -> np.ndarray:
        return self.features @ utility_vector(weights)


def build_empirical_mdp(trials: Sequence[Trial]) -> PairMDP:
    """Count only observed consecutive trials within each participant-session.

    Each retained trial contributes one outgoing count. End-of-session rows
    and rows just before missing trial numbers transition to terminal. A row
    after a gap starts a new observed fragment. No choices/rewards are read.
    This is a stationary Markov approximation to the observed fragments, not
    a reconstruction of the experiment's full time-dependent schedule.
    """
    if not trials:
        raise ValueError("Cannot build an MDP from an empty trajectory.")
    ordered = sorted(trials, key=lambda t: (t.participant, t.session, t.trial))
    keys = [(t.participant, t.session, t.trial) for t in ordered]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate participant/session/trial keys in MDP scaffold.")
    if any(not isinstance(t.session, (int, np.integer)) or t.session not in range(1, 6)
           or not isinstance(t.trial, (int, np.integer)) or t.trial < 1 for t in ordered):
        raise ValueError("MDP scaffold requires integer training sessions 1..5 and positive trial numbers.")
    states = tuple(sorted(set(t.state for t in ordered)))
    stimulus_pairs(states)
    index = {state: i for i, state in enumerate(states)}
    n = len(states)
    counts = np.zeros((n, n + 1))
    edges = session_endings = gap_endings = 0
    for i, trial in enumerate(ordered):
        following = ordered[i + 1] if i + 1 < len(ordered) else None
        same_session = following is not None and (
            trial.participant, trial.session
        ) == (following.participant, following.session)
        if same_session and following.trial == trial.trial + 1:
            destination = index[following.state]
            edges += 1
        else:
            destination = n
            if same_session:
                gap_endings += 1
            else:
                session_endings += 1
        counts[index[trial.state], destination] += 1
    kernel = counts / counts.sum(axis=1, keepdims=True)
    summary = TransitionSummary(len(ordered), len({key[:2] for key in keys}),
                                edges, session_endings, gap_endings)
    return PairMDP(states, np.repeat(kernel[:, None, :], 2, axis=1), summary)


class SoftValueError(ValueError):
    """Soft Bellman iteration did not produce a finite converged solution."""


@dataclass
class SoftSolution:
    values: np.ndarray
    q_values: np.ndarray
    log_policy: np.ndarray
    iterations: int
    bellman_residual: float

    @property
    def policy(self) -> np.ndarray:
        return np.exp(self.log_policy)


def soft_value_iteration(
    mdp: PairMDP, weights: Sequence[float], *, beta: float = 1.0,
    gamma: float = 0.95, tolerance: float = 1e-10, max_iterations: int = 10_000,
) -> SoftSolution:
    """Solve Q=r+gamma*P*V, V=logsumexp(beta*Q)/beta explicitly.

    Termination contributes zero. Shift Q before exponentiation to compute
    stable values and log policies. No logistic shortcut is used here.
    tolerance bounds the absolute Bellman residual; for gamma<1, the usual
    value error bound is residual/(1-gamma).
    """
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("Soft planning requires finite beta > 0.")
    if not np.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("Discount gamma must be in [0, 1).")
    if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError("Bellman tolerance and iteration limit must be positive.")
    rewards = mdp.rewards(weights)
    # Exclude terminal column: V(terminal)=0, not log(2)/beta.
    transitions = mdp.transitions[:, :, :-1]
    values = np.zeros(len(mdp.states))
    for iteration in range(1, max_iterations + 1):
        q = rewards + gamma * (transitions @ values)
        maximum = q.max(axis=1, keepdims=True)
        logits = beta * (q - maximum)
        normalizer = logsumexp(logits, axis=1, keepdims=True)
        updated = maximum[:, 0] + normalizer[:, 0] / beta
        if not np.isfinite(updated).all() or not np.isfinite(logits).all():
            raise SoftValueError("Soft Bellman calculation became nonfinite.")
        residual = float(np.max(np.abs(updated - values)))
        if residual <= tolerance:
            # q and policy use precisely the values returned here.
            return SoftSolution(values, q, logits - normalizer, iteration, residual)
        values = updated
    raise SoftValueError(f"Soft Bellman iteration failed after {max_iterations} iterations (residual={residual:.3g}).")


def soft_policy_jacobian(mdp: PairMDP, solution: SoftSolution, *, beta: float, gamma: float) -> np.ndarray:
    """Differentiate the converged Bellman equations, not a logistic identity.

    dV = (I-gamma*P_pi)^-1 E_pi[phi]; dQ = phi+gamma*P*dV;
    d log pi = beta*(dQ-E_pi[dQ]). Last axis contains all eight weights.
    """
    policy = solution.policy
    transitions = mdp.transitions[:, :, :-1]
    features = mdp.features
    p_policy = np.einsum("sa,san->sn", policy, transitions)
    expected_features = np.einsum("sa,sak->sk", policy, features)
    dv = np.linalg.solve(np.eye(len(mdp.states)) - gamma * p_policy, expected_features)
    dq = features + gamma * np.einsum("san,nk->sak", transitions, dv)
    return beta * (dq - np.einsum("sa,sak->sk", policy, dq)[:, None, :])
