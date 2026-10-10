"""Exact bounded-lookahead soft planning, with independent scalar and batch solvers."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .history import CHOICE_COLUMNS, ChoiceHistory
from .history_mdp import HistoryMDP, HistoryState
from .history_simulation import HistoryRollout
from .simulation import DEFAULT_UTILITIES, utility_vector


def check_planning(beta, gamma, lookahead):
    if not np.isfinite(beta) or beta <= 0 or not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("Require finite beta>0 and gamma in [0,1].")
    if not isinstance(lookahead, (int, np.integer)) or not 0 <= lookahead <= 10:
        raise ValueError("lookahead must be an integer 0..10 (explicit computation guard).")


@dataclass
class Plan:
    q: np.ndarray
    value: float
    log_policy: np.ndarray
    evaluated_states: int
    cache_hits: int


def soft_plan(mdp, state, utilities, *, alpha, beta=1., gamma=.95, lookahead=3, memoize=True):
    """h means h FUTURE decisions plus the current one; terminal/truncation V=0.

    At each future node, remaining depth decreases. At the next actual trial
    the agent replans with a fresh window: this is receding-horizon behavior.
    """
    check_planning(beta, gamma, lookahead)
    w = utility_vector(utilities)
    if not np.isfinite(alpha):
        raise ValueError("Alpha must be finite.")
    mdp.validate(state)
    if state.t == len(mdp.pairs):
        return Plan(np.array([]), 0., np.array([]), 0, 0)
    end = min(len(mdp.pairs), state.t + lookahead + 1)
    evaluations = 0

    def solve(t, chosen):
        nonlocal evaluations
        evaluations += 1
        s = HistoryState(t, chosen)
        selected = mdp.pairs[t] - 1
        q = w[selected] + alpha * mdp.history(s)[selected]
        if t + 1 < end:
            for a, label in enumerate(("left", "right")):
                child = mdp.transition(s, label)
                q[a] += gamma * visit(child.t, child.chosen)[1]
        logits = beta * (q - q.max())
        norm = logsumexp(logits)
        return q, float(q.max() + norm / beta), logits - norm

    visit = lru_cache(maxsize=None)(solve) if memoize else solve
    q, v, lp = visit(state.t, state.chosen)
    return Plan(q, v, lp, evaluations, visit.cache_info().hits if memoize else 0)


class PlanningBatch:
    """Cache exact branch reward features once; vectorize Bellman backups in fits.

    This is an unmerged binary tree, not an approximate/discretized state model.
    Scalar DP above memoizes identical states and provides an independent check.
    A hard memory guard fails BEFORE allocation instead of silently truncating.
    """

    def __init__(self, mdp, times, chosen, *, lookahead=3, max_nodes=2_000_000):
        check_planning(1., .95, lookahead)
        times = np.asarray(times)
        chosen = np.asarray(chosen)
        if times.ndim != 1 or len(times) == 0 or chosen.shape != (len(times), 8):
            raise ValueError("Provide nonempty root times and eight chosen counts per root.")
        for t, c in zip(times, chosen):
            mdp.validate(HistoryState(t, tuple(c)))
        if (times == len(mdp.pairs)).any():
            raise ValueError("Batch roots must be nonterminal.")
        depth = min(lookahead, len(mdp.pairs) - 1 - int(times.min()))
        nodes = len(times) * (2 ** (depth + 1) - 1)
        if nodes > max_nodes:
            raise ValueError(f"Planning batch would allocate {nodes} nodes; split roots into smaller batches.")
        self.lookahead, self.features, self.active = lookahead, [], []
        self.nodes = nodes
        counts = chosen.astype(int)[:, None, :]
        for d in range(depth + 1):
            valid = times + d < len(mdp.pairs)
            t = np.minimum(times + d, len(mdp.pairs) - 1).astype(int)
            pair = mdp.pairs[t] - 1
            n = mdp.presented[t, None, :]
            h = np.divide(counts, n, out=np.full(counts.shape, .5), where=n > 0) - .5
            features = np.empty((len(times), counts.shape[1], 2, 8))
            features[..., :7] = np.eye(8)[pair, 1:][:, None, :, :]
            features[..., 7] = np.take_along_axis(h, pair[:, None, :], axis=2)
            features[~valid] = 0
            self.features.append(features)
            self.active.append(valid)
            if d < depth:
                counts = (counts[:, :, None, :] + np.eye(8, dtype=int)[pair][:, None, :, :]).reshape(len(times), -1, 8)

    def evaluate(self, parameters, *, beta=1., gamma=.95, gradient=False):
        """Return root Q, log policy, and optional d log pi / d(w2..w8,alpha)."""
        check_planning(beta, gamma, self.lookahead)
        theta = np.asarray(parameters, dtype=float)
        if theta.shape != (8,) or not np.isfinite(theta).all():
            raise ValueError("Provide seven finite free utilities and finite alpha.")
        value = derivative = None
        for features, active in zip(reversed(self.features), reversed(self.active)):
            q = features @ theta
            dq = features.copy() if gradient else None
            if value is not None:
                q += gamma * value.reshape(q.shape)
                if gradient:
                    dq += gamma * derivative.reshape(dq.shape)
            maximum = q.max(axis=-1, keepdims=True)
            logits = beta * (q - maximum)
            normalizer = logsumexp(logits, axis=-1, keepdims=True)
            lp = logits - normalizer
            value = (maximum + normalizer / beta)[..., 0]
            value[~active] = 0  # no terminal entropy, including padded branches
            if gradient:
                derivative = np.sum(np.exp(lp)[..., None] * dq, axis=-2)
                derivative[~active] = 0
        jac = beta * (dq[:, 0] - derivative[:, 0, None, :]) if gradient else None
        return q[:, 0], lp[:, 0], jac


def planning_rollouts(mdp: HistoryMDP, n_rollouts: int, *, alpha: float, utilities=DEFAULT_UTILITIES,
                      beta=1., gamma=.95, lookahead=3, seed=0):
    """Independent Planning Bobs. Only chronological synthetic choices go to fits."""
    check_planning(beta, gamma, lookahead)
    if not isinstance(n_rollouts, int) or n_rollouts < 1 or not np.isfinite(alpha):
        raise ValueError("Require positive integer rollout count and finite alpha.")
    w = utility_vector(utilities)
    theta = np.r_[w[1:] - w[0], alpha]
    streams = np.random.SeedSequence(seed).spawn(n_rollouts)
    draws = np.array([np.random.default_rng(s).random(len(mdp.pairs)) for s in streams])
    chosen = np.tile(mdp.initial.chosen, (n_rollouts, 1))
    probs = np.empty_like(draws)
    actions = np.empty(draws.shape, dtype=object)
    for t, pair in enumerate(mdp.pairs):
        batch = PlanningBatch(mdp, np.full(n_rollouts, t), chosen, lookahead=lookahead)
        _, lp, _ = batch.evaluate(theta, beta=beta, gamma=gamma)
        probs[:, t] = np.exp(lp[:, 0])
        left = draws[:, t] < probs[:, t]
        actions[:, t] = np.where(left, "left", "right")
        chosen[np.arange(n_rollouts), np.where(left, pair[0], pair[1]) - 1] += 1
    result = []
    for r in range(n_rollouts):
        rows = [(r, s, trial, int(l), int(right), a)
                for (s, trial), (l, right), a in zip(mdp.chronology, mdp.pairs, actions[r])]
        result.append(HistoryRollout(pd.DataFrame(rows, columns=CHOICE_COLUMNS), probs[r],
                                    ChoiceHistory(mdp.presented[-1].copy(), chosen[r].copy())))
    return result
