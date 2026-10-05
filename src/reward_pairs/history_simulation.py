"""Sequential Fake Human with fixed utilities and a bounded causal history bonus."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from .history import CHOICE_COLUMNS, ChoiceHistory
from .simulation import DEFAULT_UTILITIES, stimulus_pairs, utility_vector
from .trajectories import Trial


@dataclass
class HistoryRollout:
    choices: pd.DataFrame  # The ONLY part handed to estimators; no truth/probabilities.
    probabilities: np.ndarray  # Generator diagnostics, kept separate.
    final_history: ChoiceHistory


def history_probability(
    left: int, right: int, history: ChoiceHistory, utilities: Sequence[float],
    *, alpha: float, beta: float = 1.0,
) -> float:
    """Probability before any update; also used for frozen-history probes."""
    values = utility_vector(utilities)
    stimulus_pairs([(left, right)])
    if not np.isfinite(alpha) or not np.isfinite(beta) or beta < 0:
        raise ValueError("Require finite alpha and finite beta >= 0.")
    h = history.centered
    return float(expit(beta * (values[left - 1] - values[right - 1] + alpha * (h[left - 1] - h[right - 1]))))


def simulate_history(
    trajectory: Sequence[Trial], *, alpha: float, beta: float = 1.0,
    seed: int | np.random.SeedSequence = 0, rollout: int = 0,
    utilities: Sequence[float] = DEFAULT_UTILITIES, reset_history_each_session: bool = False,
) -> HistoryRollout:
    """Ignore human actions; carry history across sessions/gaps by default."""
    if not trajectory or len({t.participant for t in trajectory}) != 1:
        raise ValueError("Provide a nonempty single-participant scaffold.")
    ordered = sorted(trajectory, key=lambda t: (t.session, t.trial))
    keys = [(t.session, t.trial) for t in ordered]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate session/trial scaffold keys.")
    if any(not isinstance(s, (int, np.integer)) or s not in range(1, 6)
           or not isinstance(t, (int, np.integer)) or t < 1 for s, t in keys):
        raise ValueError("Scaffold needs integer sessions 1..5 and positive trial numbers.")
    pairs = stimulus_pairs([t.state for t in ordered])
    values = utility_vector(utilities)
    if not np.isfinite(alpha) or not np.isfinite(beta) or beta < 0:
        raise ValueError("Require finite alpha and finite beta >= 0.")
    draws = np.random.default_rng(seed).random(len(ordered))
    probabilities = np.empty(len(ordered))
    history = ChoiceHistory()
    rows = []
    previous_session = None
    for i, (trial, (left, right)) in enumerate(zip(ordered, pairs)):
        if reset_history_each_session and trial.session != previous_session:
            history = ChoiceHistory()
        # Read history first; sample next; update ONLY after the action exists.
        h = history.centered
        probabilities[i] = expit(beta * (values[left - 1] - values[right - 1]
                                         + alpha * (h[left - 1] - h[right - 1])))
        action = "left" if draws[i] < probabilities[i] else "right"
        rows.append((rollout, trial.session, trial.trial, left, right, action))
        history.update(left, right, action)
        previous_session = trial.session
    return HistoryRollout(pd.DataFrame(rows, columns=CHOICE_COLUMNS), probabilities, history)


def independent_rollouts(
    trajectory: Sequence[Trial], n_rollouts: int, *, seed: int = 0,
    alpha: float, beta: float = 1.0, utilities: Sequence[float] = DEFAULT_UTILITIES,
    reset_history_each_session: bool = False,
) -> list[HistoryRollout]:
    """Spawn independent RNG streams and fresh histories, with stable prefixes.

    n=1,5,20 use nested sets of independent agents, not one long history.
    The same seed across alpha conditions supplies common random draws.
    """
    if not isinstance(n_rollouts, (int, np.integer)) or n_rollouts < 1:
        raise ValueError("n_rollouts must be a positive integer.")
    streams = np.random.SeedSequence(seed).spawn(n_rollouts)
    return [simulate_history(trajectory, alpha=alpha, beta=beta, seed=stream, rollout=i,
                             utilities=utilities, reset_history_each_session=reset_history_each_session)
            for i, stream in enumerate(streams)]
