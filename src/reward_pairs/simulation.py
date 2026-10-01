"""Fixed-utility Fake Human; no learning, history, or human actions enter it."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.special import expit

from .trajectories import Trial

DEFAULT_UTILITIES = (1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0)


def stimulus_pairs(states: Sequence[tuple[int, int]] | np.ndarray) -> np.ndarray:
    pairs = np.asarray(states)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) == 0:
        raise ValueError("States must be a nonempty (n_trials, 2) array.")
    if not np.isin(pairs, np.arange(1, 9)).all():
        raise ValueError("Stimulus labels must be integers in 1..8.")
    return pairs.astype(int)


def utility_vector(utilities: Sequence[float]) -> np.ndarray:
    values = np.asarray(utilities, dtype=float)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("Provide eight finite utilities, in stimulus 1..8 order.")
    return values


def left_probability(
    states: Sequence[tuple[int, int]] | np.ndarray,
    utilities: Sequence[float] = DEFAULT_UTILITIES,
    beta: float = 1.0,
) -> np.ndarray:
    """Stable softmax: sigmoid(beta * (U_left - U_right))."""
    pairs = stimulus_pairs(states)
    values = utility_vector(utilities)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative.")
    if beta == 0:
        return np.full(len(pairs), 0.5)
    with np.errstate(over="ignore"):
        logits = beta * (values[pairs[:, 0] - 1] - values[pairs[:, 1] - 1])
    return expit(logits)


def simulate_choices(
    states: Sequence[tuple[int, int]] | np.ndarray,
    *, seed: int = 0, beta: float = 1.0,
    utilities: Sequence[float] = DEFAULT_UTILITIES,
) -> pd.DataFrame:
    pairs = stimulus_pairs(states)
    probabilities = left_probability(pairs, utilities, beta)
    draws = np.random.default_rng(seed).random(len(pairs))
    return pd.DataFrame({
        "left_stimulus": pairs[:, 0], "right_stimulus": pairs[:, 1],
        "synthetic_action": np.where(draws < probabilities, "left", "right"),
    })


def simulate_trajectory(
    trajectory: Sequence[Trial], *, seed: int = 0, beta: float = 1.0,
    utilities: Sequence[float] = DEFAULT_UTILITIES,
) -> pd.DataFrame:
    """Read only displayed states, including rows where human action is missing."""
    return simulate_choices([trial.state for trial in trajectory], seed=seed,
                            beta=beta, utilities=utilities)
