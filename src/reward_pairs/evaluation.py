"""Truth is used here for evaluation, never by the utility estimator."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .recovery import RecoveryError, RecoveryResult, recover_utilities
from .simulation import DEFAULT_UTILITIES, simulate_choices, stimulus_pairs, utility_vector


def compare_utilities(
    result: RecoveryResult, true_utilities: Sequence[float] = DEFAULT_UTILITIES,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Reference-aligned errors; strict ordering excludes truly tied pairs.

    MAE is meaningful here only because simulation and fit use the same fixed
    beta. No post-hoc scale fitting or ground-truth information enters recovery.
    """
    truth = utility_vector(true_utilities)
    truth = truth - truth[result.reference - 1]
    recovered = result.utilities.reindex(range(1, 9)).to_numpy()
    a, b = np.triu_indices(8, k=1)
    true_difference = truth[b] - truth[a]
    estimated_difference = recovered[b] - recovered[a]
    strict = true_difference != 0
    tied = ~strict
    variable = np.ptp(truth) > 0 and np.ptp(recovered) > 0
    metrics = {
        "mae": float(np.mean(np.abs(truth - recovered))),
        "strict_order_agreement": float(np.mean(
            true_difference[strict] * estimated_difference[strict] > 0
        )) if strict.any() else float("nan"),
        "spearman": float(spearmanr(truth, recovered).statistic) if variable else float("nan"),
        "pearson": float(np.corrcoef(truth, recovered)[0, 1]) if variable else float("nan"),
        "true_tie_mean_gap": float(np.mean(np.abs(estimated_difference[tied]))) if tied.any() else float("nan"),
    }
    comparison = pd.DataFrame({
        "true_reference_aligned": truth, "recovered": recovered,
        "absolute_error": np.abs(truth - recovered),
    }, index=result.utilities.index)
    return comparison, metrics


def recovery_stability(
    states: Sequence[tuple[int, int]] | np.ndarray,
    *, seeds: Sequence[int] = (0, 1, 2, 3, 4), repeats: Sequence[int] = (1,),
    beta: float = 1.0, reference: int = 1,
    utilities: Sequence[float] = DEFAULT_UTILITIES,
) -> pd.DataFrame:
    """Fresh stochastic choices on repeated copies of a real state sequence.

    Within a seed, larger runs share the smaller run's choice prefix. Repeats
    are synthetic exposures, not additional real humans or learning sessions.
    Failures are retained as rows rather than silently excluding unlucky seeds.
    """
    pairs = stimulus_pairs(states)
    if not seeds or not repeats:
        raise ValueError("Provide at least one seed and repeat count.")
    if any(not isinstance(n, (int, np.integer)) or n < 1 for n in repeats):
        raise ValueError("Repeat counts must be positive integers.")
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("Recovery requires beta > 0.")
    rows = []
    for repeat in repeats:
        repeated = np.tile(pairs, (repeat, 1))
        for seed in seeds:
            choices = simulate_choices(repeated, seed=seed, beta=beta, utilities=utilities)
            row = {"repeats": repeat, "seed": seed, "n_trials": len(choices)}
            try:
                fit = recover_utilities(choices, beta=beta, reference=reference)
                _, metrics = compare_utilities(fit, utilities)
                row.update(status="ok", **metrics)
            except RecoveryError as exc:
                row.update(status=str(exc))
            rows.append(row)
    return pd.DataFrame(rows)
