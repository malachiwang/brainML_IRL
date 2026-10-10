"""Structural scaffold replication; the simulator and estimators are unchanged."""

from collections import Counter
from hashlib import sha256
from itertools import product
from time import perf_counter

import numpy as np
import pandas as pd

from .data import normalize_training
from .history_evaluation import compare_history_models, equal_value_probes, schedule_expected_rates
from .history_simulation import independent_rollouts
from .trajectories import build_trajectory

STRUCTURAL_COLUMNS = ["participant", "session", "trial", "left_stimulus", "right_stimulus"]
SELECTION_RULE = (
    "Reserve E11T9A. Sort all other participants by (retained row count, lexicographic ID). "
    "Select n_scaffolds-1 ranks using numpy.rint(linspace(0, remaining_count-1, n_scaffolds-1)) "
    "(nearest integer, ties to even). Return the union sorted by (row count, ID). "
    "n_scaffolds=1 selects only the reference for smoke tests. No behavioral columns are read."
)
RNG_RULE = (
    "participant_key=int.from_bytes(SHA256(UTF-8 participant)[:8], 'little'); "
    "root=int(SeedSequence([seed_batch, participant_key]).generate_state(1, dtype=uint64)[0]); "
    "existing independent_rollouts uses SeedSequence(root).spawn(max_rollouts). "
    "Scaffolds/batches have separate streams; rollout counts are nested prefixes; "
    "alpha conditions share random draws within a scaffold/batch."
)


def structural_training(source: pd.DataFrame) -> pd.DataFrame:
    """Project BEFORE normalization: human actions/rewards never enter this pipeline.

    The placeholder action only satisfies the existing trajectory API; it is
    not an imputed human response. Every source row and chronology label remains.
    """
    projected = source[["ID", "session", "trial", "stim1", "stim2"]].copy()
    return normalize_training(projected.assign(action=pd.NA))[STRUCTURAL_COLUMNS]


def select_scaffolds(frame: pd.DataFrame, n_scaffolds: int = 12) -> list[str]:
    counts = frame.groupby("participant", observed=True).size()
    ranked = sorted(((int(n), str(p)) for p, n in counts.items()))
    if "E11T9A" not in counts.index:
        raise ValueError("Reference scaffold E11T9A is absent; do not silently replace it.")
    if not isinstance(n_scaffolds, int) or not 1 <= n_scaffolds <= len(ranked):
        raise ValueError("n_scaffolds must be between 1 and the available participant count.")
    others = [p for _, p in ranked if p != "E11T9A"]
    ranks = np.rint(np.linspace(0, len(others) - 1, n_scaffolds - 1)).astype(int)
    selected = {"E11T9A", *(others[i] for i in ranks)}
    return [p for _, p in ranked if p in selected]


def scaffold_trajectory(frame: pd.DataFrame, participant: str):
    return build_trajectory(frame[STRUCTURAL_COLUMNS].assign(action=pd.NA), participant)


def scaffold_summary(trajectory) -> dict:
    ordered = sorted(trajectory, key=lambda t: (t.session, t.trial))
    counts = Counter(t.session for t in ordered)
    pairs = Counter(tuple(sorted(t.state)) for t in ordered)
    gaps = [b.trial - a.trial - 1 for a, b in zip(ordered, ordered[1:])
            if a.session == b.session and b.trial > a.trial + 1]
    return dict(participant=ordered[0].participant, retained_trials=len(ordered),
                session_counts={str(s): counts[s] for s in range(1, 6)},
                unordered_pair_counts={f"{a}-{b}": n for (a, b), n in sorted(pairs.items())},
                internal_gap_events=len(gaps), internal_missing_trial_labels=sum(gaps),
                ordered_state_count=len({t.state for t in ordered}))


def batch_root(participant: str, seed_batch: int) -> int:
    key = int.from_bytes(sha256(participant.encode("utf-8")).digest()[:8], "little")
    return int(np.random.SeedSequence([seed_batch, key]).generate_state(1, dtype=np.uint64)[0])


def experiment_grid(participants, alphas, rollouts, seed_batches, *, secondary_negative=False):
    """Validate and predeclare every configuration, before generating any choice."""
    for name, values in (("participants", participants), ("alphas", alphas),
                         ("rollouts", rollouts), ("seed_batches", seed_batches)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be nonempty and contain no duplicates.")
    if not np.isfinite(alphas).all():
        raise ValueError("Alpha values must be finite.")
    if any(not isinstance(n, int) or n < 1 for n in rollouts):
        raise ValueError("Rollout counts must be positive integers.")
    if any(not isinstance(n, int) or n < 0 for n in seed_batches):
        raise ValueError("Seed batches must be nonnegative integers.")
    if secondary_negative and -1 in alphas:
        raise ValueError("Secondary -1 must not duplicate a primary alpha condition.")
    conditions = [("primary", float(a), n) for a, n in product(alphas, sorted(rollouts))]
    if secondary_negative:
        conditions.append(("secondary", -1.0, max(rollouts)))
    return [dict(participant=p, experiment=label, alpha_true=a, rollouts=n,
                 seed_batch=seed, rng_root=batch_root(p, seed))
            for p in participants for label, a, n in conditions for seed in seed_batches]


def distribution(values) -> dict:
    """Unweighted descriptive distribution; missing results stay in denominators.

    SD/variance use ddof=1; singleton SD is null, not zero. Quantiles use NumPy's
    linear interpolation. These are not confidence intervals or independent-n SEs.
    """
    values = list(values)
    finite = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    names = ("mean", "median", "sd", "variance", "p05", "p25", "p75", "p95", "min", "max")
    result = dict(n=len(finite), missing=len(values) - len(finite), **dict.fromkeys(names))
    if len(finite):
        result.update(mean=float(finite.mean()), median=float(np.median(finite)),
                      min=float(finite.min()), max=float(finite.max()))
        result.update(zip(("p05", "p25", "p75", "p95"), map(float, np.quantile(finite, [.05, .25, .75, .95]))))
    if len(finite) > 1:
        result.update(sd=float(finite.std(ddof=1)), variance=float(finite.var(ddof=1)))
    return result


def summarize_runs(runs: list[dict]) -> dict:
    """Retain failures and pairwise denominators, including partially failed runs."""
    result = dict(configurations=len(runs), fits_attempted=sum(len(r["fits"]) for r in runs),
                  failed_configurations=sum(r["status"] != "ok" for r in runs))
    result["fit_failures"] = sum(f["status"] != "ok" for r in runs for f in r["fits"].values())
    metrics = {}
    for scope, model in product(("all", "train"), ("static", "history")):
        label = f"{scope}_{model}"
        for metric in ("alpha", "alpha_signed_error", "alpha_absolute_error", "mae", "nll",
                       "strict_order_agreement", "true_tie_mean_gap", "information_condition",
                       "information_min_eigenvalue", "test_nll", "test_log_loss"):
            if metric.startswith("information") and model == "static":
                continue
            if metric.startswith("test_") and scope == "all":
                continue
            metrics[f"{label}.{metric}"] = distribution(
                r["fits"].get(label, {}).get(metric) if r["fits"].get(label, {}).get("status") == "ok" else None
                for r in runs)
    for metric in ("heldout_log_loss_gain", "heldout_nll_gain", "mean_probability_difference", "max_probability_difference"):
        metrics[metric] = distribution(r.get(metric) for r in runs)
    result["metrics"] = metrics
    gains = [r["heldout_log_loss_gain"] for r in runs if "heldout_log_loss_gain" in r]
    wins = sum(g > 0 for g in gains)
    result.update(heldout_pairs=len(gains), heldout_missing_pairs=len(runs) - len(gains),
                  history_wins=wins, heldout_ties=sum(g == 0 for g in gains),
                  history_win_rate=wins / len(gains) if gains else None)
    zero = [r for r in runs if r["alpha_true"] == 0]
    estimates = [r["fits"]["all_history"]["alpha"] for r in zero
                 if r["fits"].get("all_history", {}).get("status") == "ok"]
    result["zero_control"] = dict(attempted=len(zero), successful=len(estimates),
        abs_alpha_ge_0_5=sum(abs(a) >= .5 for a in estimates),
        abs_alpha_ge_1=sum(abs(a) >= 1 for a in estimates))
    return result


def aggregate_results(runs: list[dict], participants: list[str]) -> dict:
    conditions = sorted({(r["experiment"], r["alpha_true"], r["rollouts"]) for r in runs})
    groups = []
    scaffolds = []
    for experiment, alpha, count in conditions:
        selected = [r for r in runs if (r["experiment"], r["alpha_true"], r["rollouts"]) == (experiment, alpha, count)]
        key = dict(experiment=experiment, alpha_true=alpha, rollouts=count)
        groups.append(dict(**key, **summarize_runs(selected)))
        for p in participants:
            scaffolds.append(dict(participant=p, **key,
                                  **summarize_runs([r for r in selected if r["participant"] == p])))
    # Empty/failed scaffolds remain visible rather than disappearing in groupby.
    overall_scaffolds = [dict(participant=p, **summarize_runs(
        [r for r in runs if r["participant"] == p and r["experiment"] == "primary"])) for p in participants]
    return dict(conditions=groups, scaffold_conditions=scaffolds, scaffolds=overall_scaffolds,
                totals=summarize_runs(runs))


def summarize_probes(probes: list[dict], *, by_scaffold=False) -> list[dict]:
    groups = {}
    for p in probes:
        key = (p["experiment"], p["alpha_true"], p["left"], p["right"])
        if by_scaffold:
            key += (p["participant"],)
        groups.setdefault(key, []).append(p)
    result = []
    for key, rows in sorted(groups.items()):
        item = dict(experiment=key[0], alpha_true=key[1], left=key[2], right=key[3], n=len(rows),
                    history_difference=distribution(p["history_difference"] for p in rows),
                    probability_left=distribution(p["probability_left"] for p in rows),
                    schedule_direction_rate=float(np.mean([p["matches_schedule_direction"] for p in rows])))
        # Requested directional comparator is evaluation ONLY, not model input.
        favored = {(2, 3): 3, (4, 5): 4, (6, 7): 7}[key[2:4]]
        item.update(reference_frequent_member=favored,
                    frequent_member_history_rate=float(np.mean([p["stronger_history"] == favored for p in rows])),
                    frequent_member_preference_rate=float(np.mean([
                        (p["probability_left"] > .5 if favored == p["left"] else p["probability_left"] < .5)
                        for p in rows])))
        if by_scaffold:
            item["participant"] = key[4]
        result.append(item)
    return result


def run_replication(trajectories: dict, grid: list[dict], *, beta=1.0, progress=None) -> dict:
    """Generate once per scaffold/alpha/batch, reuse nested sets and existing fits."""
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("Fixed beta must be finite and positive.")
    started = perf_counter()
    batches = {}
    for config in grid:
        key = (config["participant"], config["experiment"], config["alpha_true"], config["seed_batch"])
        batches.setdefault(key, []).append(config)
    runs, probes = [], []
    for (participant, experiment, alpha, seed), configs in batches.items():
        trajectory = trajectories[participant]
        if {t.session for t in trajectory} != set(range(1, 6)):
            raise ValueError(f"Scaffold {participant} lacks five sessions; report rather than exclude it.")
        maximum = max(c["rollouts"] for c in configs)
        agents = independent_rollouts(trajectory, maximum, seed=configs[0]["rng_root"], alpha=alpha, beta=beta)
        for config in sorted(configs, key=lambda c: c["rollouts"]):
            choices = pd.concat([a.choices for a in agents[:config["rollouts"]]], ignore_index=True)
            report, _ = compare_history_models(choices, alpha_true=alpha, beta=beta)
            for fit in report["fits"].values():
                if fit["status"] == "ok":
                    fit["alpha_signed_error"] = fit["alpha"] - alpha
            if "heldout_log_loss_gain" in report:
                report["heldout_nll_gain"] = report["fits"]["train_static"]["test_nll"] - report["fits"]["train_history"]["test_nll"]
            runs.append(dict(**config, **report))
        expected = schedule_expected_rates(trajectory, beta=beta)
        for probe in equal_value_probes(agents, alpha=alpha, beta=beta, expected_rates=expected):
            # Keep per-agent diagnostics without duplicating rows for nested sizes.
            fields = ("rollout", "left", "right", "left_history", "right_history", "probability_left",
                      "stronger_history", "expected_stronger", "matches_schedule_direction")
            probes.append(dict(participant=participant, experiment=experiment, alpha_true=alpha,
                               seed_batch=seed, **{k: probe[k] for k in fields},
                               history_difference=probe["left_history"] - probe["right_history"]))
        if progress:
            progress(participant, experiment, alpha, seed, len(runs), perf_counter() - started)
    return dict(runs=runs, probes=probes, summaries=aggregate_results(runs, list(trajectories)),
                probe_summaries=summarize_probes(probes),
                scaffold_probe_summaries=summarize_probes(probes, by_scaffold=True),
                experiment_seconds=perf_counter() - started)
