from itertools import combinations, product

import numpy as np
import pandas as pd
import pytest

from reward_pairs.history import causal_history
from reward_pairs.history_replication import (
    aggregate_results, batch_root, distribution, experiment_grid, run_replication,
    scaffold_summary, scaffold_trajectory, select_scaffolds, structural_training,
    summarize_probes, summarize_runs,
)
from reward_pairs.history_simulation import independent_rollouts


def structural_fixture():
    rows = []
    for i, p in enumerate(["a", "b", "c", "d", "E11T9A", "f", "g", "h", "i", "j", "k", "l", "m"]):
        for t in range(1, 6 + i):
            rows.append((p, 1, t, 1, 2))
    return pd.DataFrame(rows, columns=["participant", "session", "trial", "left_stimulus", "right_stimulus"])


def test_selection_deterministic_unique_spans_range_reference_and_ignores_actions():
    frame = structural_fixture()
    chosen = select_scaffolds(frame, 12)
    assert len(chosen) == len(set(chosen)) == 12
    assert chosen[0] == "a" and chosen[-1] == "m" and "E11T9A" in chosen
    assert chosen == select_scaffolds(frame.sample(frac=1, random_state=12).assign(action="not a real action"), 12)
    assert select_scaffolds(frame, 1) == ["E11T9A"]
    # Four others at equally spaced ranks of the remaining population.
    assert select_scaffolds(frame, 5) == ["a", "E11T9A", "f", "i", "m"]


def test_selection_lexicographic_ties_and_invalid_requests():
    frame = structural_fixture().groupby("participant", observed=True).head(3)
    assert select_scaffolds(frame, 3) == ["E11T9A", "a", "m"]
    for count in (0, 14):
        with pytest.raises(ValueError, match="n_scaffolds"):
            select_scaffolds(frame, count)
    with pytest.raises(ValueError, match="Reference"):
        select_scaffolds(frame[frame.participant != "E11T9A"])


def test_structural_projection_never_reads_human_behavior_and_counts_gaps():
    source = pd.DataFrame(dict(ID=["E11T9A"] * 5, session=[1, 1, 1, 2, 2],
                               trial=[2, 5, 6, 1, 3], stim1=[1, 2, 1, 2, 1], stim2=[2, 1, 2, 1, 2]))
    frame = structural_training(source)
    poison = source.assign(action="invalid human choice", reward1=np.nan, flag_include=0)
    pd.testing.assert_frame_equal(frame, structural_training(poison))
    assert len(frame) == len(source) and "action" not in frame
    trajectory = scaffold_trajectory(frame, "E11T9A")
    assert all(t.action is None for t in trajectory)
    s = scaffold_summary(trajectory)
    assert s["internal_gap_events"] == 2 and s["internal_missing_trial_labels"] == 3
    assert s["session_counts"] == {"1": 3, "2": 2, "3": 0, "4": 0, "5": 0}
    assert s["unordered_pair_counts"] == {"1-2": 5} and s["ordered_state_count"] == 2


def test_grid_cartesian_complete_unique_and_secondary_separate():
    participants, alphas, counts, seeds = ["a", "E11T9A"], [0., .5, 1.], [1, 5, 20], list(range(10))
    grid = experiment_grid(participants, alphas, counts, seeds, secondary_negative=True)
    observed = [(r["participant"], r["alpha_true"], r["rollouts"], r["seed_batch"]) for r in grid]
    expected = list(product(participants, alphas, counts, seeds)) + list(product(participants, [-1.], [20], seeds))
    assert set(observed) == set(expected) and len(observed) == len(set(observed)) == 200
    assert sum(r["experiment"] == "secondary" for r in grid) == 20
    with pytest.raises(ValueError, match="duplicates"):
        experiment_grid(participants, [0., 0.], counts, seeds)
    with pytest.raises(ValueError, match="duplicate"):
        experiment_grid(participants, [-1.], counts, seeds, secondary_negative=True)


def test_distribution_means_sample_sd_linear_quantiles_and_missing():
    d = distribution([0., 1., 2., 3., 4., None, np.nan])
    assert d["n"] == 5 and d["missing"] == 2
    assert d["mean"] == d["median"] == 2
    assert d["variance"] == 2.5 and d["sd"] == np.sqrt(2.5)
    assert [d[k] for k in ("p05", "p25", "p75", "p95")] == [.2, 1., 3., 3.8]
    assert d["min"] == 0 and d["max"] == 4
    assert distribution([1])["sd"] is None
    assert distribution([None])["mean"] is None


def fake_report(alpha=0., gain=None, failed=False):
    fits = {f"{scope}_{model}": dict(status="ok", alpha=alpha, mae=.1)
            for scope, model in product(("all", "train"), ("static", "history"))}
    if failed:
        fits["all_history"] = dict(status="intentional failure")
    report = dict(participant="p", experiment="primary", alpha_true=0., rollouts=1,
                  seed_batch=0, fits=fits, status="FAILED" if failed else "ok")
    if gain is not None:
        report["heldout_log_loss_gain"] = gain
    return report


def test_aggregation_preserves_failures_partial_fits_denominators_and_all_scaffolds():
    runs = [fake_report(-1, .1), fake_report(.2, -.1), fake_report(gain=0., failed=True), fake_report(failed=True)]
    summary = summarize_runs(runs)
    assert summary["fits_attempted"] == 16 and summary["fit_failures"] == 2
    assert summary["failed_configurations"] == 2
    assert summary["metrics"]["all_history.alpha"]["n"] == 2
    assert summary["metrics"]["all_history.alpha"]["missing"] == 2
    assert summary["history_wins"] == 1 and summary["heldout_pairs"] == 3
    assert summary["heldout_ties"] == 1 and summary["history_win_rate"] == 1 / 3
    assert summary["zero_control"]["abs_alpha_ge_1"] == 1
    aggregate = aggregate_results(runs, ["p", "missing_scaffold"])
    assert [s["participant"] for s in aggregate["scaffolds"]] == ["p", "missing_scaffold"]
    assert aggregate["scaffolds"][1]["configurations"] == 0


def test_probe_direction_is_measured_and_zero_is_not_counted_as_preference():
    probes = [dict(experiment="primary", alpha_true=0., participant="p", left=2, right=3,
                   history_difference=-.2, probability_left=.5, stronger_history=3,
                   matches_schedule_direction=True)]
    s = summarize_probes(probes)[0]
    assert s["frequent_member_history_rate"] == 1 and s["frequent_member_preference_rate"] == 0
    probes[0].update(alpha_true=1., stronger_history=2, history_difference=.2, probability_left=.6,
                     matches_schedule_direction=False)
    s = summarize_probes(probes, by_scaffold=True)[0]
    assert s["frequent_member_history_rate"] == s["frequent_member_preference_rate"] == 0


@pytest.fixture(scope="module")
def small_scaffold():
    rows = [("E11T9A", s, i + 1, a, b) for s in range(1, 6)
            for i, (a, b) in enumerate(list(combinations(range(1, 9), 2)) * 3)]
    return pd.DataFrame(rows, columns=structural_fixture().columns)


def test_batch_streams_separate_reproducible_prefixes_and_fresh_history(small_scaffold):
    roots = [batch_root(p, s) for p, s in product(["a", "b", "E11T9A"], range(10))]
    assert len(set(roots)) == 30
    trajectory = scaffold_trajectory(small_scaffold, "E11T9A")
    root = batch_root("E11T9A", 3)
    a = independent_rollouts(trajectory, 3, seed=root, alpha=1)
    b = independent_rollouts(trajectory, 1, seed=batch_root("E11T9A", 3), alpha=1)
    pd.testing.assert_frame_equal(a[0].choices, b[0].choices)
    assert not a[0].choices.synthetic_action.equals(a[1].choices.synthetic_action)
    histories = causal_history(pd.concat([r.choices for r in a], ignore_index=True))
    assert histories.groupby("rollout").first().left_history.eq(0).all()


def test_replication_end_to_end_reproducible_and_human_action_invariant(small_scaffold):
    grid = experiment_grid(["E11T9A"], [.5], [1, 2], [9])
    trajectory = scaffold_trajectory(small_scaffold.assign(action="left", reward1=100), "E11T9A")
    a = run_replication({"E11T9A": trajectory}, grid)
    altered = scaffold_trajectory(small_scaffold.assign(action="right", reward1=-100), "E11T9A")
    b = run_replication({"E11T9A": altered}, grid)
    assert a["runs"] == b["runs"] and a["probes"] == b["probes"]
    assert len(a["runs"]) == 2 and len(a["probes"]) == 6  # no double-counting nested agents
    assert a["summaries"]["totals"]["fit_failures"] == 0


def test_replication_failed_fits_remain_in_raw_and_aggregate_results():
    # Actual disconnected data: test infrastructure handling, not a mocked failure.
    frame = pd.DataFrame([("E11T9A", s, t, 1, 2) for s in range(1, 6) for t in range(1, 5)],
                         columns=structural_fixture().columns)
    result = run_replication({"E11T9A": scaffold_trajectory(frame, "E11T9A")},
                             experiment_grid(["E11T9A"], [0.], [1], [0]))
    assert len(result["runs"]) == 1 and result["runs"][0]["status"] == "FAILED"
    assert result["summaries"]["totals"]["fit_failures"] == 4
    assert result["summaries"]["totals"]["heldout_pairs"] == 0


def test_session_five_and_current_actions_cannot_change_earlier_replication_histories(small_scaffold):
    trajectory = scaffold_trajectory(small_scaffold, "E11T9A")
    data = independent_rollouts(trajectory, 1, seed=batch_root("E11T9A", 0), alpha=1)[0].choices
    changed = data.copy()
    changed.loc[changed.session == 5, "synthetic_action"] = "left"
    a, b = causal_history(data), causal_history(changed)
    mask = (a.session < 5) | ((a.session == 5) & (a.trial == 1))
    pd.testing.assert_frame_equal(a.loc[mask, ["left_history", "right_history"]],
                                  b.loc[mask, ["left_history", "right_history"]])
