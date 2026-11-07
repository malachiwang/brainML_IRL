"""Experiment 1 diagnostics: signs, orientation, alignment, selection, reproduction."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reward_pairs.authors_models import AuthorsTrials
from reward_pairs.human_history_evaluation import equal_pair_scores
from reward_pairs.participant_error_diagnostics import (
    PAIR_KEYS, PRIMARY_VARIANT, SENSITIVITY_VARIANT, analyze_variant, delta_summary, pair_error_summary,
    pair_rates, participant_metrics, pooled_pair_rates, reproduction_checks, select_examples, variant_rows,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'docs/human_history_results.json'
RAW = ROOT/'data/raw/02_comp_mod_RP_task_data_in.RData'


def pair(n, obs, pred, frequent=None, nll=None):
    return dict(n=n, frequent=frequent, observed_frequent_count=obs, predicted_frequent_sum=pred,
                nll=(n*.7 if nll is None else nll), q_frequent_minus_rare=0.)


def model(loss, n=100, pairs=None):
    pairs = pairs or {'2-3': pair(10, 6, 7.), '4-5': pair(10, 5, 6.), '6-7': pair(10, 4, 5.)}
    for key, f in zip(PAIR_KEYS, (3, 4, 7)):
        pairs[key]['frequent'] = f
    return dict(test=dict(n=n, nll=loss*n, mean_log_loss=loss), equal_pairs=deepcopy(pairs),
                parameters=dict(alpha_rl=.1, beta_rl=1., beta_h=1.))


def row(pid, rl, ck, cum, variant=PRIMARY_VARIANT, n=100, cum_pairs=None):
    ck_model = model(ck, n)
    ck_model['parameters'] = dict(authors_alpha_ck=.1, authors_beta_ck=1., authors_alpha_rl=.2, authors_beta_rl=1., bias=2.5)
    def comp(ref):
        diff = ref-cum
        return dict(delta_test_log_loss=diff, delta_test_LL=diff*n,
                    winner='tie' if abs(diff) <= 1e-8 else ('cumulative' if diff > 0 else None))
    c = {'rl': comp(rl), 'ck': comp(ck)}
    for name in c:
        if c[name]['winner'] is None:
            c[name]['winner'] = name
    return dict(participant=pid, variant=variant, fit_status='success', comparisons=c,
                models=dict(rl=model(rl, n), ck=ck_model, cumulative=model(cum, n, cum_pairs)))


def dataset(rows):
    ids = list(dict.fromkeys(r['participant'] for r in rows))
    return dict(participant_order=ids, rows=rows)


# 1. sign convention -----------------------------------------------------------
def test_delta_is_cumulative_minus_reference_and_converted_from_saved_sign():
    m = participant_metrics([row('a', .50, .40, .45), row('b', .50, .48, .44)])
    assert m[0]['delta_vs_ck'] == pytest.approx(.05) and m[0]['winner_vs_ck'] == 'ck'
    assert m[1]['delta_vs_ck'] == pytest.approx(-.04) and m[1]['winner_vs_ck'] == 'cumulative'
    assert m[0]['delta_vs_rl'] == pytest.approx(-.05)
    s = delta_summary([x['delta_vs_ck'] for x in m])
    assert (s['worse'], s['better'], s['tied']) == (1, 1, 0)


def test_unconverted_saved_sign_is_rejected():
    r = row('a', .5, .4, .45)
    r['comparisons']['ck']['delta_test_log_loss'] *= -1  # someone copied the wrong convention
    with pytest.raises(ValueError, match='sign conversion'):
        participant_metrics([r])


# 2. orientation ----------------------------------------------------------------
def test_frequent_rate_orientation_on_either_side():
    # pair 2-3 (frequent 3 -> index 2). Rows: 3 on left chosen, 3 on right chosen, 3 on left rejected.
    test = AuthorsTrials(np.array([[2, 1], [1, 2], [2, 1], [0, 3]]), np.array([True, False, False, True]), None)
    out = SimpleNamespace(probability_left=np.array([.8, .3, .6, .5]), log_loss=np.full(4, .1),
                          q=np.zeros((5, 8)), h=np.zeros((5, 8)))
    rec = equal_pair_scores(test, out)['2-3']
    assert rec['n'] == 3 and rec['observed_frequent_count'] == 2
    assert rec['predicted_frequent_sum'] == pytest.approx(.8+.7+.6)
    r = pair_rates(rec)
    assert r['observed_rate'] == pytest.approx(2/3)
    assert r['predicted_rate'] == pytest.approx(2.1/3)
    assert r['error'] == pytest.approx(2.1/3-2/3)  # positive = overpredicts frequent


# 3. alignment ------------------------------------------------------------------
def test_alignment_rejects_duplicates_missing_models_and_mismatched_rows():
    good = [row('a', .5, .4, .45), row('b', .5, .4, .45)]
    with pytest.raises(ValueError, match='Duplicate'):
        variant_rows(dict(rows=good+[row('a', .5, .4, .45)]))
    bad = deepcopy(good); del bad[1]['models']['ck']
    with pytest.raises(ValueError, match='missing models'):
        variant_rows(dataset(bad))
    bad = deepcopy(good); bad[0]['models']['cumulative']['test']['n'] = 99
    with pytest.raises(ValueError, match='different numbers'):
        variant_rows(dataset(bad))
    bad = deepcopy(good); bad[0]['models']['cumulative']['equal_pairs']['2-3']['observed_frequent_count'] = 1
    with pytest.raises(ValueError, match='disagree'):
        variant_rows(dataset(bad))
    with pytest.raises(ValueError, match='participant_order'):
        variant_rows(dict(participant_order=['a', 'b', 'c'], rows=good))
    failed = deepcopy(good); failed[0]['fit_status'] = 'failed'
    with pytest.raises(ValueError, match='Failed fits'):
        variant_rows(dataset(failed))


def test_saved_order_is_preserved_and_each_participant_once():
    rows = [row('b', .5, .4, .45), row('a', .5, .4, .45)]
    got = variant_rows(dict(participant_order=['a', 'b'], rows=rows))
    assert [r['participant'] for r in got] == ['a', 'b']


# 4. paper_intended excluded from primary -----------------------------------------
def test_primary_analysis_excludes_paper_intended():
    rows = [row('a', .5, .4, .45), row('b', .5, .4, .47),
            row('a', .5, .1, .9, variant=SENSITIVITY_VARIANT), row('b', .5, .1, .9, variant=SENSITIVITY_VARIANT)]
    metrics, result = analyze_variant(dataset(rows), PRIMARY_VARIANT)
    assert result['participants'] == 2
    assert result['delta_vs_ck']['mean'] == pytest.approx((.05+.07)/2)
    assert max(p['delta_vs_ck'] for p in metrics) < .5
    _, sens = analyze_variant(dataset(rows), SENSITIVITY_VARIANT)
    assert sens['delta_vs_ck']['mean'] == pytest.approx(.8)


# 5. zero observations ------------------------------------------------------------
def test_zero_observation_pair_is_explicit_not_divided():
    r = pair_rates(pair(0, 0, 0., 3))
    assert r['observed_rate'] is None and r['error'] is None and r['n'] == 0
    empty = {'2-3': pair(0, 0, 0., 3), '4-5': pair(10, 5, 6., 4), '6-7': pair(10, 4, 5., 7)}
    rows = [row('a', .5, .4, .45), row('b', .5, .4, .45)]
    for name in ('rl', 'ck', 'cumulative'):
        for key, rec in empty.items():
            rows[0]['models'][name]['equal_pairs'][key] = dict(rec, frequent=rec['frequent'])
    m = participant_metrics(variant_rows(dataset(rows)))
    s = pair_error_summary(m, '2-3', 'cumulative')
    assert s['participants'] == 1 and s['zero_observation_participants'] == 1 and s['observations'] == 10
    all_empty = [dict(p, pairs={**p['pairs'], '2-3': {k: pair_rates(pair(0, 0, 0., 3)) for k in ('rl', 'ck', 'cumulative')}}) for p in m]
    assert pair_error_summary(all_empty, '2-3', 'ck')['participants'] == 0
    assert pooled_pair_rates(all_empty, '2-3', 'ck')['observed_rate'] is None


# 6. weighted pooled reconstruction ---------------------------------------------------
def test_pooled_rates_are_observation_weighted_not_participant_means():
    p1 = {'2-3': pair(2, 2, 1.5, 3), '4-5': pair(10, 5, 6., 4), '6-7': pair(10, 4, 5., 7)}
    p2 = {'2-3': pair(18, 0, 9., 3), '4-5': pair(10, 5, 6., 4), '6-7': pair(10, 4, 5., 7)}
    rows = []
    for pid, prs in (('a', p1), ('b', p2)):
        r = row(pid, .5, .4, .45)
        for name in ('rl', 'ck', 'cumulative'):
            r['models'][name]['equal_pairs'] = deepcopy(prs)
        rows.append(r)
    m = participant_metrics(rows)
    pooled = pooled_pair_rates(m, '2-3', 'cumulative')
    assert pooled['n'] == 20
    assert pooled['observed_rate'] == pytest.approx(2/20)      # not mean(1.0, 0.0)=.5
    assert pooled['predicted_rate'] == pytest.approx(10.5/20)


# 7. deterministic selection ----------------------------------------------------------
def test_selection_is_deterministic_order_free_and_breaks_ties_by_id():
    deltas = [-.05, -.02, 0., .01, .01, .02, .03, .2, .3]
    rows = [row(f'p{i}', .5, .4, .4+d) for i, d in enumerate(deltas)]
    rows += [row('a_tie', .5, .4, .4+.01)]  # bit-identical delta to p3/p4; median target
    m = participant_metrics(rows)
    first = select_examples(m)
    again = select_examples(list(reversed(m)))
    assert [s['participant'] for s in first] == [s['participant'] for s in again]
    median = next(s for s in first if s['role'] == 'typical_p50')
    assert median['participant'] == 'a_tie'  # lexicographic tie-break among delta=.01
    assert len({s['participant'] for s in first}) == len(first)


# 8. real saved-results reproduction -----------------------------------------------------
@pytest.fixture(scope='module')
def saved():
    if not RESULTS.exists():
        pytest.skip('Saved human-history results unavailable.')
    return json.loads(RESULTS.read_text())


def test_full_cohort_reproduces_saved_comparison(saved):
    metrics, result = analyze_variant(saved, PRIMARY_VARIANT)
    assert len(metrics) == len({p['participant'] for p in metrics}) == 213
    d = result['delta_vs_ck']
    assert (d['worse'], d['better'], d['tied']) == (172, 41, 0)
    assert d['mean'] == pytest.approx(.03205211, abs=1e-8)
    assert result['trial_weighted']['ck']['test_choices'] == 28320
    checks = reproduction_checks(saved, PRIMARY_VARIANT, metrics, result)
    assert all(c['passed'] for c in checks), [c for c in checks if not c['passed']]
    published = {'2-3': (.537520, .679594, .568449), '4-5': (.512245, .693660, .565892), '6-7': (.473325, .687331, .548132)}
    for key, (obs, cum, ck) in published.items():
        assert result['pairs'][key]['cumulative']['pooled']['observed_rate'] == pytest.approx(obs, abs=5e-7)
        assert result['pairs'][key]['cumulative']['pooled']['predicted_rate'] == pytest.approx(cum, abs=5e-7)
        assert result['pairs'][key]['ck']['pooled']['predicted_rate'] == pytest.approx(ck, abs=5e-7)
    tw = result['trial_weighted']['ck']['trial_weighted_loss']
    assert (tw['rl'], tw['ck'], tw['cumulative']) == pytest.approx((.45111882, .42209503, .45411161), abs=5e-9)


@pytest.mark.skipif(not RAW.exists(), reason='Raw .RData unavailable locally.')
def test_trial_replay_reproduces_saved_deviances(saved):
    from reward_pairs.authors_audit import participant_inputs
    from reward_pairs.data import load_rdata
    from reward_pairs.participant_error_diagnostics import replay_participant
    authors = json.loads((ROOT/'docs/authors_baseline_results.json').read_text())
    pid = saved['participant_order'][0]
    ref = next(r for r in authors['rows'] if (r['participant'], r['variant'], r['model']) == (pid, PRIMARY_VARIANT, 19))
    rec = next(r for r in saved['rows'] if (r['participant'], r['variant']) == (pid, PRIMARY_VARIANT))
    rep = replay_participant(participant_inputs(load_rdata(RAW), pid), PRIMARY_VARIANT, ref, rec['models']['cumulative'])
    assert rep['deviance']['ck'] == pytest.approx(rec['models']['ck']['test']['deviance'], abs=1e-7)
    assert rep['deviance']['cumulative'] == pytest.approx(rec['models']['cumulative']['test']['deviance'], abs=1e-7)
