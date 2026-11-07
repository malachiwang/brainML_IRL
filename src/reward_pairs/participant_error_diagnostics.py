"""Experiment 1: participant-level prediction-error diagnostics from saved fits.

Everything here reads the already-validated ``docs/human_history_results.json``
rows. Nothing is refitted and no parameter, bound, or update rule is touched.

Sign convention used throughout THIS module (opposite to the saved file):

    delta = loss_cumulative - loss_reference      (nats per test choice)

so positive delta means RL+cumulative predicts held-out choices WORSE. The saved
``comparisons[ref]['delta_test_log_loss']`` is ``loss_reference - loss_cumulative``;
:func:`participant_metrics` converts it and refuses to continue if the converted
value disagrees with a direct subtraction of the stored model losses.

Equal-pair error convention: ``error = predicted_rate - observed_rate`` for the
historically frequent member (3 in 2-3, 4 in 4-5, 7 in 6-7). Positive error means
the model overpredicts choosing the frequent member.
"""
import csv
import math

import numpy as np
from scipy import stats

from .authors_evaluation import distribution
from .authors_models import initial_values
from .cumulative_history_human import PARAMETERS as CUMULATIVE_PARAMETERS, evaluate as cumulative_evaluate
from .human_history_evaluation import EQUAL_PAIRS, TIE_TOLERANCE, equal_pair_scores, replay_reference

PRIMARY_VARIANT = 'released_code_intended'
SENSITIVITY_VARIANT = 'paper_intended'
MODELS = ('rl', 'ck', 'cumulative')
HISTORY_MODELS = ('ck', 'cumulative')
MODEL_LABELS = {'rl': 'RL only', 'ck': 'RL+CK (authors)', 'cumulative': 'RL+cumulative'}
PAIR_KEYS = tuple(f'{a}-{b}' for a, b, _ in EQUAL_PAIRS)
FREQUENT = {f'{a}-{b}': f for a, b, f in EQUAL_PAIRS}
RARE = {f'{a}-{b}': (a if f == b else b) for a, b, f in EQUAL_PAIRS}
SAVED_CONSISTENCY_TOLERANCE = 1e-12  # float re-derivation of stored values only
DESCRIPTIVE_MARGIN = .01  # same descriptive (not validated) margin as the saved report
Z_REFERENCE = 1.96


# --------------------------------------------------------------------------- loading
def variant_rows(data, variant=PRIMARY_VARIANT):
    """Rows for exactly one initialization variant, validated and in saved order.

    Raises instead of silently dropping duplicates, failures, missing models, or
    models that were scored on different test rows.
    """
    rows = [r for r in data['rows'] if r['variant'] == variant]
    if not rows:
        raise ValueError(f'No saved rows for variant {variant!r}.')
    ids = [r['participant'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate participant rows in {variant}.')
    order = data.get('participant_order')
    if order is not None:
        if set(order) != set(ids) or len(order) != len(ids):
            raise ValueError('Variant participants differ from saved participant_order.')
        rank = {p: i for i, p in enumerate(order)}
        rows = sorted(rows, key=lambda r: rank[r['participant']])
    failed = [r['participant'] for r in rows if r.get('fit_status') != 'success']
    if failed:
        raise ValueError(f'Failed fits present {failed}; inspect, do not silently omit.')
    for r in rows:
        missing = [m for m in MODELS if m not in r.get('models', {})]
        if missing:
            raise ValueError(f'{r["participant"]}: missing models {missing}.')
        if len({r['models'][m]['test']['n'] for m in MODELS}) != 1:
            raise ValueError(f'{r["participant"]}: models scored on different numbers of test rows.')
        for key in PAIR_KEYS:
            recs = [r['models'][m]['equal_pairs'].get(key) for m in MODELS]
            if any(x is None for x in recs):
                raise ValueError(f'{r["participant"]}: missing equal-pair record {key}.')
            if len({(x['n'], x['observed_frequent_count']) for x in recs}) != 1:
                raise ValueError(f'{r["participant"]} {key}: models disagree on observed equal-pair rows.')
            if any(x['frequent'] != FREQUENT[key] for x in recs):
                raise ValueError(f'{r["participant"]} {key}: unexpected frequent-member label.')
    return rows


# ---------------------------------------------------------------- participant metrics
def pair_rates(record):
    """Observed/predicted frequent-member rates for one participant x pair x model.

    Zero-observation pairs return ``None`` rates (never 0/0) so they can be
    excluded explicitly and counted, not silently treated as zero error.
    """
    n = int(record['n'])
    obs = int(record['observed_frequent_count'])
    pred = float(record['predicted_frequent_sum'])
    if n < 0 or not 0 <= obs <= n or not -1e-9 <= pred <= n + 1e-9:
        raise ValueError('Inconsistent equal-pair counts.')
    if n == 0:
        return dict(n=0, observed_count=0, predicted_sum=0., observed_rate=None, predicted_rate=None,
                    error=None, abs_error=None, extremity_error=None, observed_extremity=None,
                    calibrated_expected_extremity=None, z=None, nll=0.)
    o, p = obs/n, pred/n
    variance = n*p*(1-p)  # >= Poisson-binomial variance of a calibrated model: conservative z
    k = np.arange(n+1)
    # Mean |observed rate - .5| a calibrated model would itself produce at this n
    # (binomial approximation at the participant's mean predicted rate).
    noise_extremity = float(stats.binom.pmf(k, n, min(max(p, 0.), 1.)) @ abs(k/n-.5))
    return dict(n=n, observed_count=obs, predicted_sum=pred, observed_rate=o, predicted_rate=p,
                error=p-o, abs_error=abs(p-o), extremity_error=abs(p-.5)-abs(o-.5),
                observed_extremity=abs(o-.5), calibrated_expected_extremity=noise_extremity,
                z=(pred-obs)/math.sqrt(variance) if variance > 0 else None, nll=float(record['nll']))


def pooled_participant_error(pairs, model):
    """Observation-weighted error over the three equal pairs for one participant."""
    n = sum(pairs[k][model]['n'] for k in PAIR_KEYS)
    if not n:
        return None
    return sum(pairs[k][model]['predicted_sum']-pairs[k][model]['observed_count'] for k in PAIR_KEYS)/n


def participant_metrics(rows):
    """One flat-ish dict per participant; all signs converted and cross-checked."""
    out = []
    for r in rows:
        m = r['models']
        n = m['ck']['test']['n']
        loss = {k: float(m[k]['test']['mean_log_loss']) for k in MODELS}
        nll = {k: float(m[k]['test']['nll']) for k in MODELS}
        for k in MODELS:
            if abs(nll[k]/n-loss[k]) > SAVED_CONSISTENCY_TOLERANCE:
                raise ValueError(f'{r["participant"]}: stored NLL and mean log loss disagree for {k}.')
        delta, winner = {}, {}
        for ref in ('rl', 'ck'):
            delta[ref] = loss['cumulative']-loss[ref]
            saved = r['comparisons'][ref]
            if abs(delta[ref]+saved['delta_test_log_loss']) > SAVED_CONSISTENCY_TOLERANCE:
                raise ValueError(f'{r["participant"]}: sign conversion vs saved {ref} comparison failed.')
            if abs((nll['cumulative']-nll[ref])+saved['delta_test_LL']) > 1e-9:
                raise ValueError(f'{r["participant"]}: total NLL difference vs saved {ref} comparison failed.')
            winner[ref] = 'tie' if abs(delta[ref]) <= TIE_TOLERANCE else ('cumulative' if delta[ref] < 0 else ref)
            if winner[ref] != saved['winner']:
                raise ValueError(f'{r["participant"]}: winner disagrees with saved {ref} comparison.')
        pairs = {key: {model: pair_rates(m[model]['equal_pairs'][key]) for model in MODELS} for key in PAIR_KEYS}
        equal_n = sum(pairs[k]['ck']['n'] for k in PAIR_KEYS)
        out.append(dict(
            participant=r['participant'], n_test=n, loss=loss, nll=nll,
            delta_vs_ck=delta['ck'], delta_vs_rl=delta['rl'], winner_vs_ck=winner['ck'], winner_vs_rl=winner['rl'],
            nll_gap_vs_ck=nll['cumulative']-nll['ck'],
            equal_pair_n=equal_n,
            equal_pair_nll_gap_vs_ck=sum(pairs[k]['cumulative']['nll']-pairs[k]['ck']['nll'] for k in PAIR_KEYS),
            pairs=pairs,
            pooled_error={model: pooled_participant_error(pairs, model) for model in MODELS},
            parameters=dict(cumulative=dict(m['cumulative']['parameters']),
                            ck={k.replace('authors_', ''): v for k, v in m['ck']['parameters'].items() if k != 'bias'}),
        ))
    return out


# ------------------------------------------------------------------- Analysis A: losses
def delta_summary(values):
    """Spread, sign counts, and concentration of participant loss differences."""
    d = np.asarray(values, float)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all():
        raise ValueError('Need finite participant differences.')
    ordered = np.sort(d)[::-1]
    total = float(d.sum())
    positive = ordered[ordered > 0]
    share = np.cumsum(ordered)/total if total else np.full(len(d), np.nan)
    top = {}
    for frac in (.05, .10, .20):
        k = max(1, math.ceil(frac*len(d)))
        top[f'top_{int(frac*100)}pct'] = dict(participants=k, share_of_total=float(ordered[:k].sum()/total) if total else None,
                                              mean_without=float(ordered[k:].mean()) if k < len(d) else None)
    half = None
    if len(positive) and positive.sum() > 0:
        half = int(np.searchsorted(np.cumsum(positive), .5*positive.sum())+1)
    return dict(distribution=distribution(d),
                quantiles_extra=dict(p01=float(np.quantile(d, .01)), p99=float(np.quantile(d, .99))),
                worse=int((d > TIE_TOLERANCE).sum()), better=int((d < -TIE_TOLERANCE).sum()),
                tied=int((abs(d) <= TIE_TOLERANCE).sum()),
                worse_over_margin=int((d > DESCRIPTIVE_MARGIN).sum()), better_over_margin=int((d < -DESCRIPTIVE_MARGIN).sum()),
                trimmed_mean_10pct=float(stats.trim_mean(d, .1)), mean=float(d.mean()), sum=total,
                top_contributors=top, participants_for_half_of_positive_sum=half,
                positive_participants=int(len(positive)), cumulative_share=share.tolist())


def trial_weighted(metrics, ref):
    nll = sum(p['nll']['cumulative']-p['nll'][ref] for p in metrics)
    n = sum(p['n_test'] for p in metrics)
    return dict(total_nll_gap=nll, test_choices=n, trial_weighted_delta=nll/n,
                trial_weighted_loss={k: sum(p['nll'][k] for p in metrics)/n for k in MODELS})


# --------------------------------------------------------- Analysis B: equal pairs
def pooled_pair_rates(metrics, key, model):
    """Observation-weighted pooled rates, reconstructing the published table."""
    recs = [p['pairs'][key][model] for p in metrics if p['pairs'][key][model]['n']]
    n = sum(r['n'] for r in recs)
    if not n:
        return dict(n=0, participants=0, observed_rate=None, predicted_rate=None, log_loss=None)
    return dict(n=n, participants=len(recs), observed_rate=sum(r['observed_count'] for r in recs)/n,
                predicted_rate=sum(r['predicted_sum'] for r in recs)/n, log_loss=sum(r['nll'] for r in recs)/n)


def pair_error_summary(metrics, key, model):
    recs = [p['pairs'][key][model] for p in metrics]
    valid = [r for r in recs if r['n']]
    if not valid:
        return dict(participants=0, zero_observation_participants=len(recs), observations=0)
    e = np.array([r['error'] for r in valid])
    o = np.array([r['observed_rate'] for r in valid])
    pos, neg = int((e > 0).sum()), int((e < 0).sum())
    z = np.array([r['z'] for r in valid if r['z'] is not None])
    side = np.where(o > .5, 'frequent', np.where(o < .5, 'rare', 'indifferent'))
    sign = np.where(e > 0, 'over', np.where(e < 0, 'under', 'exact'))
    crosstab = {s: {g: int(((side == s) & (sign == g)).sum()) for g in ('over', 'under', 'exact')}
                for s in ('frequent', 'indifferent', 'rare')}
    pred = np.array([r['predicted_rate'] for r in valid])
    return dict(
        participants=len(valid), zero_observation_participants=len(recs)-len(valid),
        observations=int(sum(r['n'] for r in valid)),
        n_per_participant=distribution([r['n'] for r in valid]),
        error=distribution(e), abs_error=distribution(abs(e)),
        positive=pos, negative=neg, exact_zero=int((e == 0).sum()),
        positive_fraction=pos/len(valid), negative_fraction=neg/len(valid),
        abs_error_over_010=int((abs(e) > .10).sum()), abs_error_over_020=int((abs(e) > .20).sum()),
        sign_test_p=float(stats.binomtest(pos, pos+neg).pvalue) if pos+neg else None,
        z_over=int((z > Z_REFERENCE).sum()), z_under=int((z < -Z_REFERENCE).sum()), z_defined=int(len(z)),
        observed_side_by_error_sign=crosstab,
        predicted_frequent_majority=int((pred > .5).sum()),
        observed_frequent_majority=int((o > .5).sum()), observed_rare_majority=int((o < .5).sum()),
        rare_observed_but_frequent_predicted=int(((o < .5) & (pred > .5)).sum()),
        direction_agreement=int((((o > .5) & (pred > .5)) | ((o < .5) & (pred < .5))).sum()),
        direction_decided=int((o != .5).sum()),
        mean_observed_extremity=float(np.mean([r['observed_extremity'] for r in valid])),
        mean_calibrated_expected_extremity=float(np.mean([r['calibrated_expected_extremity'] for r in valid])),
        mean_predicted_extremity=float(np.mean(abs(pred-.5))),
        extremity_error=distribution([r['extremity_error'] for r in valid]),
        pooled=pooled_pair_rates(metrics, key, model))


def poisson_binomial(probabilities):
    dist = np.array([1.])
    for p in probabilities:
        dist = np.convolve(dist, [1-p, p])
    return dist


def cross_pair_consistency(metrics, model):
    """Are the same people overpredicted on several pairs? Descriptive only."""
    complete = [p for p in metrics if all(p['pairs'][k][model]['n'] for k in PAIR_KEYS)]
    if not complete:
        return dict(participants=0)
    E = np.array([[p['pairs'][k][model]['error'] for k in PAIR_KEYS] for p in complete])
    positive = E > 0
    counts = np.bincount(positive.sum(axis=1), minlength=4)
    expected = poisson_binomial(positive.mean(axis=0))*len(complete)
    correlations = {}
    for i in range(3):
        for j in range(i+1, 3):
            rho = stats.spearmanr(E[:, i], E[:, j])
            correlations[f'{PAIR_KEYS[i]} vs {PAIR_KEYS[j]}'] = dict(spearman=float(rho.statistic), p=float(rho.pvalue))
    return dict(participants=len(complete), positive_pair_count=counts.tolist(),
                expected_if_independent=expected.tolist(), spearman=correlations)


def gap_link(metrics, model='cumulative'):
    """Descriptive association between equal-pair overprediction and log-loss gap."""
    use = [p for p in metrics if p['pooled_error'][model] is not None]
    err = np.array([p['pooled_error'][model] for p in use])
    err_diff = np.array([p['pooled_error']['cumulative']-p['pooled_error']['ck'] for p in use])
    delta = np.array([p['delta_vs_ck'] for p in use])
    eq_gap = np.array([p['equal_pair_nll_gap_vs_ck'] for p in use])
    tot_gap = np.array([p['nll_gap_vs_ck'] for p in use])
    rho = stats.spearmanr(err, delta)
    rho_diff = stats.spearmanr(err_diff, delta)
    rho_rest = stats.spearmanr(err, (tot_gap-eq_gap)/np.array([p['n_test']-p['equal_pair_n'] for p in use]))
    q75 = np.quantile(err, .75)
    k = max(1, math.ceil(.2*len(use)))
    top_delta = set(np.argsort(-delta, kind='stable')[:k])
    top_err = set(np.argsort(-err, kind='stable')[:k])
    return dict(
        participants=len(use),
        spearman_error_vs_delta=dict(rho=float(rho.statistic), p=float(rho.pvalue)),
        pearson_error_vs_delta=float(np.corrcoef(err, delta)[0, 1]),
        spearman_error_difference_vs_delta=dict(rho=float(rho_diff.statistic), p=float(rho_diff.pvalue)),
        spearman_error_vs_non_equal_pair_gap=dict(rho=float(rho_rest.statistic), p=float(rho_rest.pvalue)),
        top_quartile_error=dict(threshold=float(q75), n=int((err >= q75).sum()),
                                median_delta=float(np.median(delta[err >= q75])),
                                rest_median_delta=float(np.median(delta[err < q75]))),
        top20pct_overlap=dict(k=k, overlap=len(top_delta & top_err), expected_by_chance=k*k/len(use)),
        equal_pair_share_of_total_gap=float(eq_gap.sum()/tot_gap.sum()),
        equal_pair_gap_nats=float(eq_gap.sum()), total_gap_nats=float(tot_gap.sum()),
        equal_pair_share_of_test_choices=float(sum(p['equal_pair_n'] for p in use)/sum(p['n_test'] for p in use)),
        participants_with_positive_equal_pair_gap=int((eq_gap > 0).sum()),
        participants_with_positive_other_gap=int(((tot_gap-eq_gap) > 0).sum()),
    )


# ------------------------------------------------------ Analysis C: example selection
SELECTION_QUANTILES = (('poor_cumulative_p95', .95), ('typical_p50', .50), ('good_cumulative_p05', .05))


def select_examples(metrics, *, include_overprediction=True):
    """Deterministic, outcome-documented example choice; no manual inspection.

    For each target quantile q of delta (numpy 'linear' quantile), pick the
    participant whose delta is closest to it; ties -> lexicographically smallest
    participant ID; already-chosen participants are skipped. Optionally add the
    participant with the largest observation-weighted pooled cumulative error
    across the three equal pairs (ties -> smallest ID), if not already chosen.
    """
    ids = [p['participant'] for p in metrics]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate participants.')
    d = np.array([p['delta_vs_ck'] for p in metrics])
    chosen = []
    for role, q in SELECTION_QUANTILES:
        target = float(np.quantile(d, q))
        order = sorted(range(len(ids)), key=lambda i: (abs(d[i]-target), ids[i]))
        i = next(i for i in order if ids[i] not in {c['participant'] for c in chosen})
        chosen.append(dict(role=role, participant=ids[i], quantile=q, target_delta=target, delta=float(d[i]),
                           rule=f'|delta - quantile({q:g})| minimal; ties by ID'))
    if include_overprediction:
        cand = [(-p['pooled_error']['cumulative'], p['participant']) for p in metrics if p['pooled_error']['cumulative'] is not None]
        for neg, pid in sorted(cand):
            if pid not in {c['participant'] for c in chosen}:
                chosen.append(dict(role='largest_cumulative_overprediction', participant=pid, quantile=None, target_delta=None,
                                   delta=float(d[ids.index(pid)]), pooled_error=-neg,
                                   rule='max observation-weighted pooled cumulative equal-pair error; ties by ID'))
                break
    return chosen


# ----------------------------------------------------------- optional trial replay
def replay_participant(inputs, variant, ck_reference, cumulative_record):
    """Trial-level test predictions for one participant, reproduced and verified.

    Uses the validated replays only: ``replay_reference`` (authors RL+CK; raises if
    saved deviances are not reproduced) and ``cumulative_history_human.evaluate``
    with the saved parameters, training -> test carry-over of Q and counts.
    """
    train, test, ratings, pretest, _ = inputs
    q0, h0 = initial_values(ratings, pretest, variant)
    _, ck = replay_reference(train, test, q0, h0, ck_reference)
    np.testing.assert_allclose(q0, cumulative_record['initial_q'], rtol=0, atol=1e-12)
    theta = [cumulative_record['parameters'][p] for p in CUMULATIVE_PARAMETERS]
    tr = cumulative_evaluate(train, theta, q0)
    te = cumulative_evaluate(test, theta, tr.q[-1], initial_history=tr.final_history)
    for phase, out in (('training', tr), ('test', te)):
        if abs(out.deviance-cumulative_record[phase]['deviance']) > 1e-7 or len(out.left) != cumulative_record[phase]['n']:
            raise ValueError(f'Cumulative {phase} replay does not reproduce saved deviance/rows.')
    for model, out, saved in (('ck', ck, None), ('cumulative', te, cumulative_record['equal_pairs'])):
        if saved is None:
            continue
        for key, rec in equal_pair_scores(test, out).items():
            if rec['n'] != saved[key]['n'] or abs(rec['predicted_frequent_sum']-saved[key]['predicted_frequent_sum']) > 1e-9:
                raise ValueError('Replayed equal-pair predictions differ from saved record.')
    return dict(pairs=test.pairs+1, left=test.left.copy(), p_left=dict(ck=ck.probability_left, cumulative=te.probability_left),
                loss=dict(ck=ck.log_loss, cumulative=te.log_loss),
                deviance=dict(ck=ck.deviance, cumulative=te.deviance))


# ----------------------------------------------------------------------- outputs
def csv_rows(metrics, selected=()):
    roles = {s['participant']: s['role'] for s in selected}
    out = []
    for p in metrics:
        row = dict(participant=p['participant'], n_test=p['n_test'],
                   loss_rl=p['loss']['rl'], loss_ck=p['loss']['ck'], loss_cumulative=p['loss']['cumulative'],
                   delta_cum_minus_ck=p['delta_vs_ck'], delta_cum_minus_rl=p['delta_vs_rl'],
                   winner_vs_ck=p['winner_vs_ck'], winner_vs_rl=p['winner_vs_rl'],
                   nll_gap_cum_minus_ck=p['nll_gap_vs_ck'], equal_pair_n=p['equal_pair_n'],
                   equal_pair_nll_gap_cum_minus_ck=p['equal_pair_nll_gap_vs_ck'],
                   pooled_error_ck=p['pooled_error']['ck'], pooled_error_cumulative=p['pooled_error']['cumulative'])
        for key in PAIR_KEYS:
            tag = key.replace('-', '_')
            ck, cum = p['pairs'][key]['ck'], p['pairs'][key]['cumulative']
            row.update({f'pair{tag}_n': ck['n'], f'pair{tag}_observed_frequent_count': ck['observed_count'],
                        f'pair{tag}_observed_rate': ck['observed_rate'],
                        f'pair{tag}_predicted_rate_ck': ck['predicted_rate'], f'pair{tag}_predicted_rate_cumulative': cum['predicted_rate'],
                        f'pair{tag}_error_ck': ck['error'], f'pair{tag}_error_cumulative': cum['error']})
        for k, v in p['parameters']['cumulative'].items():
            row[f'cumulative_{k}'] = v
        for k, v in p['parameters']['ck'].items():
            row[f'ck_{k}'] = v
        row['example_role'] = roles.get(p['participant'], '')
        out.append(row)
    return out


def write_csv(rows, path):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: '' if v is None else (f'{v:.10g}' if isinstance(v, float) else v) for k, v in row.items()})


def analyze_variant(data, variant):
    rows = variant_rows(data, variant)
    metrics = participant_metrics(rows)
    result = dict(variant=variant, participants=len(metrics),
                  delta_vs_ck=delta_summary([p['delta_vs_ck'] for p in metrics]),
                  delta_vs_rl=delta_summary([p['delta_vs_rl'] for p in metrics]),
                  trial_weighted=dict(ck=trial_weighted(metrics, 'ck'), rl=trial_weighted(metrics, 'rl')),
                  loss_distribution={k: distribution([p['loss'][k] for p in metrics]) for k in MODELS},
                  pairs={key: {model: pair_error_summary(metrics, key, model) for model in MODELS} for key in PAIR_KEYS},
                  consistency={model: cross_pair_consistency(metrics, model) for model in HISTORY_MODELS},
                  gap_link=gap_link(metrics))
    return metrics, result


def reproduction_checks(data, variant, metrics, result, *, tolerance=1e-9):
    """Compare re-derived cohort numbers with the saved summary block."""
    saved = data['summary'][variant]
    checks = []

    def add(name, ours, theirs, tol=tolerance):
        ok = (ours == theirs) if isinstance(theirs, int) else abs(ours-theirs) <= tol
        checks.append(dict(check=name, ours=ours, saved=theirs, passed=bool(ok)))

    add('participants', result['participants'], saved['complete_pairs'])
    for ref in ('ck', 'rl'):
        c = saved['comparisons'][ref]
        mine = result[f'delta_vs_{ref}']
        add(f'{ref}: cumulative worse count', mine['worse'], c['cumulative_worse'])
        add(f'{ref}: cumulative better count', mine['better'], c['cumulative_better'])
        add(f'{ref}: ties', mine['tied'], c['ties'])
        add(f'{ref}: mean delta (sign-flipped)', mine['mean'], -c['delta_test_log_loss']['mean'])
        add(f'{ref}: median delta (sign-flipped)', mine['distribution']['median'], -c['delta_test_log_loss']['median'])
        add(f'{ref}: total NLL gap (sign-flipped)', result['trial_weighted'][ref]['total_nll_gap'], -c['total_delta_test_LL'], 1e-6)
    for model in MODELS:
        s = saved['models'][model]
        add(f'{model}: trial-weighted test log loss', result['trial_weighted']['ck']['trial_weighted_loss'][model], s['trial_weighted_test_log_loss'])
        add(f'{model}: participant mean test log loss', result['loss_distribution'][model]['mean'], s['participant_test_log_loss']['mean'])
        add(f'{model}: test choices', result['trial_weighted']['ck']['test_choices'], s['test_choices'])
    for key in PAIR_KEYS:
        for model in MODELS:
            ours, theirs = result['pairs'][key][model]['pooled'], saved['equal_pairs'][key][model]
            add(f'{key} {model}: n', ours['n'], theirs['n'])
            add(f'{key} {model}: observed frequent rate', ours['observed_rate'], theirs['observed_frequent_rate'])
            add(f'{key} {model}: predicted frequent rate', ours['predicted_rate'], theirs['predicted_frequent_rate'])
            add(f'{key} {model}: log loss', ours['log_loss'], theirs['log_loss'])
    return checks
