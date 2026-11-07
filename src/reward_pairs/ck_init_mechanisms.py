"""Experiment 5: what in the released choice-kernel initialization drives the fitted alpha_CK?

Part A (real data): K0 variants that separate the *level*, *spread* and stimulus-specific
*pattern* of the released initialization v (= Q0).  Write v = v_bar*1 + d (d centred).
Because choice depends on K_L - K_R, and an unseen-update stimulus keeps
(1-alpha)^n of its start, K0 enters the logit only through
    beta_CK [ (1-alpha)^{n_L} K0_L - (1-alpha)^{n_R} K0_R ],
so a uniform level matters only through unequal appearance counts n_L != n_R, whereas
the centred pattern d matters from trial 1.  Variants:

  released      v                              level v_bar, pattern d (spread ~1.8), = authors
  level_shift   0.5 + d                        level 0.5, same pattern and spread      (isolates level)
  shrink_0.5    0.5 + 0.5 d                    level 0.5, same pattern, half spread    (spread dial)
  shrink_0.25   0.5 + 0.25 d                   level 0.5, same pattern, quarter spread (spread dial)
  uniform_0.5   0.5                            no pattern  (= E4 C1)
  uniform_mean  v_bar                          participant's own level, no pattern
  permuted      v[pi], pi a derangement        same values/level/spread, pattern destroyed
  minmax        (v - min v)/(max v - min v)    rank-preserving map into [0,1] (spread -> 1)

Q0 is the released vector in every variant.  Fitting: S1-S4 only, unchanged
authors' objective/optimizer; S5 scored by causal replay.

Part B (synthetic): choices simulated from the unchanged authors' decision process on
real S1-S5 stimulus/reward sequences (rewards are deterministic per stimulus and do not
depend on choices, so real sequences are valid), with known parameters and a known K0
rule; then refitted under several K0 assumptions.
"""
import numpy as np

from .authors_audit import random_starts as authors_starts
from .authors_evaluation import distribution
from .authors_fitting import AuthorsFitError, fit_authors
from .authors_models import FREE, LIMITS, AuthorsTrials, TrainingObjective, evaluate as authors_evaluate
from .ck_ablation import effect_summary, paired_bootstrap
from .ck_alpha_profile import FixedAlphaObjective, minimize_fixed
from .ck_initialization import score

MODEL = 19
VARIANTS = ('released', 'level_shift', 'shrink_0.5', 'shrink_0.25', 'uniform_0.5', 'uniform_mean', 'permuted', 'minmax')
DESCRIPTION = {
    'released': 'v (authors; level v̄, pattern d, spread ≈ 1.8)',
    'level_shift': '0.5 + d (same pattern and spread, level 0.5)',
    'shrink_0.5': '0.5 + 0.5·d (same pattern, half spread)',
    'shrink_0.25': '0.5 + 0.25·d (same pattern, quarter spread)',
    'uniform_0.5': '0.5 for all (no pattern; = E4 C1)',
    'uniform_mean': 'v̄ for all (own level, no pattern)',
    'permuted': 'v shuffled across stimuli (same values, pattern destroyed)',
    'minmax': '(v − min)/(max − min) (rank-preserving, in [0,1])'}
PARAMETERS = ('alpha_ck', 'beta_ck', 'alpha_rl', 'beta_rl')
PERMUTATION_SEED = 20261108
SYNTH_SEED = 20261109
SYNTH_STARTS = 50
FIT_INITS = ('released', 'uniform_0.5', 'minmax')
GENERATORS = ('released', 'uniform_0.5')
GRID_ALPHA = (.02, .05, .10, .20, .40)
GRID_BETA_CK = (1., 2.5)
TOL = 1e-6


def derangement(index, seed=PERMUTATION_SEED):
    """Deterministic permutation of 8 stimuli with no fixed point."""
    rng = np.random.default_rng([seed, int(index)])
    while True:
        p = rng.permutation(8)
        if np.all(p != np.arange(8)):
            return p


def initial_kernel(variant, v, index):
    v = np.asarray(v, float)
    if v.shape != (8,) or not np.isfinite(v).all():
        raise ValueError('Need eight finite released values.')
    d = v-v.mean()
    if variant == 'released':
        return v.copy()
    if variant == 'level_shift':
        return .5+d
    if variant.startswith('shrink_'):
        return .5+float(variant.split('_')[1])*d
    if variant == 'uniform_0.5':
        return np.full(8, .5)
    if variant == 'uniform_mean':
        return np.full(8, v.mean())
    if variant == 'permuted':
        return v[derangement(index)]
    if variant == 'minmax':
        span = np.ptp(v)
        if span <= 0:
            raise ValueError('Released values are constant; min-max undefined.')
        return (v-v.min())/span
    raise ValueError(f'Unknown K0 variant {variant!r}.')


def k0_descriptors(k0, v):
    d, e = k0-k0.mean(), v-v.mean()
    return dict(level=float(k0.mean()), spread=float(np.ptp(k0)), sd=float(k0.std()),
                pattern_corr=float(np.corrcoef(d, e)[0, 1]) if np.ptp(k0) > 0 else None,
                above_1=int(np.sum(k0 > 1)), below_0=int(np.sum(k0 < 0)))


def _fit(fold, q0, k0, starts):
    fitted = fit_authors([fold.fit_trials], MODEL, q0, k0, starts=starts)
    rec = fitted.record(MODEL)
    return dict(theta=fitted.theta.tolist(), deviance=fitted.objective_deviance, failed_starts=rec['failed_starts'],
                boundary=rec['boundary_flags'],
                reaching_best=int(sum(d is not None and d <= fitted.objective_deviance+TOL for d in rec['restart_deviances'])))


# ------------------------------------------------------------------------------------------- Part A

def fit_real(job):
    participant, index, fold, q0, v, saved, n_starts, use_r = job
    starts = authors_starts(index, n_starts, use_r=use_r)[:, list(FREE[MODEL])]
    starts = np.vstack([starts, np.asarray(saved['theta'])[None]])  # + saved released optimum, identical for all variants
    row = dict(participant=participant, source_index=index, n_train=len(fold.fit_trials.left),
               n_validation=len(fold.validation_trials.left), q0=list(map(float, q0)), variants={})
    for name in VARIANTS:
        k0 = initial_kernel(name, v, index)
        try:
            f = _fit(fold, q0, k0, starts)
        except AuthorsFitError as exc:
            row['variants'][name] = dict(status='failed', error=str(exc)[:300]); continue
        s, _ = score(fold, f['theta'], q0, k0)
        if abs(s['train_deviance']-f['deviance']) > 1e-7:
            raise RuntimeError('Objective disagrees with scalar replay.')
        s.pop('final_q'); s.pop('final_k')
        row['variants'][name] = f | s | dict(status='ok', k0=k0.tolist(), k0_descriptors=k0_descriptors(k0, v),
                                            parameters=dict(zip(PARAMETERS, f['theta'])))
    rel = row['variants']['released']
    row['checks'] = dict(released_vs_saved_deviance=abs(rel['deviance']-saved['optimization']['optimizer_objective_deviance']),
                         released_s5_nll=abs(rel['s5_nll']-saved['validation']['nll']))
    return row


# ------------------------------------------------------------------------------------------- Part B

def simulate(fold, theta, q0, k0, seed):
    """Simulated S1-S4 then S5 choices (states carried); observed human choices are never used."""
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, 2**32, 2)
    ft, vt = fold.fit_trials, fold.validation_trials
    blank = lambda t: AuthorsTrials(t.pairs, np.zeros(len(t.left), bool), t.rewards)
    tr = authors_evaluate(blank(ft), MODEL, theta, q0, k0, simulate_seed=int(a))
    va = authors_evaluate(blank(vt), MODEL, theta, tr.q[-1], tr.h[-1], simulate_seed=int(b))
    return AuthorsTrials(ft.pairs, tr.left, ft.rewards), AuthorsTrials(vt.pairs, va.left, vt.rewards)


class _Fold:
    def __init__(self, fit_trials, validation_trials, fit_sessions):
        self.fit_trials, self.validation_trials, self.fit_sessions = fit_trials, validation_trials, fit_sessions


def recover(fold, q0, k0_fit, truth, k0_true, starts, rng):
    """Multistart fit, separate truth-start fit (optimization check), fixed-true-alpha fit (identifiability)."""
    multi = _fit(fold, q0, k0_fit, starts)
    try:
        from_truth = _fit(fold, q0, k0_fit, np.asarray(truth)[None])
    except AuthorsFitError:  # recorded, never silently dropped: the truth start alone did not converge
        from_truth = dict(deviance=np.inf, failed=True)
    best = multi if multi['deviance'] <= from_truth['deviance'] else from_truth
    problem = TrainingObjective([fold.fit_trials], MODEL, q0, k0_fit)
    fixed = FixedAlphaObjective(problem, truth[0])
    nuis = [np.clip(truth[1:], LIMITS[1:4, 0], LIMITS[1:4, 1])] + list(rng.uniform([.001, .001, .001], [9.999, .999, 9.999], (8, 3)))
    attempts = [minimize_fixed(fixed, x, f'start_{i}') for i, x in enumerate(nuis)]
    ok = [a for a in attempts if a['success']]
    fixed_dev = min(a['deviance'] for a in ok) if ok else None
    s, (tr, va) = score(fold, best['theta'], q0, k0_fit)
    true_s, (ttr, tva) = score(fold, truth, q0, k0_true)
    return dict(theta=best['theta'], deviance=best['deviance'], boundary=best['boundary'],
                multistart_deviance=multi['deviance'],
                truth_start_deviance=None if from_truth.get('failed') else from_truth['deviance'],
                truth_start_failed=bool(from_truth.get('failed', False)),
                optimization_failure=bool(from_truth['deviance'] < multi['deviance']-TOL),
                reaching_best=multi['reaching_best'],
                nll_at_truth_minus_fit=(problem(np.asarray(truth))[0]-best['deviance'])/2,
                delta_nll_true_alpha=None if fixed_dev is None else (fixed_dev-best['deviance'])/2,
                session_loss=s['session_loss'], s1_early_loss=s['s1_early_loss'], train_loss=s['train_loss'],
                s5_loss=s['s5_loss'], s5_loss_truth=true_s['s5_loss'],
                s5_mean_abs_dp_vs_truth=float(abs(va.probability_left-tva.probability_left).mean()),
                train_mean_abs_dp_vs_truth=float(abs(tr.probability_left-ttr.probability_left).mean()),
                segments={k: dict(zk=v['median_abs_zk'], zq=v['median_abs_zq']) for k, v in s['segments'].items()},
                s5_probability=va.probability_left)


def synthetic_dataset(fold, q0, v, index, generator, truth, seed, fit_inits=FIT_INITS, n_starts=SYNTH_STARTS):
    k0_true = initial_kernel(generator, v, index)
    fit_trials, val_trials = simulate(fold, truth, q0, k0_true, seed)
    sf = _Fold(fit_trials, val_trials, fold.fit_sessions)
    rng = np.random.default_rng([SYNTH_SEED, index, seed % (2**31)])
    starts = rng.uniform([.001, .001, .001, .001], [.999, 9.999, .999, 9.999], (n_starts, 4))
    fits = {f: recover(sf, q0, initial_kernel(f, v, index), truth, k0_true, starts, rng) for f in fit_inits}
    ref = fits.get(generator)
    for f, r in fits.items():
        r['s5_mean_abs_dp_vs_matched'] = None if ref is None else float(abs(r['s5_probability']-ref['s5_probability']).mean())
    for r in fits.values():
        r.pop('s5_probability')
    return dict(generator=generator, truth=list(map(float, truth)), seed=int(seed),
                simulated_left_rate=float(fit_trials.left.mean()), fits=fits)


def synth_participant(job):
    """Realistic regime: each participant's own fitted parameters under each generator, several seeds."""
    participant, index, fold, q0, v, truths, n_seeds = job
    out = []
    for g in GENERATORS:
        for r in range(n_seeds):
            seed = int(np.random.default_rng([SYNTH_SEED, index, GENERATORS.index(g), r]).integers(2**31))
            out.append(synthetic_dataset(fold, q0, v, index, g, truths[g], seed))
    return dict(participant=participant, source_index=index, datasets=out)


def grid_participant(job):
    """Controlled regime: designed true alpha_CK x beta_CK, nuisance at cohort medians."""
    participant, index, fold, q0, v, base = job
    out = []
    for g in GENERATORS:
        for a in GRID_ALPHA:
            for b in GRID_BETA_CK:
                truth = [a, b, base[g]['alpha_rl'], base[g]['beta_rl']]
                seed = int(np.random.default_rng([SYNTH_SEED+1, index, GENERATORS.index(g), int(a*1000), int(b*10)]).integers(2**31))
                out.append(synthetic_dataset(fold, q0, v, index, g, truth, seed, fit_inits=('released', 'uniform_0.5')))
    return dict(participant=participant, source_index=index, datasets=out)


# ------------------------------------------------------------------------------------------- summaries

def usable(rows):
    good = [r for r in rows if all(r['variants'][v]['status'] == 'ok' for v in VARIANTS)]
    return good, [r['participant'] for r in rows if r not in good]


def _col(rows, v, fn):
    return np.array([fn(r['variants'][v]) for r in rows], float)


def real_summary(rows, draws=10000):
    out = dict(variants={}, contrasts={})
    for v in VARIANTS:
        a = _col(rows, v, lambda x: x['parameters']['alpha_ck'])
        out['variants'][v] = dict(
            description=DESCRIPTION[v],
            parameters={p: distribution(_col(rows, v, lambda x, p=p: x['parameters'][p])) for p in PARAMETERS},
            boundary={k.replace('authors_', ''): int(sum(r['variants'][v]['boundary'][k] for r in rows)) for k in rows[0]['variants'][v]['boundary']},
            alpha_below_001=int(np.sum(a < .01)), alpha_at_1=int(np.sum(a > 1-1e-6)),
            half_life_median=float(np.log(.5)/np.log1p(-np.median(a))),
            train_nll=distribution(_col(rows, v, lambda x: x['train_nll'])),
            s5_loss=distribution(_col(rows, v, lambda x: x['s5_loss'])),
            k0=dict(level=distribution(_col(rows, v, lambda x: x['k0_descriptors']['level'])),
                    spread=distribution(_col(rows, v, lambda x: x['k0_descriptors']['spread'])),
                    pattern_corr=distribution(np.array([r['variants'][v]['k0_descriptors']['pattern_corr'] for r in rows
                                                        if r['variants'][v]['k0_descriptors']['pattern_corr'] is not None], float))),
            segments={s: dict(zk=distribution(_col(rows, v, lambda x, s=s: x['segments'][s]['median_abs_zk'])),
                              zq=distribution(_col(rows, v, lambda x, s=s: x['segments'][s]['median_abs_zq'])))
                      for s in ('S1 early', 'S1 late', 'S2', 'S4', 'S5')},
            first20=dict(zk=distribution(_col(rows, v, lambda x: x['first20_mean_abs_zk'])),
                         zq=distribution(_col(rows, v, lambda x: x['first20_mean_abs_zq']))),
            failed_starts=int(sum(r['variants'][v]['failed_starts'] for r in rows)),
            reaching_best=distribution(_col(rows, v, lambda x: x['reaching_best'])))
    for ref in ('released', 'uniform_0.5'):
        for v in VARIANTS:
            if v == ref:
                continue
            la = np.log(np.clip(_col(rows, v, lambda x: x['parameters']['alpha_ck']), 1e-4, None)) - \
                np.log(np.clip(_col(rows, ref, lambda x: x['parameters']['alpha_ck']), 1e-4, None))
            dar = _col(rows, v, lambda x: x['parameters']['alpha_rl'])-_col(rows, ref, lambda x: x['parameters']['alpha_rl'])
            out['contrasts'][f'{v}-vs-{ref}'] = dict(
                log_alpha_ck=effect_summary(la, draws=draws) | dict(median_ratio=float(np.exp(np.median(la)))),
                alpha_ck_decreases=int(np.sum(la < -1e-6)), alpha_ck_increases=int(np.sum(la > 1e-6)),
                alpha_rl=effect_summary(dar, draws=draws),
                beta_ck=effect_summary(_col(rows, v, lambda x: x['parameters']['beta_ck'])-_col(rows, ref, lambda x: x['parameters']['beta_ck']), draws=draws),
                train_nll=effect_summary(_col(rows, v, lambda x: x['train_nll'])-_col(rows, ref, lambda x: x['train_nll']), draws=draws),
                s1_loss=effect_summary(_col(rows, v, lambda x: x['session_loss']['1'])-_col(rows, ref, lambda x: x['session_loss']['1']), draws=draws),
                s2_s4_loss=effect_summary(np.mean([_col(rows, v, lambda x, s=s: x['session_loss'][s])-_col(rows, ref, lambda x, s=s: x['session_loss'][s])
                                                   for s in '234'], axis=0), draws=draws),
                s5_loss=effect_summary(_col(rows, v, lambda x: x['s5_loss'])-_col(rows, ref, lambda x: x['s5_loss']), draws=draws))
    # Spread dial at fixed level 0.5 and fixed pattern: lambda = 0, .25, .5, 1.
    dial = ('uniform_0.5', 'shrink_0.25', 'shrink_0.5', 'level_shift')
    out['spread_dial'] = dict(lambdas=[0, .25, .5, 1.],
                              alpha_ck_median=[out['variants'][v]['parameters']['alpha_ck']['median'] for v in dial],
                              alpha_rl_median=[out['variants'][v]['parameters']['alpha_rl']['median'] for v in dial],
                              beta_ck_median=[out['variants'][v]['parameters']['beta_ck']['median'] for v in dial],
                              train_nll_vs_uniform=[0.]+[out['contrasts'][f'{v}-vs-uniform_0.5']['train_nll']['mean'] for v in dial[1:]],
                              s5_vs_uniform=[0.]+[out['contrasts'][f'{v}-vs-uniform_0.5']['s5_loss']['mean'] for v in dial[1:]])
    # Best-fitting variant per participant on TRAINING likelihood (descriptive; equal parameter counts).
    nll = np.array([[r['variants'][v]['train_nll'] for v in VARIANTS] for r in rows])
    winners = np.array(VARIANTS)[nll.argmin(axis=1)]
    out['training_winner_counts'] = {v: int(np.sum(winners == v)) for v in VARIANTS}
    return out


def recovery_summary(rows, key='participants'):
    """Bias = fitted - true (log scale for rates), per generator x fit init."""
    data = [d for r in rows for d in r['datasets']]
    out = {}
    for g in GENERATORS:
        for f in FIT_INITS:
            ds = [d for d in data if d['generator'] == g and f in d['fits']]
            if not ds:
                continue
            true = np.array([d['truth'] for d in ds]); fit = np.array([d['fits'][f]['theta'] for d in ds])
            la = np.log(np.clip(fit[:, 0], 1e-4, None))-np.log(np.clip(true[:, 0], 1e-4, None))
            rec = dict(n=len(ds),
                       log_alpha_ck_bias=distribution(la), alpha_ck_ratio_median=float(np.exp(np.median(la))),
                       alpha_ck_overestimated=int(np.sum(la > .1)), alpha_ck_underestimated=int(np.sum(la < -.1)),
                       spearman_alpha_ck=float(_spearman(true[:, 0], fit[:, 0])),
                       parameter_error={p: distribution(fit[:, i]-true[:, i]) for i, p in enumerate(PARAMETERS)},
                       optimization_failures=int(sum(d['fits'][f]['optimization_failure'] for d in ds)),
                       truth_start_failures=int(sum(d['fits'][f].get('truth_start_failed', False) for d in ds)),
                       truth_beats_fit=int(sum(d['fits'][f]['nll_at_truth_minus_fit'] < -TOL for d in ds)),
                       nll_at_truth_minus_fit=distribution(np.array([d['fits'][f]['nll_at_truth_minus_fit'] for d in ds])),
                       delta_nll_true_alpha=distribution(np.array([d['fits'][f]['delta_nll_true_alpha'] for d in ds
                                                                   if d['fits'][f]['delta_nll_true_alpha'] is not None])),
                       true_alpha_within_1_nat=int(sum((d['fits'][f]['delta_nll_true_alpha'] or 0) <= 1 for d in ds)),
                       s5_loss_minus_truth=distribution(np.array([d['fits'][f]['s5_loss']-d['fits'][f]['s5_loss_truth'] for d in ds])),
                       s5_mean_abs_dp_vs_truth=distribution(np.array([d['fits'][f]['s5_mean_abs_dp_vs_truth'] for d in ds])),
                       s1_early_minus_matched=None)
            if g in FIT_INITS and f != g:
                rec['s1_early_minus_matched'] = distribution(np.array([d['fits'][f]['s1_early_loss']-d['fits'][g]['s1_early_loss'] for d in ds]))
                rec['train_nll_minus_matched'] = distribution(np.array([(d['fits'][f]['train_loss']-d['fits'][g]['train_loss']) for d in ds]))
                rec['s5_mean_abs_dp_vs_matched'] = distribution(np.array([d['fits'][f]['s5_mean_abs_dp_vs_matched'] for d in ds]))
                rec['s5_loss_minus_matched'] = distribution(np.array([d['fits'][f]['s5_loss']-d['fits'][g]['s5_loss'] for d in ds]))
                # participant-level paired bootstrap (average over seeds within participant)
                per = {}
                for r in rows:
                    v = [np.log(max(d['fits'][f]['theta'][0], 1e-4))-np.log(max(d['fits'][g]['theta'][0], 1e-4))
                         for d in r['datasets'] if d['generator'] == g and f in d['fits']]
                    if v:
                        per[r['participant']] = np.mean(v)
                if len(per) > 1:
                    rec['log_alpha_mismatch_minus_matched'] = effect_summary(np.array(list(per.values())), draws=2000)
            out[f'{g}->{f}'] = rec
    return out


def grid_summary(rows):
    data = [d for r in rows for d in r['datasets']]
    out = []
    for g in GENERATORS:
        for a in GRID_ALPHA:
            for b in GRID_BETA_CK:
                ds = [d for d in data if d['generator'] == g and np.isclose(d['truth'][0], a) and np.isclose(d['truth'][1], b)]
                cell = dict(generator=g, alpha=a, beta_ck=b, n=len(ds))
                for f in ('released', 'uniform_0.5'):
                    fit = np.array([d['fits'][f]['theta'] for d in ds])
                    cell[f] = dict(alpha_median=float(np.median(fit[:, 0])), alpha_q25=float(np.quantile(fit[:, 0], .25)),
                                   alpha_q75=float(np.quantile(fit[:, 0], .75)),
                                   alpha_rl_median=float(np.median(fit[:, 2])), beta_ck_median=float(np.median(fit[:, 1])),
                                   true_alpha_within_1_nat=int(sum((d['fits'][f]['delta_nll_true_alpha'] or 0) <= 1 for d in ds)),
                                   optimization_failures=int(sum(d['fits'][f]['optimization_failure'] for d in ds)),
                       truth_start_failures=int(sum(d['fits'][f].get('truth_start_failed', False) for d in ds)),
                                   mean_train_loss=float(np.mean([d['fits'][f]['train_loss'] for d in ds])))
                cell['released_minus_uniform_train_loss'] = float(np.mean([d['fits']['released']['train_loss']-d['fits']['uniform_0.5']['train_loss'] for d in ds]))
                out.append(cell)
    return out


def _spearman(x, y):
    from scipy import stats
    return stats.spearmanr(x, y).statistic


# ------------------------------------------------------------------------------------------- Part C follow-up

def pretest_proportions(objects, participant):
    """Pre-training (pretest) choice proportions per stimulus 1..8: empirical choice rates in [0, 1].

    Same source table and checks as ``history_diagnostics.training_initialization``; pretest data
    precede training, so using them as K0 involves no leakage."""
    import pandas as pd
    p = objects['grouped_data_RP_pretest2_clean']
    p = p.loc[p.ID.astype(str).eq(participant), ['stim_chosen', 'perc_stim_chosen']].copy()
    p['stim_chosen'] = pd.to_numeric(p.stim_chosen).astype(int)
    if len(p) != 8 or set(p.stim_chosen) != set(range(1, 9)):
        raise ValueError('Incomplete pretest proportions; no exclusions added.')
    out = p.sort_values('stim_chosen').perc_stim_chosen.to_numpy(float)
    if np.any((out < 0) | (out > 1)):
        raise ValueError('Pretest proportions must lie in [0, 1].')
    return out


FOLLOWUP = ('pretest_choice', 'pretest_choice_centered')
FOLLOWUP_DESCRIPTION = {'pretest_choice': 'K0 = pretest choice proportions p (empirical choice rates, in [0,1])',
                        'pretest_choice_centered': 'K0 = 0.5 + (p − p̄) (same pattern, level 0.5)'}


def fit_followup(job):
    """Same protocol as fit_real for K0 built from pretest choice proportions."""
    participant, index, fold, q0, v, saved, n_starts, use_r, p = job
    starts = authors_starts(index, n_starts, use_r=use_r)[:, list(FREE[MODEL])]
    starts = np.vstack([starts, np.asarray(saved['theta'])[None]])
    row = dict(participant=participant, source_index=index, pretest=list(map(float, p)), variants={})
    for name, k0 in (('pretest_choice', np.asarray(p, float)), ('pretest_choice_centered', .5+p-np.mean(p))):
        try:
            f = _fit(fold, q0, k0, starts)
        except AuthorsFitError as exc:
            row['variants'][name] = dict(status='failed', error=str(exc)[:300]); continue
        s, _ = score(fold, f['theta'], q0, k0)
        s.pop('final_q'); s.pop('final_k')
        row['variants'][name] = f | s | dict(status='ok', k0=k0.tolist(), k0_descriptors=k0_descriptors(k0, v),
                                            parameters=dict(zip(PARAMETERS, f['theta'])))
    return row


def followup_summary(real_rows, follow_rows, draws=10000):
    by = {r['participant']: r for r in follow_rows}
    rows = [r for r in real_rows if r['participant'] in by and all(by[r['participant']]['variants'][f]['status'] == 'ok' for f in FOLLOWUP)]
    merged = [dict(r, variants=dict(r['variants'], **by[r['participant']]['variants'])) for r in rows]
    out = dict(participants=len(merged), variants={}, contrasts={})
    for v in FOLLOWUP:
        a = _col(merged, v, lambda x: x['parameters']['alpha_ck'])
        out['variants'][v] = dict(description=FOLLOWUP_DESCRIPTION[v],
                                  parameters={p: distribution(_col(merged, v, lambda x, p=p: x['parameters'][p])) for p in PARAMETERS},
                                  boundary={k.replace('authors_', ''): int(sum(r['variants'][v]['boundary'][k] for r in merged))
                                            for k in merged[0]['variants'][v]['boundary']},
                                  half_life_median=float(np.log(.5)/np.log1p(-np.median(a))),
                                  train_nll=distribution(_col(merged, v, lambda x: x['train_nll'])),
                                  s5_loss=distribution(_col(merged, v, lambda x: x['s5_loss'])),
                                  k0=dict(spread=distribution(_col(merged, v, lambda x: x['k0_descriptors']['spread'])),
                                          pattern_corr_with_released=distribution(np.array([r['variants'][v]['k0_descriptors']['pattern_corr']
                                                                                             for r in merged], float))),
                                  first20=dict(zk=distribution(_col(merged, v, lambda x: x['first20_mean_abs_zk']))),
                                  failed_starts=int(sum(r['variants'][v]['failed_starts'] for r in merged)))
        for ref in ('released', 'uniform_0.5', 'shrink_0.5'):
            la = np.log(np.clip(a, 1e-4, None))-np.log(np.clip(_col(merged, ref, lambda x: x['parameters']['alpha_ck']), 1e-4, None))
            out['contrasts'][f'{v}-vs-{ref}'] = dict(
                log_alpha_ck=effect_summary(la, draws=draws) | dict(median_ratio=float(np.exp(np.median(la)))),
                alpha_rl=effect_summary(_col(merged, v, lambda x: x['parameters']['alpha_rl'])-_col(merged, ref, lambda x: x['parameters']['alpha_rl']), draws=draws),
                train_nll=effect_summary(_col(merged, v, lambda x: x['train_nll'])-_col(merged, ref, lambda x: x['train_nll']), draws=draws),
                s1_loss=effect_summary(_col(merged, v, lambda x: x['session_loss']['1'])-_col(merged, ref, lambda x: x['session_loss']['1']), draws=draws),
                s5_loss=effect_summary(_col(merged, v, lambda x: x['s5_loss'])-_col(merged, ref, lambda x: x['s5_loss']), draws=draws))
    return out


def signature_comparison(real_rows, synth_rows):
    """Which generator reproduces the real-data contrast between released and uniform fits?

    For each (real participant | synthetic dataset): log(alpha_CK released / uniform), alpha_RL
    released - uniform, and training loss per choice released - uniform.  Synthetic values are
    averaged over seeds within participant so all three columns are participant-level."""
    def triple(rel, uni, tr_rel, tr_uni):
        return [np.log(max(rel[0], 1e-4))-np.log(max(uni[0], 1e-4)), rel[2]-uni[2], tr_rel-tr_uni]
    real = np.array([triple(r['variants']['released']['theta'], r['variants']['uniform_0.5']['theta'],
                            r['variants']['released']['train_loss'], r['variants']['uniform_0.5']['train_loss']) for r in real_rows])
    out = dict(quantities=['log_alpha_ck_ratio', 'alpha_rl_difference', 'train_loss_difference'],
               real={q: distribution(real[:, i]) for i, q in enumerate(['log_alpha_ck_ratio', 'alpha_rl_difference', 'train_loss_difference'])})
    for g in GENERATORS:
        per = []
        for r in synth_rows:
            ds = [d for d in r['datasets'] if d['generator'] == g]
            per.append(np.mean([triple(d['fits']['released']['theta'], d['fits']['uniform_0.5']['theta'],
                                       d['fits']['released']['train_loss'], d['fits']['uniform_0.5']['train_loss']) for d in ds], axis=0))
        per = np.array(per)
        out[g] = {q: distribution(per[:, i]) for i, q in enumerate(out['quantities'])}
        out[g]['released_fits_better_share'] = float(np.mean(per[:, 2] < 0))
    out['real']['released_fits_better_share'] = float(np.mean(real[:, 2] < 0))
    return out
