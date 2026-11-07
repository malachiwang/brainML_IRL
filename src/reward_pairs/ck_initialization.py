"""Experiment 4: choice-kernel initialization (K0) in the authors' RL+CK, all else unchanged.

Conditions differ ONLY in the initial kernel vector passed to the unchanged
authors' machinery (``TrainingObjective``/``fit_authors``/``evaluate`` accept
``h0`` externally):

  C0  released_code_intended: K0 = Q0 = pretraining value vector (primary reference)
  C1  K0 = 0.5 for all eight stimuli (primary intervention)
  C2  K0 = 0   for all eight stimuli (secondary robustness check)

Q0, update rules, bounds, objective, optimizer, starts, chronology and pi=0 are
identical.  Fitting uses training sessions 1-4 only; session 5 is scored by
causal replay with training-final Q and K carried in.
"""
import numpy as np

from .authors_audit import random_starts as authors_starts
from .authors_evaluation import distribution
from .authors_fitting import AuthorsFitError, fit_authors
from .authors_models import FREE, LIMITS, TrainingObjective, evaluate as authors_evaluate
from .ck_ablation import TIE_TOLERANCE, effect_summary, paired_bootstrap
from .ck_alpha_profile import NUISANCE, profile_curve, random_nuisance_starts

MODEL = 19
CONDITIONS = ('C0', 'C1', 'C2')
LABELS = {'C0': 'released K0 = Q0', 'C1': 'K0 = 0.5 (neutral)', 'C2': 'K0 = 0 (zero)'}
UNIFORM_K0 = {'C1': .5, 'C2': 0.}
PROFILE_GRID = (.02, .03, .05, .075, .10, .125, .15, .20, .30)
SEGMENTS = ('S1 early', 'S1 late', 'S2', 'S3', 'S4', 'S5')
PARAMETERS = ('alpha_ck', 'beta_ck', 'alpha_rl', 'beta_rl')
CROSS_TOL = 1e-6          # deviance: a cross-condition warm start "improves" the multistart optimum
THRESHOLD = .005          # nats/choice, descriptive only
EARLY_FRACTION = 1/3      # first third of each participant's session-1 trials


def initial_kernel(condition, released_k0):
    released_k0 = np.asarray(released_k0, float)
    if released_k0.shape != (8,):
        raise ValueError('Need eight released K0 values.')
    if condition == 'C0':
        return released_k0.copy()
    if condition not in UNIFORM_K0:
        raise ValueError(f'Unknown initialization condition {condition!r}.')
    return np.full(8, UNIFORM_K0[condition])


def session_losses(log_loss, sessions):
    """Mean log loss per fitted session (participant-level quantity)."""
    sessions = np.asarray(sessions)
    if len(sessions) != len(log_loss):
        raise ValueError('Session labels must align with trials.')
    return {str(int(s)): float(log_loss[sessions == s].mean()) for s in np.unique(sessions)}


def segment_labels(sessions):
    """S1 early (first third of S1 trials), S1 late, S2, S3, S4 for fitted trials."""
    sessions = np.asarray(sessions)
    labels = np.array([f'S{int(s)}' for s in sessions], dtype=object)
    s1 = np.flatnonzero(sessions == 1)
    cut = int(np.ceil(len(s1)*EARLY_FRACTION))
    labels[s1[:cut]] = 'S1 early'; labels[s1[cut:]] = 'S1 late'
    return labels


def contributions(trials, output, theta):
    t = np.arange(len(trials.left)); l, r = trials.pairs.T
    zq = theta[3]*(output.q[t, l]-output.q[t, r])
    zk = theta[1]*(output.h[t, l]-output.h[t, r])
    np.testing.assert_allclose(1/(1+np.exp(-(zq+zk))), output.probability_left, rtol=0, atol=1e-12)
    return zq, zk


def _segment_stats(zq, zk, labels):
    out = {}
    for s in SEGMENTS:
        m = labels == s
        if m.any():
            out[s] = dict(n=int(m.sum()), median_abs_zq=float(np.median(abs(zq[m]))), median_abs_zk=float(np.median(abs(zk[m]))),
                          mean_abs_zq=float(abs(zq[m]).mean()), mean_abs_zk=float(abs(zk[m]).mean()),
                          kernel_reverses_reward=float(np.mean((abs(zq[m]) > 1e-9) & (np.sign(zq[m]+zk[m]) != np.sign(zq[m])))))
    return out


def replay(fold, theta, q0, k0):
    """Unchanged authors' evaluate: S1-S4 then S5 with carried states (pi = 0)."""
    tr = authors_evaluate(fold.fit_trials, MODEL, theta, q0, k0, scoring_pi=0.)
    va = authors_evaluate(fold.validation_trials, MODEL, theta, tr.q[-1], tr.h[-1], scoring_pi=0.)
    return tr, va


def score(fold, theta, q0, k0):
    tr, va = replay(fold, theta, q0, k0)
    labels = segment_labels(fold.fit_sessions)
    zq, zk = contributions(fold.fit_trials, tr, theta)
    vq, vk = contributions(fold.validation_trials, va, theta)
    seg = _segment_stats(np.r_[zq, vq], np.r_[zk, vk], np.r_[labels, np.full(len(vq), 'S5', dtype=object)])
    first = np.arange(min(20, len(zk)))
    return dict(train_deviance=tr.deviance, train_nll=tr.deviance/2, train_loss=float(tr.log_loss.mean()),
                session_loss=session_losses(tr.log_loss, fold.fit_sessions),
                session_nll={str(int(s)): float(tr.log_loss[fold.fit_sessions == s].sum()) for s in np.unique(fold.fit_sessions)},
                s1_early_loss=float(tr.log_loss[labels == 'S1 early'].mean()),
                s5_nll=float(va.log_loss.sum()), s5_loss=float(va.log_loss.mean()),
                segments=seg, first20_mean_abs_zk=float(abs(zk[first]).mean()), first20_mean_abs_zq=float(abs(zq[first]).mean()),
                final_k_range=float(np.ptp(tr.h[-1])), final_k=tr.h[-1].tolist(), final_q=tr.q[-1].tolist(),
                carried_ok=bool(np.array_equal(tr.h[-1], va.h[0]) and np.array_equal(tr.q[-1], va.q[0]))), (tr, va)


def _fit(fold, q0, k0, starts):
    fitted = fit_authors([fold.fit_trials], MODEL, q0, k0, starts=starts)
    rec = fitted.record(MODEL)
    return dict(theta=fitted.theta.tolist(), deviance=fitted.objective_deviance, failed_starts=rec['failed_starts'],
                starts=rec['restart_count'], boundary=rec['boundary_flags'],
                reaching_best=int(sum(d is not None and d <= fitted.objective_deviance+CROSS_TOL for d in rec['restart_deviances'])))


def fit_participant(job):
    """All three conditions with identical R-seeded starts, then cross-condition warm starts."""
    participant, index, fold, q0, released_k0, saved, n_starts, use_r, profile = job
    starts = authors_starts(index, n_starts, use_r=use_r)[:, list(FREE[MODEL])]
    k0s = {c: initial_kernel(c, released_k0) for c in CONDITIONS}
    fits = {}
    for c in CONDITIONS:
        try:
            fits[c] = _fit(fold, q0, k0s[c], starts) | dict(status='ok')
        except AuthorsFitError as exc:
            fits[c] = dict(status='failed', error=str(exc)[:500])
    # Cross-condition warm starts: each condition also starts from the other conditions' optima.
    for c in CONDITIONS:
        if fits[c]['status'] != 'ok':
            continue
        others = np.array([fits[o]['theta'] for o in CONDITIONS if o != c and fits[o]['status'] == 'ok'])
        if len(others):
            try:
                cross = _fit(fold, q0, k0s[c], np.clip(others, LIMITS[:4, 0], LIMITS[:4, 1]))
            except AuthorsFitError:
                fits[c]['cross_start_improvement'] = None; continue
            fits[c]['cross_start_improvement'] = max(0., fits[c]['deviance']-cross['deviance'])
            if cross['deviance'] < fits[c]['deviance']-CROSS_TOL:
                fits[c].update(theta=cross['theta'], deviance=cross['deviance'], boundary=cross['boundary'], won_by='cross_start')
    row = dict(participant=participant, source_index=index, n_train=len(fold.fit_trials.left),
               n_validation=len(fold.validation_trials.left), q0=list(map(float, q0)),
               k0={c: k0s[c].tolist() for c in CONDITIONS}, conditions={})
    for c in CONDITIONS:
        f = fits[c]
        if f['status'] != 'ok':
            row['conditions'][c] = f; continue
        s, _ = score(fold, f['theta'], q0, k0s[c])
        if abs(s['train_deviance']-f['deviance']) > 1e-7:
            raise RuntimeError('Fitted objective disagrees with unchanged scalar replay.')
        row['conditions'][c] = f | s | dict(parameters=dict(zip(PARAMETERS, f['theta'])))
    c0 = row['conditions']['C0']
    row['checks'] = dict(
        refit_vs_saved_deviance=abs(c0['deviance']-saved['optimization']['optimizer_objective_deviance']),
        refit_vs_saved_theta=float(np.max(abs(np.array(c0['theta'])-saved['theta']))),
        carried_q=float(np.max(abs(np.array(c0['final_q'])-saved['initial_validation_q']))),
        carried_k=float(np.max(abs(np.array(c0['final_k'])-saved['initial_validation_h']))),
        s5_nll=abs(c0['s5_nll']-saved['validation']['nll']),
        q0_identical=all(np.array_equal(q0, row['q0']) for _ in CONDITIONS))
    if profile:
        row['profile'] = {c: profile_condition(fold, q0, k0s[c], index, row['conditions'][c]['theta'][1:]) for c in ('C0', 'C1')
                          if row['conditions'][c]['status'] == 'ok'}
    return row


def profile_condition(fold, q0, k0, index, nuisance_start):
    """Fixed-alpha profile (nuisance re-optimized) on PROFILE_GRID; session losses at each point."""
    problem = TrainingObjective([fold.fit_trials], MODEL, q0, k0)
    pts = profile_curve(problem, PROFILE_GRID, np.nan, np.asarray(nuisance_start), random_nuisance_starts(index))
    out = []
    for p in pts:
        rec = dict(alpha_ck=p['alpha_ck'], status=p['status'])
        if p['status'] == 'ok':
            s, _ = score(fold, p['theta'], q0, k0)
            rec.update(theta=p['theta'], deviance=p['deviance'], session_loss=s['session_loss'], s1_early_loss=s['s1_early_loss'],
                       s5_loss=s['s5_loss'], projected_gradient=p['projected_gradient'], random_reaching_best=p['random_reaching_best'])
        out.append(rec)
    return out


# ------------------------------------------------------------------------------------------- summaries

def usable(rows):
    good = [r for r in rows if all(r['conditions'][c]['status'] == 'ok' for c in CONDITIONS)]
    return good, [r['participant'] for r in rows if r not in good]


def _col(rows, c, fn):
    return np.array([fn(r['conditions'][c]) for r in rows], float)


def alpha_summary(rows):
    out = {}
    a0 = _col(rows, 'C0', lambda x: x['parameters']['alpha_ck'])
    for c in CONDITIONS:
        a = _col(rows, c, lambda x: x['parameters']['alpha_ck'])
        out[c] = dict(alpha=distribution(a),
                      boundary={k.replace('authors_', ''): int(sum(r['conditions'][c]['boundary'][k] for r in rows))
                                for k in rows[0]['conditions'][c]['boundary']},
                      alpha_ck_at_1=int(np.sum(a > 1-1e-6)), alpha_ck_at_0=int(np.sum(a < 1e-6)),
                      share_in_005_010=float(np.mean((a >= .05) & (a <= .10))),
                      parameters={p: distribution(_col(rows, c, lambda x, p=p: x['parameters'][p])) for p in PARAMETERS})
        if c != 'C0':
            d = a-a0
            ratio = np.log(np.clip(a, 1e-6, None)/np.clip(a0, 1e-6, None))
            out[c]['paired_change'] = dict(difference=effect_summary(d), decreases=int(np.sum(d < -1e-6)),
                                           increases=int(np.sum(d > 1e-6)), unchanged=int(np.sum(abs(d) <= 1e-6)),
                                           median_log_ratio=float(np.median(ratio)), median_ratio=float(np.exp(np.median(ratio))),
                                           median_difference_ci=paired_bootstrap([d], lambda x: np.median(x, axis=1))['ci95'])
    return out


def session_summary(rows):
    out = {}
    for c in ('C1', 'C2'):
        res = {}
        for s in ('1', '2', '3', '4'):
            res[f'S{s}'] = effect_summary(_col(rows, c, lambda x: x['session_loss'][s])-_col(rows, 'C0', lambda x: x['session_loss'][s]))
        res['S1 early'] = effect_summary(_col(rows, c, lambda x: x['s1_early_loss'])-_col(rows, 'C0', lambda x: x['s1_early_loss']))
        n24 = np.array([sum(r['conditions']['C0']['segments'][f'S{s}']['n'] for s in (2, 3, 4)) for r in rows], float)
        l24 = lambda cc: np.array([sum(r['conditions'][cc]['session_nll'][s] for s in ('2', '3', '4')) for r in rows])/n24
        res['S2-S4'] = effect_summary(l24(c)-l24('C0'))
        total = _col(rows, c, lambda x: x['train_loss'])-_col(rows, 'C0', lambda x: x['train_loss'])
        res['total'] = effect_summary(total)
        res['total_nll'] = effect_summary(_col(rows, c, lambda x: x['train_nll'])-_col(rows, 'C0', lambda x: x['train_nll']))
        s1 = _col(rows, c, lambda x: x['session_nll']['1'])-_col(rows, 'C0', lambda x: x['session_nll']['1'])
        rest = (_col(rows, c, lambda x: x['train_nll'])-_col(rows, 'C0', lambda x: x['train_nll']))-s1
        res['offset'] = dict(s1_nll_change=distribution(s1), s2_s4_nll_change=distribution(rest),
                             s1_better_and_later_worse=int(np.sum((s1 < -TIE_TOLERANCE) & (rest > TIE_TOLERANCE))),
                             s1_worse_and_later_better=int(np.sum((s1 > TIE_TOLERANCE) & (rest < -TIE_TOLERANCE))),
                             correlation=float(np.corrcoef(s1, rest)[0, 1]))
        res['concentration'] = concentration(total*np.array([r['n_train'] for r in rows]))
        out[c] = res
    return out


def concentration(values):
    """Share of the summed absolute change carried by the top 10% of participants."""
    v = np.sort(abs(np.asarray(values, float)))[::-1]
    k = max(1, int(round(.1*len(v))))
    return dict(top10pct_share=float(v[:k].sum()/v.sum()) if v.sum() > 0 else None, n_top=k)


def validation_summary(rows):
    out = {}
    n = np.array([r['n_validation'] for r in rows], float)
    for c in ('C1', 'C2'):
        d = _col(rows, c, lambda x: x['s5_loss'])-_col(rows, 'C0', lambda x: x['s5_loss'])
        out[c] = effect_summary(d) | dict(
            median=float(np.median(d)), trial_weighted=float((d*n).sum()/n.sum()),
            trial_weighted_ci=paired_bootstrap([d*n, n], lambda a, b: a.sum(axis=1)/b.sum(axis=1))['ci95'],
            over_threshold_worse=int(np.sum(d > THRESHOLD)), over_threshold_better=int(np.sum(d < -THRESHOLD)),
            concentration=concentration(d*n))
    d = _col(rows, 'C2', lambda x: x['s5_loss'])-_col(rows, 'C1', lambda x: x['s5_loss'])
    out['C2_minus_C1'] = effect_summary(d)
    out['losses'] = {c: distribution(_col(rows, c, lambda x: x['s5_loss'])) for c in CONDITIONS}
    return out


def contribution_summary(rows):
    out = {}
    for c in CONDITIONS:
        out[c] = {s: dict(median_abs_zq=distribution(_col(rows, c, lambda x: x['segments'][s]['median_abs_zq'])),
                          median_abs_zk=distribution(_col(rows, c, lambda x: x['segments'][s]['median_abs_zk'])),
                          kernel_reverses_reward=distribution(_col(rows, c, lambda x: x['segments'][s]['kernel_reverses_reward'])))
                  for s in SEGMENTS}
        out[c]['first20'] = dict(mean_abs_zk=distribution(_col(rows, c, lambda x: x['first20_mean_abs_zk'])),
                                 mean_abs_zq=distribution(_col(rows, c, lambda x: x['first20_mean_abs_zq'])))
    for c in ('C1', 'C2'):
        out[f'{c}_minus_C0'] = {s: dict(
            zk=effect_summary(_col(rows, c, lambda x: x['segments'][s]['median_abs_zk'])-_col(rows, 'C0', lambda x: x['segments'][s]['median_abs_zk'])),
            zq=effect_summary(_col(rows, c, lambda x: x['segments'][s]['median_abs_zq'])-_col(rows, 'C0', lambda x: x['segments'][s]['median_abs_zq'])))
            for s in SEGMENTS}
    k0 = np.array([r['k0']['C0'] for r in rows])
    out['released_k0'] = dict(values=distribution(k0.ravel()), range_per_participant=distribution(np.ptp(k0, axis=1)),
                              above_1=int(np.sum(k0 > 1)), cells=int(k0.size))
    return out


def profile_summary(rows):
    rows = [r for r in rows if 'profile' in r and all(p['status'] == 'ok' for c in ('C0', 'C1') for p in r['profile'][c])]
    if not rows:
        return None
    out = dict(participants=len(rows), grid=list(PROFILE_GRID), conditions={})
    for c in ('C0', 'C1'):
        dev = np.array([[p['deviance'] for p in r['profile'][c]] for r in rows])
        best = np.minimum(dev.min(axis=1), [r['conditions'][c]['deviance'] for r in rows])
        dnll = (dev-best[:, None])/2
        per = {}
        for key, fn in (('S1 early', lambda p: p['s1_early_loss']), ('S1', lambda p: p['session_loss']['1']),
                        ('S2', lambda p: p['session_loss']['2']), ('S3', lambda p: p['session_loss']['3']),
                        ('S4', lambda p: p['session_loss']['4']), ('S5', lambda p: p['s5_loss'])):
            vals = np.array([[fn(p) for p in r['profile'][c]] for r in rows])
            ref = vals[:, PROFILE_GRID.index(.15)][:, None]   # predetermined reference point alpha = 0.15
            per[key] = dict(mean_minus_alpha015=(vals-ref).mean(axis=0).tolist(),
                            ci95=[paired_bootstrap([(vals-ref)[:, j]], lambda x: x.mean(axis=1), draws=2000)['ci95'] for j in range(len(PROFILE_GRID))])
        out['conditions'][c] = dict(delta_nll=dict(median=np.median(dnll, axis=0).tolist(), q25=np.quantile(dnll, .25, axis=0).tolist(),
                                                   q75=np.quantile(dnll, .75, axis=0).tolist()),
                                    within_1_nat=(dnll <= 1).sum(axis=0).tolist(),
                                    argmin_alpha=distribution(np.array(PROFILE_GRID)[dnll.argmin(axis=1)]),
                                    session_relative_to_alpha015=per,
                                    beyond_multistart=int(np.sum(dev.min(axis=1) < np.array([r['conditions'][c]['deviance'] for r in rows])-CROSS_TOL)))
    # The E3 pattern: session-1 penalty of a slow kernel (0.05 vs 0.15), per participant, C1 minus C0.
    j05, j15 = PROFILE_GRID.index(.05), PROFILE_GRID.index(.15)
    pen = {c: np.array([r['profile'][c][j05]['session_loss']['1']-r['profile'][c][j15]['session_loss']['1'] for r in rows]) for c in ('C0', 'C1')}
    late = {c: np.array([np.mean([r['profile'][c][j05]['session_loss'][s]-r['profile'][c][j15]['session_loss'][s] for s in '234'])
                         for r in rows]) for c in ('C0', 'C1')}
    out['s1_penalty_005_vs_015'] = {c: effect_summary(pen[c]) for c in pen} | dict(C1_minus_C0=effect_summary(pen['C1']-pen['C0']))
    out['s2_s4_penalty_005_vs_015'] = {c: effect_summary(late[c]) for c in late} | dict(C1_minus_C0=effect_summary(late['C1']-late['C0']))
    return out


def summarize(rows):
    good, excluded = usable(rows)
    return dict(participants=len(rows), analysed=len(good), excluded=excluded,
                optimization={c: dict(failed_starts=int(sum(r['conditions'][c].get('failed_starts', 0) for r in rows)),
                                      failed_conditions=int(sum(r['conditions'][c]['status'] != 'ok' for r in rows)),
                                      reaching_best=distribution(np.array([r['conditions'][c]['reaching_best'] for r in good], float)),
                                      won_by_cross_start=int(sum(r['conditions'][c].get('won_by') == 'cross_start' for r in good)),
                                      max_cross_start_improvement=float(max(r['conditions'][c].get('cross_start_improvement') or 0. for r in good)))
                              for c in CONDITIONS},
                alpha=alpha_summary(good), sessions=session_summary(good), validation=validation_summary(good),
                contributions=contribution_summary(good), profile=profile_summary(good))
