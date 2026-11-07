"""Experiment 3: profile likelihood of the authors' RL+CK choice-kernel learning rate.

The objective is the unchanged authors' ``TrainingObjective`` (model 19,
released-code-intended initialization, pi = 0).  ``FixedAlphaObjective`` only
reconstructs the full four-parameter vector [alpha_CK (fixed), beta_CK,
alpha_RL, beta_RL] and drops the alpha component of the analytic gradient.  The
optimizer call (L-BFGS-B, bounds, tolerances) mirrors ``authors_fitting``.

Fitting sees only sessions 1-4 (``history_diagnostics.make_fold`` rejects any row
that is not phase=training).  Session 5 is scored afterwards by the original
causal replay (``history_diagnostics.replay_fold``): Q and K carried in, both keep
updating, parameters fixed, each prediction precedes its own update.

Two analyses are kept strictly apart:
  * profile: nuisance parameters re-optimized at every fixed alpha (primary);
  * fixed_nuisance: saved unconstrained nuisance held fixed, only alpha varied
    (secondary diagnostic; it is NOT a profile likelihood).
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize

from .authors_models import LIMITS, TrainingObjective, unpack
from .authors_evaluation import distribution
from .ck_ablation import TIE_TOLERANCE, effect_summary, paired_bootstrap
from .history_diagnostics import decompose, half_life, replay_fold

MODEL = 19
GRID = (0., .005, .01, .02, .03, .05, .075, .10, .125, .15, .20, .30, .50, .75, 1.)
NUISANCE = ('authors_beta_ck', 'authors_alpha_rl', 'authors_beta_rl')
NUISANCE_BOUNDS = LIMITS[[1, 2, 3]]
OPTIONS = dict(ftol=1e-12, gtol=1e-7, maxiter=1000, maxls=40)  # identical to authors_fitting.fit_authors
RANDOM_STARTS = 20
START_SEED = 20261107
AGREEMENT_TOL = 1e-6        # deviance units: a restart "reaches" the best solution
VIOLATION_TOL = 1e-6        # deviance units: constrained beating saved unconstrained optimum
BOUNDARY_TOL = 1e-6
NEAR_OPTIMAL_NATS = (.5, 1., 1.92)  # descriptive training-NLL tolerances (nats), NOT confidence levels
PRIMARY_TOL = 1.


class FixedAlphaObjective:
    """Authors' deviance as a function of the three nuisance parameters only."""

    def __init__(self, problem, alpha):
        alpha = float(alpha)
        if not np.isfinite(alpha) or not 0. <= alpha <= 1.:
            raise ValueError('Fixed alpha_CK must lie in [0, 1].')
        if problem.model != MODEL:
            raise ValueError('Profile wrapper is defined for authors model 19 only.')
        self.problem, self.alpha = problem, alpha

    def full(self, nuisance):
        theta = np.empty(4)
        theta[0] = self.alpha
        theta[1:] = np.asarray(nuisance, float)
        return theta

    def __call__(self, nuisance):
        deviance, gradient = self.problem(self.full(nuisance))
        return deviance, gradient[1:]


def random_nuisance_starts(participant_index, n=RANDOM_STARTS, seed=START_SEED):
    """Deterministic interior starts, same ranges as the authors' driver."""
    rng = np.random.default_rng([seed, int(participant_index)])
    return rng.uniform([.001, .001, .001], [9.999, .999, 9.999], (n, 3))


def projected_gradient(x, g, bounds=NUISANCE_BOUNDS, tol=1e-10):
    g = np.array(g, float)
    g[(x <= bounds[:, 0]+tol) & (g > 0)] = 0.
    g[(x >= bounds[:, 1]-tol) & (g < 0)] = 0.
    return float(np.max(abs(g)))


def minimize_fixed(objective, start, label, *, minimizer=minimize):
    start = np.asarray(start, float)
    unpack(MODEL, objective.full(start))  # bounds check of the start
    fit = minimizer(objective, start, jac=True, method='L-BFGS-B', bounds=NUISANCE_BOUNDS, options=OPTIONS)
    x = np.asarray(fit.x, float)
    valid = bool(fit.success and np.isfinite(fit.fun) and np.isfinite(x).all()
                 and np.all(x >= NUISANCE_BOUNDS[:, 0]) and np.all(x <= NUISANCE_BOUNDS[:, 1]))
    out = dict(label=label, initial=start.tolist(), success=valid, message=str(fit.message),
               deviance=float(fit.fun) if np.isfinite(fit.fun) else None,
               nuisance=x.tolist() if np.isfinite(x).all() else None,
               iterations=int(getattr(fit, 'nit', -1)), evaluations=int(getattr(fit, 'nfev', -1)))
    if valid:
        dev, grad = objective(x)
        out['projected_gradient'] = projected_gradient(x, grad)
        if abs(dev-out['deviance']) > 1e-9:
            raise RuntimeError('Optimizer value disagrees with re-evaluated objective.')
    return out


def _best(attempts):
    ok = [a for a in attempts if a['success']]
    return min(ok, key=lambda a: a['deviance']) if ok else None


def point_record(alpha, attempts, anchor):
    best = _best(attempts)
    record = dict(alpha_ck=float(alpha), anchor=bool(anchor), starts=len(attempts),
                  successful_starts=sum(a['success'] for a in attempts),
                  failed_starts=[dict(label=a['label'], message=a['message']) for a in attempts if not a['success']])
    if best is None:
        return record | dict(status='failed')
    devs = np.array([a['deviance'] for a in attempts if a['success']])
    random_ok = [a for a in attempts if a['success'] and a['label'].startswith('random')]
    x = np.array(best['nuisance'])
    return record | dict(
        status='ok', deviance=best['deviance'], nuisance=dict(zip(NUISANCE, x.tolist())),
        theta=[float(alpha)]+x.tolist(), best_label=best['label'], best_message=best['message'],
        projected_gradient=best['projected_gradient'],
        reaching_best=int(np.sum(devs <= best['deviance']+AGREEMENT_TOL)),
        random_reaching_best=sum(a['deviance'] <= best['deviance']+AGREEMENT_TOL for a in random_ok),
        random_successful=len(random_ok), worst_successful_gap=float(devs.max()-best['deviance']),
        boundary=dict(zip(NUISANCE, [bool(min(v-lo, hi-v) < BOUNDARY_TOL) for v, (lo, hi) in zip(x, NUISANCE_BOUNDS)])))


def profile_curve(problem, alphas, anchor_alpha, saved_nuisance, random_starts, *, minimizer=minimize):
    """Multistart at every alpha, then ascending and descending warm-start sweeps."""
    alphas = np.asarray(alphas, float)
    if np.any(np.diff(alphas) <= 0):
        raise ValueError('Alpha values must be strictly increasing.')
    objectives = [FixedAlphaObjective(problem, a) for a in alphas]
    attempts = [[] for _ in alphas]
    for i, obj in enumerate(objectives):
        saved = np.clip(saved_nuisance, NUISANCE_BOUNDS[:, 0], NUISANCE_BOUNDS[:, 1])
        attempts[i].append(minimize_fixed(obj, saved, 'saved_baseline', minimizer=minimizer))
        for j, start in enumerate(random_starts):
            attempts[i].append(minimize_fixed(obj, start, f'random_{j}', minimizer=minimizer))
    for order, label in ((range(1, len(alphas)), 'warm_up'), (range(len(alphas)-2, -1, -1), 'warm_down')):
        for i in order:
            neighbour = _best(attempts[i-1] if label == 'warm_up' else attempts[i+1])
            if neighbour is not None:
                attempts[i].append(minimize_fixed(objectives[i], neighbour['nuisance'], label, minimizer=minimizer))
    return [point_record(a, att, a == anchor_alpha) for a, att in zip(alphas, attempts)]


def _components(trials, output, theta):
    parts = decompose(trials, output, theta[3], theta[1])
    reward, kernel = abs(parts['reward_logit']), abs(parts['history_logit'])
    return dict(median_abs_reward=float(np.median(reward)), median_abs_kernel=float(np.median(kernel)),
                mean_abs_reward=float(reward.mean()), mean_abs_kernel=float(kernel.mean()),
                kernel_flips_fraction=float(np.mean((reward > 1e-9) & (np.sign(parts['reward_logit']) != np.sign(parts['reward_logit']+parts['history_logit'])))))


def score(fold, theta, q0, h0, reference=None):
    """Causal replay of S1-S4 then S5 with fixed parameters (unchanged authors evaluate)."""
    tr, va = replay_fold(fold, 'ck', list(theta), q0, h0)
    out = dict(train_deviance=tr.deviance, train_nll=tr.deviance/2, train_loss=float(tr.log_loss.mean()),
               s5_nll=float(va.log_loss.sum()), s5_loss=float(va.log_loss.mean()),
               train_components=_components(fold.fit_trials, tr, theta),
               s5_components=_components(fold.validation_trials, va, theta),
               carried_k_spread=float(np.ptp(va.h[0])),
               train_session_loss={str(k): float(tr.log_loss[fold.fit_sessions == k].mean()) for k in np.unique(fold.fit_sessions)})
    if reference is not None:
        out['train_mean_abs_probability_change'] = float(abs(tr.probability_left-reference[0]).mean())
        out['s5_mean_abs_probability_change'] = float(abs(va.probability_left-reference[1]).mean())
        out['s5_max_abs_probability_change'] = float(abs(va.probability_left-reference[1]).max())
    return out, (tr.probability_left, va.probability_left)


def profile_participant(job):
    """One participant: verify saved baseline, profile alpha, score S5, fixed-nuisance diagnostic."""
    participant, index, fold, q0, h0, saved, grid, n_random = job
    theta_hat = np.asarray(saved['theta'], float)
    problem = TrainingObjective([fold.fit_trials], MODEL, q0, h0)
    base, reference = score(fold, theta_hat, q0, h0)
    objective_dev = problem(theta_hat)[0]
    checks = dict(saved_fit_deviance=abs(base['train_deviance']-saved['optimization']['optimizer_objective_deviance']),
                  objective_vs_replay=abs(objective_dev-base['train_deviance']),
                  saved_s5_nll=abs(base['s5_nll']-saved['validation']['nll']))
    alphas = np.unique(np.r_[np.asarray(grid, float), theta_hat[0]])
    points = profile_curve(problem, alphas, theta_hat[0], theta_hat[1:], random_nuisance_starts(index, n_random))
    saved_dev = base['train_deviance']
    for p in points:
        if p['status'] != 'ok':
            continue
        p.update(score(fold, p['theta'], q0, h0, reference)[0])
        if abs(p['train_deviance']-p['deviance']) > 1e-7:
            raise RuntimeError('Constrained objective disagrees with unchanged scalar replay.')
        p['delta_nll_vs_saved'] = (p['deviance']-saved_dev)/2
        p['half_life'] = half_life(p['alpha_ck'])
    ok = [p for p in points if p['status'] == 'ok']
    best_found = min([saved_dev]+[p['deviance'] for p in ok])
    for p in ok:
        p['delta_nll'] = (p['deviance']-best_found)/2
        p['delta_train_loss_per_choice'] = p['delta_nll']/len(fold.fit_trials.left)
        p['delta_s5_loss'] = p['s5_loss']-base['s5_loss']
        p['delta_train_session_loss'] = {k: v-base['train_session_loss'][k] for k, v in p['train_session_loss'].items()}
    violations = [dict(alpha_ck=p['alpha_ck'], improvement_nll=-p['delta_nll_vs_saved'])
                  for p in ok if p['deviance'] < saved_dev-VIOLATION_TOL]
    anchor = next(p for p in points if p['anchor'])
    checks['anchor_vs_saved_deviance'] = None if anchor['status'] != 'ok' else anchor['deviance']-saved_dev
    fixed = []
    for a in alphas:  # secondary: other parameters held at saved values (NOT a profile)
        theta = np.r_[a, theta_hat[1:]]
        s, _ = score(fold, theta, q0, h0, reference)
        fixed.append(dict(alpha_ck=float(a), delta_nll=(s['train_deviance']-best_found)/2,
                          delta_s5_loss=s['s5_loss']-base['s5_loss'], s5_loss=s['s5_loss'],
                          s5_mean_abs_probability_change=s['s5_mean_abs_probability_change']))
    return dict(participant=participant, source_index=index, n_train=len(fold.fit_trials.left),
                n_validation=len(fold.validation_trials.left), saved_theta=theta_hat.tolist(),
                saved_boundary=saved['optimization']['boundary_flags'], baseline=base, checks=checks,
                best_found_deviance=best_found, violations=violations, points=points, fixed_nuisance=fixed)


# ----------------------------------------------------------------------------- summaries

def grid_points(row, grid=GRID):
    """Profile points at the predetermined grid (anchor excluded unless it lies on the grid)."""
    want = set(map(float, grid))
    return [p for p in row['points'] if p['alpha_ck'] in want]


def usable(rows, grid=GRID):
    """Participants whose every grid point optimized successfully; others are reported, never used."""
    good, bad = [], []
    for r in rows:
        pts = grid_points(r, grid)
        (good if len(pts) == len(grid) and all(p['status'] == 'ok' for p in pts) else bad).append(r)
    return good, [r['participant'] for r in bad]


def near_optimal(row, tol, grid=GRID):
    pts = grid_points(row, grid)
    inside = np.array([p['delta_nll'] <= tol for p in pts])
    alphas = np.array([p['alpha_ck'] for p in pts])
    idx = np.flatnonzero(inside)
    if not len(idx):
        return dict(count=0, lower=np.nan, upper=np.nan, contiguous=True, flat=False, includes_zero=False,
                    includes_one=False, s5_loss_range=np.nan, s5_delta_values=[])
    return dict(count=int(len(idx)), lower=float(alphas[idx[0]]), upper=float(alphas[idx[-1]]),
                contiguous=bool(np.all(np.diff(idx) == 1)), flat=bool(inside.all()),
                includes_zero=bool(inside[0]), includes_one=bool(inside[-1]),
                s5_loss_range=float(np.ptp([pts[i]['s5_loss'] for i in idx])),
                s5_delta_values=[pts[i]['delta_s5_loss'] for i in idx])


def _hl(alpha):
    h = half_life(alpha)
    return None if h['infinite'] else h['updates']


def local_minima(row, grid=GRID, tol=1e-6):
    """Strict local minima of the grid profile (ΔNLL values); >1 indicates a multimodal profile."""
    y = np.array([p['delta_nll'] for p in grid_points(row, grid)])
    return [float(y[i]) for i in range(len(y))
            if (i == 0 or y[i] < y[i-1]-tol) and (i == len(y)-1 or y[i] < y[i+1]-tol)]


def identifiability(rows, grid=GRID):
    minima = [sorted(local_minima(r, grid)) for r in rows]
    second = np.array([m[1] if len(m) > 1 else np.inf for m in minima])
    out = dict(multimodal=dict(participants_with_multiple_local_minima=int(sum(len(m) > 1 for m in minima)),
                               second_minimum_within_1_nat=int(np.sum(second <= 1.)),
                               second_minimum_within_2_nats=int(np.sum(second <= 2.))))
    for tol in NEAR_OPTIMAL_NATS:
        sets = [near_optimal(r, tol, grid) for r in rows]
        lo = np.array([s['lower'] for s in sets], float); hi = np.array([s['upper'] for s in sets], float)
        ratio = np.where(lo > 0, hi/np.where(lo > 0, lo, 1), np.inf)
        lo_ok = (lo > 0) & (hi < 1)
        out[str(tol)] = dict(
            lower=distribution(lo[np.isfinite(lo)]), upper=distribution(hi[np.isfinite(hi)]),
            half_life_at_upper_alpha=distribution(np.log(.5)/np.log1p(-hi[lo_ok])),
            half_life_at_lower_alpha=distribution(np.log(.5)/np.log1p(-lo[lo_ok])),
            tolerance_nats=tol, participants=len(rows),
            grid_points_inside=distribution(np.array([s['count'] for s in sets], float)),
            flat_all_grid=sum(s['flat'] for s in sets), includes_zero=sum(s['includes_zero'] for s in sets),
            includes_one=sum(s['includes_one'] for s in sets), non_contiguous=sum(not s['contiguous'] for s in sets),
            single_grid_point=sum(s['count'] == 1 for s in sets),
            no_grid_point_inside=sum(s['count'] == 0 for s in sets),
            upper_over_lower_ratio=dict(finite=distribution(ratio[np.isfinite(ratio)]), lower_is_zero=int(np.sum(lo == 0)),
                                        at_least_3x=int(np.sum(ratio >= 3)), at_least_10x=int(np.sum(ratio >= 10))),
            half_life_upper_over_lower=distribution(np.array([_hl(a)/_hl(b) for a, b in zip(lo, hi)
                                                              if 0 < a and b < 1 and _hl(b)], float)),
            s5_loss_range_within_set=distribution(np.array([s['s5_loss_range'] for s in sets if s['count']])),
            s5_range_over_001=int(sum(s['s5_loss_range'] > .01 for s in sets)),
            s5_range_over_0005=int(sum(s['s5_loss_range'] > .005 for s in sets)))
    return out


def _spearman(x, y):
    if np.ptp(y) < 1e-12:
        return 0.
    return float(stats.spearmanr(x, y).statistic)


def compensation(rows, grid=GRID):
    """Within-participant rank association between fixed alpha and each re-optimized nuisance parameter."""
    out = {}
    for name in NUISANCE:
        rho = np.array([_spearman(grid, [p['nuisance'][name] for p in grid_points(r, grid)]) for r in rows])
        out[name] = dict(spearman=distribution(rho), positive=int(np.sum(rho > .5)), negative=int(np.sum(rho < -.5)))
    for key in ('train_components', 's5_components'):
        for part in ('median_abs_kernel', 'median_abs_reward'):
            rho = np.array([_spearman(grid, [p[key][part] for p in grid_points(r, grid)]) for r in rows])
            out[f'{key}.{part}'] = dict(spearman=distribution(rho), positive=int(np.sum(rho > .5)), negative=int(np.sum(rho < -.5)))
    return out


def by_alpha(rows, grid=GRID, *, seed=20261015, draws=10000):
    table = []
    for k, a in enumerate(grid):
        pts = [grid_points(r, grid)[k] for r in rows]
        fixed = [next(f for f in r['fixed_nuisance'] if f['alpha_ck'] == a) for r in rows]
        d_s5 = np.array([p['delta_s5_loss'] for p in pts])
        n_val = np.array([r['n_validation'] for r in rows], float)
        entry = dict(alpha_ck=float(a), half_life=half_life(a),
                     delta_nll=distribution(np.array([p['delta_nll'] for p in pts])),
                     delta_train_loss_per_choice=distribution(np.array([p['delta_train_loss_per_choice'] for p in pts])),
                     within_1_nat=int(sum(p['delta_nll'] <= 1 for p in pts)),
                     delta_s5_loss=effect_summary(d_s5, seed=seed, draws=draws),
                     delta_train_session_loss={k: effect_summary(np.array([p['delta_train_session_loss'][k] for p in pts]), seed=seed, draws=draws)
                                               for k in pts[0]['delta_train_session_loss']},
                     delta_s5_loss_trial_weighted=float((d_s5*n_val).sum()/n_val.sum()),
                     s5_mean_abs_probability_change=distribution(np.array([p['s5_mean_abs_probability_change'] for p in pts])),
                     train_mean_abs_probability_change=distribution(np.array([p['train_mean_abs_probability_change'] for p in pts])),
                     nuisance={n: distribution(np.array([p['nuisance'][n] for p in pts])) for n in NUISANCE},
                     nuisance_at_bound={n: int(sum(p['boundary'][n] for p in pts)) for n in NUISANCE},
                     components={f'{k2}.{c}': distribution(np.array([p[k2][c] for p in pts]))
                                 for k2 in ('train_components', 's5_components')
                                 for c in ('median_abs_kernel', 'median_abs_reward', 'kernel_flips_fraction')},
                     fixed_nuisance=dict(delta_nll=distribution(np.array([f['delta_nll'] for f in fixed])),
                                         delta_s5_loss=distribution(np.array([f['delta_s5_loss'] for f in fixed])),
                                         s5_mean_abs_probability_change=distribution(np.array([f['s5_mean_abs_probability_change'] for f in fixed]))))
        table.append(entry)
    return table


def training_validation_link(rows, grid=GRID, *, seed=20261015, draws=10000):
    """Within the 1-nat near-optimal set: does S5 loss still vary?"""
    rho = []
    for r in rows:
        pts = grid_points(r, grid)
        rho.append(_spearman([p['delta_nll'] for p in pts], [p['delta_s5_loss'] for p in pts]))
    sets = [near_optimal(r, PRIMARY_TOL, grid) for r in rows]
    worst = np.array([max(s['s5_delta_values']) for s in sets if s['count']])
    return dict(spearman_train_vs_s5=distribution(np.array(rho)),
                worst_s5_delta_within_1nat=effect_summary(worst, seed=seed, draws=draws),
                worst_over_001=int(np.sum(worst > .01)), worst_over_0005=int(np.sum(worst > .005)))


def optimization_summary(rows):
    pts = [p for r in rows for p in r['points']]
    ok = [p for p in pts if p['status'] == 'ok']
    rand = np.array([p['random_reaching_best']/max(p['random_successful'], 1) for p in ok])
    return dict(points=len(pts), failed_points=len(pts)-len(ok),
                failed_starts=sum(len(p['failed_starts']) for p in pts),
                starts=sum(p['starts'] for p in pts),
                winning_start={str(k): int(v) for k, v in zip(*np.unique(
                    ['random' if p['best_label'].startswith('random') else p['best_label'] for p in ok], return_counts=True))},
                random_fraction_reaching_best=distribution(rand),
                points_where_no_random_start_reached_best=int(sum(p['random_reaching_best'] == 0 for p in ok)),
                max_projected_gradient=float(max(p['projected_gradient'] for p in ok)),
                projected_gradient_over_1e_3=int(sum(p['projected_gradient'] > 1e-3 for p in ok)),
                violations=[dict(participant=r['participant'], **v) for r in rows for v in r['violations']],
                anchor_vs_saved=distribution(np.array([r['checks']['anchor_vs_saved_deviance'] for r in rows
                                                       if r['checks']['anchor_vs_saved_deviance'] is not None])))


def summarize(rows, grid=GRID, *, seed=20261015, draws=10000):
    good, excluded = usable(rows, grid)
    for r in good:  # failed points can never reach the analysis
        assert all(p['status'] == 'ok' for p in grid_points(r, grid))
    if len(good) < 2:
        raise ValueError('Too few participants with complete successful profiles.')
    saved_alpha = np.array([r['saved_theta'][0] for r in good])
    return dict(participants=len(rows), analysed=len(good), excluded_incomplete_profiles=excluded,
                saved_alpha=distribution(saved_alpha),
                optimization=optimization_summary(rows),
                identifiability=identifiability(good, grid),
                compensation=compensation(good, grid),
                by_alpha=by_alpha(good, grid, seed=seed, draws=draws),
                training_validation=training_validation_link(good, grid, seed=seed, draws=draws),
                baseline_s5_loss=float(np.mean([r['baseline']['s5_loss'] for r in good])))


def representatives(rows, k=6):
    """Reproducible: participants nearest to evenly spaced quantiles of the saved alpha."""
    order = sorted(rows, key=lambda r: (r['saved_theta'][0], r['participant']))
    idx = np.unique(np.round(np.linspace(0, len(order)-1, k)).astype(int))
    return [order[i]['participant'] for i in idx]
