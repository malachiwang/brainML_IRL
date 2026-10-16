"""Authors' unchanged reward learning plus signed causal cumulative history.

This is myopic sequential choice likelihood, not planning or a Bellman model.
The authors modules remain untouched. Q replay and its encounter-time derivative
are delegated to the validated RL implementation; only the history feature differs.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .authors_models import AuthorsTrials, TrainingObjective, evaluate as authors_evaluate
from .history import ChoiceHistory

PARAMETERS = ('alpha_rl', 'beta_rl', 'beta_h')
BOUNDS = np.array([[0., 1.], [0., 10.], [-10., 10.]])
OPTIONS = dict(ftol=1e-12, gtol=1e-7, maxiter=1000, maxls=40)


def validate_parameters(theta):
    p = np.asarray(theta, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or np.any(p < BOUNDS[:, 0]) or np.any(p > BOUNDS[:, 1]):
        raise ValueError('Expected alpha_rl in [0,1], beta_rl in [0,10], beta_h in [-10,10].')
    return p


def copy_history(history=None):
    if history is None:
        return ChoiceHistory()
    c, n = np.asarray(history.chosen), np.asarray(history.presented)
    if c.shape != (8,) or n.shape != (8,) or not np.isfinite(np.r_[c,n]).all():
        raise ValueError('History requires eight finite count pairs.')
    if np.any(c < 0) or np.any(n < c) or np.any(c != np.floor(c)) or np.any(n != np.floor(n)):
        raise ValueError('Counts must be nonnegative integers with chosen <= presented.')
    return ChoiceHistory(presented=n.astype(int).copy(), chosen=c.astype(int).copy())


def history_features(trials: AuthorsTrials, initial_history=None):
    """n+1 states: initial, then each post-response state. No session reset.

    The first n rows are pre-choice features. The final row may seed test only,
    never earlier training features. The caller's initial state is not mutated.
    """
    state = copy_history(initial_history)
    h = np.empty((len(trials.left)+1, 8))
    h[0] = state.centered
    for t, ((left, right), action) in enumerate(zip(trials.pairs, trials.left)):
        state.update(int(left)+1, int(right)+1, 'left' if action else 'right')
        h[t+1] = state.centered
    return h, state


@dataclass
class CumulativeEvaluation:
    probability_left: np.ndarray
    log_loss: np.ndarray
    q: np.ndarray
    h: np.ndarray
    final_history: ChoiceHistory
    left: np.ndarray

    @property
    def deviance(self):
        return float(2*self.log_loss.sum())


def evaluate(trials, theta, q0, *, initial_history=None, simulate_seed=None):
    """Replay or simulate; observed test actions update counts, never parameters.

    Q depends on displayed feedback, not which option was chosen, because both
    displayed Qs update. We reuse authors RL for that entire Q trajectory. Test
    trials have rewards=None and therefore cannot change Q.
    """
    alpha, beta, weight = validate_parameters(theta)
    rl = authors_evaluate(trials, 3, [alpha, beta], q0, np.zeros(8))
    left = trials.left.copy()
    if simulate_seed is None:
        h, state = history_features(trials, initial_history)
    else:
        state = copy_history(initial_history)
        h = np.empty_like(rl.q)
        h[0] = state.centered
        draws = np.random.default_rng(simulate_seed).random(len(left))
        for t, (l, r) in enumerate(trials.pairs):
            z = beta*(rl.q[t,l]-rl.q[t,r]) + weight*(h[t,l]-h[t,r])
            left[t] = draws[t] < expit(z)
            state.update(int(l)+1, int(r)+1, 'left' if left[t] else 'right')
            h[t+1] = state.centered
    t = np.arange(len(left))
    l, r = trials.pairs.T
    z = beta*(rl.q[t,l]-rl.q[t,r]) + weight*(h[t,l]-h[t,r])
    return CumulativeEvaluation(expit(z), np.logaddexp(0., -np.where(left, 1., -1.)*z),
                                rl.q, h, state, left)


class CumulativeObjective:
    """Training-only deviance and analytic gradient; histories are observed past.

    The validated authors TrainingObjective prepares encounters and supplies the
    exact RL recurrence/derivative. Independent episodes are only for small
    synthetic validation; real participants are fitted separately, once each.
    """
    def __init__(self, episodes, q0):
        self.rl = TrainingObjective(episodes, 3, q0, np.zeros(8))
        self.features = []
        for trials, _ in self.rl.episodes:
            h, _ = history_features(trials)
            t = np.arange(len(trials.left))
            self.features.append(h[t, trials.pairs[:,0]]-h[t, trials.pairs[:,1]])

    def __call__(self, theta):
        alpha, beta, weight = validate_parameters(theta)
        dev, gradient = 0., np.zeros(3)
        for (trials, encounters), dh in zip(self.rl.episodes, self.features):
            dq, derivative = np.zeros((2, len(trials.left)))
            for stimulus, (t, sign, _, rewards) in enumerate(encounters):
                if not len(t):
                    continue
                q, da = self.rl.trace(rewards, alpha, self.rl.q0[stimulus])
                dq[t] += sign*q
                derivative[t] += sign*da
            signed = np.where(trials.left, 1., -1.)
            z = beta*dq + weight*dh
            residual = -2*signed*expit(-signed*z)
            dev += float(2*np.logaddexp(0., -signed*z).sum())
            gradient += [residual@(beta*derivative), residual@dq, residual@dh]
        return dev, gradient


class CumulativeFitError(RuntimeError):
    def __init__(self, attempts):
        super().__init__('All cumulative-history optimization starts failed; attempts retained.')
        self.attempts = attempts


def random_starts(participant_index, restarts, seed=20261015):
    """New-model RNG only; prefix-stable across restart counts, no truth/test input."""
    if participant_index < 1 or restarts < 1:
        raise ValueError('Need positive participant index and restart count.')
    rng = np.random.default_rng(np.random.SeedSequence([seed, participant_index]))
    return rng.uniform([.001,.001,-9.999], [.999,9.999,9.999], (restarts,3))


def fit(episodes, q0, *, starts, max_iterations=1000):
    starts = np.asarray(starts, float)
    if starts.ndim != 2 or starts.shape[1] != 3 or not len(starts):
        raise ValueError('Need restart-by-three explicit starts.')
    problem = CumulativeObjective(episodes, q0)
    attempts = []
    best = None
    start_time = perf_counter()
    for theta in starts:
        validate_parameters(theta)
        result = minimize(problem, theta, jac=True, method='L-BFGS-B', bounds=BOUNDS,
                          options=OPTIONS | {'maxiter': max_iterations})
        valid = bool(result.success and np.isfinite(result.fun) and np.isfinite(result.x).all())
        attempts.append(dict(success=valid, initial=theta.tolist(),
                             theta=result.x.tolist() if np.isfinite(result.x).all() else None,
                             deviance=float(result.fun) if np.isfinite(result.fun) else None,
                             message=str(result.message), iterations=int(result.nit), evaluations=int(result.nfev)))
        if valid and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise CumulativeFitError(attempts)
    boundary = {p:dict(lower=bool(abs(v-lo)<1e-6), upper=bool(abs(v-hi)<1e-6))
                for p,v,(lo,hi) in zip(PARAMETERS,best.x,BOUNDS)}
    return dict(parameters=dict(zip(PARAMETERS,map(float,best.x))), optimization_success=True,
                objective_deviance=float(best.fun), restart_count=len(starts),
                restart_deviances=[a['deviance'] for a in attempts], restart_success=[a['success'] for a in attempts],
                failed_starts=sum(not a['success'] for a in attempts),
                unsuccessful_starts=[dict(index=i,**a) for i,a in enumerate(attempts) if not a['success']],
                boundary_flags=boundary, best_start=min((a for a in attempts if a['success']),key=lambda a:a['deviance']),
                fit_seconds=perf_counter()-start_time)


def criteria(deviance, n, k=3):
    if n <= k+1 or not np.isfinite(deviance):
        raise ValueError('Need finite deviance and n > k+1.')
    return dict(n=int(n), k=k, deviance=float(deviance), nll=float(deviance/2),
                mean_log_loss=float(deviance/(2*n)), AIC=float(deviance+2*k),
                AICc=float(deviance+2*k*n/(n-k-1)), BIC=float(deviance+k*np.log(n)))
