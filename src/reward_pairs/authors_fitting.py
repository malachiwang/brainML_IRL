"""Bounded multistart likelihood fitting for the separate authors benchmark."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from .authors_models import FREE, LIMITS, PARAMETERS, TrainingObjective, unpack


class AuthorsFitError(ValueError):
    pass


@dataclass
class AuthorsFit:
    theta: np.ndarray
    objective_deviance: float
    starts: list[dict]
    seconds: float

    def record(self, model):
        p = unpack(model, self.theta)
        bounds = LIMITS[list(FREE[model])]
        return dict(free_parameters={PARAMETERS[i]:float(v) for i,v in zip(FREE[model], self.theta)},
                    parameters=dict(zip(PARAMETERS, map(float, p))),
                    optimizer_objective_deviance=self.objective_deviance,
                    optimization_success=True, restart_count=len(self.starts),
                    failed_starts=sum(not s['success'] for s in self.starts),
                    boundary_flags={PARAMETERS[i]:bool(min(v-lo, hi-v)<1e-6)
                                    for i,v,(lo,hi) in zip(FREE[model],self.theta,bounds)},
                    restart_deviances=[s['deviance'] for s in self.starts],
                    restart_success=[s['success'] for s in self.starts],
                    unsuccessful_starts=[dict(restart_index=i,**s) for i,s in enumerate(self.starts) if not s['success']],
                    best_start=min((s for s in self.starts if s['success']),key=lambda s:s['deviance']),
                    fit_seconds=self.seconds)


def fit_authors(episodes, model, q0, h0, *, starts, max_iterations=1000):
    """No test inputs. Starts are explicit, recorded, and independent of truth."""
    starts = np.asarray(starts, float)
    if starts.ndim != 2 or starts.shape[1] != len(FREE[model]) or len(starts)<1:
        raise ValueError('Need a nonempty restart-by-free-parameter matrix.')
    problem = TrainingObjective(episodes, model, q0, h0)
    t = perf_counter(); attempts = []; best = None
    for start in starts:
        unpack(model, start)
        fit = minimize(problem, start, jac=True, method='L-BFGS-B',
                       bounds=LIMITS[list(FREE[model])],
                       options=dict(ftol=1e-12, gtol=1e-7, maxiter=max_iterations, maxls=40))
        valid = bool(fit.success and np.isfinite(fit.fun) and np.isfinite(fit.x).all())
        attempts.append(dict(initial=start.tolist(), success=valid, message=str(fit.message),
                             deviance=float(fit.fun) if np.isfinite(fit.fun) else None,
                             theta=fit.x.tolist() if np.isfinite(fit.x).all() else None,
                             iterations=int(fit.nit), evaluations=int(fit.nfev)))
        if valid and (best is None or fit.fun < best.fun):
            best = fit
    if best is None:
        raise AuthorsFitError(f'All {len(starts)} authors-model starts failed: {attempts}')
    return AuthorsFit(best.x.copy(), float(best.fun), attempts, perf_counter()-t)


def information_criteria(deviance, model, n):
    k = len(FREE[model])
    if n <= k+1 or not np.isfinite(deviance):
        raise ValueError('Information criteria need finite deviance and n>k+1.')
    return dict(n=int(n), k=k, deviance=float(deviance), nll=float(deviance/2),
                AIC=float(deviance+2*k), AICc=float(deviance+2*k*n/(n-k-1)),
                BIC=float(deviance+k*np.log(n)))
