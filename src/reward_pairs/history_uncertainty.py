"""Profile-likelihood uncertainty for the unchanged myopic history likelihood.

The chi-square cutoff is nominal: repeated-simulation coverage is a separate
evaluation, not assumed by the interval constructor. No synthetic truth enters.
"""

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit
from scipy.stats import chi2, norm

from .history import causal_history
from .history_recovery import history_design, recover_history
from .history_replication import distribution
from .recovery import RecoveryError


def lr_endpoints(lr, center, *, cutoff, step=.25, max_distance=64.):
    """Double distance until a crossing, then Brent root; None means unresolved.

    A search-limit endpoint is never substituted for an interval endpoint.
    'Open' means no crossing found in the searched range, NOT proven infinite.
    """
    if not all(np.isfinite([center, cutoff, step, max_distance])) or min(cutoff, step, max_distance) <= 0:
        raise ValueError("Require finite center and positive cutoff/search settings.")
    center_lr = lr(center)
    if not np.isfinite(center_lr) or abs(center_lr) > 1e-5:
        raise RecoveryError("Profile LR at unrestricted estimate is not zero.")
    endpoints = []
    for direction in (-1, 1):
        distance = min(step, max_distance)
        inner = center
        while True:
            outer = center + direction * distance
            value = lr(outer)
            if not np.isfinite(value) or value < -1e-5:
                raise RecoveryError("Nonfinite/negative profile LR during endpoint search.")
            if value >= cutoff:
                root = brentq(lambda a: lr(a) - cutoff, min(inner, outer), max(inner, outer),
                              xtol=1e-7, rtol=1e-10, maxiter=100)
                endpoints.append(dict(value=float(root), lr=float(lr(root)), status="closed",
                                      searched_to=float(outer)))
                break
            if distance == max_distance:
                endpoints.append(dict(value=None, lr=None, status="open_search_limit",
                                      searched_to=float(outer), lr_at_search_limit=float(value)))
                break
            inner = outer
            distance = min(2 * distance, max_distance)
    return endpoints


class AlphaProfile:
    """Fit once, then reoptimize SEVEN nuisance utilities at every fixed alpha.

    Alternative reference is a numerical-invariance diagnostic, not a model
    change. The original unrestricted estimator still fits with w1=0.
    """

    def __init__(self, choices, *, beta=1., reference=1, max_iterations=1000):
        if reference not in range(1, 9) or max_iterations < 1:
            raise ValueError("Require reference 1..8 and positive iteration limit.")
        self.fit = recover_history(choices, beta=beta)
        frame = causal_history(choices)
        design = history_design(frame, beta)
        # Recover all eight stimulus columns (their row sums are zero), then
        # omit the desired reference. The history contrast is unchanged.
        full = np.column_stack([-design[:, :7].sum(axis=1), design[:, :7]])
        self.free = np.arange(8) != reference - 1
        self.design = np.column_stack([full[:, self.free], design[:, 7]])
        self.signed = (2 * frame.synthetic_action.eq("left").to_numpy(dtype=float) - 1)[:, None] * self.design
        self.n = len(frame)
        w = self.fit.utilities.to_numpy()
        self.initial = (w - w[reference - 1])[self.free]
        self.max_iterations = max_iterations
        self.cache = {}
        self.attempts = 0
        theta = np.r_[self.initial, self.fit.alpha]
        z = self.design @ theta
        information = self.design.T @ ((expit(z) * expit(-z))[:, None] * self.design)
        self.alpha_se = float(np.sqrt(np.linalg.inv(information)[-1, -1]))

    def conditional(self, alpha):
        if not np.isfinite(alpha):
            raise ValueError("Fixed alpha must be finite.")
        alpha = float(alpha)
        if alpha not in self.cache:
            # A previously optimized neighbor is merely a warm start, not a
            # substitution for conditional optimization or a truth-based start.
            start = (self.cache[min(self.cache, key=lambda a: abs(a-alpha))]["free_utilities"]
                     if self.cache else self.initial)
            offset = self.signed[:, -1] * alpha

            def objective(w):
                margins = self.signed[:, :7] @ w + offset
                return float(np.logaddexp(0, -margins).mean()), -self.signed[:, :7].T @ expit(-margins) / self.n

            self.attempts += 1
            fit = minimize(objective, start, jac=True, method="BFGS",
                           options={"gtol": 1e-8, "maxiter": self.max_iterations})
            if not fit.success or not np.isfinite(fit.fun) or not np.isfinite(fit.x).all():
                raise RecoveryError(f"Conditional profile optimization failed at alpha={alpha}: {fit.message}")
            self.cache[alpha] = dict(alpha=alpha, nll=float(fit.fun*self.n), free_utilities=fit.x.tolist(),
                                     gradient_max_abs=float(np.max(np.abs(fit.jac))), iterations=int(fit.nit))
        return self.cache[alpha]

    def lr(self, alpha):
        value = 2 * (self.conditional(alpha)["nll"] - self.fit.negative_log_likelihood)
        if value < -1e-5:
            raise RecoveryError("Conditional profile exceeds unrestricted maximum; inspect optimization.")
        return max(0., float(value))  # numerical roundoff only, checked above

    def interval(self, *, confidence=.95, max_distance=64.):
        if not 0 < confidence < 1:
            raise ValueError("Confidence must lie strictly between zero and one.")
        cutoff = float(chi2.ppf(confidence, 1))
        lower, upper = lr_endpoints(self.lr, self.fit.alpha, cutoff=cutoff,
                                    step=max(.1, self.alpha_se), max_distance=max_distance)
        z = float(norm.ppf((1+confidence)/2))
        return dict(alpha_hat=self.fit.alpha, utilities=self.fit.utilities.tolist(),
                    nll=self.fit.negative_log_likelihood, n_trials=self.n, cutoff=cutoff,
                    confidence=confidence, lower=lower, upper=upper,
                    width=upper['value']-lower['value'] if lower['value'] is not None and upper['value'] is not None else None,
                    wald_lower=self.fit.alpha-z*self.alpha_se, wald_upper=self.fit.alpha+z*self.alpha_se,
                    alpha_se=self.alpha_se, conditional_fits=self.attempts,
                    evaluations=[self.cache[a] for a in sorted(self.cache)],
                    information_condition=self.fit.information_condition)


def contains(interval, value):
    lo, hi = interval['lower']['value'], interval['upper']['value']
    return (lo is None or lo <= value) and (hi is None or value <= hi)


def interval_outcomes(interval, truth):
    """Evaluation only. Open sides extend outward for descriptive coverage."""
    lo, hi = interval['lower']['value'], interval['upper']['value']
    return dict(covered=bool(contains(interval, truth)), contains_zero=bool(contains(interval, 0)),
                positive_detection=bool(lo is not None and lo > 0),
                negative_detection=bool(hi is not None and hi < 0),
                open=lo is None or hi is None,
                wald_covered=bool(interval['wald_lower'] <= truth <= interval['wald_upper']))


def summarize_uncertainty(rows):
    ok = [r for r in rows if r['status']=='ok']
    outcomes = [interval_outcomes(r['interval'], r['alpha_true']) for r in ok]
    result = dict(attempted=len(rows), successful=len(ok), failed=len(rows)-len(ok))
    for key in ('covered','contains_zero','positive_detection','negative_detection','open','wald_covered'):
        count = sum(o[key] for o in outcomes)
        result[key+'_count'] = count
        result[key+'_rate'] = count/len(ok) if ok else None
    result['zero_exclusion_rate'] = 1-result['contains_zero_rate'] if ok else None
    result['alpha_hat'] = distribution(r['interval']['alpha_hat'] for r in ok)
    result['signed_error'] = distribution(r['interval']['alpha_hat']-r['alpha_true'] for r in ok)
    result['absolute_error'] = distribution(abs(r['interval']['alpha_hat']-r['alpha_true']) for r in ok)
    result['closed_interval_width'] = distribution(r['interval']['width'] for r in ok)
    result['wald_width'] = distribution(r['interval']['wald_upper']-r['interval']['wald_lower'] for r in ok)
    return result


def aggregate_uncertainty(rows, participants):
    conditions, scaffolds = [], []
    for a,n in sorted({(r['alpha_true'],r['rollouts']) for r in rows}):
        selected = [r for r in rows if (r['alpha_true'],r['rollouts'])==(a,n)]
        conditions.append(dict(alpha_true=a, rollouts=n, **summarize_uncertainty(selected)))
        for p in participants:
            scaffolds.append(dict(participant=p,alpha_true=a,rollouts=n,
                **summarize_uncertainty([r for r in selected if r['participant']==p])))
    return dict(conditions=conditions, scaffold_conditions=scaffolds, totals=summarize_uncertainty(rows))
