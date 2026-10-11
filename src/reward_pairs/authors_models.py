"""Authors' RL/CK benchmark, separate from our cumulative-history models."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter
from scipy.special import expit

VARIANTS = ('paper_intended', 'released_code_intended', 'released_code_literal_clean_r')
PARAMETERS = ('authors_alpha_ck', 'authors_beta_ck', 'authors_alpha_rl', 'authors_beta_rl', 'bias')
FREE = {1: (4,), 3: (2, 3), 11: (0, 1), 19: (0, 1, 2, 3), 20: (0, 1, 2)}
LIMITS = np.array([[0., 1.], [0., 10.], [0., 1.], [0., 10.], [0., 5.]])


def unpack(model: int, theta) -> np.ndarray:
    if model not in FREE:
        raise ValueError('Supported authors models: 1, 3, 11, 19, 20.')
    theta = np.asarray(theta, dtype=float)
    bounds = LIMITS[list(FREE[model])]
    if theta.shape != (len(bounds),) or not np.isfinite(theta).all() or np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1]):
        raise ValueError('Invalid authors parameter shape, finiteness, or bounds.')
    p = np.array([0., 0., 0., 0., 2.5])
    p[list(FREE[model])] = theta
    if model == 20:
        p[3] = 10 - p[1]
    return p


def initial_values(ratings, pretest_proportions, variant):
    r = np.asarray(ratings, dtype=float)
    if r.shape != (8,) or not np.isfinite(r).all() or np.any((r < 0) | (r > 100)):
        raise ValueError('Need eight finite pre-training ratings in [0,100].')
    if variant == 'paper_intended':
        return (r-r.min())/11.15, np.zeros(8)
    if variant not in VARIANTS:
        raise ValueError('Unknown authors variant.')
    p = np.asarray(pretest_proportions, dtype=float)
    if p.shape != (8,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Released initialization needs eight pretest proportions in [0,1].')
    value = ((r-r.min())/20 + (p-p.min())*5)/2
    return value.copy(), value.copy()


@dataclass
class AuthorsTrials:
    """One participant/phase or one synthetic episode; arrays are in row order.

    Test rewards are deliberately None and cannot update Q. Chronology validation
    occurs in the source adapter; fitting never combines participant histories.
    """
    pairs: np.ndarray  # zero-based stimulus indices
    left: np.ndarray  # observed choice
    rewards: np.ndarray | None

    def __post_init__(self):
        self.pairs = np.array(self.pairs, copy=True)
        self.left = np.array(self.left, copy=True)
        n = len(self.pairs)
        if self.pairs.shape != (n, 2) or n == 0 or not np.isin(self.pairs, range(8)).all():
            raise ValueError('Nonempty pairs must contain integer indices 0..7.')
        if np.any(self.pairs[:, 0] == self.pairs[:, 1]):
            raise ValueError('Identical displayed stimuli unsupported; inspect source, do not drop rows.')
        if self.left.shape != (n,) or self.left.dtype != bool:
            raise ValueError('Observed actions must be a boolean left-choice vector.')
        self.pairs = self.pairs.astype(int)
        if self.rewards is not None:
            self.rewards = np.array(self.rewards, dtype=float, copy=True)
            if self.rewards.shape != (n, 2) or not np.isfinite(self.rewards).all():
                raise ValueError('Training requires two finite feedback values per row.')


@dataclass
class Evaluation:
    probability_left: np.ndarray
    log_loss: np.ndarray
    q: np.ndarray  # includes initial and every post-choice state
    h: np.ndarray
    left: np.ndarray

    @property
    def deviance(self):
        return float(2*self.log_loss.sum())


def evaluate(trials, model, theta, q0, h0, *, scoring_pi=0., simulate_seed=None):
    """Transparent scalar replay, also used as an independent fast-objective check.

    Simulation uses only pairs/rewards, never supplied observed actions. Each call
    resets positional previous response, as the original train/test functions do.
    """
    ac, bc, ar, br, bias = unpack(model, theta)
    q0, h0 = np.asarray(q0, float), np.asarray(h0, float)
    if q0.shape != (8,) or h0.shape != (8,) or not np.isfinite(np.r_[q0, h0, scoring_pi]).all():
        raise ValueError('Initial states and pi must be finite, with eight stimulus values.')
    n = len(trials.left)
    q, h = np.empty((n+1, 8)), np.empty((n+1, 8))
    q[0], h[0] = q0, h0
    prob, loss, left = np.empty(n), np.empty(n), trials.left.copy()
    draws = None if simulate_seed is None else np.random.default_rng(simulate_seed).random(n)
    previous = 0.
    for t, (l, r) in enumerate(trials.pairs):
        z = bc*(h[t,l]-h[t,r]) + br*(q[t,l]-q[t,r]) + 2*bias-5 + 2*scoring_pi*previous
        prob[t] = expit(z)
        if draws is not None:
            left[t] = draws[t] < prob[t]
        loss[t] = np.logaddexp(0., -z if left[t] else z)
        q[t+1], h[t+1] = q[t], h[t]
        h[t+1,l] += ac*(float(left[t])-h[t,l])
        h[t+1,r] += ac*(float(not left[t])-h[t,r])
        if trials.rewards is not None:
            q[t+1,[l,r]] += ar*(trials.rewards[t]-q[t,[l,r]])
        previous = 1. if left[t] else -1.
    return Evaluation(prob, loss, q, h, left)


class TrainingObjective:
    """Same recurrence in encounter time, evaluated by exact linear filters.

    Each stimulus only changes when displayed. lfilter runs that recurrence in C;
    its derivative is d_next=(1-alpha)*d + target-value_before. No approximation,
    altered likelihood, hidden updates or regularization. Supports independent
    synthetic episodes by summing likelihoods, with fresh initial states each.
    """
    def __init__(self, episodes, model, q0, h0):
        if model not in FREE:
            raise ValueError('Unsupported authors model.')
        unpack(model, np.mean(LIMITS[list(FREE[model])], axis=1))
        self.model, self.q0, self.h0 = model, np.asarray(q0, float), np.asarray(h0, float)
        if self.q0.shape!=(8,) or self.h0.shape!=(8,) or not np.isfinite(np.r_[self.q0,self.h0]).all():
            raise ValueError('Need eight finite initial Q and H values.')
        self.episodes = []
        for trials in episodes:
            if trials.rewards is None:
                raise ValueError('Training objective requires training rewards, never test-only rows.')
            encounters = []
            for stimulus in range(8):
                t, side = np.nonzero(trials.pairs == stimulus)
                target = np.where(side == 0, trials.left[t], ~trials.left[t]).astype(float)
                encounters.append((t, np.where(side == 0, 1., -1.), target, trials.rewards[t, side]))
            self.episodes.append((trials, encounters))
        if not self.episodes:
            raise ValueError('Need at least one independent training episode.')

    @staticmethod
    def trace(target, alpha, initial):
        after = lfilter([alpha], [1., alpha-1], target, zi=[(1-alpha)*initial])[0]
        before = np.r_[initial, after[:-1]]
        derivative_after = lfilter([1.], [1., alpha-1], target-before)
        return before, np.r_[0., derivative_after[:-1]]

    def __call__(self, theta):
        ac, bc, ar, br, bias = unpack(self.model, theta)
        dev, grad = 0., np.zeros(5)
        for trials, encounters in self.episodes:
            n = len(trials.left)
            dh, dq, da_h, da_q = np.zeros((4, n))
            for i, (t, sign, target_h, target_q) in enumerate(encounters):
                if not len(t):
                    continue
                h, ha = self.trace(target_h, ac, self.h0[i])
                q, qa = self.trace(target_q, ar, self.q0[i])
                dh[t] += sign*h; da_h[t] += sign*ha
                dq[t] += sign*q; da_q[t] += sign*qa
            z = bc*dh + br*dq + 2*bias-5
            signed = np.where(trials.left, 1., -1.)
            dev += float(2*np.logaddexp(0., -signed*z).sum())
            residual = -2*signed*expit(-signed*z)
            grad += np.array([residual@(bc*da_h), residual@dh, residual@(br*da_q), residual@dq, 2*residual.sum()])
        if self.model == 20:
            grad[1] -= grad[3]
        return dev, grad[list(FREE[self.model])]
