"""Experiment 2: evaluation-phase choice-kernel ablations of the authors' RL+CK.

Fixed, previously saved parameters only; nothing is refitted. The authors'
implementation (``authors_models``) is imported, never modified. C0 is scored by
that unchanged ``evaluate``; C1-C3 use :func:`kernel_replay`, a line-for-line copy
of the same recurrence whose ONLY differences are the three flagged interventions.
With no intervention it reproduces ``authors_models.evaluate`` exactly (tested).

Conditions, applied from the first evaluation trial onward:

* C0  original replay (carried-in kernel state, kernel updates on).
* C1  kernel frozen at its carried-in state (no kernel updates during evaluation).
* C2  kernel reset to a common neutral value, then normal updates.
* C3  kernel term weight beta_CK set to 0 (Q and its updates untouched).

Because only K_left - K_right enters the logit, "reset + frozen" gives a constant
kernel with zero difference on every trial, i.e. exactly C3. The four conditions
therefore form a 2 x 2 factorial: carried-in state {kept, removed} x updating {on, off}.

Q never depends on choices (both displayed options receive feedback), so Q
trajectories are identical across conditions; this is asserted, not assumed.

Sign convention: ``effect = loss(condition) - loss(C0)`` in nats per choice.
Positive = removing that kernel ingredient HURTS prediction (the ingredient helped).
"""
import numpy as np
from scipy import stats
from scipy.special import expit

from .authors_evaluation import distribution
from .authors_models import Evaluation, evaluate as authors_evaluate, unpack
from .human_history_evaluation import EQUAL_PAIRS, TIE_TOLERANCE

MODEL = 19
CONDITIONS = ('C0', 'C1', 'C2', 'C3')
CONDITION_LABELS = {
    'C0': 'C0 original', 'C1': 'C1 kernel frozen', 'C2': 'C2 kernel reset (0.5)', 'C3': 'C3 kernel weight 0'}
NEUTRAL_K = .5
RESET_SENSITIVITY = (0., 1.)
PHASES = ('early', 'middle', 'late')
TRAINED_PAIRS = ((1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 8))
HIGH_FREQUENCY_TRAINING_PAIRS = ((1, 3), (2, 4), (5, 7), (6, 8))  # ~3x more training presentations (verified in script)
EQUAL = tuple((a, b) for a, b, _ in EQUAL_PAIRS)
FLIP_FLOOR = 1e-9


# ------------------------------------------------------------------- replay
def kernel_replay(trials, theta, q0, h0, *, freeze=False, reset_to=None, kernel_weight=None):
    """Authors model-19 recurrence with optional evaluation-phase kernel interventions.

    ``theta`` is the saved free-parameter vector (alpha_CK, beta_CK, alpha_RL, beta_RL).
    ``reset_to``: replace the carried-in kernel by this common value before trial 1.
    ``freeze``: skip every kernel update. ``kernel_weight``: override beta_CK in the
    logit only. Q initialization and updates are never altered.
    """
    ac, bc, ar, br, bias = unpack(MODEL, theta)
    if kernel_weight is not None:
        if not np.isfinite(kernel_weight):
            raise ValueError('Kernel weight must be finite.')
        bc = float(kernel_weight)
    q0, h0 = np.asarray(q0, float), np.asarray(h0, float)
    if reset_to is not None:
        if not np.isfinite(reset_to):
            raise ValueError('Reset value must be finite.')
        h0 = np.full(8, float(reset_to))
    if q0.shape != (8,) or h0.shape != (8,) or not np.isfinite(np.r_[q0, h0]).all():
        raise ValueError('Need eight finite initial Q and K values.')
    n = len(trials.left)
    q, h = np.empty((n+1, 8)), np.empty((n+1, 8))
    q[0], h[0] = q0, h0
    prob, loss, left = np.empty(n), np.empty(n), trials.left.copy()
    for t, (l, r) in enumerate(trials.pairs):
        z = bc*(h[t, l]-h[t, r]) + br*(q[t, l]-q[t, r]) + 2*bias-5
        prob[t] = expit(z)
        loss[t] = np.logaddexp(0., -z if left[t] else z)
        q[t+1], h[t+1] = q[t], h[t]
        if not freeze:
            h[t+1, l] += ac*(float(left[t])-h[t, l])
            h[t+1, r] += ac*(float(not left[t])-h[t, r])
        if trials.rewards is not None:
            q[t+1, [l, r]] += ar*(trials.rewards[t]-q[t, [l, r]])
    return Evaluation(prob, loss, q, h, left)


def run_conditions(trials, theta, q0, h0, *, reset_values=RESET_SENSITIVITY):
    """All four conditions (+ secondary reset values) on one evaluation segment."""
    out = {'C0': authors_evaluate(trials, MODEL, theta, q0, h0, scoring_pi=0.),
           'C1': kernel_replay(trials, theta, q0, h0, freeze=True),
           'C2': kernel_replay(trials, theta, q0, h0, reset_to=NEUTRAL_K),
           'C3': kernel_replay(trials, theta, q0, h0, kernel_weight=0.)}
    for c in reset_values:
        out[f'C2_reset_{c:g}'] = kernel_replay(trials, theta, q0, h0, reset_to=c)
    for name, ev in out.items():
        np.testing.assert_array_equal(ev.q, out['C0'].q, err_msg=f'{name}: Q trajectory changed.')
        np.testing.assert_array_equal(ev.left, trials.left)
    return out


def decompose(trials, ev, theta, condition):
    """Weighted reward and kernel logits actually used on each trial."""
    _, bc, _, br, bias = unpack(MODEL, theta)
    if condition == 'C3':
        bc = 0.
    t = np.arange(len(trials.left)); l, r = trials.pairs.T
    reward = br*(ev.q[t, l]-ev.q[t, r])
    kernel = bc*(ev.h[t, l]-ev.h[t, r])
    np.testing.assert_allclose(expit(reward+kernel+2*bias-5), ev.probability_left, rtol=0, atol=1e-12)
    return reward, kernel


# ---------------------------------------------------------------- labelling
def terciles(n):
    labels = np.empty(n, dtype=int)
    for j, idx in enumerate(np.array_split(np.arange(n), 3)):
        labels[idx] = j
    return labels


def pair_classes(pairs_zero_based, scheme):
    """'validation': high/low training frequency; 'test': trained/novel_unequal/novel_equal."""
    p = np.sort(np.asarray(pairs_zero_based)+1, axis=1)
    member = lambda group: np.array([tuple(x) in group for x in map(tuple, p)])
    if scheme == 'validation':
        if not member(TRAINED_PAIRS).all():
            raise ValueError('Validation contains an untrained pair; inspect source.')
        return np.where(member(HIGH_FREQUENCY_TRAINING_PAIRS), 'high_freq_pair', 'low_freq_pair')
    if scheme == 'test':
        return np.where(member(TRAINED_PAIRS), 'trained_pair', np.where(member(EQUAL), 'novel_equal', 'novel_unequal'))
    raise ValueError('Unknown pair scheme.')


# ------------------------------------------------------ participant summary
def _mean(x, mask):
    return float(x[mask].mean()) if mask.any() else None


def participant_summary(trials, theta, outputs, scheme):
    """Per-participant losses and logit contributions by condition, phase, pair class."""
    n = len(trials.left)
    phase = terciles(n)
    klass = pair_classes(trials.pairs, scheme)
    choice_sign = np.where(trials.left, 1., -1.)
    classes = sorted(set(klass))
    res = dict(n=n, class_counts={c: int((klass == c).sum()) for c in classes}, conditions={})
    for name, ev in outputs.items():
        loss = ev.log_loss
        rec = dict(nll=float(loss.sum()), loss=float(loss.mean()),
                   phase_loss={p: _mean(loss, phase == j) for j, p in enumerate(PHASES)},
                   class_loss={c: _mean(loss, klass == c) for c in classes},
                   class_nll={c: float(loss[klass == c].sum()) for c in classes})
        base = name.split('_')[0]
        reward, kernel = decompose(trials, ev, theta, base)
        ar, ak = abs(reward), abs(kernel)
        total = reward+kernel
        flips = (ar > FLIP_FLOOR) & (np.sign(total) != np.sign(reward))
        def contrib(mask):
            if not mask.any():
                return None
            return dict(median_abs_reward=float(np.median(ar[mask])), median_abs_kernel=float(np.median(ak[mask])),
                        mean_abs_reward=float(ar[mask].mean()), mean_abs_kernel=float(ak[mask].mean()),
                        kernel_larger_fraction=float(np.mean(ak[mask] > ar[mask])),
                        kernel_flips_reward_direction_fraction=float(np.mean(flips[mask])),
                        reward_points_to_choice_fraction=float(np.mean((reward*choice_sign)[mask] > 0)),
                        kernel_points_to_choice_fraction=float(np.mean((kernel*choice_sign)[mask] > 0)),
                        kernel_nonzero_fraction=float(np.mean(ak[mask] > 1e-12)))
        rec['contributions'] = dict(all=contrib(np.ones(n, bool)),
                                    phases={p: contrib(phase == j) for j, p in enumerate(PHASES)},
                                    classes={c: contrib(klass == c) for c in classes})
        res['conditions'][name] = rec
    return res


# -------------------------------------------------------------- cohort stats
def paired_bootstrap(columns, statistic, *, seed=20261015, draws=10000):
    """Resample whole participants (same indices for every column)."""
    cols = [np.asarray(c, float) for c in columns]
    n = len(cols[0])
    if any(len(c) != n for c in cols) or n < 2 or any(not np.isfinite(c).all() for c in cols):
        raise ValueError('Paired bootstrap needs equal-length finite participant columns.')
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for offset in range(0, draws, 500):
        k = min(500, draws-offset)
        idx = rng.integers(0, n, size=(k, n))
        values[offset:offset+k] = statistic(*[c[idx] for c in cols])
    return dict(ci95=np.quantile(values, [.025, .975]).tolist(), standard_error=float(values.std(ddof=1)),
                seed=seed, draws=draws, unit='participant')


def effect_summary(diff, *, seed=20261015, draws=10000):
    d = np.asarray(diff, float)
    boot = paired_bootstrap([d], lambda x: x.mean(axis=1), seed=seed, draws=draws)
    return dict(distribution=distribution(d), mean=float(d.mean()),
                hurts=int((d > TIE_TOLERANCE).sum()), helps=int((d < -TIE_TOLERANCE).sum()),
                tied=int((abs(d) <= TIE_TOLERANCE).sum()), ci95=boot['ci95'], standard_error=boot['standard_error'],
                sign_test_p=float(stats.binomtest(int((d > TIE_TOLERANCE).sum()), int((abs(d) > TIE_TOLERANCE).sum())).pvalue)
                if (abs(d) > TIE_TOLERANCE).any() else None)


def factorial(rows, key='loss', *, seed=20261015, draws=10000):
    """2x2 decomposition of the kernel's total value, participant-weighted.

    L00=C0 (state kept, updating on), L01=C1 (kept, off), L10=C2 (removed, on), L11=C3.
    """
    L = {c: np.array([r['conditions'][c][key] for r in rows]) for c in CONDITIONS}
    total = L['C3']-L['C0']
    pieces = dict(
        total_kernel_value=total,
        updating_given_carried_state=L['C1']-L['C0'],
        carried_state_given_updating=L['C2']-L['C0'],
        carried_state_without_updating=L['C3']-L['C1'],
        updating_without_carried_state=L['C3']-L['C2'],
        interaction=(L['C3']-L['C1'])-(L['C2']-L['C0']),
    )
    out = {k: effect_summary(v, seed=seed, draws=draws) for k, v in pieces.items()}
    # Shares of the kernel's total value: ratio of participant means, bootstrapped jointly.
    for name, num in (('updating_given_carried_state', pieces['updating_given_carried_state']),
                      ('carried_state_given_updating', pieces['carried_state_given_updating'])):
        share = float(num.mean()/total.mean()) if total.mean() else None
        boot = paired_bootstrap([num, total], lambda a, b: a.mean(axis=1)/b.mean(axis=1), seed=seed, draws=draws)
        out[name]['share_of_total'] = share
        out[name]['share_ci95'] = boot['ci95']
    out['main_effect_carried_state'] = effect_summary(((L['C2']-L['C0'])+(L['C3']-L['C1']))/2, seed=seed, draws=draws)
    out['main_effect_updating'] = effect_summary(((L['C1']-L['C0'])+(L['C3']-L['C2']))/2, seed=seed, draws=draws)
    return out


def trial_weighted(rows, condition, reference='C0'):
    nll = sum(r['conditions'][condition]['nll']-r['conditions'][reference]['nll'] for r in rows)
    n = sum(r['n'] for r in rows)
    return dict(total_nll_difference=nll, choices=n, per_choice=nll/n)


def stratified_effects(rows, field, levels, *, seed=20261015, draws=10000):
    """Condition-minus-C0 losses within phase or pair class (participants with data)."""
    out = {}
    for level in levels:
        keep = [r for r in rows if r['conditions']['C0'][field].get(level) is not None]
        out[level] = {c: effect_summary([r['conditions'][c][field][level]-r['conditions']['C0'][field][level] for r in keep],
                                        seed=seed, draws=draws) | dict(participants=len(keep))
                      for c in ('C1', 'C2', 'C3')}
    return out


def class_trial_weighted(rows, classes):
    out = {}
    for c in classes:
        n = sum(r['class_counts'].get(c, 0) for r in rows)
        out[c] = dict(choices=n, **{k: sum(r['conditions'][k]['class_nll'].get(c, 0.)-r['conditions']['C0']['class_nll'].get(c, 0.)
                                         for r in rows)/n if n else None for k in ('C1', 'C2', 'C3')},
                      total_nll={k: sum(r['conditions'][k]['class_nll'].get(c, 0.)-r['conditions']['C0']['class_nll'].get(c, 0.)
                                        for r in rows) for k in ('C1', 'C2', 'C3')})
    return out


def contribution_summary(rows, condition='C0'):
    def collect(getter):
        vals = [getter(r) for r in rows]
        vals = [v for v in vals if v is not None]
        return distribution(vals)
    keys = ('median_abs_reward', 'median_abs_kernel', 'kernel_larger_fraction', 'kernel_flips_reward_direction_fraction',
            'reward_points_to_choice_fraction', 'kernel_points_to_choice_fraction')
    c = lambda r: r['conditions'][condition]['contributions']
    out = dict(all={k: collect(lambda r, k=k: c(r)['all'][k]) for k in keys})
    out['phases'] = {p: {k: collect(lambda r, k=k, p=p: (c(r)['phases'][p] or {}).get(k)) for k in keys} for p in PHASES}
    classes = sorted({k for r in rows for k in c(r)['classes']})
    out['classes'] = {cl: {k: collect(lambda r, k=k, cl=cl: (c(r)['classes'].get(cl) or {}).get(k)) for k in keys} for cl in classes}
    return out


def consistency(validation_rows, test_rows):
    v = {r['participant']: r for r in validation_rows}
    t = {r['participant']: r for r in test_rows}
    common = sorted(set(v) & set(t))
    out = dict(participants=len(common))
    for name, (a, b) in dict(total_kernel_value=('C3', 'C0'), updating_given_carried_state=('C1', 'C0'),
                             carried_state_given_updating=('C2', 'C0')).items():
        x = [v[p]['conditions'][a]['loss']-v[p]['conditions'][b]['loss'] for p in common]
        y = [t[p]['conditions'][a]['loss']-t[p]['conditions'][b]['loss'] for p in common]
        rho = stats.spearmanr(x, y)
        out[name] = dict(spearman=float(rho.statistic), p=float(rho.pvalue))
    return out


def summarize(rows, scheme, *, seed=20261015, draws=10000):
    classes = sorted({c for r in rows for c in r['class_counts']})
    extra = sorted({c for r in rows for c in r['conditions']} - set(CONDITIONS))
    return dict(
        participants=len(rows), choices=sum(r['n'] for r in rows),
        losses={c: distribution([r['conditions'][c]['loss'] for r in rows]) for c in list(CONDITIONS)+extra},
        trial_weighted={c: trial_weighted(rows, c) for c in list(CONDITIONS[1:])+extra},
        trial_weighted_loss={c: sum(r['conditions'][c]['nll'] for r in rows)/sum(r['n'] for r in rows) for c in list(CONDITIONS)+extra},
        factorial=factorial(rows, seed=seed, draws=draws),
        reset_sensitivity={c: effect_summary([r['conditions'][c]['loss']-r['conditions']['C0']['loss'] for r in rows], seed=seed, draws=draws)
                           for c in extra},
        phases=stratified_effects(rows, 'phase_loss', PHASES, seed=seed, draws=draws),
        classes=stratified_effects(rows, 'class_loss', classes, seed=seed, draws=draws),
        class_trial_weighted=class_trial_weighted(rows, classes),
        contributions={c: contribution_summary(rows, c) for c in ('C0', 'C1', 'C2')},
        scheme=scheme, class_names=classes)
