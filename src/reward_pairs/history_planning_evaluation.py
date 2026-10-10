"""Synthetic planning diagnostics; answer-key parameters are evaluation-only."""

from time import perf_counter

import numpy as np
import pandas as pd

from .evaluation import compare_utilities
from .history import ChoiceHistory
from .history_irl import demonstration_states, predict_planning, recover_planning
from .history_mdp import HistoryMDP, HistoryState, reachable_state_audit
from .history_planning import PlanningBatch, soft_plan
from .history_recovery import predict_history, recover_history
from .recovery import RecoveryError
from .simulation import DEFAULT_UTILITIES

TOY_PAIRS = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (1, 8)]


def fit_record(mdp, choices, *, model, alpha_true, beta, gamma, lookahead, train=False):
    """Truth scores completed fits; it never enters either recovery function."""
    started = perf_counter()
    sessions = [1, 2, 3, 4] if train else None
    try:
        if model == 'planning':
            fit = recover_planning(mdp, choices, beta=beta, gamma=gamma, lookahead=lookahead, fit_sessions=sessions)
        elif model == 'myopic':
            fit = recover_history(choices, beta=beta, fit_sessions=sessions)
        else:
            raise ValueError('Unknown synthetic model.')
        _, metrics = compare_utilities(fit)
        record = dict(status='ok', utilities=fit.utilities.tolist(), alpha=fit.alpha,
                      alpha_signed_error=fit.alpha-alpha_true, alpha_absolute_error=abs(fit.alpha-alpha_true),
                      nll=fit.negative_log_likelihood, n_trials=fit.n_trials, **metrics,
                      information_min_eigenvalue=fit.information_min_eigenvalue,
                      information_condition=fit.information_condition,
                      optimizer_iterations=fit.optimizer_iterations)
        if model == 'planning':
            record.update(optimizer_evaluations=fit.optimizer_evaluations, gradient_max_abs=fit.gradient_max_abs)
        if train:
            prediction = (predict_planning(mdp, fit, choices, sessions=[5]) if model == 'planning' else
                          predict_history(fit, choices, sessions=[5]))
            if prediction.empty:
                raise ValueError('Held-out scoring requires session 5.')
            record.update(test_nll=float(prediction.log_loss.sum()), test_log_loss=float(prediction.log_loss.mean()),
                          test_trials=len(prediction))
    except RecoveryError as exc:
        record = dict(status='FAILED', error=str(exc))
    record['seconds'] = perf_counter() - started
    return record


def benchmark(mdp, choices, *, beta, gamma, horizons=(0, 1, 3, 5, 8)):
    frame, times, counts = demonstration_states(mdp, choices)
    theta = np.r_[np.array(DEFAULT_UTILITIES)[1:]-DEFAULT_UTILITIES[0], 1.]
    rows = []
    for h in horizons:
        t = perf_counter()
        batch = PlanningBatch(mdp, times, counts, lookahead=h)
        compile_seconds = perf_counter()-t
        t = perf_counter()
        for _ in range(10):
            batch.evaluate(theta, beta=beta, gamma=gamma, gradient=True)
        evaluation_seconds = (perf_counter()-t)/10
        plans = {}
        for memo in (False, True):
            t = perf_counter()
            for _ in range(10):
                plan = soft_plan(mdp, mdp.initial, DEFAULT_UTILITIES, alpha=1, beta=beta,
                                 gamma=gamma, lookahead=h, memoize=memo)
            plans[str(memo)] = dict(seconds=(perf_counter()-t)/10,
                                    evaluated_states=plan.evaluated_states, cache_hits=plan.cache_hits)
        rows.append(dict(lookahead=h, root_count=len(frame), compiled_nodes=batch.nodes,
                         feature_bytes=sum(f.nbytes for f in batch.features), compile_seconds=compile_seconds,
                         batch_policy_gradient_seconds=evaluation_seconds, scalar=plans))
    return dict(horizons=rows, reachable_states=reachable_state_audit(mdp),
                projection_note='Scale measured primary recovery time by batch-gradient time ratios; assumes equal optimizer evaluations.')


def sensitivity(mdp, rollout, *, alpha, beta, gamma, horizons):
    _, times, counts = demonstration_states(mdp, rollout.choices)
    theta = np.r_[np.array(DEFAULT_UTILITIES)[1:] - DEFAULT_UTILITIES[0], alpha]
    baseline = np.exp(PlanningBatch(mdp, times, counts, lookahead=0).evaluate(theta, beta=beta, gamma=gamma)[1][:, 0])
    results = []
    for h in horizons:
        p = np.exp(PlanningBatch(mdp, times, counts, lookahead=h).evaluate(theta, beta=beta, gamma=gamma)[1][:, 0])
        delta = np.abs(p-baseline)
        results.append(dict(alpha_true=alpha, lookahead=h, mean_absolute_probability_difference=float(delta.mean()),
                            max_absolute_probability_difference=float(delta.max()), max_difference_t=int(delta.argmax()),
                            most_likely_action_change_fraction=float(np.mean((p > .5) != (baseline > .5))),
                            probability_left=p.tolist(), myopic_probability_left=baseline.tolist()))
    probes = []
    for t in (10, len(mdp.pairs)//2, len(mdp.pairs)):
        c = counts[t] if t < len(mdp.pairs) else rollout.final_history.chosen
        initial = ChoiceHistory(mdp.presented[t].copy(), np.array(c).copy())
        for pair in ((2, 3), (4, 5), (6, 7)):
            # Replace the current displayed pair, then use the actual next pairs.
            # At T append a final terminal probe; never fabricate a continuation.
            continuation = mdp.pairs[t+1:t+1+max(horizons)]
            probe_mdp = HistoryMDP([pair, *continuation], initial_history=initial)
            for h in horizons:
                plan = soft_plan(probe_mdp, probe_mdp.initial, DEFAULT_UTILITIES, alpha=alpha,
                                 beta=beta, gamma=gamma, lookahead=h)
                immediate_gap = alpha * (initial.centered[pair[0]-1] - initial.centered[pair[1]-1])
                probes.append(dict(alpha_true=alpha, retained_t=t, pair=list(pair), lookahead=h,
                                   history_left=float(initial.centered[pair[0]-1]), history_right=float(initial.centered[pair[1]-1]),
                                   immediate_gap=float(immediate_gap), q_gap=float(plan.q[0]-plan.q[1]),
                                   continuation_gap=float(plan.q[0]-plan.q[1]-immediate_gap),
                                   probability_left=float(np.exp(plan.log_policy[0]))))
    return results, probes


def summarize(runs):
    rows = []
    for run in runs:
        for label, fit in run['fits'].items():
            row = {k: run[k] for k in ('generator', 'alpha_true', 'rollouts', 'seed_batch')}
            row.update(fit=label, status=fit['status'])
            if fit['status'] == 'ok':
                row.update({k: fit[k] for k in ('alpha', 'alpha_absolute_error', 'mae', 'nll', 'seconds', 'information_condition')})
                row['test_log_loss'] = fit.get('test_log_loss', np.nan)
            rows.append(row)
    frame = pd.DataFrame(rows)
    # Keep raw failed rows separately; summary explicitly reports success/attempt counts.
    output = []
    for keys, group in frame.groupby(['generator', 'alpha_true', 'rollouts', 'fit']):
        ok = group[group.status == 'ok']
        result = dict(generator=keys[0], alpha_true=float(keys[1]), rollouts=int(keys[2]), fit=keys[3])
        result.update(attempted=len(group), successful=len(ok))
        for metric in ('alpha', 'alpha_absolute_error', 'mae', 'nll', 'seconds', 'information_condition', 'test_log_loss'):
            values = ok[metric].dropna() if metric in ok else pd.Series(dtype=float)
            result[metric+'_mean'] = float(values.mean()) if len(values) else None
            result[metric+'_sd'] = float(values.std()) if len(values)>1 else None
        output.append(result)
    return output
