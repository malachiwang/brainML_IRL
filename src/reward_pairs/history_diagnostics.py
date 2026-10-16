"""Training-only orchestration and diagnostics; no new behavioral model.

The final-test table is not an input. Phase labels, not session numbers alone,
guard this boundary: the supplied final-test table also uses session label 5.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import expit

from .authors_audit import phase_trials, random_starts as authors_starts
from .authors_models import AuthorsTrials, FREE, initial_values, evaluate as authors_evaluate
from .authors_fitting import AuthorsFitError, fit_authors
from .authors_evaluation import distribution
from .cumulative_history_human import (
    CumulativeFitError, evaluate as cumulative_evaluate, fit as cumulative_fit,
    random_starts as cumulative_starts,
)
from .human_history_evaluation import EQUAL_PAIRS, TIE_TOLERANCE

MODELS = ('rl', 'ck', 'cumulative')
PHASES = ('early', 'middle', 'late')
COMPARISONS = (('cumulative', 'rl'), ('cumulative', 'ck'), ('ck', 'rl'))


def guard_training(frame):
    """Reject test phase even when mislabeled as session 5 (or supplied as S6)."""
    if frame.empty or 'phase' not in frame or not frame.phase.astype(str).eq('training').all():
        raise ValueError('Only explicit phase=training rows are allowed; final test is forbidden.')
    s = pd.to_numeric(frame.session, errors='raise').to_numpy(float)
    if not np.isfinite(s).all() or not np.isin(s, [1, 2, 3, 4, 5]).all():
        raise ValueError('Only training sessions 1..5 are allowed, never session 6.')


def permitted_inputs(objects):
    """Read only named training and initialization tables from an RData container."""
    permitted={key:objects[key] for key in ('data_4_analysis_RP_training','ratings_RP2','grouped_data_RP_pretest2_clean')}
    guard_training(permitted['data_4_analysis_RP_training'])
    order=list(dict.fromkeys(objects['data_4_analysis_RP_pretest'].ID.astype(str)))
    if set(order)!=set(permitted['data_4_analysis_RP_training'].ID.astype(str)):
        raise ValueError('Source participant sets differ; no new exclusions permitted.')
    return permitted,order


def training_initialization(objects, participant):
    """Same named/positional source checks as authors adapter, without test access."""
    ratings = objects['ratings_RP2']
    columns = [f's{i}_used_pre' for i in range(1, 9)]
    if list(ratings.columns[1:16:2]) != columns:
        raise ValueError('Rating schema differs from authors adapter.')
    r = ratings.loc[ratings.ID.astype(str).eq(participant), columns]
    p = objects['grouped_data_RP_pretest2_clean']
    p = p.loc[p.ID.astype(str).eq(participant), ['stim_chosen', 'perc_stim_chosen']].copy()
    p['stim_chosen'] = pd.to_numeric(p.stim_chosen).astype(int)
    if len(r) != 1 or len(p) != 8 or set(p.stim_chosen) != set(range(1, 9)):
        raise ValueError('Incomplete supplied pre-training initialization; no exclusions added.')
    return initial_values(r.iloc[0].to_numpy(float),
                          p.sort_values('stim_chosen').perc_stim_chosen.to_numpy(float),
                          'released_code_intended')


@dataclass
class TrainingFold:
    fit_trials: object
    validation_trials: object
    fit_sessions: np.ndarray
    validation_trial_labels: np.ndarray
    structure: dict


def make_fold(frame, participant, validation_session):
    guard_training(frame)
    if validation_session not in (2, 3, 4, 5):
        raise ValueError('Validation target must be training session 2..5.')
    p = frame.loc[frame.ID.astype(str).eq(participant)].copy()
    p['session'] = pd.to_numeric(p.session).astype(int)
    p['trial'] = pd.to_numeric(p.trial)  # phase_trials validates integers; never truncate
    if set(p.session) != set(range(1, 6)):
        raise ValueError('Expected all five training sessions; inspect rather than exclude.')
    before, now = p.loc[p.session < validation_session], p.loc[p.session == validation_session]
    fit_trials, fit_info = phase_trials(before, participant, training=True)
    validation, validation_info = phase_trials(now, participant, training=True)
    return TrainingFold(fit_trials, validation,
                        before.sort_values(['session', 'trial']).session.to_numpy(int),
                        now.sort_values('trial').trial.to_numpy(int),
                        dict(fit=fit_info, validation=validation_info))


def terciles(n):
    if n < 1:
        raise ValueError('Need retained observations.')
    labels = np.empty(n, dtype=int)
    for j, indices in enumerate(np.array_split(np.arange(n), 3)):
        labels[indices] = j
    return labels


def reward_classes(rewards):
    """Natural supplied reward units: equal=0, small=1, larger>1. No error tuning."""
    gap = np.abs(np.diff(np.asarray(rewards, float), axis=1)[:, 0])
    if not np.isfinite(gap).all() or np.any((gap > 1e-10) & (gap < 1-1e-10)):
        raise ValueError('Unexpected reward units; inspect before categorizing.')
    return np.where(np.isclose(gap, 0), 'equal', np.where(np.isclose(gap, 1), 'small', 'larger'))


def half_life(alpha):
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('CK alpha must lie in [0,1].')
    if alpha == 0:
        return dict(updates=None, infinite=True, case='no_updating', first_halving_update=None)
    if alpha == 1:
        return dict(updates=1., infinite=False, case='immediate_one_update', first_halving_update=1)
    value = float(np.log(.5)/np.log1p(-alpha))
    return dict(updates=value, infinite=False, case='interior', first_halving_update=int(np.ceil(value)))


def decompose(trials, output, beta_rl, beta_history):
    t = np.arange(len(trials.left)); l, r = trials.pairs.T
    raw = output.h[t, l]-output.h[t, r]
    reward = beta_rl*(output.q[t, l]-output.q[t, r])
    history = beta_history*raw
    np.testing.assert_allclose(expit(reward+history), output.probability_left, rtol=0, atol=1e-13)
    return dict(raw_history_delta=raw, reward_logit=reward, history_logit=history)


def calibration(probability, left):
    p = np.asarray(probability, float); y = np.asarray(left, bool)
    if p.shape != y.shape or np.any((p < 0) | (p > 1)) or not np.isfinite(p).all():
        raise ValueError('Invalid calibration observations.')
    confidence = np.maximum(p, 1-p)
    correct = ((p >= .5) == y)  # exactly .5 deterministically favors left
    edges = np.linspace(.5, 1., 11)
    which = np.clip(np.searchsorted(edges, confidence, side='right')-1, 0, 9)
    bins = []
    for i in range(10):
        mask = which == i; n = int(mask.sum())
        pred, obs = float(confidence[mask].sum()), int(correct[mask].sum())
        bins.append(dict(lower=float(edges[i]), upper=float(edges[i+1]), n=n,
                         confidence_sum=pred, correct_count=obs,
                         mean_confidence=pred/n if n else None, accuracy=obs/n if n else None,
                         overconfidence=(pred-obs)/n if n else None))
    return bins


def score_slice(trials, output, mask):
    n = int(mask.sum())
    if not n:
        return dict(n=0, nll=0., mean_log_loss=None, brier=None)
    return dict(n=n, nll=float(output.log_loss[mask].sum()),
                mean_log_loss=float(output.log_loss[mask].mean()),
                brier=float(np.mean((output.probability_left[mask]-trials.left[mask])**2)))


def equal_pair_diagnostic(trials, output, components, beta_rl, beta_h):
    """Observed comparisons, if present; unobserved state probes kept separate."""
    result = {}
    for a, b, frequent in EQUAL_PAIRS:
        rare = a if frequent == b else b
        f, r = frequent-1, rare-1
        mask = np.all(np.sort(trials.pairs, axis=1) == [a-1, b-1], axis=1)
        left_f = trials.pairs[:, 0] == f
        predicted = np.where(left_f, output.probability_left, 1-output.probability_left)
        observed = np.where(left_f, trials.left, ~trials.left)
        n = int(mask.sum())
        empirical = float(observed[mask].mean()) if n else None
        pred = float(predicted[mask].mean()) if n else None
        # No hypothetical action is sampled/scored, and no hypothetical update occurs.
        dh = output.h[:-1, f]-output.h[:-1, r]
        reward = beta_rl*(output.q[:-1, f]-output.q[:-1, r])
        history = beta_h*dh
        probe = expit(reward+history)
        result[f'{a}-{b}'] = dict(
            observed=dict(n=n, frequent_rate=empirical, predicted_frequent=pred,
                          mean_log_loss=float(output.log_loss[mask].mean()) if n else None,
                          calibration_error=pred-empirical if n else None,
                          observed_abs_from_half=abs(empirical-.5) if n else None,
                          predicted_mean_abs_from_half=float(abs(predicted[mask]-.5).mean()) if n else None),
            unobserved_state_probe=dict(raw_history_delta=float(dh.mean()),
                weighted_history=float(history.mean()), reward_logit=float(reward.mean()),
                mean_probability=float(probe.mean()), mean_abs_probability_from_half=float(abs(probe-.5).mean()),
                history_dominates_fraction=float(np.mean(abs(history)>abs(reward))),
                phase_weighted_history=[float(history[terciles(len(history))==j].mean())
                                        if np.any(terciles(len(history))==j) else None for j in range(3)]))
    return result


def memory_updates(trials, output, weight):
    changes = abs(np.diff(output.h, axis=0))
    displayed = np.zeros_like(changes, dtype=bool)
    displayed[np.arange(len(trials.left))[:, None], trials.pairs] = True
    return dict(mean_abs_raw_update=float(changes[displayed].mean()),
                mean_abs_weighted_update=float((abs(weight)*changes[displayed]).mean()),
                per_stimulus=[dict(stimulus=i+1, presentations=int(displayed[:, i].sum()),
                    mean_abs_raw_update=float(changes[displayed[:, i], i].mean()) if displayed[:, i].any() else None,
                    mean_abs_weighted_update=float(abs(weight)*changes[displayed[:, i], i].mean()) if displayed[:, i].any() else None)
                    for i in range(8)])


def disagreement(trials, ck, cumulative, phases):
    difference = abs(ck.probability_left-cumulative.probability_left)
    signed = np.where(trials.left, 1., -1.)*(ck.probability_left-cumulative.probability_left)
    pairs = np.sort(trials.pairs+1, axis=1)
    gaps = abs(trials.rewards[:, 0]-trials.rewards[:, 1])
    out = {}
    for cutoff in (.10, .20):
        mask = difference >= cutoff
        out[str(cutoff)] = dict(n=int(mask.sum()), denominator=len(mask),
            ck_assigns_higher=int(np.sum(mask & (signed>1e-12))),
            cumulative_assigns_higher=int(np.sum(mask & (signed< -1e-12))),
            ties=int(np.sum(mask & (abs(signed)<=1e-12))),
            ck_log_likelihood_gain=float((cumulative.log_loss[mask]-ck.log_loss[mask]).sum()),
            phases={name:int(np.sum(mask & (phases==j))) for j, name in enumerate(PHASES)},
            pair_counts={f'{a}-{b}':int(np.sum(mask & np.all(pairs==[a,b],axis=1))) for a,b in np.unique(pairs,axis=0)},
            reward_gap_counts={str(float(g)):int(np.sum(mask & (gaps==g))) for g in np.unique(gaps)})
    return out


def replay_fold(fold, name, theta, q0, h0):
    """Fit parameters are already fixed; validation feedback remains present."""
    if fold.fit_trials.rewards is None or fold.validation_trials.rewards is None:
        raise ValueError('Both fit and validation must have training feedback.')
    if name == 'cumulative':
        tr = cumulative_evaluate(fold.fit_trials, theta, q0)
        va = cumulative_evaluate(fold.validation_trials, theta, tr.q[-1], initial_history=tr.final_history)
    else:
        model = {'rl':3, 'ck':19}[name]
        tr = authors_evaluate(fold.fit_trials, model, theta, q0, h0, scoring_pi=0.)
        va = authors_evaluate(fold.validation_trials, model, theta, tr.q[-1], tr.h[-1], scoring_pi=0.)
    np.testing.assert_array_equal(tr.q[-1], va.q[0])
    np.testing.assert_array_equal(tr.h[-1], va.h[0])
    return tr, va


def fit_fold(fold, q0, h0, r_starts, c_starts):
    """Only prefix training trials enter existing optimizers; no validation inputs."""
    fits = {}
    for name, model in [('rl', 3), ('ck', 19), ('cumulative', None)]:
        try:
            if model is None:
                result = cumulative_fit([fold.fit_trials], q0, starts=c_starts)
                theta = list(result['parameters'].values())
            else:
                fitted = fit_authors([fold.fit_trials], model, q0, h0, starts=r_starts[:, list(FREE[model])])
                result, theta = fitted.record(model), fitted.theta.tolist()
            fits[name] = dict(status='success', theta=theta, optimization=result)
        except (AuthorsFitError, CumulativeFitError) as exc:
            fits[name] = dict(status='failed', error=str(exc), attempts=getattr(exc, 'attempts', None))
    return fits


def analyze_fold(fold, fits, q0, h0, session):
    row = dict(session=session, structure=fold.structure, models={}, status='success')
    traces = {}; outputs = {}; phases = terciles(len(fold.validation_trials.left))
    trials = fold.validation_trials
    for name, fitted in fits.items():
        if fitted['status'] != 'success':
            row['models'][name] = fitted; row['status'] = 'failed'; continue
        theta = fitted['theta']
        tr, va = replay_fold(fold, name, theta, q0, h0)
        obj = fitted['optimization'].get('objective_deviance', fitted['optimization'].get('optimizer_objective_deviance'))
        if abs(tr.deviance-obj)>1e-7:
            raise RuntimeError('Fit objective disagrees with unchanged scalar replay.')
        br, bh = (theta[1], theta[2]) if name=='cumulative' else ((theta[3],theta[1]) if name=='ck' else (theta[1],0.))
        components = decompose(trials, va, br, bh)
        reward, history = abs(components['reward_logit']), abs(components['history_logit'])
        sensible = reward>1e-6
        model = fitted | dict(validation=score_slice(trials,va,np.ones(len(trials.left),bool)),
            phases={p:score_slice(trials,va,phases==j) for j,p in enumerate(PHASES)},
            reward_classes={c:score_slice(trials,va,reward_classes(trials.rewards)==c) for c in ('equal','small','larger')},
            calibration=calibration(va.probability_left,trials.left),
            contributions=dict(raw_history_delta=distribution(components['raw_history_delta']),
                abs_reward=distribution(reward), abs_history=distribution(history),
                history_to_reward_ratio=distribution(history[sensible]/reward[sensible]),
                ratio_excluded=int((~sensible).sum()), history_dominates_fraction=float(np.mean(history>reward))),
            equal_pairs=equal_pair_diagnostic(trials,va,components,br,bh),
            initial_validation_q=va.q[0].tolist(), initial_validation_h=va.h[0].tolist(),
            final_validation_q=va.q[-1].tolist(), final_validation_h=va.h[-1].tolist())
        if name=='ck': model['ck_half_life']=half_life(theta[0])
        if name!='rl':
            model['validation_memory_updates']=memory_updates(trials,va,bh)
            if session==5:
                # Fixed S1-S4-estimated parameters: descriptive replay, NOT early-session validation.
                model['fixed_primary_parameter_memory']={}
                for s in range(1,6):
                    if s==5: segment_trials, segment = trials, va
                    else:
                        idx=np.flatnonzero(fold.fit_sessions==s)
                        segment_trials=AuthorsTrials(fold.fit_trials.pairs[idx],fold.fit_trials.left[idx],fold.fit_trials.rewards[idx])
                        segment=SimpleNamespace(h=tr.h[np.r_[idx,idx[-1]+1]])
                    model['fixed_primary_parameter_memory'][str(s)]=memory_updates(segment_trials,segment,bh)
        row['models'][name]=model; outputs[name]=va
        traces.update({f'{name}_{k}':v for k,v in components.items()})
        traces.update({f'{name}_probability_left':va.probability_left, f'{name}_log_loss':va.log_loss,
                       f'{name}_q_before':va.q[:-1], f'{name}_h_before':va.h[:-1], f'{name}_h_change':np.diff(va.h,axis=0)})
    if row['status']=='success':
        row['disagreement']=disagreement(trials,outputs['ck'],outputs['cumulative'],phases)
    traces.update(left=trials.left,pairs=trials.pairs,rewards=trials.rewards,
                  trial_labels=fold.validation_trial_labels,tercile=phases)
    return row,traces


def run_participant(job):
    participant,index,frame,q0,h0,folds,primary_starts,rolling_starts,use_r=job
    maximum=max(primary_starts,rolling_starts)
    rs=authors_starts(index,maximum,use_r=use_r)
    cs=cumulative_starts(index,maximum,seed=20261015)
    rows=[]; all_traces={}
    for session in folds:
        fold=make_fold(frame,participant,session)
        n=primary_starts if session==5 else rolling_starts
        fits=fit_fold(fold,q0,h0,rs[:n],cs[:n])
        row,traces=analyze_fold(fold,fits,q0,h0,session)
        prefix=f'p{index}_s{session}'
        row.update(participant=participant,source_index=index,trace_prefix=prefix)
        rows.append(row)
        all_traces.update({f'{prefix}_{k}':v for k,v in traces.items()})
    return rows,all_traces
