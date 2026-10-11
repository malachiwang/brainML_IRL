"""Paired predictive comparison. Generator truth is never a fitting input."""

import numpy as np
import pandas as pd

from .history_irl import predict_planning, recover_planning
from .history_mdp import HistoryMDP
from .history_planning import planning_rollouts
from .history_recovery import predict_history, recover_history
from .history_replication import batch_root, distribution
from .history_simulation import independent_rollouts
from .recovery import RecoveryError

WINDOWS = ('early', 'mid', 'late', 'all', 'session1_remainder', 'sessions2_4')


def evaluation_root(participant, seed_batch):
    """Domain-separated from existing training root; shared across paired counts."""
    return int(np.random.SeedSequence([batch_root(participant, seed_batch), 0x4556414C]).generate_state(1, dtype=np.uint64)[0])


def generate_agents(trajectory, n, *, generator, alpha, seed, beta=1., gamma=.95, horizon=3, evaluation=False):
    if generator == 'myopic':
        agents = independent_rollouts(trajectory, n, alpha=alpha, beta=beta, seed=seed)
    elif generator == 'planning':
        agents = planning_rollouts(HistoryMDP.from_trials(trajectory), n, alpha=alpha, beta=beta,
                                   gamma=gamma, lookahead=horizon, seed=seed)
    else:
        raise ValueError('Generator must be myopic or planning.')
    frame = pd.concat([a.choices for a in agents], ignore_index=True)
    # Disjoint IDs prevent accidental overlap/concatenation being invisible.
    if evaluation:
        frame['rollout'] = 'eval_' + frame.rollout.astype(str).str.zfill(6)
    return frame


def fit_models(mdp, choices, *, beta=1., gamma=.95, horizon=3):
    """Neither generator label nor truth nor held-out agents is accepted here."""
    fits, records = {}, {}
    for scope, sessions in (('all',None), ('train',[1,2,3,4])):
        for model in ('myopic','planning'):
            key = f'{scope}_{model}'
            try:
                fit = (recover_history(choices, beta=beta, fit_sessions=sessions) if model=='myopic' else
                       recover_planning(mdp, choices, beta=beta, gamma=gamma, lookahead=horizon, fit_sessions=sessions))
                fits[key] = fit
                records[key] = dict(status='ok', alpha=fit.alpha, utilities=fit.utilities.tolist(),
                    nll=fit.negative_log_likelihood, n_trials=fit.n_trials,
                    information_condition=fit.information_condition)
            except RecoveryError as exc:
                records[key] = dict(status='FAILED', error=str(exc))
    return fits, records


def window_masks(frame):
    """Retained position, NOT original trial label; disjoint early/mid/late."""
    position = frame.groupby('rollout', sort=False).cumcount().to_numpy()
    early = position < 20
    late = frame.session.eq(5).to_numpy()
    if np.any(early & late):
        raise ValueError('First 20 retained trials overlap session 5; inspect scaffold.')
    return dict(early=early, mid=~early & ~late, late=late, all=np.ones(len(frame),dtype=bool),
                session1_remainder=~early & frame.session.eq(1).to_numpy(),
                sessions2_4=frame.session.isin([2,3,4]).to_numpy())


def paired_score(myopic, planning, mask, *, tie_tolerance=1e-8):
    """Delta=NLL_myopic-NLL_planning. Small absolute-NLL ties stay explicit."""
    if len(myopic)!=len(planning) or not myopic[['rollout','session','trial']].equals(planning[['rollout','session','trial']]):
        raise ValueError('Predictions must refer to identical ordered observations.')
    a,b = myopic.loc[mask],planning.loc[mask]
    if not len(a):
        raise ValueError('Empty evaluation window.')
    nll_a,nll_b = float(a.log_loss.sum()),float(b.log_loss.sum())
    delta = nll_a-nll_b
    gap = np.abs(a.probability_left.to_numpy()-b.probability_left.to_numpy())
    return dict(n=len(a), nll_myopic=nll_a,nll_planning=nll_b,
                log_loss_myopic=nll_a/len(a),log_loss_planning=nll_b/len(a),
                delta_nll=delta,delta_log_loss=delta/len(a),
                selected='tie' if abs(delta)<=tie_tolerance else ('planning' if delta>0 else 'myopic'),
                probability_difference_mean=float(gap.mean()),probability_difference_max=float(gap.max()),
                probability_difference_gt_01=float(np.mean(gap>.01)),
                probability_difference_gt_05=float(np.mean(gap>.05)),
                different_modal_action=float(np.mean((a.probability_left.to_numpy()>.5)!=(b.probability_left.to_numpy()>.5))))


def score_fitted(mdp, fits, training_choices, fresh_choices):
    """No fitting here. All-session fits score fresh agents; train fits score S5."""
    scores = {}
    for scope, data in (('all',fresh_choices),('train',training_choices)):
        if not all(f'{scope}_{m}' in fits for m in ('myopic','planning')):
            continue
        a = predict_history(fits[f'{scope}_myopic'],data)
        b = predict_planning(mdp,fits[f'{scope}_planning'],data)
        if scope=='all':
            for name,mask in window_masks(a).items():
                scores['fresh_'+name] = paired_score(a,b,mask)
        else:
            scores['within_session5'] = paired_score(a,b,a.session.eq(5).to_numpy())
    return scores


def selection_summary(rows, window):
    scores = [r['scores'][window] for r in rows if window in r.get('scores',{})]
    counts = {m:sum(s['selected']==m for s in scores) for m in ('myopic','planning','tie')}
    result = dict(attempted=len(rows),paired=len(scores),missing=len(rows)-len(scores), counts=counts,
                  rates={m:n/len(scores) if scores else None for m,n in counts.items()})
    for metric in ('delta_nll','delta_log_loss','nll_myopic','nll_planning','log_loss_myopic','log_loss_planning',
                   'probability_difference_mean','probability_difference_max','probability_difference_gt_01',
                   'probability_difference_gt_05','different_modal_action'):
        result[metric] = distribution(s[metric] for s in scores)
    return result


def confusion_matrix(rows, window):
    """Null alpha=0 is excluded from classification, including denominators."""
    return {g:selection_summary([r for r in rows if r['generator']==g and r['alpha_true']!=0],window)
            for g in ('myopic','planning')}


def aggregate_selection(rows, participants):
    conditions, scaffolds, parameters = [],[],[]
    windows = ['fresh_'+w for w in WINDOWS]+['within_session5']
    for g,a,n in sorted({(r['generator'],r['alpha_true'],r['rollouts']) for r in rows}):
        group = [r for r in rows if (r['generator'],r['alpha_true'],r['rollouts'])==(g,a,n)]
        for window in windows:
            key = dict(generator=g,alpha_true=a,rollouts=n,window=window,null_control=a==0)
            conditions.append(dict(**key,**selection_summary(group,window)))
            for p in participants:
                scaffolds.append(dict(participant=p,**key,**selection_summary([r for r in group if r['participant']==p],window)))
        for model in ('myopic','planning'):
            fits = [r['fits']['all_'+model] for r in group if r['fits']['all_'+model]['status']=='ok']
            parameters.append(dict(generator=g,alpha_true=a,rollouts=n,model=model,attempted=len(group),successful=len(fits),
                alpha=distribution(f['alpha'] for f in fits),
                alpha_bias=distribution(f['alpha']-a for f in fits),
                utility_mae=distribution(float(np.mean(np.abs(np.array(f['utilities'])-[0,1,1,2,2,3,3,4]))) for f in fits)))
    matrices=[]
    for a,n in sorted({(r['alpha_true'],r['rollouts']) for r in rows if r['alpha_true']>0}):
        selected=[r for r in rows if r['alpha_true']==a and r['rollouts']==n]
        for window in windows:
            matrices.append(dict(alpha_true=a,rollouts=n,window=window,matrix=confusion_matrix(selected,window)))
    return dict(conditions=conditions,scaffold_conditions=scaffolds,parameters=parameters,confusion_matrices=matrices,
                fits_attempted=sum(len(r['fits']) for r in rows),
                fit_failures=sum(f['status']!='ok' for r in rows for f in r['fits'].values()))
