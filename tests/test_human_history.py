"""Causal, matched-RL, held-out, fitting, and participant-bootstrap checks."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from types import SimpleNamespace
from copy import deepcopy

from reward_pairs.authors_audit import phase_trials
from reward_pairs.authors_models import AuthorsTrials, evaluate as rl_evaluate, initial_values
from reward_pairs.authors_fitting import information_criteria
from reward_pairs.history import ChoiceHistory
from reward_pairs.cumulative_history_human import (
    CumulativeObjective, CumulativeFitError, copy_history, criteria, evaluate,
    fit, history_features, random_starts, validate_parameters,
)
from reward_pairs.human_history_evaluation import (
    compare_participant, equal_pair_scores, history_summary, participant_bootstrap,
    replay_reference, summarize, training_history_diagnostics,
)


@pytest.fixture
def trials():
    return AuthorsTrials(np.array([[0,1],[1,0],[0,2],[2,1],[3,2],[1,3]]),
                        np.array([True,False,False,True,False,True]),
                        np.array([[1.,2.],[2.,1.],[1.,3.],[3.,2.],[4.,3.],[2.,4.]]))


def test_counts_and_unseen_history(trials):
    h,state=history_features(trials)
    np.testing.assert_array_equal(h[0],np.zeros(8))
    np.testing.assert_allclose(h[1,:3],[.5,-.5,0])
    np.testing.assert_allclose(h[2,:3],[.5,-.5,0])
    assert state.presented.sum()==2*len(trials.left)
    assert state.chosen.sum()==len(trials.left)
    np.testing.assert_array_equal(state.presented,[3,4,3,2,0,0,0,0])
    np.testing.assert_array_equal(state.chosen,[2,1,3,0,0,0,0,0])


def test_current_and_future_choices_no_leakage(trials):
    a=evaluate(trials,[.2,1.,2.],np.zeros(8))
    changed=AuthorsTrials(trials.pairs,trials.left.copy(),trials.rewards)
    changed.left[2:]=~changed.left[2:]
    b=evaluate(changed,[.2,1.,2.],np.zeros(8))
    np.testing.assert_array_equal(a.h[:3],b.h[:3])
    np.testing.assert_array_equal(a.probability_left[:3],b.probability_left[:3])
    np.testing.assert_array_equal(a.q,b.q)  # RL does not depend on selected option
    assert not np.array_equal(a.h[3:],b.h[3:])


def test_session_gaps_isolation_and_carryover():
    frame=pd.DataFrame(dict(ID=['p','p','p','q'],session=[2,1,1,1],trial=[1,4,1,1],
        stim1=[1]*4,stim2=[2]*4,action=['left']*4,reward1=[1]*4,reward2=[2]*4))
    data,info=phase_trials(frame,'p',training=True)
    h,state=history_features(data)
    assert len(data.left)==3 and not info['original_order_chronological']
    np.testing.assert_array_equal(state.chosen,[3,0,0,0,0,0,0,0])
    np.testing.assert_allclose(h[2,:2],[.5,-.5])
    other,_=phase_trials(frame,'q',training=True)
    np.testing.assert_array_equal(history_features(other)[0][0],0)


def test_test_carryover_freezes_q_and_does_not_mutate_training(trials):
    theta=[.3,2.,1.5]; q0=np.arange(8.)/2
    train=evaluate(trials,theta,q0)
    before=train.final_history.chosen.copy()
    test=AuthorsTrials(trials.pairs,~trials.left,None)
    output=evaluate(test,theta,train.q[-1],initial_history=train.final_history)
    np.testing.assert_array_equal(output.h[0],train.h[-1])
    np.testing.assert_array_equal(output.q,np.tile(train.q[-1],(len(test.left)+1,1)))
    np.testing.assert_array_equal(train.final_history.chosen,before)
    assert output.final_history.chosen.sum()==2*len(trials.left)
    test.left=~test.left
    changed=evaluate(test,theta,train.q[-1],initial_history=train.final_history)
    assert changed.probability_left[0]==output.probability_left[0]


@pytest.mark.parametrize('alpha',[0.,.2,1.])
@pytest.mark.parametrize('beta',[0.,1.,10.])
def test_zero_history_weight_exact_rl_limit(trials,alpha,beta):
    q0=np.arange(8.)/3
    c=evaluate(trials,[alpha,beta,0.],q0)
    r=rl_evaluate(trials,3,[alpha,beta],q0,np.zeros(8))
    np.testing.assert_allclose(c.probability_left,r.probability_left,rtol=0,atol=1e-14)
    np.testing.assert_allclose(c.log_loss,r.log_loss,rtol=0,atol=1e-14)
    np.testing.assert_array_equal(c.q,r.q)
    assert CumulativeObjective([trials],q0)([alpha,beta,0])[0]==pytest.approx(r.deviance,abs=1e-10)


@pytest.mark.parametrize('weight',[-3.,0.,3.])
def test_signed_weight_effect(weight):
    history=ChoiceHistory(presented=np.array([6,6,0,0,0,0,0,0]),chosen=np.array([5,2,0,0,0,0,0,0]))
    t=AuthorsTrials(np.array([[0,1]]),np.array([True]),None)
    o=evaluate(t,[.5,1.,weight],np.ones(8),initial_history=history)
    assert o.probability_left[0]==pytest.approx(expit(weight*.5))


def test_fast_likelihood_gradient_and_determinism(trials):
    theta=np.array([.3,1.4,-1.2]); q0=np.arange(8.)/4
    objective=CumulativeObjective([trials],q0)
    dev,grad=objective(theta)
    assert dev==objective(theta)[0]
    assert dev==pytest.approx(evaluate(trials,theta,q0).deviance,abs=1e-12)
    numeric=[(objective(theta+np.eye(3)[i]*1e-6)[0]-objective(theta-np.eye(3)[i]*1e-6)[0])/2e-6 for i in range(3)]
    np.testing.assert_allclose(grad,numeric,atol=1e-7)
    assert CumulativeObjective([trials,trials],q0)(theta)[0]==pytest.approx(2*dev)


def test_simulation_reproducible_and_ignores_supplied_choices(trials):
    a=evaluate(trials,[.2,1.,2.],np.zeros(8),simulate_seed=2)
    changed=AuthorsTrials(trials.pairs,~trials.left,trials.rewards)
    b=evaluate(changed,[.2,1.,2.],np.zeros(8),simulate_seed=2)
    np.testing.assert_array_equal(a.left,b.left)
    np.testing.assert_array_equal(a.probability_left,b.probability_left)


@pytest.mark.parametrize('weight',[-2.,0.,2.])
def test_small_matched_recovery(weight):
    # Controlled varying feedback provides RL-rate information without a huge campaign.
    rng=np.random.default_rng(32)
    pairs=np.array([rng.choice(8,2,replace=False) for _ in range(200)])
    rewards=rng.uniform(0,3,(200,2))
    scaffold=AuthorsTrials(pairs,np.zeros(200,dtype=bool),rewards)
    q0=np.zeros(8); theta=[.3,1.,weight]; episodes=[]
    for seed in range(30):
        out=evaluate(scaffold,theta,q0,simulate_seed=seed)
        episodes.append(AuthorsTrials(pairs,out.left,rewards))
    result=fit(episodes,q0,starts=random_starts(1,20))
    estimate=list(result['parameters'].values())
    np.testing.assert_allclose(estimate,theta,atol=.3)
    assert result['objective_deviance']<=CumulativeObjective(episodes,q0)(theta)[0]+1e-7


def test_failure_and_invalid_parameters(monkeypatch,trials):
    monkeypatch.setattr('reward_pairs.cumulative_history_human.minimize',lambda *a,**k:SimpleNamespace(
        success=False,fun=2.,x=np.array([.5,1.,0.]),message='injected',nit=1,nfev=1))
    with pytest.raises(CumulativeFitError) as exc: fit([trials],np.zeros(8),starts=[[.5,1.,0.]])
    assert len(exc.value.attempts)==1
    with pytest.raises(ValueError): validate_parameters([.2,1,10.1])
    with pytest.raises(ValueError): copy_history(ChoiceHistory(chosen=np.ones(8)))
    with pytest.raises(ValueError): CumulativeObjective([AuthorsTrials(trials.pairs,trials.left,None)],np.zeros(8))


def make_references(train,test,ratings,pretest):
    variant='released_code_intended'; q,h=initial_values(ratings,pretest,variant)
    out={}
    for model,theta in [(3,[.3,1.]),(19,[.2,1.,.3,1.])]:
        tr=rl_evaluate(train,model,theta,q,h); te=rl_evaluate(test,model,theta,tr.q[-1],tr.h[-1])
        out[variant,model]=dict(variant=variant,model=model,scoring_pi=0,optimization_success=True,included=True,
            initial_q=q.tolist(),initial_h=h.tolist(),free_parameters={str(i):v for i,v in enumerate(theta)},
            parameters={},training=information_criteria(tr.deviance,model,len(train.left)),
            test=information_criteria(te.deviance,model,len(test.left)),restart_count=200,boundary_flags={})
    return out


def test_fit_inputs_no_test_and_same_initialization(monkeypatch,trials):
    test=AuthorsTrials(trials.pairs,~trials.left,None)
    ratings=np.arange(8.)*10; pre=np.arange(8.)/8
    references=make_references(trials,test,ratings,pre)
    seen=[]
    from reward_pairs.human_history_evaluation import fit as actual_fit
    def spy(episodes,q0,**kwargs):
        assert len(episodes)==1 and episodes[0] is trials
        np.testing.assert_array_equal(q0,initial_values(ratings,pre,'released_code_intended')[0])
        seen.append(q0.copy())
        return actual_fit(episodes,q0,**kwargs)
    monkeypatch.setattr('reward_pairs.human_history_evaluation.fit',spy)
    inputs=(trials,test,ratings,pre,dict(training={},test={}))
    first=compare_participant(('p',1,inputs,references,['released_code_intended'],2,7))[0]
    test.left=~test.left
    references=make_references(trials,test,ratings,pre)
    second=compare_participant(('p',1,inputs,references,['released_code_intended'],2,7))[0]
    assert first['models']['cumulative']['parameters']==second['models']['cumulative']['parameters']
    assert len(seen)==2
    np.testing.assert_array_equal(first['history']['chosen'],second['history']['chosen'])


@pytest.mark.parametrize('corruption',['literal_pi','row_count','deviance'])
def test_reference_replay_rejects_inconsistent_benchmark(trials,corruption):
    test=AuthorsTrials(trials.pairs,~trials.left,None)
    ratings=np.arange(8.)*10; pre=np.arange(8.)/8
    reference=make_references(trials,test,ratings,pre)['released_code_intended',3]
    if corruption=='literal_pi': reference['scoring_pi']=np.pi
    if corruption=='row_count': reference['test']['n']+=1
    if corruption=='deviance': reference['training']['deviance']+=.1
    q0,h0=initial_values(ratings,pre,'released_code_intended')
    with pytest.raises(ValueError): replay_reference(trials,test,q0,h0,reference)


def test_bootstrap_paired_units_and_counts():
    a=participant_bootstrap([1.,2.,3.],seed=5,draws=1000)
    b=participant_bootstrap([1.,2.,3.],seed=5,draws=1000)
    assert a==b and a['mean']==2
    constant=participant_bootstrap([.25]*4,seed=5,draws=100)
    assert constant['ci95']==[.25,.25]
    assert a['ci95'][0]<2<a['ci95'][1]


def test_orientation_equal_probe_and_criteria():
    test=AuthorsTrials(np.array([[1,2],[2,1],[3,4],[6,5]]),np.array([False,True,True,True]),None)
    result=evaluate(test,[0,0,0],np.zeros(8))
    scores=equal_pair_scores(test,result)
    assert scores['2-3']['observed_frequent_count']==2
    assert scores['2-3']['predicted_frequent_sum']==1
    ic=criteria(30,100)
    assert ic['k']==3 and ic['AIC']==36 and ic['AICc']==30+6*100/96
    assert ic['BIC']==30+3*np.log(100)


def test_restart_prefix_and_history_summary(trials):
    np.testing.assert_array_equal(random_starts(1,10),random_starts(1,20)[:10])
    assert not np.array_equal(random_starts(1,10),random_starts(2,10))
    diag=training_history_diagnostics(trials)
    s=history_summary([diag,diag])
    assert s['participants']==2
    assert s['final_h_by_stimulus'][0]['mean']==diag['final_h'][0]


def test_aggregation_sign_ties_failures_and_participant_weighting(trials):
    test=AuthorsTrials(trials.pairs,~trials.left,None)
    ratings=np.arange(8.)*10; pre=np.arange(8.)/8
    references=make_references(trials,test,ratings,pre)
    base=compare_participant(('p',1,(trials,test,ratings,pre,{}),references,
                             ['released_code_intended'],2,7))[0]
    rows=[]
    for index,gain in enumerate([-.2,0.,.1]):
        row=deepcopy(base); row['participant']=str(index)
        for name in ('rl','ck'):
            row['comparisons'][name].update(delta_test_log_loss=gain,delta_test_LL=gain*(index+1),
                winner='tie' if gain==0 else 'cumulative' if gain>0 else name)
        rows.append(row)
    rows.append(dict(participant='failed',variant='released_code_intended',fit_status='failed'))
    out=summarize(rows,bootstrap_seed=3,bootstrap_draws=100)['released_code_intended']
    assert out['participants']==4 and out['complete_pairs']==3
    assert out['failed_participants']==['failed']
    for name in ('rl','ck'):
        c=out['comparisons'][name]
        assert (c['cumulative_better'],c['cumulative_worse'],c['ties'])==(1,1,1)
        assert c['bootstrap']['mean']==pytest.approx(-.1/3)
        assert c['bootstrap']['n_participants']==3
