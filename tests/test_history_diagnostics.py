"""Training-only boundaries, causal validation, and descriptive diagnostics."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from reward_pairs.authors_models import AuthorsTrials, evaluate, initial_values
from reward_pairs.history_diagnostics import (
    analyze_fold, calibration, decompose, disagreement, equal_pair_diagnostic,
    fit_fold, guard_training, half_life, make_fold, memory_updates, permitted_inputs,
    replay_fold, reward_classes, terciles, training_initialization,
)
from reward_pairs.history_diagnostics_summary import paired_summary, pool_calibration


@pytest.fixture
def frame():
    rng=np.random.default_rng(4)
    pairs=np.array([rng.choice(8,2,replace=False) for _ in range(60)])
    return pd.DataFrame(dict(ID=['p']*60,phase=['training']*60,
        session=np.repeat(np.arange(1,6),12),trial=np.tile(np.arange(1,13),5),
        stim1=pairs[:,0]+1,stim2=pairs[:,1]+1,
        action=np.where(rng.random(60)<.5,'left','right'),
        reward1=pairs[:,0]+1.,reward2=pairs[:,1]+1.))


@pytest.mark.parametrize('session',[2,3,4,5])
def test_prefix_fit_boundaries_and_participant_isolation(frame,session):
    other=frame.assign(ID='other',action='left')
    f=make_fold(pd.concat([frame,other]),'p',session)
    assert len(f.fit_trials.left)==12*(session-1)
    assert set(f.fit_sessions)==set(range(1,session))
    assert len(f.validation_trials.left)==12
    changed=frame.copy(); changed.loc[changed.session>=session,'action']='right'
    g=make_fold(changed,'p',session)
    np.testing.assert_array_equal(f.fit_trials.left,g.fit_trials.left)
    np.testing.assert_array_equal(f.fit_trials.rewards,g.fit_trials.rewards)


@pytest.mark.parametrize('bad_phase,bad_session',[('test',5),('test',6),('training',6),('pretest',1)])
def test_final_test_guard_even_if_labelled_session_five(frame,bad_phase,bad_session):
    frame.loc[0,['phase','session']]=[bad_phase,bad_session]
    with pytest.raises(ValueError): make_fold(frame,'p',5)


def test_noninteger_trial_not_silently_truncated(frame):
    frame['trial']=frame.trial.astype(float); frame.loc[0,'trial']=1.5
    with pytest.raises(ValueError): make_fold(frame,'p',5)


def test_loader_cannot_read_final_test_and_pretest_initialization_only(frame):
    class Guarded(dict):
        def __getitem__(self,key):
            assert key!='data_4_analysis_RP_test','final test accessed'
            return super().__getitem__(key)
    ratings={'ID':['p']}
    for i in range(1,9):
        ratings[f's{i}_used_pre']=[i*10.]; ratings[f's{i}_used_post']=[999.]
    objects=Guarded(data_4_analysis_RP_training=frame,ratings_RP2=pd.DataFrame(ratings),
        grouped_data_RP_pretest2_clean=pd.DataFrame(dict(ID=['p']*8,stim_chosen=range(1,9),perc_stim_chosen=np.arange(8.)/8)),
        data_4_analysis_RP_pretest=pd.DataFrame({'ID':['p']}))
    allowed,order=permitted_inputs(objects)
    assert order==['p'] and 'data_4_analysis_RP_test' not in allowed
    q,h=training_initialization(allowed,'p')
    expected=initial_values(np.arange(1,9)*10,np.arange(8.)/8,'released_code_intended')
    np.testing.assert_array_equal(q,expected[0]); np.testing.assert_array_equal(h,expected[1])


def test_optimizer_only_receives_fit_prefix(monkeypatch,frame):
    import reward_pairs.history_diagnostics as module
    fold=make_fold(frame,'p',5); seen=[]
    def authors(episodes,model,q,h,**kw):
        assert len(episodes)==1 and episodes[0] is fold.fit_trials
        seen.append(model)
        return SimpleNamespace(record=lambda m:{},theta=np.array([.2,1.] if model==3 else [.2,1.,.2,1.]))
    def cumulative(episodes,q,**kw):
        assert len(episodes)==1 and episodes[0] is fold.fit_trials
        seen.append('cumulative')
        return {'parameters':dict(alpha_rl=.2,beta_rl=1.,beta_h=1.)}
    monkeypatch.setattr(module,'fit_authors',authors); monkeypatch.setattr(module,'cumulative_fit',cumulative)
    fit_fold(fold,np.zeros(8),np.zeros(8),np.array([[.2,1.,.2,1.,2.5]]),np.array([[.2,1.,1.]]))
    assert seen==[3,19,'cumulative']


@pytest.mark.parametrize('name,theta',[('rl',[.3,1.]),('ck',[.2,1.,.3,1.]),('cumulative',[.3,1.,1.])])
def test_validation_state_carryover_causality_and_feedback(frame,name,theta):
    fold=make_fold(frame,'p',5)
    tr,va=replay_fold(fold,name,theta,np.zeros(8),np.zeros(8))
    np.testing.assert_array_equal(tr.q[-1],va.q[0]); np.testing.assert_array_equal(tr.h[-1],va.h[0])
    l,r=fold.validation_trials.pairs[0]
    np.testing.assert_allclose(va.q[1,[l,r]],va.q[0,[l,r]]+.3*(fold.validation_trials.rewards[0]-va.q[0,[l,r]]))
    assert not np.array_equal(va.q[0],va.q[1])
    if name!='rl': assert not np.array_equal(va.h[0],va.h[1])
    altered=deepcopy(fold); altered.validation_trials.left[3:]=~altered.validation_trials.left[3:]
    tr2,va2=replay_fold(altered,name,theta,np.zeros(8),np.zeros(8))
    np.testing.assert_array_equal(tr.h,tr2.h)
    np.testing.assert_array_equal(va.probability_left[:4],va2.probability_left[:4])
    if name=='cumulative':
        assert va.final_history.chosen.sum()==60
        assert va.final_history.presented.sum()==120


@pytest.mark.parametrize('alpha,expected',[(0,None),(1,1),(.5,1),(.25,np.log(.5)/np.log(.75))])
def test_ck_half_life(alpha,expected):
    result=half_life(alpha)
    if expected is None: assert result['updates'] is None and result['infinite']
    else: assert result['updates']==pytest.approx(expected) and not result['infinite']
    if 0<alpha<1: assert (1-alpha)**result['updates']==pytest.approx(.5)
    with pytest.raises(ValueError): half_life(-.01)


def test_terciles_are_chronological_equal_count():
    labels=terciles(8)
    np.testing.assert_array_equal(labels,[0,0,0,1,1,1,2,2])
    assert np.bincount(terciles(160)).max()-np.bincount(terciles(160)).min()<=1
    with pytest.raises(ValueError): terciles(0)


def test_reward_and_equal_pair_classification():
    rewards=np.array([[2.,2.],[1.,2.],[2.,5.]])
    np.testing.assert_array_equal(reward_classes(rewards),['equal','small','larger'])
    trials=AuthorsTrials(np.array([[1,2],[0,1],[1,7]]),np.array([False,True,False]),rewards)
    out=evaluate(trials,19,[.2,1.,.3,1.],np.zeros(8),np.zeros(8))
    comp=decompose(trials,out,1,1)
    result=equal_pair_diagnostic(trials,out,comp,1,1)
    assert result['2-3']['observed']['n']==1
    assert result['2-3']['observed']['frequent_rate']==1
    assert result['4-5']['observed']['n']==0
    assert result['4-5']['observed']['calibration_error'] is None


def test_decomposition_identity_and_displayed_update_magnitudes(frame):
    fold=make_fold(frame,'p',5)
    _,out=replay_fold(fold,'ck',[.2,1.5,.3,2.],np.zeros(8),np.zeros(8))
    comp=decompose(fold.validation_trials,out,2,1.5)
    np.testing.assert_allclose(expit(comp['reward_logit']+comp['history_logit']),out.probability_left,atol=1e-14)
    update=memory_updates(fold.validation_trials,out,1.5)
    assert sum(s['presentations'] for s in update['per_stimulus'])==24
    assert update['mean_abs_weighted_update']==pytest.approx(1.5*update['mean_abs_raw_update'])


def test_confidence_bins_and_empty_bins():
    p=np.array([.5,.55,.95,1.,.1]); y=np.array([True,False,True,True,False])
    bins=calibration(p,y)
    assert sum(b['n'] for b in bins)==5
    assert bins[0]['n']==1 and bins[1]['n']==1 and bins[-1]['n']==2
    assert sum(b['correct_count'] for b in bins)==4
    pooled=pool_calibration([bins,bins])
    assert sum(b['n'] for b in pooled['bins'])==10
    assert pooled['pooled_ECE']>=0


def test_disagreement_thresholds_and_chosen_probability():
    trials=AuthorsTrials(np.tile([0,1],(4,1)),np.array([True,False,True,False]),np.tile([1.,2.],(4,1)))
    ck=np.array([.8,.2,.75,.8]); cu=np.array([.5,.5,.5,.79])
    def result(p): return SimpleNamespace(probability_left=p,log_loss=-np.log(np.where(trials.left,p,1-p)))
    result=disagreement(trials,result(ck),result(cu),terciles(4))
    assert result['0.1']['n']==3 and result['0.2']['n']==3
    assert result['0.2']['ck_assigns_higher']==3
    assert result['0.2']['ck_log_likelihood_gain']>0


def test_disagreement_includes_exact_cutoffs():
    trials=AuthorsTrials(np.tile([0,1],(2,1)),np.array([True,True]),np.tile([1.,2.],(2,1)))
    ck=SimpleNamespace(probability_left=np.array([.1,.2]),log_loss=np.ones(2))
    cum=SimpleNamespace(probability_left=np.zeros(2),log_loss=np.ones(2)*2)
    result=disagreement(trials,ck,cum,terciles(2))
    assert result['0.1']['n']==2 and result['0.2']['n']==1


def test_validation_rejects_rewardless_phase(frame):
    fold=make_fold(frame,'p',5)
    fold.validation_trials.rewards=None
    with pytest.raises(ValueError,match='training feedback'):
        replay_fold(fold,'rl',[.2,1.],np.zeros(8),np.zeros(8))


def test_participant_aggregation_gain_sign_and_bootstrap():
    rows=[]
    for i in range(3):
        rows.append(dict(participant=str(i),status='success',models={name:{'validation':dict(
            n=10*(i+1),nll=loss*10*(i+1),mean_log_loss=loss,brier=.2)}
            for name,loss in [('rl',.6),('ck',.4),('cumulative',.5)]}))
    rows.append(dict(participant='failure',status='failed'))
    a=paired_summary(rows); b=paired_summary(rows)
    assert a==b and a['failed_participants']==['failure']
    assert a['gains']['cumulative_over_ck']['distribution']['mean']==pytest.approx(-.1)
    assert a['gains']['ck_over_rl']['better']==3
    assert a['gains']['ck_over_rl']['bootstrap']['seed']==20261015
    assert a['gains']['ck_over_rl']['bootstrap']['n_participants']==3
