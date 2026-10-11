"""Independent recurrence, derivative, causal-order and original-source checks."""
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from reward_pairs.authors_models import (
    AuthorsTrials, FREE, LIMITS, TrainingObjective, evaluate, initial_values, unpack,
)
from reward_pairs.authors_fitting import AuthorsFitError, fit_authors, information_criteria
from reward_pairs.authors_audit import REFERENCE, original_r_evaluate, phase_trials, random_starts
from reward_pairs.authors_evaluation import bic_category, distribution, run_participant, summarize


@pytest.fixture
def trials():
    return AuthorsTrials(np.array([[0,1],[1,2],[0,2],[3,1]]),
                        np.array([True,False,False,True]),
                        np.array([[1.,3.],[3.,5.],[1.,5.],[7.,3.]]))


@pytest.mark.parametrize('alpha',[0.,.3,1.])
def test_both_updates_and_limits(trials,alpha):
    q0=np.arange(8.)/10; h0=np.arange(8.)/20
    o=evaluate(trials,19,[alpha,2,alpha,1],q0,h0)
    np.testing.assert_allclose(o.q[1,:2],q0[:2]+alpha*(trials.rewards[0]-q0[:2]))
    np.testing.assert_allclose(o.h[1,:2],h0[:2]+alpha*(np.array([1,0])-h0[:2]))
    np.testing.assert_array_equal(o.q[1,2:],q0[2:])
    np.testing.assert_array_equal(o.h[1,2:],h0[2:])
    assert o.h[2,2]==pytest.approx(h0[2]+alpha*(1-h0[2]))
    assert o.h[2,1]==pytest.approx((1-alpha)*o.h[1,1])


def test_test_freezes_only_q(trials):
    o=evaluate(AuthorsTrials(trials.pairs,trials.left,None),19,[.5,2,.7,1],np.ones(8),np.zeros(8))
    np.testing.assert_array_equal(o.q,np.ones((5,8)))
    assert o.h[1,0]==.5 and o.h[2,2]==.5


def test_initialization():
    r=np.arange(8.)*10; p=np.arange(8.)/10
    q,h=initial_values(r,None,'paper_intended')
    np.testing.assert_allclose(q,r/11.15); np.testing.assert_array_equal(h,0)
    for variant in ('released_code_intended','released_code_literal_clean_r'):
        q,h=initial_values(r,p,variant)
        np.testing.assert_allclose(q,(r/20+p*5)/2)
        np.testing.assert_array_equal(h,q)
    q2,_=initial_values(r+20,p,'paper_intended')
    np.testing.assert_allclose(q2,r/11.15)


@pytest.mark.parametrize('model,k',[(1,1),(3,2),(11,2),(19,4),(20,3)])
def test_specs_bounds_and_ic(model,k):
    assert len(FREE[model])==k
    bounds=LIMITS[list(FREE[model])]
    p=unpack(model,bounds.mean(axis=1))
    if model!=1: assert p[4]==2.5
    if model in (1,3): assert p[0]==p[1]==0
    if model in (1,11): assert p[2]==p[3]==0
    if model==20: assert p[3]==10-p[1]
    with pytest.raises(ValueError): unpack(model,bounds[:,1]+.01)
    ic=information_criteria(42,model,100)
    assert ic['AIC']==42+2*k
    assert ic['AICc']==42+2*k*100/(100-k-1)
    assert ic['BIC']==42+k*np.log(100)


def test_bias_stability_and_no_leakage(trials):
    q=np.arange(8.)*1000; h=np.zeros(8)
    o=evaluate(trials,3,[0,10],q,h)
    assert np.isfinite(o.log_loss).all()
    bias=evaluate(trials,1,[3],q,h)
    np.testing.assert_allclose(bias.probability_left,expit(1))
    changed=AuthorsTrials(trials.pairs,~trials.left,trials.rewards)
    a=evaluate(trials,19,[.5,3,.3,2],q/1000,h)
    b=evaluate(changed,19,[.5,3,.3,2],q/1000,h)
    assert a.probability_left[0]==b.probability_left[0]
    changed.left[:3]=trials.left[:3]
    b=evaluate(changed,19,[.5,3,.3,2],q/1000,h)
    np.testing.assert_array_equal(a.probability_left,b.probability_left)
    np.testing.assert_array_equal(a.q,b.q)
    np.testing.assert_array_equal(a.h[:4],b.h[:4])


@pytest.mark.parametrize('model,theta',[(1,[2.1]),(3,[.2,1.4]),(11,[.3,2.]),(19,[.3,2,.2,1.4]),(20,[.3,6,.2])])
def test_filtered_objective_scalar_parity_and_gradient(trials,model,theta):
    q=np.arange(8.)/10; h=q/2
    objective=TrainingObjective([trials],model,q,h)
    dev,grad=objective(theta)
    assert dev==pytest.approx(evaluate(trials,model,theta,q,h).deviance,abs=1e-12)
    theta=np.array(theta,float); numeric=[]
    for j in range(len(theta)):
        step=np.eye(len(theta))[j]*1e-6
        numeric.append((objective(theta+step)[0]-objective(theta-step)[0])/2e-6)
    np.testing.assert_allclose(grad,numeric,atol=1e-7,rtol=1e-6)
    pooled=TrainingObjective([trials,trials],model,q,h)
    assert pooled(theta)[0]==pytest.approx(2*dev)


@pytest.mark.parametrize('alpha',[0.,1.])
def test_filter_boundaries(trials,alpha):
    p=[alpha,3,alpha,2]; q=np.arange(8.)/10; h=q/2
    assert TrainingObjective([trials],19,q,h)(p)[0]==pytest.approx(evaluate(trials,19,p,q,h).deviance)


def test_chronology_retained_only():
    f=pd.DataFrame(dict(ID=['p','p','p','other'],session=[2,1,1,1],trial=[1,4,1,1],
                        stim1=[1]*4,stim2=[2]*4,action=['left']*4,reward1=[1]*4,reward2=[3]*4))
    data,info=phase_trials(f,'p',training=True)
    assert len(data.left)==3 and not info['original_order_chronological']
    o=evaluate(data,11,[.5,2],np.zeros(8),np.zeros(8))
    np.testing.assert_allclose(o.h[:,0],[0,.5,.75,.875])
    test,_=phase_trials(f,'p',training=False)
    assert test.rewards is None
    f.loc[0,'trial']=0
    with pytest.raises(ValueError): phase_trials(f,'p',training=True)


def test_optimizer_failure_explicit(monkeypatch,trials):
    monkeypatch.setattr('reward_pairs.authors_fitting.minimize',lambda *a,**kw:SimpleNamespace(
        success=False,fun=1.,x=np.array([2.5]),message='injected',nit=0,nfev=1))
    with pytest.raises(AuthorsFitError,match='failed'):
        fit_authors([trials],1,np.zeros(8),np.zeros(8),starts=[[2.5]])


@pytest.mark.skipif(not shutil.which('Rscript') or not (REFERENCE/'02_functions4modelling.R').exists(),reason='Local unmodified R reference unavailable')
@pytest.mark.parametrize('model,theta',[(1,[2.1]),(3,[.2,1.4]),(11,[.3,2.]),(19,[.3,2,.2,1.4]),(20,[.3,6,.2])])
@pytest.mark.parametrize('pi',[0.,None])
def test_original_r_trial_parity(trials,model,theta,pi):
    q=np.arange(8.)/10; h=q/2
    r=original_r_evaluate(trials,model,theta,q,h,scoring_pi=pi)
    o=evaluate(trials,model,theta,q,h,scoring_pi=np.pi if pi is None else pi)
    assert r['objective']==pytest.approx(evaluate(trials,model,theta,q,h).deviance,abs=1e-10)
    assert r['deviance']==pytest.approx(o.deviance,abs=1e-10)
    for key in ('probability_left','q','h'):
        np.testing.assert_allclose(r[key],getattr(o,key),atol=1e-14)
    test=AuthorsTrials(trials.pairs,trials.left,None)
    r=original_r_evaluate(test,model,theta,o.q[-1],o.h[-1],scoring_pi=pi)
    o=evaluate(test,model,theta,o.q[-1],o.h[-1],scoring_pi=np.pi if pi is None else pi)
    assert r['deviance']==pytest.approx(o.deviance,abs=1e-10)
    for key in ('probability_left','q','h'):
        np.testing.assert_allclose(r[key],getattr(o,key),atol=1e-14)


def test_simulation_ignores_actions_reproducible(trials):
    a=evaluate(trials,19,[.3,2,.2,1],np.zeros(8),np.zeros(8),simulate_seed=7)
    trials.left=~trials.left
    b=evaluate(trials,19,[.3,2,.2,1],np.zeros(8),np.zeros(8),simulate_seed=7)
    np.testing.assert_array_equal(a.left,b.left)
    np.testing.assert_array_equal(a.probability_left,b.probability_left)


def test_random_fit_known_likelihood():
    left=np.array([True]*70+[False]*30)
    trials=AuthorsTrials(np.tile([0,1],(100,1)),left,np.tile([1.,2.],(100,1)))
    fit=fit_authors([trials],1,np.zeros(8),np.zeros(8),starts=[[1.],[4.]])
    assert fit.theta[0]==pytest.approx((5+np.log(7/3))/2,abs=1e-6)
    assert fit.objective_deviance==pytest.approx(-2*(70*np.log(.7)+30*np.log(.3)),abs=1e-8)


def test_test_choices_never_affect_fit_and_variants_share_fit():
    pairs=np.tile([[0,1],[1,0]],(10,1)); left=np.tile([True,False,False,True],5)
    train=AuthorsTrials(pairs,left,np.tile([1.,2.],(20,1)))
    test=AuthorsTrials(pairs,left,None)
    inputs=(train,test,np.arange(8.),np.arange(8.)/8,dict(training={},test={}))
    variants=['released_code_intended','released_code_literal_clean_r']
    a=run_participant(('x',inputs,np.array([[.2,2.,.3,1.,2.3]]),[1],variants,np.pi))
    test.left=~test.left
    b=run_participant(('x',inputs,np.array([[.2,2.,.3,1.,2.3]]),[1],variants,np.pi))
    assert a[0]['parameters']==a[1]['parameters']==b[0]['parameters']
    assert a[0]['optimizer_objective_deviance']==b[0]['optimizer_objective_deviance']
    assert abs(a[0]['rescore_minus_objective'])<1e-9
    assert abs(a[1]['rescore_minus_objective'])>1


def test_summary_counts_and_failures():
    rows=[]
    for model in (1,3,11,19,20):
        rows.append(dict(participant='x',variant='v',model=model,included=True,optimization_success=True,
            rescore_minus_objective=0,parameters={'bias':2.5},boundary_flags={'bias':False},
            final_training_q=[1]*8,final_training_h=[0]*8,
            training=information_criteria(100 if model!=19 else 50,model,100),
            test=information_criteria(100 if model!=3 else 50,model,30)))
    rows.append(dict(participant='failed',variant='v',model=19,included=True,optimization_success=False))
    summary,matrices=summarize(rows)
    s=summary['v']; assert s['failed_fits']==1
    assert s['comparisons']['preregistered']['incomplete_participants']==['failed']
    assert s['comparisons']['preregistered']['phases']['training']['winner_counts']['19']==1
    assert s['comparisons']['preregistered']['phases']['test']['winner_counts']['3']==1
    m=matrices['v/preregistered/test']; assert m['LL'][1]==[-25.]
    assert bic_category(-11)=='very_strong_RL_CK'
    assert bic_category(7)=='strong_RL'
    assert bic_category(6)=='no_strong_evidence'
    d=distribution([0,2,4]); assert d['mean']==2 and d['sd']==2 and d['median']==2


def test_invalid_training_objective(trials):
    with pytest.raises(ValueError): TrainingObjective([],1,np.zeros(8),np.zeros(8))
    with pytest.raises(ValueError): TrainingObjective([trials],1,np.zeros(7),np.zeros(8))
    with pytest.raises(ValueError): TrainingObjective([AuthorsTrials(trials.pairs,trials.left,None)],1,np.zeros(8),np.zeros(8))


@pytest.mark.skipif(not shutil.which('Rscript'),reason='R unavailable')
def test_r_start_stream_prefix():
    a=random_starts(2,3); b=random_starts(2,7)
    np.testing.assert_array_equal(a,b[:3])
    assert not np.array_equal(a,random_starts(3,3))


def test_literal_pi_positional_convention_and_phase_reset(trials):
    o=evaluate(trials,1,[2.5],np.zeros(8),np.zeros(8),scoring_pi=np.pi)
    assert o.probability_left[0]==.5
    assert o.probability_left[1]==pytest.approx(expit(2*np.pi))
    assert o.probability_left[2]==pytest.approx(expit(-2*np.pi))
    test=AuthorsTrials(trials.pairs,trials.left,None)
    t=evaluate(test,1,[2.5],o.q[-1],o.h[-1],scoring_pi=np.pi)
    assert t.probability_left[0]==.5


def test_released_pretest_only_initialization():
    from reward_pairs.authors_audit import participant_inputs
    f=pd.DataFrame(dict(ID=['p']*2,session=[1,2],trial=[1,1],stim1=[1,2],stim2=[2,1],
                        action=['left','right'],reward1=[1,2],reward2=[2,1]))
    ratings={'ID':['p']}
    for i in range(1,9):
        ratings[f's{i}_used_pre']=[i*10.]
        ratings[f's{i}_used_post']=[999.]
    objects=dict(data_4_analysis_RP_training=f,data_4_analysis_RP_test=f.copy(),
        ratings_RP2=pd.DataFrame(ratings),grouped_data_RP_pretest2_clean=pd.DataFrame(
        dict(ID=['p']*8,stim_chosen=range(1,9),perc_stim_chosen=np.arange(8.)/8)))
    first=participant_inputs(objects,'p')
    objects['data_4_analysis_RP_test'].loc[:,'action']='right'
    objects['data_4_analysis_RP_test'].loc[:,'reward1']=999
    objects['ratings_RP2'].loc[:,'s1_used_post']=-999
    second=participant_inputs(objects,'p')
    np.testing.assert_array_equal(first[0].left,second[0].left)
    for a,b in zip(first[2:4],second[2:4]): np.testing.assert_array_equal(a,b)
    assert second[1].rewards is None
