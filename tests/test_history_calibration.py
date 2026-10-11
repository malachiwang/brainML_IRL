from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from reward_pairs.history import causal_history
from reward_pairs.history_mdp import HistoryMDP
from reward_pairs.history_model_selection import (
    aggregate_selection, confusion_matrix, evaluation_root, fit_models, generate_agents,
    paired_score, score_fitted, window_masks,
)
from reward_pairs.history_replication import batch_root, structural_training, scaffold_trajectory
from reward_pairs.history_uncertainty import (
    AlphaProfile, aggregate_uncertainty, contains, interval_outcomes, lr_endpoints,
)
from reward_pairs.recovery import RecoveryError
from reward_pairs.trajectories import Trial


@pytest.fixture(scope='module')
def scaffold():
    pairs=list(combinations(range(1,9),2))*2
    rng=np.random.default_rng(31)
    return [Trial('E11T9A',s,i+1,int(a),int(b),None) for s in range(1,6)
            for i,(a,b) in enumerate(np.array(pairs)[rng.permutation(len(pairs))])]


@pytest.fixture(scope='module')
def choices(scaffold):
    return generate_agents(scaffold,10,generator='myopic',alpha=.5,seed=71)


def test_profile_center_endpoints_and_fixed_alpha(choices):
    p=AlphaProfile(choices)
    assert p.lr(p.fit.alpha)==pytest.approx(0,abs=1e-5)
    interval=p.interval()
    assert contains(interval,p.fit.alpha)
    for end in ('lower','upper'):
        assert interval[end]['lr']==pytest.approx(chi2.ppf(.95,1),abs=1e-5)
    for a in (p.fit.alpha-1,p.fit.alpha+.7):
        c=p.conditional(a)
        assert c['alpha']==a and len(c['free_utilities'])==7
        assert p.lr(a)>=0
        # Independent likelihood calculation and nonzero nuisance adjustment.
        theta=np.r_[c['free_utilities'],a]
        assert np.logaddexp(0,-p.signed@theta).sum()==pytest.approx(c['nll'],abs=1e-9)
        assert not np.allclose(c['free_utilities'],p.initial)


def test_reference_invariance_and_no_truth_inputs(choices):
    one=AlphaProfile(choices)
    eight=AlphaProfile(choices.assign(alpha_true=999,reward1=-999,action='poison'),reference=8)
    for a in (-.2,.5,1.2):
        assert one.lr(a)==pytest.approx(eight.lr(a),abs=1e-5)
        c=eight.conditional(a)
        w=np.zeros(8); w[eight.free]=c['free_utilities']
        assert w[7]==0
    x,y=one.interval(),eight.interval()
    assert x['lower']['value']==pytest.approx(y['lower']['value'],abs=1e-5)
    assert x['upper']['value']==pytest.approx(y['upper']['value'],abs=1e-5)


def test_more_independent_agents_narrower_interval(choices):
    small=AlphaProfile(choices[choices.rollout<1]).interval()
    large=AlphaProfile(choices).interval()
    assert small['width']>2*large['width']


def test_adaptive_brackets_and_open_sides():
    # Exact quadratic profile with known endpoints, not implementation outputs.
    ends=lr_endpoints(lambda a:(a-2)**2/4,2,cutoff=4,step=.01)
    assert [e['value'] for e in ends]==pytest.approx([-2,6])
    ends=lr_endpoints(lambda a:a*a,0,cutoff=4,max_distance=.1)
    assert all(e['value'] is None and e['status']=='open_search_limit' for e in ends)
    ends=lr_endpoints(lambda a:a*a if a>0 else 1-np.exp(a),0,cutoff=4,max_distance=10)
    assert ends[0]['value'] is None and ends[1]['value']==pytest.approx(2)
    with pytest.raises(RecoveryError,match='not zero'):
        lr_endpoints(lambda a:1,0,cutoff=4)


def test_profile_optimizer_failure_explicit(choices,monkeypatch):
    p=AlphaProfile(choices)
    monkeypatch.setattr('reward_pairs.history_uncertainty.minimize',lambda *a,**k:SimpleNamespace(
        success=False,message='deliberate failure'))
    with pytest.raises(RecoveryError,match='Conditional profile optimization failed'):
        p.interval()


def test_conditional_alpha_zero_agrees_with_independent_bt_fit(choices):
    from reward_pairs.recovery import recover_utilities
    profile=AlphaProfile(choices)
    fixed=profile.conditional(0.)
    bt=recover_utilities(choices)
    np.testing.assert_allclose(fixed['free_utilities'],bt.utilities.iloc[1:],atol=1e-5,rtol=0)
    assert fixed['nll']==pytest.approx(bt.negative_log_likelihood,abs=1e-7)


def test_real_profile_search_limit_is_not_a_fake_endpoint(choices):
    p=AlphaProfile(choices)
    result=p.interval(max_distance=1e-3)
    assert result['width'] is None
    assert result['lower']['value'] is None and result['upper']['value'] is None
    assert result['lower']['searched_to']==pytest.approx(p.fit.alpha-1e-3)
    assert result['upper']['lr_at_search_limit']<result['cutoff']


def test_nonfinite_conditional_optimizer_rejected(choices,monkeypatch):
    profile=AlphaProfile(choices)
    monkeypatch.setattr('reward_pairs.history_uncertainty.minimize',lambda *a,**k:SimpleNamespace(
        success=True,fun=np.nan,x=np.zeros(7),message='nonfinite stub'))
    with pytest.raises(RecoveryError,match='Conditional profile optimization failed'):
        profile.conditional(0)


def interval(lo,hi,hat=0):
    return dict(lower={'value':lo},upper={'value':hi},alpha_hat=hat,width=hi-lo if None not in (lo,hi) else None,
                wald_lower=-1,wald_upper=1)


def test_coverage_zero_and_wrong_direction_open_and_failure_aggregation():
    rows=[dict(participant='p',alpha_true=.5,rollouts=1,status='ok',interval=i) for i in
          (interval(.1,.9,.5),interval(-2,-.1,-1),interval(None,1,.1))]
    rows.append(dict(participant='p',alpha_true=.5,rollouts=1,status='FAILED'))
    result=aggregate_uncertainty(rows,['p','absent'])
    s=result['totals']
    assert s['attempted']==4 and s['successful']==3 and s['failed']==1
    assert s['covered_rate']==2/3 and s['positive_detection_rate']==1/3
    assert s['negative_detection_rate']==1/3 and s['contains_zero_rate']==1/3
    assert s['open_rate']==1/3 and s['closed_interval_width']['missing']==1
    assert result['scaffold_conditions'][1]['attempted']==0
    assert interval_outcomes(interval(None,None),-100)['covered']


def test_controlled_coverage_sanity(scaffold):
    # Broad check, not a claim of calibrated 95% coverage from only 20 batches.
    covered=0
    for seed in range(20):
        data=generate_agents(scaffold,10,generator='myopic',alpha=.5,seed=200+seed)
        covered+=contains(AlphaProfile(data).interval(),.5)
    assert covered>=16


def test_rng_fresh_history_disjoint_ids_and_null_generators(scaffold):
    train=batch_root('E11T9A',2); test=evaluation_root('E11T9A',2)
    assert train!=test and test==evaluation_root('E11T9A',2)
    roots=[evaluation_root(p,s) for p in ('p','q') for s in range(10)]
    assert len(set(roots))==20
    a=generate_agents(scaffold,3,generator='myopic',alpha=0,seed=train)
    b=generate_agents(scaffold,3,generator='planning',alpha=0,seed=train)
    pd.testing.assert_frame_equal(a,b)
    fresh=generate_agents(scaffold,3,generator='myopic',alpha=0,seed=test,evaluation=True)
    assert not set(a.rollout)&set(fresh.rollout)
    assert not np.array_equal(a.synthetic_action,fresh.synthetic_action)
    h=causal_history(fresh).groupby('rollout').first()
    assert h.left_history.eq(0).all() and h.right_history.eq(0).all()


def test_windows_retained_indices_partition(choices):
    altered=choices.copy(); altered['trial']*=3  # gaps do not alter position
    masks=window_masks(altered)
    assert masks['early'].sum()==20*10
    assert masks['late'].sum()==56*10
    np.testing.assert_array_equal(masks['early'].astype(int)+masks['mid']+masks['late'],1)
    np.testing.assert_array_equal(masks['mid'],masks['session1_remainder']|masks['sessions2_4'])


def test_delta_winners_confusion_and_null_control():
    a=pd.DataFrame(dict(rollout=[0,0],session=[1,1],trial=[1,2],probability_left=[.2,.4],log_loss=[1.,2.]))
    b=a.copy(); b['log_loss']=[.9,1.9]; b['probability_left']=[.3,.6]
    score=paired_score(a,b,[True,True])
    assert score['delta_nll']==pytest.approx(.2) and score['selected']=='planning'
    assert score['different_modal_action']==.5 and score['probability_difference_gt_05']==1
    assert paired_score(b,a,[True,True])['selected']=='myopic'
    assert paired_score(a,a,[True,True])['selected']=='tie'
    rows=[dict(generator=g,alpha_true=alpha,scores={'fresh_all':score}) for g,alpha in
          [('myopic',1),('planning',1),('planning',0)]]
    matrix=confusion_matrix(rows,'fresh_all')
    assert matrix['planning']['counts']['planning']==1 and matrix['planning']['paired']==1
    assert matrix['myopic']['counts']['planning']==1


def test_identical_data_to_both_fitters_and_no_generator_or_truth(monkeypatch,choices,scaffold):
    seen=[]
    fake=SimpleNamespace(alpha=.1,utilities=pd.Series(np.arange(8)),negative_log_likelihood=1,n_trials=len(choices),information_condition=1)
    def myopic(data,**kwargs):
        seen.append((data,kwargs)); return fake
    def planning(mdp,data,**kwargs):
        seen.append((data,kwargs)); return fake
    monkeypatch.setattr('reward_pairs.history_model_selection.recover_history',myopic)
    monkeypatch.setattr('reward_pairs.history_model_selection.recover_planning',planning)
    fit_models(HistoryMDP.from_trials(scaffold),choices)
    assert len(seen)==4 and all(data is choices for data,_ in seen)
    assert all(not {'generator','alpha_true','utilities'}&set(k) for _,k in seen)


def test_scoring_no_refit_and_no_future_choice_leakage(choices,scaffold,monkeypatch):
    mdp=HistoryMDP.from_trials(scaffold)
    fits,_=fit_models(mdp,choices)
    before={k:(f.alpha,f.utilities.copy()) for k,f in fits.items()}
    fresh=generate_agents(scaffold,3,generator='planning',alpha=.5,seed=456,evaluation=True)
    def forbidden(*a,**kw):
        raise AssertionError('Scoring must not refit')
    monkeypatch.setattr('reward_pairs.history_model_selection.recover_history',forbidden)
    monkeypatch.setattr('reward_pairs.history_model_selection.recover_planning',forbidden)
    first=score_fitted(mdp,fits,choices,fresh)
    changed=fresh.copy()
    pos=changed.groupby('rollout').cumcount()
    changed.loc[pos>=20,'synthetic_action']='right'
    second=score_fitted(mdp,fits,choices,changed)
    assert first['fresh_early']==second['fresh_early']
    for k,f in fits.items():
        assert before[k][0]==f.alpha
        pd.testing.assert_series_equal(before[k][1],f.utilities)


def test_calibration_projection_reads_only_structure(scaffold):
    class GuardedFrame(pd.DataFrame):
        def __getitem__(self,key):
            assert isinstance(key,list) and set(key)=={'ID','session','trial','stim1','stim2'}
            return super().__getitem__(key)
    source=GuardedFrame([(t.participant,t.session,t.trial,*t.state,'poison',np.nan) for t in scaffold],
        columns=['ID','session','trial','stim1','stim2','action','reward_chosen'])
    trajectory=scaffold_trajectory(structural_training(source),'E11T9A')
    assert all(t.action is None for t in trajectory)
    a=generate_agents(trajectory,1,generator='planning',alpha=1,seed=9)
    b=generate_agents(scaffold,1,generator='planning',alpha=1,seed=9)
    pd.testing.assert_frame_equal(a,b)


def test_selection_failures_retained_for_each_scaffold():
    fits={f'{scope}_{model}':dict(status='FAILED',error='intentional') for scope in ('all','train') for model in ('myopic','planning')}
    row=dict(participant='p',generator='myopic',alpha_true=0.,rollouts=1,seed_batch=0,fits=fits,scores={})
    summary=aggregate_selection([row],['p','absent'])
    assert summary['fits_attempted']==summary['fit_failures']==4
    assert summary['conditions'][0]['missing']==1
    assert summary['confusion_matrices']==[]
    assert any(s['participant']=='absent' and s['attempted']==0 for s in summary['scaffold_conditions'])
