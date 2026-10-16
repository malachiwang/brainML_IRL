"""Participant-level inference and descriptive aggregation for training folds."""
import numpy as np
from scipy.stats import pearsonr, spearmanr

from .authors_evaluation import distribution
from .human_history_evaluation import participant_bootstrap, TIE_TOLERANCE
from .history_diagnostics import MODELS, PHASES, COMPARISONS


def paired_summary(rows, phase=None, bootstrap=True):
    complete=[r for r in rows if r['status']=='success']
    result=dict(total_participants=len(rows),complete_participants=len(complete),
                failed_participants=[r['participant'] for r in rows if r['status']!='success'],
                models={},gains={})
    if not complete: return result
    def scores(row,name):
        m=row['models'][name]
        return m['validation'] if phase is None else m['phases'][phase]
    for name in MODELS:
        ss=[scores(r,name) for r in complete]
        n=sum(s['n'] for s in ss); nll=sum(s['nll'] for s in ss)
        result['models'][name]=dict(participant_log_loss=distribution([s['mean_log_loss'] for s in ss]),
            participant_brier=distribution([s['brier'] for s in ss]), n=n,nll=nll,
            trial_weighted_log_loss=nll/n)
    for a,b in COMPARISONS:
        gain=np.array([scores(r,b)['mean_log_loss']-scores(r,a)['mean_log_loss'] for r in complete])
        record=dict(distribution=distribution(gain),better=int(np.sum(gain>TIE_TOLERANCE)),
                    worse=int(np.sum(gain< -TIE_TOLERANCE)),tied=int(np.sum(abs(gain)<=TIE_TOLERANCE)))
        if bootstrap:
            ci=participant_bootstrap(gain,seed=20261015,draws=10000)
            ci.pop('replicate_means')  # draws reproducible; save full precision endpoints, not 10k repetitions
            record['bootstrap']=ci
        result['gains'][f'{a}_over_{b}']=record
    winners={name:0 for name in MODELS}; ties=0
    for row in complete:
        values=np.array([scores(row,name)['mean_log_loss'] for name in MODELS])
        best=np.flatnonzero(values<=values.min()+TIE_TOLERANCE)
        if len(best)==1: winners[MODELS[best[0]]]+=1
        else: ties+=1
    result['best_model_counts']=winners; result['best_model_ties']=ties
    return result


def pool_calibration(records):
    bins=[]
    for i in range(10):
        n=sum(r[i]['n'] for r in records)
        p=sum(r[i]['confidence_sum'] for r in records)
        y=sum(r[i]['correct_count'] for r in records)
        bins.append(dict(lower=.5+.05*i,upper=.5+.05*(i+1),n=n,
            mean_confidence=p/n if n else None,accuracy=y/n if n else None,
            overconfidence=(p-y)/n if n else None))
    n=sum(b['n'] for b in bins)
    return dict(bins=bins,pooled_ECE=sum(b['n']*abs(b['overconfidence']) for b in bins if b['n'])/n if n else None,
        participant_ECE=distribution([sum(abs(b['confidence_sum']-b['correct_count']) for b in r)/sum(b['n'] for b in r) for r in records]))


def pool_disagreement(rows,cutoff):
    records=[r['disagreement'][str(cutoff)] for r in rows]
    if not records: return dict(n=0)
    n=sum(r['n'] for r in records); total=sum(r['denominator'] for r in records)
    out={k:sum(r[k] for r in records) for k in ('n','denominator','ck_assigns_higher','cumulative_assigns_higher','ties','ck_log_likelihood_gain')}
    out.update(fraction=n/total,ck_higher_fraction=out['ck_assigns_higher']/n if n else None,
               ck_gain_per_disagreement=out['ck_log_likelihood_gain']/n if n else None)
    for k in ('phases','pair_counts','reward_gap_counts'):
        keys=sorted({key for r in records for key in r[k]})
        out[k]={key:sum(r[k].get(key,0) for r in records) for key in keys}
    return out


def finite_correlation(x,y):
    valid=[(a,b) for a,b in zip(x,y) if a is not None and np.isfinite(a) and np.isfinite(b)]
    if len(valid)<3: return dict(n=len(valid),pearson=None,spearman=None)
    a,b=np.array(valid).T
    if np.ptp(a)==0 or np.ptp(b)==0: return dict(n=len(valid),pearson=None,spearman=None)
    return dict(n=len(valid),pearson=float(pearsonr(a,b).statistic),spearman=float(spearmanr(a,b).statistic))


def summarize(rows):
    out=dict(sessions={},fixed_primary_parameter_memory={},restart_stability=[])
    for session in sorted({r['session'] for r in rows}):
        group=[r for r in rows if r['session']==session]
        good=[r for r in group if r['status']=='success']
        summary=paired_summary(group)
        summary['phases']={p:paired_summary(group,p,bootstrap=session==5) for p in PHASES}
        summary['calibration']={name:pool_calibration([r['models'][name]['calibration'] for r in good]) for name in MODELS} if good else {}
        summary['disagreement']={str(c):pool_disagreement(good,c) for c in (.10,.20)}
        summary['reward_classes']={}
        summary['equal_pairs']={}
        summary['contributions']={}; summary['memory_updates']={}
        for category in ('equal','small','larger'):
            summary['reward_classes'][category]={}
            for name in MODELS:
                rr=[r['models'][name]['reward_classes'][category] for r in good]
                present=[r for r in rr if r['n']]
                n=sum(r['n'] for r in present)
                summary['reward_classes'][category][name]=dict(n=n,
                    participant_log_loss=distribution([r['mean_log_loss'] for r in present]),
                    pooled_log_loss=sum(r['nll'] for r in present)/n if n else None)
        for name in MODELS:
            records=[r['models'][name] for r in good]
            summary['contributions'][name]={key:distribution([r['contributions'][key]['mean'] for r in records])
                                            for key in ('raw_history_delta','abs_reward','abs_history')}
            summary['contributions'][name]['participant_mean_history_reward_ratio']=distribution(
                [r['contributions']['history_to_reward_ratio']['mean'] for r in records if r['contributions']['history_to_reward_ratio']['n']])
            summary['contributions'][name]['ratio_excluded_trials']=sum(r['contributions']['ratio_excluded'] for r in records)
            summary['contributions'][name]['history_dominates_fraction']=distribution([r['contributions']['history_dominates_fraction'] for r in records])
            if name!='rl':
                summary['memory_updates'][name]={key:distribution([r['validation_memory_updates'][key] for r in records])
                    for key in ('mean_abs_raw_update','mean_abs_weighted_update')}
            for key in ('2-3','4-5','6-7'):
                rr=[r['equal_pairs'][key] for r in records]
                observed=[r['observed'] for r in rr if r['observed']['n']]
                n=sum(r['n'] for r in observed)
                summary['equal_pairs'].setdefault(key,{})[name]=dict(observed_n=n,
                    observed_frequent_rate=sum(r['n']*r['frequent_rate'] for r in observed)/n if n else None,
                    observed_comparison='unavailable: pair never displayed' if not n else 'available',
                    observed_metrics={k:distribution([r[k] for r in observed]) for k in
                        ('predicted_frequent','mean_log_loss','calibration_error','observed_abs_from_half','predicted_mean_abs_from_half')},
                    unobserved_probes={k:distribution([r['unobserved_state_probe'][k] for r in rr]) for k in
                        ('raw_history_delta','weighted_history','reward_logit','mean_probability','mean_abs_probability_from_half','history_dominates_fraction')})
        alphas=[r['models']['ck']['theta'][0] for r in good]
        halves=[r['models']['ck']['ck_half_life']['updates'] for r in good]
        gain=[r['models']['cumulative']['validation']['mean_log_loss']-r['models']['ck']['validation']['mean_log_loss'] for r in good]
        summary['ck_timescale']=dict(alpha=distribution(alphas),finite_half_life=distribution([h for h in halves if h is not None]),
            infinite_count=sum(h is None for h in halves),alpha_one_count=sum(a==1 for a in alphas),
            alpha_ck_vs_ck_gain=finite_correlation(alphas,gain),half_life_vs_ck_gain=finite_correlation(halves,gain))
        summary['parameters']={name:{key:distribution([r['models'][name]['optimization']['parameters'][key] for r in good])
            for key in (good[0]['models'][name]['optimization']['parameters'] if good else [])} for name in MODELS}
        out['sessions'][str(session)]=summary
    primary=[r for r in rows if r['session']==5 and r['status']=='success']
    for s in range(1,6):
        out['fixed_primary_parameter_memory'][str(s)]={}
        for name in ('ck','cumulative'):
            rr=[r['models'][name]['fixed_primary_parameter_memory'][str(s)] for r in primary]
            out['fixed_primary_parameter_memory'][str(s)][name]={k:distribution([r[k] for r in rr])
                for k in ('mean_abs_raw_update','mean_abs_weighted_update')}
            out['fixed_primary_parameter_memory'][str(s)][name]['per_stimulus']=[
                dict(stimulus=i+1,raw=distribution([r['per_stimulus'][i]['mean_abs_raw_update'] for r in rr
                    if r['per_stimulus'][i]['presentations']]),
                     weighted=distribution([r['per_stimulus'][i]['mean_abs_weighted_update'] for r in rr
                    if r['per_stimulus'][i]['presentations']])) for i in range(8)]
    for row in rows:
        for name,model in row['models'].items():
            if model['status']!='success': continue
            opt=model['optimization']; minima={}
            for n in (10,20,50,100,200):
                if n>opt['restart_count']: continue
                values=[v for v,ok in zip(opt['restart_deviances'][:n],opt['restart_success'][:n]) if ok]
                minima[str(n)]=min(values) if values else None
            out['restart_stability'].append(dict(participant=row['participant'],session=row['session'],model=name,
                prefix_minima=minima,best=min(v for v,ok in zip(opt['restart_deviances'],opt['restart_success']) if ok)))
    out['fit_counts']=dict(attempted=len(rows)*3,failed=sum(m['status']!='success' for r in rows for m in r['models'].values()),
        successful_starts=sum(sum(m['optimization']['restart_success']) for r in rows for m in r['models'].values() if m['status']=='success'),
        unsuccessful_starts=sum(m['optimization']['failed_starts'] for r in rows for m in r['models'].values() if m['status']=='success'))
    # Paired within-person phase/session trend contrasts, descriptive (no model selection).
    for session,s in out['sessions'].items():
        rr=[r for r in rows if r['session']==int(session) and r['status']=='success']
        trend=[]
        for r in rr:
            gain=lambda phase:r['models']['ck']['phases'][phase]['mean_log_loss']-r['models']['cumulative']['phases'][phase]['mean_log_loss']
            trend.append(gain('late')-gain('early'))
        s['late_minus_early_cumulative_over_ck_gain']=distribution(trend)
    return out
