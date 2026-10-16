"""Paired human predictive comparisons; participants are the resampling units."""
import numpy as np

from .authors_models import evaluate as authors_evaluate, initial_values
from .authors_evaluation import distribution
from .cumulative_history_human import (
    CumulativeFitError, criteria, evaluate, fit, history_features, random_starts,
)

EQUAL_PAIRS = ((2,3,3), (4,5,4), (6,7,7))  # diagnostic expected frequent member only
TIE_TOLERANCE = 1e-8  # nats per choice, numerical tie, NOT practical equivalence


def training_history_diagnostics(train):
    h, state = history_features(train)
    return dict(final_h=h[-1].tolist(), chosen=state.chosen.tolist(), presented=state.presented.tolist(),
                equal_pairs={f'{a}-{b}':dict(frequent=f, rare=a if f==b else b,
                    final_difference=float(h[-1,f-1]-h[-1,(a if f==b else b)-1])) for a,b,f in EQUAL_PAIRS})


def history_summary(records):
    return dict(participants=len(records), final_h_by_stimulus=[distribution([r['final_h'][i] for r in records]) for i in range(8)],
        equal_pairs={key:dict(difference=distribution([r['equal_pairs'][key]['final_difference'] for r in records]),
            expected_order_count=sum(r['equal_pairs'][key]['final_difference']>0 for r in records),
            tie_count=sum(r['equal_pairs'][key]['final_difference']==0 for r in records),
            expected_order_fraction=float(np.mean([r['equal_pairs'][key]['final_difference']>0 for r in records])))
            for key in ('2-3','4-5','6-7')})


def equal_pair_scores(test, output):
    result = {}
    for a,b,f in EQUAL_PAIRS:
        mask = np.all(np.sort(test.pairs,axis=1)==np.array([a-1,b-1]),axis=1)
        left_frequent = test.pairs[mask,0]==f-1
        observed = np.where(left_frequent,test.left[mask],~test.left[mask])
        predicted = np.where(left_frequent,output.probability_left[mask],1-output.probability_left[mask])
        rare = a if f==b else b
        result[f'{a}-{b}'] = dict(n=int(mask.sum()), frequent=f, observed_frequent_count=int(observed.sum()),
            predicted_frequent_sum=float(predicted.sum()), nll=float(output.log_loss[mask].sum()),
            q_frequent_minus_rare=float(output.q[0,f-1]-output.q[0,rare-1]),
            mean_abs_probability_from_half=float(np.mean(abs(predicted-.5))) if len(predicted) else None,
            mean_history_frequent_minus_rare=float(np.mean(output.h[:-1][mask,f-1]-output.h[:-1][mask,rare-1])) if len(predicted) else None)
    return result


def replay_reference(train, test, q0, h0, record):
    """Verify the saved benchmark on these exact inputs; never refit or use pi."""
    if record['variant'] not in ('released_code_intended','paper_intended') or record['scoring_pi']!=0:
        raise ValueError('Only consistent pi=0 references are permitted.')
    if not record['optimization_success'] or not record['included']:
        raise ValueError('Reference fit unsuccessful or excluded; inspect, do not silently omit.')
    np.testing.assert_allclose(q0,record['initial_q'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(h0,record['initial_h'],rtol=0,atol=1e-12)
    theta=list(record['free_parameters'].values())
    tr=authors_evaluate(train,record['model'],theta,q0,h0)
    te=authors_evaluate(test,record['model'],theta,tr.q[-1],tr.h[-1])
    for phase,output in [('training',tr),('test',te)]:
        if abs(output.deviance-record[phase]['deviance'])>1e-7:
            raise ValueError(f'Saved {phase} reference fails exact-row replay.')
        if record[phase]['n']!=len(output.left):
            raise ValueError('Benchmark row count mismatch.')
    return tr,te


def compare_participant(job):
    participant,index,inputs,references,variants,restarts,seed=job
    train,test,ratings,pretest,structure=inputs
    diagnostics=training_history_diagnostics(train)
    rows=[]
    for variant in variants:
        q0,h0=initial_values(ratings,pretest,variant)
        row=dict(participant=participant,variant=variant,structure=structure,history=diagnostics,models={})
        predictions={}
        for name,model in [('rl',3),('ck',19)]:
            saved=references[variant,model]
            tr,te=replay_reference(train,test,q0,h0,saved)
            predictions[name]=te
            row['models'][name]=dict(parameters=saved['parameters'],training=criteria(tr.deviance,len(train.left),2 if model==3 else 4),
                test=criteria(te.deviance,len(test.left),2 if model==3 else 4),
                final_q=tr.q[-1].tolist(),final_h=tr.h[-1].tolist(),equal_pairs=equal_pair_scores(test,te),
                reference=dict(file='docs/authors_baseline_results.json',participant=participant,variant=variant,model=model,
                               restart_count=saved['restart_count'],boundary_flags=saved['boundary_flags']))
        # Only training and initialization enter this call. No test observations,
        # held-out scores, benchmark parameters, or truth are optimizer inputs.
        try:
            fitted=fit([train],q0,starts=random_starts(index,restarts,seed))
            theta=list(fitted['parameters'].values())
            tr=evaluate(train,theta,q0)
            te=evaluate(test,theta,tr.q[-1],initial_history=tr.final_history)
            if abs(fitted['objective_deviance']-tr.deviance)>1e-7:
                raise RuntimeError('Fast objective and independent causal replay disagree.')
            predictions['cumulative']=te
            fitted.update(training=criteria(tr.deviance,len(train.left)),test=criteria(te.deviance,len(test.left)),
                initial_q=q0.tolist(),final_q=tr.q[-1].tolist(),final_h=tr.h[-1].tolist(),
                final_test_h=te.h[-1].tolist(),final_test_chosen=te.final_history.chosen.tolist(),
                final_test_presented=te.final_history.presented.tolist(),equal_pairs=equal_pair_scores(test,te))
            row['models']['cumulative']=fitted
            row['fit_status']='success'
            row['comparisons']={}
            for name in ('rl','ck'):
                other=row['models'][name]
                difference=other['test']['mean_log_loss']-fitted['test']['mean_log_loss']
                pdiff=abs(predictions[name].probability_left-te.probability_left)
                row['comparisons'][name]=dict(delta_test_LL=other['test']['nll']-fitted['test']['nll'],
                    delta_test_log_loss=difference,
                    winner='tie' if abs(difference)<=TIE_TOLERANCE else 'cumulative' if difference>0 else name,
                    mean_abs_probability_difference=float(pdiff.mean()),max_probability_difference=float(pdiff.max()),
                    probability_difference_over_01_fraction=float(np.mean(pdiff>.01)),
                    probability_difference_over_05_fraction=float(np.mean(pdiff>.05)))
        except CumulativeFitError as exc:
            row.update(fit_status='failed',error=str(exc),attempts=exc.attempts)
        rows.append(row)
    return rows


def participant_bootstrap(values, *, seed=20261015, draws=10000):
    a=np.asarray(values,float)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all() or draws<1:
        raise ValueError('Bootstrap requires finite participant differences and positive draws.')
    rng=np.random.default_rng(seed)
    means=np.empty(draws)
    for offset in range(0,draws,500):
        n=min(500,draws-offset)
        means[offset:offset+n]=a[rng.integers(0,len(a),size=(n,len(a)))].mean(axis=1)
    return dict(unit='participant',method='paired percentile bootstrap, 95%',seed=seed,draws=draws,
                n_participants=len(a),mean=float(a.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
                standard_error=float(means.std(ddof=1)) if draws>1 else 0.,replicate_means=means.tolist())


def summarize(rows, *, bootstrap_seed=20261015, bootstrap_draws=10000):
    output={}
    for variant in sorted({r['variant'] for r in rows}):
        all_rows=[r for r in rows if r['variant']==variant]
        good=[r for r in all_rows if r['fit_status']=='success']
        summary=dict(participants=len(all_rows),failed_participants=[r['participant'] for r in all_rows if r['fit_status']!='success'],
                     complete_pairs=len(good),models={},comparisons={},equal_pairs={})
        if not good:
            output[variant]=summary
            continue
        for name in ('rl','ck','cumulative'):
            models=[r['models'][name] for r in good]
            n=sum(m['test']['n'] for m in models)
            nll=sum(m['test']['nll'] for m in models)
            summary['models'][name]=dict(total_test_nll=nll,test_choices=n,trial_weighted_test_log_loss=nll/n,
                participant_test_log_loss=distribution([m['test']['mean_log_loss'] for m in models]),
                training_nll=distribution([m['training']['nll'] for m in models]),
                training_total_nll=sum(m['training']['nll'] for m in models))
        for name in ('rl','ck'):
            comp=[r['comparisons'][name] for r in good]
            d=[c['delta_test_log_loss'] for c in comp]
            summary['comparisons'][name]=dict(delta_test_log_loss=distribution(d),
                delta_test_LL=distribution([c['delta_test_LL'] for c in comp]),
                total_delta_test_LL=sum(c['delta_test_LL'] for c in comp),
                cumulative_better=sum(c['winner']=='cumulative' for c in comp),
                cumulative_worse=sum(c['winner']==name for c in comp),ties=sum(c['winner']=='tie' for c in comp),
                gain_over_001_count=sum(v>.01 for v in d),loss_over_001_count=sum(v<-.01 for v in d),
                bootstrap=participant_bootstrap(d,seed=bootstrap_seed,draws=bootstrap_draws),
                mean_abs_probability_difference=distribution([c['mean_abs_probability_difference'] for c in comp]))
        values=[r['models']['cumulative']['parameters']['beta_h'] for r in good]
        summary['beta_h']=dict(distribution=distribution(values),positive=sum(v>1e-8 for v in values),
                              negative=sum(v< -1e-8 for v in values),near_zero=sum(abs(v)<=1e-8 for v in values))
        summary['boundary_counts']={p:{side:sum(r['models']['cumulative']['boundary_flags'][p][side] for r in good)
                                      for side in ('lower','upper')} for p in ('alpha_rl','beta_rl','beta_h')}
        summary['failed_starts']=sum(r['models']['cumulative']['failed_starts'] for r in good)
        for key in ('2-3','4-5','6-7'):
            pair={}
            for name in ('rl','ck','cumulative'):
                records=[r['models'][name]['equal_pairs'][key] for r in good]
                valid=[r for r in records if r['n']]
                n=sum(r['n'] for r in valid)
                pair[name]=dict(n=n,participants=len(valid),observed_frequent_rate=sum(r['observed_frequent_count'] for r in valid)/n if n else None,
                    predicted_frequent_rate=sum(r['predicted_frequent_sum'] for r in valid)/n if n else None,
                    log_loss=sum(r['nll'] for r in valid)/n if n else None,
                    participant_prediction=distribution([r['predicted_frequent_sum']/r['n'] for r in valid]),
                    q_difference=distribution([r['q_frequent_minus_rare'] for r in valid]))
            summary['equal_pairs'][key]=pair
        output[variant]=summary
    return output
