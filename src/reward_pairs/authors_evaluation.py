"""Benchmark execution and descriptive comparisons, not psychological inference."""
from collections import Counter

import numpy as np

from .authors_audit import CLINICAL_AGE, TASK_EXCLUSIONS
from .authors_fitting import AuthorsFitError, fit_authors, information_criteria
from .authors_models import FREE, evaluate, initial_values


def run_participant(job):
    participant, inputs, starts, models, variants, literal_pi = job
    train,test,ratings,pretest,structure=inputs
    rows=[]; fitted={}
    for variant in variants:
        family='paper_intended' if variant=='paper_intended' else 'released_code_intended'
        q0,h0=initial_values(ratings,pretest,family)
        for model in models:
            row=dict(participant=participant,variant=variant,model=model,structure=structure,
                     included=participant not in CLINICAL_AGE+TASK_EXCLUSIONS,
                     exclusion_category='clinical_or_age' if participant in CLINICAL_AGE else
                     'task_missing_accuracy_knowledge' if participant in TASK_EXCLUSIONS else None,
                     initial_q=q0.tolist(),initial_h=h0.tolist())
            key=(family,model)
            try:
                if key not in fitted:
                    fitted[key]=fit_authors([train],model,q0,h0,starts=starts[:,list(FREE[model])])
                fit=fitted[key]
                pi=literal_pi if variant=='released_code_literal_clean_r' else 0.
                training=evaluate(train,model,fit.theta,q0,h0,scoring_pi=pi)
                testing=evaluate(test,model,fit.theta,training.q[-1],training.h[-1],scoring_pi=pi)
                row.update(fit.record(model))
                row.update(scoring_pi=pi,postfit_rescored_training_deviance=training.deviance,
                           rescore_minus_objective=training.deviance-fit.objective_deviance,
                           training=information_criteria(training.deviance,model,len(train.left)),
                           test=information_criteria(testing.deviance,model,len(test.left)),
                           final_training_q=training.q[-1].tolist(),final_training_h=training.h[-1].tolist(),
                           final_test_q=testing.q[-1].tolist(),final_test_h=testing.h[-1].tolist(),
                           reused_fit=variant!=family)
            except AuthorsFitError as exc:
                row.update(optimization_success=False,error=str(exc),restart_count=len(starts))
            rows.append(row)
    return rows


def distribution(values):
    a=np.asarray(values,float)
    if not len(a): return dict(n=0)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
                sd=float(a.std(ddof=1)) if len(a)>1 else 0.,minimum=float(a.min()),maximum=float(a.max()),
                quantiles=dict(zip(('p05','p25','p75','p95'),map(float,np.quantile(a,[.05,.25,.75,.95])))))


def bic_category(delta):
    """Source delta = combined BIC - RL BIC; exact ±6 is no strong evidence."""
    if abs(delta)<=6: return 'no_strong_evidence'
    direction='RL_CK' if delta<0 else 'RL'
    return ('very_strong_' if abs(delta)>10 else 'strong_')+direction


def summarize(rows):
    summaries={}; matrices={}
    for variant in sorted({r['variant'] for r in rows}):
        all_rows=[r for r in rows if r['variant']==variant]
        good=[r for r in all_rows if r['optimization_success'] and r['included']]
        lookup={(r['participant'],r['model']):r for r in good}
        participants=sorted({r['participant'] for r in all_rows if r['included']})
        report=dict(fit_rows=len(all_rows),failed_fits=sum(not r['optimization_success'] for r in all_rows),
                    participants=len({r['participant'] for r in all_rows}),included_participants=len(participants),
                    rescore_delta=distribution([r['rescore_minus_objective'] for r in good]),
                    models={},comparisons={})
        for model in sorted({r['model'] for r in good}):
            selected=[r for r in good if r['model']==model]
            report['models'][str(model)]=dict(
                parameters={p:distribution([r['parameters'][p] for r in selected]) for p in selected[0]['parameters']},
                training_deviance=distribution([r['training']['deviance'] for r in selected]),
                test_deviance=distribution([r['test']['deviance'] for r in selected]),
                boundary_counts={p:sum(r['boundary_flags'][p] for r in selected) for p in selected[0]['boundary_flags']},
                participants_with_boundary=sum(any(r['boundary_flags'].values()) for r in selected),
                final_q_by_stimulus=[distribution([r['final_training_q'][i] for r in selected]) for i in range(8)],
                final_h_by_stimulus=[distribution([r['final_training_h'][i] for r in selected]) for i in range(8)])
        for name,models in [('preregistered',[1,3,11,19]),('exploratory',[1,3,11,19,20])]:
            complete=[p for p in participants if all((p,m) in lookup for m in models)]
            comparison=dict(complete_participants=len(complete),incomplete_participants=sorted(set(participants)-set(complete)),phases={})
            for phase in ('training','test'):
                winner={p:min(models,key=lambda m:lookup[p,m][phase]['BIC']) for p in complete}
                delta={p:lookup[p,19][phase]['BIC']-lookup[p,3][phase]['BIC'] for p in complete}
                phase_report=dict(winner_counts={str(m):sum(v==m for v in winner.values()) for m in models},
                                  participant_winners=winner,combined_minus_rl_bic=delta,
                                  rl_comparison_counts=dict(Counter(bic_category(d) for d in delta.values())),
                                  rl_comparison_distribution=distribution(list(delta.values())))
                if 20 in models:
                    reduced={p:lookup[p,20][phase]['BIC']-lookup[p,3][phase]['BIC'] for p in complete}
                    phase_report.update(reduced_minus_rl_bic=reduced,reduced_vs_rl_counts=dict(Counter(bic_category(d) for d in reduced.values())))
                comparison['phases'][phase]=phase_report
                matrices[f'{variant}/{name}/{phase}']=dict(models=models,participants=complete,
                    **{metric:[[lookup[p,m][phase][metric] for p in complete] for m in models]
                       for metric in ('AIC','AICc','BIC','deviance')},
                    LL=[[-lookup[p,m][phase]['deviance']/2 for p in complete] for m in models])
            report['comparisons'][name]=comparison
        summaries[variant]=report
    return summaries,matrices
