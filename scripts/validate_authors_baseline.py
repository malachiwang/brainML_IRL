#!/usr/bin/env python3
"""Original-source participant parity and limited synthetic matched recovery."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from reward_pairs.authors_audit import original_r_evaluate, original_r_fit, participant_inputs, random_starts
from reward_pairs.authors_models import AuthorsTrials, FREE, evaluate, initial_values
from reward_pairs.authors_fitting import fit_authors, information_criteria
from reward_pairs.data import load_rdata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fits',type=Path,required=True)
    parser.add_argument('--participant',default='E11T9A')
    parser.add_argument('--output',type=Path,default=Path('docs/authors_validation_results.json'))
    parser.add_argument('--r-optimization',action='store_true')
    parser.add_argument('--r-restarts',type=int,default=10)
    parser.add_argument('--r-models',type=int,nargs='+',choices=list(FREE),default=list(FREE))
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--refine-r-only',action='store_true',help='Diagnose existing R/Python optimizer gaps with smaller R finite differences.')
    args=parser.parse_args(); started=perf_counter()
    fits=json.loads(args.fits.read_text()); objects=load_rdata()
    participant=args.participant; train,test,ratings,pretest,_=participant_inputs(objects,participant)
    if args.refine_r_only:
        payload=json.loads(args.output.read_text())
        order=list(dict.fromkeys(objects['data_4_analysis_RP_pretest'].ID.astype(str)))
        starts=random_starts(order.index(participant)+1,args.r_restarts)
        for row in payload['original_r_optimization']:
            if abs(row['minus_python_deviance'])<=1e-5: continue
            model=row['model']; variant=row['variant']; q0,h0=initial_values(ratings,pretest,variant)
            r=original_r_fit(train,model,q0,h0,starts[:,list(FREE[model])],finite_difference_step=1e-6)
            py=next(x for x in fits['rows'] if x['participant']==participant and x['variant']==variant and x['model']==model)
            r.update(finite_difference_step=1e-6,minus_python_deviance=r['deviance']-py['optimizer_objective_deviance'])
            row['fine_difference_diagnostic']=r
            print(variant,model,r,flush=True)
        payload['refinement_runtime_seconds']=perf_counter()-started
        args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
        return
    results=[]
    fixed={1:[2.1],3:[.2,1.4],11:[.3,2.],19:[.3,2,.2,1.4],20:[.3,6,.2]}
    for row in fits['rows']:
        if row['participant']!=participant: continue
        model=row['model']; variant=row['variant']
        q0,h0=initial_values(ratings,pretest,variant)
        for label,theta in [('fixed',fixed[model]),('fitted',list(row['free_parameters'].values()))]:
            py=evaluate(train,model,theta,q0,h0,scoring_pi=row['scoring_pi'])
            r=original_r_evaluate(train,model,theta,q0,h0,scoring_pi=None if row['scoring_pi'] else 0.)
            pyt=evaluate(test,model,theta,py.q[-1],py.h[-1],scoring_pi=row['scoring_pi'])
            rt=original_r_evaluate(test,model,theta,py.q[-1],py.h[-1],scoring_pi=None if row['scoring_pi'] else 0.)
            comparison=dict(variant=variant,model=model,parameters=label,
                objective_deviance_difference=r['objective']-evaluate(train,model,theta,q0,h0).deviance,
                training_deviance_difference=r['deviance']-py.deviance,test_deviance_difference=rt['deviance']-pyt.deviance,
                probability_max_difference=max(float(np.max(abs(r['probability_left']-py.probability_left))),float(np.max(abs(rt['probability_left']-pyt.probability_left)))),
                q_max_difference=float(np.max(abs(r['q']-py.q))),h_max_difference=float(np.max(abs(r['h']-py.h))))
            results.append(comparison)
            if max(abs(comparison[k]) for k in ('objective_deviance_difference','training_deviance_difference','test_deviance_difference'))>1e-7:
                raise RuntimeError(f'R deviance parity failed: {comparison}')
        print(f'Parity {variant} {model} passed',flush=True)
    payload=dict(participant=participant,parity=results,original_r_optimization=[],synthetic=[])
    if args.r_optimization:
        # Independent starts; original objective with base optim, not unavailable optimr.
        order=list(dict.fromkeys(objects['data_4_analysis_RP_pretest'].ID.astype(str)))
        starts=random_starts(order.index(participant)+1,args.r_restarts)
        for variant in ('paper_intended','released_code_intended'):
            q0,h0=initial_values(ratings,pretest,variant)
            for model in args.r_models:
                r=original_r_fit(train,model,q0,h0,starts[:,list(FREE[model])])
                py=next(row for row in fits['rows'] if row['participant']==participant and row['variant']==variant and row['model']==model)
                r.update(variant=variant,model=model,starts=args.r_restarts,minus_python_deviance=r['deviance']-py['optimizer_objective_deviance'])
                replay=evaluate(train,model,r['theta'],q0,h0)
                r['rescored_test']={str(pi):information_criteria(evaluate(test,model,r['theta'],replay.q[-1],replay.h[-1],scoring_pi=pi).deviance,model,len(test.left)) for pi in (0.,np.pi)}
                payload['original_r_optimization'].append(r)
                print('R optim',variant,model,r['minus_python_deviance'],flush=True)
    if args.synthetic:
        # Predeclared interior settings, two per model; fresh independent histories.
        settings={3:[[.15,1.2],[.6,.8]],11:[[.15,2.],[.5,1.2]],
                  19:[[.15,1.5,.25,1.],[.4,2.,.6,.7]],20:[[.15,8.,.25],[.5,9.,.6]]}
        q0,h0=initial_values(ratings,pretest,'released_code_intended')
        for model,truths in settings.items():
            for setting,truth in enumerate(truths):
                episodes=[]
                for seed in np.random.SeedSequence([20261010,model,setting]).spawn(30):
                    sim=evaluate(train,model,truth,q0,h0,simulate_seed=seed)
                    episodes.append(AuthorsTrials(train.pairs,sim.left,train.rewards))
                fit=fit_authors(episodes,model,q0,h0,starts=random_starts(model+setting,10)[:,list(FREE[model])])
                truth_dev=sum(evaluate(e,model,truth,q0,h0).deviance for e in episodes)
                row=dict(model=model,setting=setting,independent_agents=30,trials_per_agent=len(train.left),truth=truth,
                         estimate=fit.theta.tolist(),absolute_error=np.abs(fit.theta-truth).tolist(),
                         truth_deviance=truth_dev,**fit.record(model))
                payload['synthetic'].append(row)
                if fit.objective_deviance>truth_dev+1e-6: raise RuntimeError('Fit worse than generating parameters.')
                print('Synthetic',model,setting,'truth',truth,'estimate',fit.theta,flush=True)
    payload['runtime_seconds']=perf_counter()-started
    args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(f'Saved {args.output}, {payload["runtime_seconds"]:.1f}s',flush=True)


if __name__=='__main__': main()
