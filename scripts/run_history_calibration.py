"""Predeclared synthetic alpha coverage and myopic/planning distinguishability."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

from reward_pairs.data import load_rdata, select_training_object
from reward_pairs.history_mdp import HistoryMDP
from reward_pairs.history_model_selection import (
    aggregate_selection, evaluation_root, fit_models, generate_agents, score_fitted,
)
from reward_pairs.history_replication import batch_root, scaffold_summary, scaffold_trajectory, structural_training, experiment_grid
from reward_pairs.history_uncertainty import AlphaProfile, aggregate_uncertainty
from reward_pairs.recovery import RecoveryError


def choice_hash(frame):
    return sha256(pd.util.hash_pandas_object(frame,index=False).to_numpy().tobytes()).hexdigest()


def run_batch(job):
    trajectory, alpha, seed, sizes, do_uncertainty, do_selection, options = job
    participant = trajectory[0].participant
    root = batch_root(participant,seed)
    eval_root = evaluation_root(participant,seed)
    mdp = HistoryMDP.from_trials(trajectory)
    start = perf_counter()
    uncertainty, selection = [],[]
    for generator in (('myopic','planning') if do_selection else ('myopic',)):
        data = generate_agents(trajectory,max(sizes),generator=generator,alpha=alpha,seed=root,**options['model'])
        fresh = (generate_agents(trajectory,options['fresh_agents'],generator=generator,alpha=alpha,
                                seed=eval_root,evaluation=True,**options['model']) if do_selection else None)
        for count in sorted(sizes):
            choices = data[data.rollout<count].copy()
            config = dict(participant=participant,alpha_true=alpha,rollouts=count,seed_batch=seed,
                          training_rng_root=root,choices_sha256=choice_hash(choices))
            if generator=='myopic' and do_uncertainty:
                t = perf_counter()
                profile=None
                try:
                    profile=AlphaProfile(choices,beta=options['model']['beta'])
                    interval=profile.interval()
                    row=dict(**config,status='ok',interval=interval)
                except RecoveryError as exc:
                    row=dict(**config,status='FAILED',error=str(exc),stage='profile' if profile is not None else 'unrestricted')
                row.update(seconds=perf_counter()-t,conditional_fits_attempted=profile.attempts if profile is not None else 0)
                uncertainty.append(row)
            if do_selection:
                t=perf_counter()
                fits,records=fit_models(mdp,choices,**options['model'])
                scores=score_fitted(mdp,fits,choices,fresh)
                selection.append(dict(**config,generator=generator,fresh_rng_root=eval_root,
                    fresh_agents=options['fresh_agents'],fresh_sha256=choice_hash(fresh),fits=records,scores=scores,
                    seconds=perf_counter()-t))
    return dict(uncertainty=uncertainty,selection=selection,seconds=perf_counter()-start,
                participant=participant,alpha_true=alpha,seed_batch=seed)


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',choices=['all','uncertainty','distinguishability'],default='all')
    parser.add_argument('--data',type=Path,default=root/'data/raw/02_comp_mod_RP_task_data_in.RData')
    parser.add_argument('--prior-results',type=Path,default=root/'docs/history_replication_results.json')
    parser.add_argument('--participants',nargs='+',help='Explicit smoke-test subset of the stored list; default all 12')
    parser.add_argument('--beta',type=float,default=1.)
    parser.add_argument('--gamma',type=float,default=.95)
    parser.add_argument('--planning-horizon',type=int,default=3)
    parser.add_argument('--alphas',type=float,nargs='+',default=[0.,.5,1.])
    parser.add_argument('--rollouts',type=int,nargs='+',default=[1,5,20])
    parser.add_argument('--seed-batches',type=int,nargs='+',default=list(range(5)),help='Model comparison batches')
    parser.add_argument('--uncertainty-seed-batches',type=int,nargs='+',default=list(range(10)))
    parser.add_argument('--fresh-eval-agents',type=int,default=20)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--output',type=Path,default=root/'docs/history_calibration_results.json')
    args=parser.parse_args()
    from reward_pairs.history_planning import check_planning
    check_planning(args.beta,args.gamma,args.planning_horizon)
    if args.workers<1 or args.fresh_eval_agents<1:
        parser.error('workers and fresh evaluation agents must be positive.')
    started=perf_counter()
    prior=json.loads(args.prior_results.read_text())
    participants=args.participants or [s['participant'] for s in prior['scaffolds']]
    if not set(participants)<=set(s['participant'] for s in prior['scaffolds']):
        parser.error('Requested scaffold not in previous stored selection.')
    # Reuse existing grid validation, but do not reselect or rank participants.
    for seeds in (args.seed_batches,args.uncertainty_seed_batches):
        experiment_grid(participants,args.alphas,args.rollouts,seeds)
    name,source=select_training_object(load_rdata(args.data))
    structural=structural_training(source)
    trajectories={p:scaffold_trajectory(structural,p) for p in participants}
    scaffolds=[scaffold_summary(t) for t in trajectories.values()]
    previous={s['participant']:s for s in prior['scaffolds']}
    if any(s!=previous[s['participant']] for s in scaffolds):
        raise ValueError('Scaffold structure differs from previous results; inspect before proceeding.')
    source_hash=sha256(args.data.read_bytes()).hexdigest()
    if source_hash!=prior['source_sha256']:
        raise ValueError('Source file hash differs from the replication; inspect rather than silently proceeding.')
    do_u=args.study in ('all','uncertainty')
    do_s=args.study in ('all','distinguishability')
    seeds=sorted(set(args.uncertainty_seed_batches if do_u else [])|set(args.seed_batches if do_s else []))
    options=dict(model=dict(beta=args.beta,gamma=args.gamma,horizon=args.planning_horizon),fresh_agents=args.fresh_eval_agents)
    jobs=[(trajectories[p],a,s,args.rollouts,do_u and s in args.uncertainty_seed_batches,
           do_s and s in args.seed_batches,options) for p in participants for a in args.alphas for s in seeds]
    payload=dict(schema_version=1,complete=False,command=[sys.executable,*sys.argv],study=args.study,
        beta=args.beta,gamma=args.gamma,horizon=args.planning_horizon,alphas=args.alphas,rollouts=args.rollouts,
        uncertainty_seed_batches=args.uncertainty_seed_batches,selection_seed_batches=args.seed_batches,
        fresh_eval_agents=args.fresh_eval_agents,workers=args.workers,training_object=name,source_sha256=source_hash,
        prior_results_sha256=sha256(args.prior_results.read_bytes()).hexdigest(),scaffolds=scaffolds,
        selection_rule='Reuse ordered scaffold list and validate structure from previous replication JSON; no reselection.',
        rng='Training batch_root(participant,batch) unchanged from replication; evaluation SeedSequence([training_root,0x4556414C]) -> uint64 root; spawn separate agents. Nested counts and common draws across alpha/generator conditions; evaluation never overlaps training.',
        windows='early=first20 retained; mid=rest of sessions1-4; late=session5; all=union. Also report session1 remainder and sessions2-4.',
        selection_rule_nll='delta=NLL_myopic-NLL_planning; positive favors planning; abs(delta)<=1e-8 is a numerical tie. Alpha0 is null, excluded from classification accuracy.',
        profile_rule='95% chi2_1 LR; seven nuisance utilities refit; geometric brackets to +/-64 from MLE; Brent roots xtol1e-7; unresolved sides open, not finite endpoints.',
        environment={p:version(p) for p in ('numpy','pandas','scipy','rdata','pytest')},
        code_sha256={p.name:sha256(p.read_bytes()).hexdigest() for p in sorted((root/'src/reward_pairs').glob('*.py'))},
        uncertainty=[],selection=[],batches=[])
    print('Scaffolds:',[(s['participant'],s['retained_trials']) for s in scaffolds],flush=True)
    print('Predeclared jobs:',len(jobs),'uncertainty datasets:',len(participants)*len(args.alphas)*len(args.rollouts)*len(args.uncertainty_seed_batches) if do_u else 0,
          'selection datasets:',len(participants)*len(args.alphas)*len(args.rollouts)*len(args.seed_batches)*2 if do_s else 0,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)

    def accept(batch):
        payload['uncertainty'].extend(batch.pop('uncertainty'))
        payload['selection'].extend(batch.pop('selection'))
        payload['batches'].append(batch)
        print('Completed',len(payload['batches']),'/',len(jobs),batch,'wall',perf_counter()-started,flush=True)
        # Periodic checkpoint; complete flag remains false until all jobs.
        args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')

    if args.workers==1:
        for job in jobs:
            accept(run_batch(job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_batch,j) for j in jobs]
            for future in as_completed(futures):
                accept(future.result())
    for key in ('uncertainty','selection'):
        payload[key].sort(key=lambda r:(participants.index(r['participant']),r['alpha_true'],r['seed_batch'],r.get('generator',''),r['rollouts']))
    payload['uncertainty_summary']=aggregate_uncertainty(payload['uncertainty'],participants)
    payload['selection_summary']=aggregate_selection(payload['selection'],participants)
    payload.update(complete=True,completed_utc=datetime.now(timezone.utc).isoformat(),seconds=perf_counter()-started)
    args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print('Uncertainty:',payload['uncertainty_summary']['totals'],flush=True)
    print('Model fits:',payload['selection_summary']['fits_attempted'],'failures:',payload['selection_summary']['fit_failures'],flush=True)
    print('Total seconds:',perf_counter()-started,'output:',args.output,flush=True)
    if payload['uncertainty_summary']['totals']['failed'] or payload['selection_summary']['fit_failures']:
        parser.exit(1,'Failures retained; inspect calibration report before interpreting successful-only summaries.\n')


if __name__=='__main__':
    main()
