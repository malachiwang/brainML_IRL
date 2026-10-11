#!/usr/bin/env python3
"""Audit or fit the separate authors benchmark; real actions are intentional here."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from reward_pairs.authors_audit import participant_inputs, random_starts, source_audit
from reward_pairs.authors_evaluation import run_participant, summarize
from reward_pairs.authors_models import VARIANTS
from reward_pairs.data import DEFAULT_DATA_PATH, load_rdata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA_PATH)
    parser.add_argument('--variant',choices=(*VARIANTS,'all'),default='released_code_intended')
    parser.add_argument('--models',type=int,nargs='+',choices=[1,3,11,19,20],default=[1,3,11,19,20])
    parser.add_argument('--participant',nargs='+')
    parser.add_argument('--restarts',type=int,default=200)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--start-rng',choices=['R','numpy'],default='R')
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('docs/authors_baseline_results.json'))
    args=parser.parse_args()
    if args.restarts<1 or args.workers<1: parser.error('Positive restarts/workers required.')
    started=perf_counter(); objects=load_rdata(args.data)
    audit=source_audit(objects,args.data)
    if args.audit_only:
        print(json.dumps(audit,indent=2)); return
    order=list(dict.fromkeys(objects['data_4_analysis_RP_pretest'].ID.astype(str)))
    if set(order)!=set(objects['data_4_analysis_RP_training'].ID.astype(str)) or set(order)!=set(objects['data_4_analysis_RP_test'].ID.astype(str)):
        parser.error('Pretest/training/test ID sets differ: inspect explicitly rather than silently omitting participants.')
    participants=order if args.participant is None else args.participant
    if len(set(participants))!=len(participants) or any(p not in order for p in participants):
        parser.error('Participants must be distinct IDs present in source pretest order.')
    variants=list(VARIANTS) if args.variant=='all' else [args.variant]
    models=list(dict.fromkeys(args.models))
    print(f'Authors benchmark: {len(participants)} participants, {models}, {variants}, {args.restarts} starts; pi audit = {audit["literal_scoring_pi"]}',flush=True)
    jobs=[(p,participant_inputs(objects,p),random_starts(order.index(p)+1,args.restarts,use_r=args.start_rng=='R'),
           models,variants,audit['literal_scoring_pi']) for p in participants]
    rows=[]
    if args.workers==1:
        results=map(run_participant,jobs)
        executor=None
    else:
        executor=ProcessPoolExecutor(args.workers); results=executor.map(run_participant,jobs)
    try:
        for i,result in enumerate(results,1):
            rows.extend(result)
            print(f'{i}/{len(jobs)} {result[0]["participant"]}: {sum(not r["optimization_success"] for r in result)} failed fits; elapsed {perf_counter()-started:.1f}s',flush=True)
    finally:
        if executor: executor.shutdown()
    summary,matrices=summarize(rows)
    payload=dict(configuration=vars(args)|{'data':str(args.data),'output':str(args.output)},
                 protocol='200-start, R-draw / SciPy-optimizer reproduction' if args.restarts==200 else 'accelerated reproduction (restart convergence must be separately validated)',
                 runtime_seconds=perf_counter()-started,audit=audit,participant_source_order=order,
                 optimizer=dict(method='SciPy L-BFGS-B',ftol=1e-12,gtol=1e-7,maxiter=1000,maxls=40,
                                difference='R optimr unavailable; analytic gradients and explicit convergence checking'),
                 rows=rows,summary=summary,vba_export_matrices=matrices,vba_exceedance_reproduced=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    for variant,report in summary.items():
        print(variant, json.dumps({name:{phase:details['winner_counts'] for phase,details in comp['phases'].items()}
                                  for name,comp in report['comparisons'].items()}))
    print(f'Saved {args.output}; {payload["runtime_seconds"]:.1f}s. Benchmark reproduction only; no novel human model fitted.')


if __name__=='__main__': main()
