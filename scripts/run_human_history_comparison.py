#!/usr/bin/env python3
"""Training-only cumulative-history fits against frozen, validated RL/CK fits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import scipy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from reward_pairs.authors_audit import participant_inputs
from reward_pairs.data import DEFAULT_DATA_PATH, load_rdata
from reward_pairs.human_history_evaluation import (
    TIE_TOLERANCE, compare_participant, history_summary, summarize, training_history_diagnostics,
)
from reward_pairs.human_history_report import write_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA_PATH)
    parser.add_argument('--authors-results',type=Path,default=Path('docs/authors_baseline_results.json'))
    parser.add_argument('--participant',nargs='+')
    parser.add_argument('--restarts',type=int,default=200)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=20261015)
    parser.add_argument('--bootstrap-draws',type=int,default=10000)
    parser.add_argument('--paper-sensitivity',action='store_true')
    parser.add_argument('--stability-check',action='store_true',help='Shortest, median, longest retained length ranks plus E11T9A; no behavior-based selection.')
    parser.add_argument('--output',type=Path,default=Path('docs/human_history_results.json'))
    args=parser.parse_args()
    if args.restarts<1 or args.workers<1 or args.bootstrap_draws<1 or args.seed<0:
        parser.error('Positive restarts/workers/bootstrap draws and nonnegative seed required.')
    if args.stability_check and args.participant:
        parser.error('Use either deterministic stability subset or participant selection.')
    start=perf_counter()
    references=json.loads(args.authors_results.read_text())
    raw_hash=sha256(args.data.read_bytes()).hexdigest()
    if raw_hash!=references['audit']['source_sha256']:
        raise ValueError('Raw data do not match the validated benchmark; do not silently reuse fits.')
    objects=load_rdata(args.data)
    order=list(dict.fromkeys(objects['data_4_analysis_RP_pretest'].ID.astype(str)))
    if set(order)!=set(objects['data_4_analysis_RP_training'].ID.astype(str)) or set(order)!=set(objects['data_4_analysis_RP_test'].ID.astype(str)):
        raise ValueError('Phase participant sets differ; no exclusions are permitted.')
    participants=order if args.participant is None else args.participant
    if args.stability_check:
        counts=objects['data_4_analysis_RP_training'].groupby('ID',observed=True).size()
        ranked=sorted(order,key=lambda p:(int(counts[p]),p))
        participants=list(dict.fromkeys([ranked[0],ranked[(len(ranked)-1)//2],ranked[-1],'E11T9A']))
    if len(set(participants))!=len(participants) or any(p not in order for p in participants):
        parser.error('Need distinct existing participant IDs.')
    variants=['released_code_intended']+(['paper_intended'] if args.paper_sensitivity else [])
    lookup={(r['participant'],r['variant'],r['model']):r for r in references['rows']}
    source_paths=list(Path('src/reward_pairs').glob('authors_*.py'))
    author_hashes={str(p):sha256(p.read_bytes()).hexdigest() for p in source_paths}
    inputs={p:participant_inputs(objects,p) for p in participants}
    histories={p:training_history_diagnostics(inputs[p][0]) for p in participants}
    history=history_summary(list(histories.values()))
    print(f'{len(participants)} participants, {args.restarts} starts, {variants}. Authors parameters reused, pi=0.',flush=True)
    print('BEFORE FITTING: final retained-training history diagnostics',json.dumps(history['equal_pairs']),flush=True)
    jobs=[(p,order.index(p)+1,inputs[p],{(v,m):lookup[p,v,m] for v in variants for m in (3,19)},
           variants,args.restarts,args.seed) for p in participants]
    rows=[]
    executor=ProcessPoolExecutor(args.workers) if args.workers>1 else None
    results=executor.map(compare_participant,jobs) if executor else map(compare_participant,jobs)
    try:
        for index,result in enumerate(results,1):
            rows.extend(result)
            print(f'{index}/{len(jobs)} {result[0]["participant"]}: {sum(r["fit_status"]!="success" for r in result)} failures, {perf_counter()-start:.1f}s',flush=True)
    finally:
        if executor: executor.shutdown()
    if author_hashes!={str(p):sha256(p.read_bytes()).hexdigest() for p in source_paths}:
        raise RuntimeError('Authors implementation changed during experiment.')
    stability=[]
    for row in rows:
        if row['fit_status']!='success': continue
        fit=row['models']['cumulative']
        minima={}
        for n in (10,20,50,100,200):
            if n>args.restarts: continue
            values=[d for d,s in zip(fit['restart_deviances'][:n],fit['restart_success'][:n]) if s]
            minima[str(n)]=min(values) if values else None
        stability.append(dict(participant=row['participant'],variant=row['variant'],prefix_minima=minima,
                              full_minimum=fit['objective_deviance']))
    summary=summarize(rows,bootstrap_seed=args.seed,bootstrap_draws=args.bootstrap_draws)
    payload=dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        protocol=dict(variants=variants,fit='five training sessions only',evaluation='final test, no refitting; causal one-step prediction',
                      history_initialization='zero chosen/presented counts; no pretest counts',
                      difference_sign='positive delta = cumulative better; LL_cum-LL_ref = NLL_ref-NLL_cum',
                      numerical_tie_nats_per_choice=TIE_TOLERANCE,descriptive_magnitude_nats_per_choice=.01,
                      bootstrap='paired participant percentile bootstrap, 95%; participant means equally weighted',
                      optional_shared_weight='deferred; joint nuisance optimization is a separate secondary analysis'),
        provenance=dict(raw_sha256=raw_hash,authors_results_sha256=sha256(args.authors_results.read_bytes()).hexdigest(),
                        authors_source_sha256=author_hashes,python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
        participant_order=participants,history_summary=history,rows=rows,summary=summary,
        restart_stability=stability,runtime_seconds=perf_counter()-start)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    write_report(payload,args.output.with_suffix('.md'))
    for v,s in summary.items():
        print(v,{m:s['models'][m]['participant_test_log_loss']['mean'] for m in s['models']},flush=True)
        for name,c in s['comparisons'].items():
            print('vs',name,'mean gain',c['delta_test_log_loss']['mean'],'CI',c['bootstrap']['ci95'],
                  'better/worse/tied',c['cumulative_better'],c['cumulative_worse'],c['ties'],flush=True)
    print(f'Saved {args.output} and Markdown report; {payload["runtime_seconds"]:.2f}s.',flush=True)


if __name__=='__main__': main()
