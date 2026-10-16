#!/usr/bin/env python3
"""Training-only rolling-origin diagnostics. Final-test scoring is forbidden."""
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
from reward_pairs.data import DEFAULT_DATA_PATH, load_rdata
from reward_pairs.history_diagnostics import permitted_inputs, run_participant, training_initialization
from reward_pairs.history_diagnostics_summary import summarize
from reward_pairs.history_diagnostics_report import write_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA_PATH)
    parser.add_argument('--participant',nargs='+')
    parser.add_argument('--sessions',type=int,nargs='+',choices=(2,3,4,5),default=[2,3,4,5])
    parser.add_argument('--primary-restarts',type=int,default=200)
    parser.add_argument('--rolling-restarts',type=int,default=50)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--start-rng',choices=('r','numpy'),default='r')
    parser.add_argument('--stability-check',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('docs/history_diagnostics_results.json'))
    parser.add_argument('--trace-output',type=Path)
    parser.add_argument('--report-only',action='store_true',help='Render existing JSON; no data loading or fitting.')
    args=parser.parse_args()
    if args.report_only:
        write_report(json.loads(args.output.read_text()),args.output.with_suffix('.md')); return
    if min(args.primary_restarts,args.rolling_restarts,args.workers)<1 or len(set(args.sessions))!=len(args.sessions):
        parser.error('Positive starts/workers and distinct validation sessions required.')
    if args.stability_check and args.participant:
        parser.error('Stability participants are selected structurally, not manually.')
    start=perf_counter()
    objects,order=permitted_inputs(load_rdata(args.data))
    frame=objects['data_4_analysis_RP_training']
    participants=order if not args.participant else args.participant
    if args.stability_check:
        counts=frame.groupby('ID',observed=True).size()
        ranked=sorted(order,key=lambda p:(int(counts[p]),p))
        participants=list(dict.fromkeys([ranked[0],ranked[(len(ranked)-1)//2],ranked[-1],'E11T9A']))
        args.sessions=[5]
        if args.primary_restarts!=200: parser.error('Stability check requires 200 primary starts.')
    if len(set(participants))!=len(participants) or any(p not in order for p in participants):
        parser.error('Need distinct supplied participants.')
    protected=list(Path('src/reward_pairs').glob('authors_*.py'))+[Path('src/reward_pairs/cumulative_history_human.py')]
    hashes={str(p):sha256(p.read_bytes()).hexdigest() for p in protected}
    jobs=[]
    for p in participants:
        q0,h0=training_initialization(objects,p)
        jobs.append((p,order.index(p)+1,frame.loc[frame.ID.astype(str).eq(p)].copy(),q0,h0,
                     args.sessions,args.primary_restarts,args.rolling_restarts,args.start_rng=='r'))
    subset=frame.loc[frame.ID.astype(str).isin(participants)]
    pair_counts={f'{int(a)}-{int(b)}':int(n) for (a,b),n in
                 subset.assign(lo=np.minimum(subset.stim1.astype(int),subset.stim2.astype(int)),
                               hi=np.maximum(subset.stim1.astype(int),subset.stim2.astype(int))).groupby(['lo','hi']).size().items()}
    gaps=abs(subset.reward1-subset.reward2)
    support=dict(training_rows=len(subset),participants=len(participants),
                 session_counts={str(k):int(v) for k,v in subset.groupby('session',observed=True).size().items()},
                 unordered_pairs=pair_counts,reward_gap_counts={str(k):int(v) for k,v in gaps.value_counts().items()})
    print('PREDECLARED: S1..S(t-1) -> St; sessions',args.sessions,'participants',len(participants),flush=True)
    print('Primary starts',args.primary_restarts,'rolling starts',args.rolling_restarts,'RNG',args.start_rng,flush=True)
    print('Structural support',json.dumps(support),'FINAL TEST FORBIDDEN',flush=True)
    rows=[]; traces={}
    executor=ProcessPoolExecutor(args.workers) if args.workers>1 else None
    results=executor.map(run_participant,jobs) if executor else map(run_participant,jobs)
    try:
        for i,(records,arrays) in enumerate(results,1):
            rows.extend(records); traces.update(arrays)
            print(f'{i}/{len(jobs)} {records[0]["participant"]}: {sum(r["status"]!="success" for r in records)} failed folds; {perf_counter()-start:.1f}s',flush=True)
    finally:
        if executor: executor.shutdown()
    if hashes!={str(p):sha256(p.read_bytes()).hexdigest() for p in protected}:
        raise RuntimeError('Validated model implementation changed during run.')
    summary=summarize(rows)
    args.trace_output=args.trace_output or Path('data/derived')/(args.output.stem+'_traces.npz')
    args.trace_output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.trace_output,**traces)
    result=dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        design=dict(variant='released_code_intended',scoring_pi=0,
            gain='gain_A_over_B = log_loss_B - log_loss_A; positive favors A',
            bootstrap_seed=20261015,bootstrap_draws=10000,bootstrap_unit='participant',tie_tolerance=1e-8,
            phases='np.array_split chronological retained indices into three near-equal counts',
            calibration='favored-option confidence, .50:.05:1.00 bins; .5 favors left; last bin closed',
            reward_categories='equal=0, small=1 supplied reward unit, larger>1; never fabricate absent categories',
            ratio_denominator_threshold=1e-6,disagreement_thresholds=[.1,.2],
            equal_pair_probes='counterfactual state contrasts only; no simulated action, update, or observed preference',
            fixed_parameter_memory='S1-S4-fitted parameters replay S1-S5; S1-S4 portions descriptive, not held out',
            final_test='RData container deserialized; final-test object never accessed by pipeline; phase and session guards'),
        provenance=dict(raw_sha256=sha256(args.data.read_bytes()).hexdigest(),protected_source_sha256=hashes,
                        python=sys.version,numpy=np.__version__,scipy=scipy.__version__),
        support=support,participants=participants,rows=rows,summary=summary,
        trace_archive=dict(path=str(args.trace_output),sha256=sha256(args.trace_output.read_bytes()).hexdigest(),
                           schema='row.trace_prefix + _field; model h_before/q_before have eight stimulus columns; positions are retained order'),
        runtime_seconds=perf_counter()-start)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    write_report(result,args.output.with_suffix('.md'))
    print('Completed',summary['fit_counts'],'runtime',result['runtime_seconds'],flush=True)
    for session,s in summary['sessions'].items():
        print('S'+session,{m:r['participant_log_loss']['mean'] for m,r in s['models'].items()},flush=True)
        print({k:(v['distribution']['mean'],v['bootstrap']['ci95']) for k,v in s['gains'].items()},flush=True)


if __name__=='__main__': main()
