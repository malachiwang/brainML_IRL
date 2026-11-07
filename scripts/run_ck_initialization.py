#!/usr/bin/env python3
"""Experiment 4: choice-kernel initialization (K0) conditions of the authors' RL+CK, S1-S4 -> S5.

Training-only: only the training and initialization tables pass ``permitted_inputs``;
the final-test table is never read.  The C0 refit is verified against the saved
S1-S4 fits (docs/history_diagnostics_results.json).  Per-participant results are
cached so an interrupted run resumes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from reward_pairs import ck_initialization as ci  # noqa: E402
from reward_pairs.ck_alpha_profile import RANDOM_STARTS, START_SEED  # noqa: E402
from reward_pairs.data import load_rdata  # noqa: E402
from reward_pairs.history_diagnostics import make_fold, permitted_inputs, training_initialization  # noqa: E402

TOLERANCE = dict(refit_vs_saved_deviance=1e-6, refit_vs_saved_theta=1e-4, carried_q=1e-6, carried_k=1e-4, s5_nll=1e-6)


def build_jobs(data, folds, participants, n_starts, use_r, profile):
    objects, order = permitted_inputs(load_rdata(data))  # training + initialization tables only
    frame = objects['data_4_analysis_RP_training']
    s5 = {r['participant']: r for r in folds['rows'] if r['session'] == 5}
    jobs = []
    for p in participants:
        saved = s5[p]['models']['ck']
        if saved['status'] != 'success':
            raise ValueError(f'Saved S5-fold fit missing for {p}; inspect, do not omit.')
        fold = make_fold(frame.loc[frame.ID.astype(str).eq(p)].copy(), p, 5)
        q0, k0 = training_initialization(objects, p)
        if not np.array_equal(q0, k0):
            raise ValueError('Released initialization should set K0 = Q0.')
        jobs.append((p, order.index(p)+1, fold, q0, k0, saved, n_starts, use_r, profile))
    return jobs


def run(jobs, workers, cache, tag):
    cache.mkdir(parents=True, exist_ok=True)
    rows, todo = {}, []
    for job in jobs:
        f = cache/f'{job[0]}.json'
        if f.exists() and json.loads(f.read_text()).get('config_tag') == tag:
            rows[job[0]] = json.loads(f.read_text())['row']
        else:
            todo.append(job)
    start = perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        for i, row in enumerate(pool.map(ci.fit_participant, todo), 1):
            (cache/f'{row["participant"]}.json').write_text(json.dumps(dict(config_tag=tag, row=row), allow_nan=False))
            rows[row['participant']] = row
            if i % 20 == 0 or i == len(todo):
                print(f'  {i}/{len(todo)} participants, {perf_counter()-start:.0f}s', flush=True)
    log_file = cache/'compute_log.json'
    log = json.loads(log_file.read_text()) if log_file.exists() else []
    if todo:
        log.append(dict(config_tag=tag, participants_computed=len(todo), workers=workers, seconds=perf_counter()-start))
        log_file.write_text(json.dumps(log, indent=2))
    return [rows[j[0]] for j in jobs], [e for e in log if e['config_tag'] == tag]


def verify(rows):
    worst = {k: max(r['checks'][k] for r in rows) for k in TOLERANCE}
    failed = {k: v for k, v in worst.items() if v > TOLERANCE[k]}
    if not all(r['checks']['q0_identical'] for r in rows):
        failed['q0_identical'] = False
    if not all(r['conditions'][c]['carried_ok'] for r in rows for c in ci.CONDITIONS if r['conditions'][c]['status'] == 'ok'):
        failed['carried_states'] = False
    if failed:
        raise RuntimeError(f'C0 reproduction failed {failed}; conditions NOT interpreted.')
    return dict(worst=worst, tolerance=TOLERANCE)


def write_csv(path, rows):
    out = []
    for r in rows:
        row = dict(participant=r['participant'], n_train=r['n_train'], n_validation=r['n_validation'],
                   released_k0_min=min(r['k0']['C0']), released_k0_max=max(r['k0']['C0']))
        for c in ci.CONDITIONS:
            x = r['conditions'][c]
            row[f'{c}_status'] = x['status']
            if x['status'] != 'ok':
                continue
            row.update({f'{c}_{k}': v for k, v in x['parameters'].items()})
            row.update({f'{c}_at_bound_{k.replace("authors_", "")}': v for k, v in x['boundary'].items()})
            row.update({f'{c}_train_nll': x['train_nll'], f'{c}_train_loss': x['train_loss'], f'{c}_s1_early_loss': x['s1_early_loss'],
                        f'{c}_s5_loss': x['s5_loss'], f'{c}_reaching_best': x['reaching_best']})
            row.update({f'{c}_S{s}_loss': v for s, v in x['session_loss'].items()})
            for seg in ci.SEGMENTS:
                key = seg.replace(' ', '_')
                row[f'{c}_{key}_median_abs_zq'] = x['segments'][seg]['median_abs_zq']
                row[f'{c}_{key}_median_abs_zk'] = x['segments'][seg]['median_abs_zk']
        for c in ('C1', 'C2'):
            if r['conditions'][c]['status'] == r['conditions']['C0']['status'] == 'ok':
                row[f'{c}_minus_C0_alpha_ck'] = r['conditions'][c]['parameters']['alpha_ck']-r['conditions']['C0']['parameters']['alpha_ck']
                row[f'{c}_minus_C0_s5_loss'] = r['conditions'][c]['s5_loss']-r['conditions']['C0']['s5_loss']
                row[f'{c}_minus_C0_train_nll'] = r['conditions'][c]['train_nll']-r['conditions']['C0']['train_nll']
        out.append(row)
    keys = list(dict.fromkeys(k for r in out for k in r))
    with open(path, 'w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=keys); w.writeheader()
        for r in out:
            w.writerow({k: '' if r.get(k) is None else (f'{r[k]:.10g}' if isinstance(r.get(k), float) else r[k]) for k in keys})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/raw/02_comp_mod_RP_task_data_in.RData'))
    parser.add_argument('--fold-results', type=Path, default=Path('docs/history_diagnostics_results.json'))
    parser.add_argument('--participant', nargs='+', help='Pilot subset; full cohort otherwise.')
    parser.add_argument('--starts', type=int, default=200, help='R-seeded starts per condition (saved C0 protocol: 200).')
    parser.add_argument('--start-rng', choices=('r', 'numpy'), default='r')
    parser.add_argument('--no-profile', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cache-dir', type=Path, default=Path('data/derived/experiment4_ck_initialization'))
    parser.add_argument('--output-dir', type=Path, default=Path('docs'))
    parser.add_argument('--prefix', default='experiment4_ck_initialization')
    args = parser.parse_args()
    start = perf_counter()
    folds = json.loads(args.fold_results.read_text())
    raw_hash = sha256(args.data.read_bytes()).hexdigest()
    if raw_hash != folds['provenance']['raw_sha256']:
        raise ValueError('Raw data differ from the data behind the saved fits.')
    if folds['configuration']['primary_restarts'] != args.starts or folds['configuration']['start_rng'] != args.start_rng:
        print('NOTE: start protocol differs from the saved C0 fits; C0 reproduction may not be exact.', flush=True)
    protected = sorted(Path('src/reward_pairs').glob('authors_*.py'))
    hashes = {str(p): sha256(p.read_bytes()).hexdigest() for p in protected}
    order = list(dict.fromkeys(r['participant'] for r in folds['rows'] if r['session'] == 5))
    participants = args.participant or order
    if any(p not in order for p in participants) or len(set(participants)) != len(participants):
        parser.error('Unknown or duplicate participants.')
    profile = not args.no_profile
    tag = f'starts={args.starts};rng={args.start_rng};profile={profile};grid={ci.PROFILE_GRID};prof_random={RANDOM_STARTS};seed={START_SEED}'
    rows, compute = run(build_jobs(args.data, folds, participants, args.starts, args.start_rng == 'r', profile),
                        args.workers, args.cache_dir, tag)
    if hashes != {str(p): sha256(p.read_bytes()).hexdigest() for p in protected}:
        raise RuntimeError('Validated authors implementation changed during the run.')
    verification = verify(rows)
    summary = ci.summarize(rows)
    result = dict(
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        design=dict(model='authors model 19 (RL+CK), released Q0, scoring pi=0, bias fixed 2.5',
                    conditions={c: ci.LABELS[c] for c in ci.CONDITIONS}, only_difference='initial kernel vector K0',
                    fold='fit S1-S4 training choices; score S5 by causal replay with carried Q and K (previously inspected validation)',
                    starts=f'{args.starts} R-seeded starts per condition (seed 1234+17*index; identical across conditions) '
                           'plus warm starts from the other two conditions\' optima',
                    optimizer='authors_fitting.fit_authors (L-BFGS-B, ftol 1e-12, gtol 1e-7, maxiter 1000)',
                    s1_early=f'first {ci.EARLY_FRACTION:.3g} of each participant\'s S1 trials',
                    profile=dict(grid=list(ci.PROFILE_GRID), conditions=['C0', 'C1'], participants='all (cost permitted full cohort)',
                                 starts=f'saved/refit nuisance + {RANDOM_STARTS} random (seed {START_SEED}) + warm sweeps (E3 protocol)',
                                 reference_alpha=.15),
                    effect_sign='alternative minus C0; positive = alternative worse', threshold=ci.THRESHOLD,
                    final_test='never read: permitted_inputs exposes only training/initialization tables'),
        provenance=dict(raw_sha256=raw_hash, fold_results_sha256=sha256(args.fold_results.read_bytes()).hexdigest(),
                        protected_source_sha256=hashes, python=sys.version.split()[0], numpy=np.__version__,
                        scipy=scipy.__version__, platform=platform.platform()),
        verification=verification, compute_log=compute, summary=summary, runtime_seconds=perf_counter()-start)
    out = args.output_dir; (out/'figures').mkdir(parents=True, exist_ok=True)
    if args.participant:
        out = args.cache_dir/'pilot_output'; (out/'figures').mkdir(parents=True, exist_ok=True)
    (out/f'{args.prefix}.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    write_csv(out/f'{args.prefix}_participants.csv', rows)
    from reward_pairs import ck_initialization_report as report
    report.render(result, rows, out, args.prefix)
    a = summary['alpha']; v = summary['validation']
    print(f'n={summary["analysed"]}/{summary["participants"]}; alpha median ' +
          ', '.join(f'{c} {a[c]["alpha"]["median"]:.4f}' for c in ci.CONDITIONS) +
          '; S5 delta ' + ', '.join(f'{c} {v[c]["mean"]:+.5f} [{v[c]["ci95"][0]:+.5f},{v[c]["ci95"][1]:+.5f}]' for c in ('C1', 'C2')) +
          f'; verification {verification["worst"]}; runtime {result["runtime_seconds"]:.0f}s')


if __name__ == '__main__':
    main()
