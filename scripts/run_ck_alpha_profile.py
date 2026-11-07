#!/usr/bin/env python3
"""Experiment 3: profile likelihood of alpha_CK in the authors' RL+CK (S1-S4 -> S5).

Training-only: the RData container is deserialized, but only the training and
initialization tables pass ``permitted_inputs``; the final-test table is never
read.  Saved unconstrained S1-S4 fits (docs/history_diagnostics_results.json)
are verified and reused, never refitted.  Per-participant results are cached so
an interrupted run resumes.
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
from reward_pairs import ck_alpha_profile as cap  # noqa: E402
from reward_pairs.data import load_rdata  # noqa: E402
from reward_pairs.history_diagnostics import make_fold, permitted_inputs, training_initialization  # noqa: E402

TOLERANCE = dict(saved_fit_deviance=1e-7, objective_vs_replay=1e-7, saved_s5_nll=1e-9)
ANCHOR_WORSE_TOL = 1e-6  # constrained fit at the saved alpha may not be worse than the saved optimum


def build_jobs(data, folds, participants, n_random):
    objects, order = permitted_inputs(load_rdata(data))  # training + initialization tables only
    frame = objects['data_4_analysis_RP_training']
    s5 = {r['participant']: r for r in folds['rows'] if r['session'] == 5}
    jobs = []
    for p in participants:
        if s5[p]['status'] != 'success' or s5[p]['models']['ck']['status'] != 'success':
            raise ValueError(f'Saved S5-fold fit missing for {p}; inspect, do not omit.')
        fold = make_fold(frame.loc[frame.ID.astype(str).eq(p)].copy(), p, 5)
        q0, h0 = training_initialization(objects, p)
        jobs.append((p, order.index(p)+1, fold, q0, h0, s5[p]['models']['ck'], cap.GRID, n_random))
    return jobs


def run(jobs, workers, cache, tag):
    """Resumable: a cached participant is reused only if its configuration tag matches."""
    cache.mkdir(parents=True, exist_ok=True)
    rows, todo = {}, []
    for job in jobs:
        f = cache/f'{job[0]}.json'
        if f.exists():
            saved = json.loads(f.read_text())
            if saved.get('config_tag') == tag:
                rows[job[0]] = saved['row']; continue
        todo.append(job)
    start = perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        for i, row in enumerate(pool.map(cap.profile_participant, todo), 1):
            (cache/f'{row["participant"]}.json').write_text(json.dumps(dict(config_tag=tag, row=row), allow_nan=False))
            rows[row['participant']] = row
            if i % 20 == 0 or i == len(todo):
                print(f'  {i}/{len(todo)} new participants, {perf_counter()-start:.0f}s', flush=True)
    manifest = cache/'compute_log.json'
    log = json.loads(manifest.read_text()) if manifest.exists() else []
    if todo:
        log.append(dict(config_tag=tag, participants_computed=len(todo), workers=workers, seconds=perf_counter()-start))
        manifest.write_text(json.dumps(log, indent=2))
    return [rows[j[0]] for j in jobs], dict(reused=len(jobs)-len(todo), compute_log=[e for e in log if e['config_tag'] == tag])


def verify(rows):
    worst = {k: max(r['checks'][k] for r in rows) for k in TOLERANCE}
    anchor = max(r['checks']['anchor_vs_saved_deviance'] for r in rows if r['checks']['anchor_vs_saved_deviance'] is not None)
    failed = {k: v for k, v in worst.items() if v > TOLERANCE[k]}
    if anchor > ANCHOR_WORSE_TOL:
        failed['anchor_vs_saved_deviance'] = anchor
    if failed:
        raise RuntimeError(f'Baseline reproduction failed {failed}; profiles NOT interpreted.')
    return dict(worst=worst | dict(anchor_minus_saved_deviance_max=anchor), tolerance=TOLERANCE | dict(anchor=ANCHOR_WORSE_TOL))


def write_csv(path, rows):
    out = []
    for r in rows:
        fixed = {f['alpha_ck']: f for f in r['fixed_nuisance']}
        for p in r['points']:
            row = dict(participant=r['participant'], n_train=r['n_train'], n_validation=r['n_validation'],
                       alpha_ck=p['alpha_ck'], on_grid=p['alpha_ck'] in cap.GRID, saved_alpha_anchor=p['anchor'],
                       status=p['status'], starts=p['starts'], successful_starts=p['successful_starts'])
            if p['status'] == 'ok':
                row.update({n: p['nuisance'][n] for n in cap.NUISANCE})
                row.update(train_nll=p['train_nll'], train_deviance=p['train_deviance'],
                           delta_nll_vs_saved_optimum=p['delta_nll_vs_saved'], delta_nll_vs_best_found=p['delta_nll'],
                           delta_train_loss_per_choice=p['delta_train_loss_per_choice'],
                           s5_loss=p['s5_loss'], delta_s5_loss_vs_saved=p['delta_s5_loss'],
                           s5_mean_abs_probability_change=p['s5_mean_abs_probability_change'],
                           train_mean_abs_probability_change=p['train_mean_abs_probability_change'],
                           train_median_abs_kernel_logit=p['train_components']['median_abs_kernel'],
                           train_median_abs_reward_logit=p['train_components']['median_abs_reward'],
                           s5_median_abs_kernel_logit=p['s5_components']['median_abs_kernel'],
                           s5_median_abs_reward_logit=p['s5_components']['median_abs_reward'],
                           best_start=p['best_label'], reaching_best=p['reaching_best'],
                           random_reaching_best=p['random_reaching_best'], projected_gradient=p['projected_gradient'],
                           boundary=';'.join(k for k, v in p['boundary'].items() if v))
            f = fixed[p['alpha_ck']]
            row.update(fixed_nuisance_delta_nll=f['delta_nll'], fixed_nuisance_delta_s5_loss=f['delta_s5_loss'])
            out.append(row)
    keys = list(dict.fromkeys(k for r in out for k in r))
    with open(path, 'w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=keys); w.writeheader()
        for r in out:
            w.writerow({k: '' if r.get(k) is None else (f'{r[k]:.10g}' if isinstance(r.get(k), float) else r[k]) for k in keys})


def pilot(args, folds, order):
    """Reproducible subset spanning the saved alpha distribution; 20 vs 100 random starts."""
    saved = [dict(participant=r['participant'], saved_theta=r['models']['ck']['theta']) for r in folds['rows'] if r['session'] == 5]
    chosen = cap.representatives(saved, 9)
    report = dict(participants=chosen, runs={})
    for n in (args.random_starts, 100):
        t = perf_counter()
        rows, info = run(build_jobs(args.data, folds, chosen, n), args.workers, args.cache_dir/f'pilot_{n}', f'pilot{n}')
        report['runs'][str(n)] = dict(seconds_this_invocation=perf_counter()-t, compute_log=info['compute_log'], verification=verify(rows),
                                      optimization=cap.optimization_summary(rows), rows=rows)
    a, b = (report['runs'][str(n)]['rows'] for n in (args.random_starts, 100))
    gaps = [pa['deviance']-pb['deviance'] for ra, rb in zip(a, b) for pa, pb in zip(ra['points'], rb['points'])
            if pa['status'] == pb['status'] == 'ok']
    report['restart_sensitivity'] = dict(points=len(gaps), max_deviance_gain_from_100_starts=float(max(gaps)),
                                         points_gaining_over_1e_6=int(sum(g > 1e-6 for g in gaps)))
    for n in report['runs']:  # keep the pilot file compact: per-point optimization outcome only
        report['runs'][n]['rows'] = [dict(participant=r['participant'], saved_alpha_ck=r['saved_theta'][0], checks=r['checks'],
            points=[{k: p.get(k) for k in ('alpha_ck', 'status', 'deviance', 'theta', 'best_label', 'random_reaching_best',
                                            'projected_gradient', 'delta_nll', 'delta_s5_loss')} for p in r['points']])
            for r in report['runs'][n]['rows']]
    (args.output_dir/f'{args.prefix}_pilot.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({n: dict(compute=v['compute_log'], violations=v['optimization']['violations'],
                              failed=v['optimization']['failed_points']) for n, v in report['runs'].items()}))
    print('restart sensitivity', report['restart_sensitivity'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/raw/02_comp_mod_RP_task_data_in.RData'))
    parser.add_argument('--fold-results', type=Path, default=Path('docs/history_diagnostics_results.json'))
    parser.add_argument('--participant', nargs='+')
    parser.add_argument('--random-starts', type=int, default=cap.RANDOM_STARTS)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--cache-dir', type=Path, default=Path('data/derived/experiment3_alpha_profile'))
    parser.add_argument('--output-dir', type=Path, default=Path('docs'))
    parser.add_argument('--prefix', default='experiment3_alpha')
    args = parser.parse_args()
    start = perf_counter()
    folds = json.loads(args.fold_results.read_text())
    raw_hash = sha256(args.data.read_bytes()).hexdigest()
    if raw_hash != folds['provenance']['raw_sha256']:
        raise ValueError('Raw data differ from the data behind the saved fits; refusing to reuse them.')
    protected = sorted(Path('src/reward_pairs').glob('authors_*.py'))
    hashes = {str(p): sha256(p.read_bytes()).hexdigest() for p in protected}
    order = list(dict.fromkeys(r['participant'] for r in folds['rows'] if r['session'] == 5))
    if args.pilot:
        return pilot(args, folds, order)
    participants = args.participant or order
    if any(p not in order for p in participants) or len(set(participants)) != len(participants):
        parser.error('Unknown or duplicate participants.')
    tag = f'grid={cap.GRID};random={args.random_starts};seed={cap.START_SEED};options={sorted(cap.OPTIONS.items())}'
    rows, cache_info = run(build_jobs(args.data, folds, participants, args.random_starts), args.workers, args.cache_dir, tag)
    if hashes != {str(p): sha256(p.read_bytes()).hexdigest() for p in protected}:
        raise RuntimeError('Validated authors implementation changed during the run.')
    verification = verify(rows)
    summary = cap.summarize(rows, seed=args.seed, draws=args.bootstrap_draws)
    result = dict(
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        design=dict(model='authors model 19 (RL+CK), released_code_intended, scoring pi=0, bias fixed 2.5',
                    fold='fit S1-S4 training choices; score S5 by causal replay (previously inspected validation)',
                    grid=list(cap.GRID), anchor="each participant's saved unconstrained alpha_CK added to its grid",
                    nuisance=list(cap.NUISANCE), bounds=cap.NUISANCE_BOUNDS.tolist(), optimizer='scipy L-BFGS-B',
                    options=cap.OPTIONS, starts=dict(saved_baseline=1, random=args.random_starts, warm_up=1, warm_down=1,
                                                     random_seed=f'numpy default_rng([{cap.START_SEED}, source_index])'),
                    agreement_tol=cap.AGREEMENT_TOL, violation_tol=cap.VIOLATION_TOL, boundary_tol=cap.BOUNDARY_TOL,
                    near_optimal_nats=list(cap.NEAR_OPTIMAL_NATS), primary_tolerance=cap.PRIMARY_TOL,
                    delta_nll='(deviance(alpha) - min(saved deviance, all constrained deviances))/2, nats',
                    s5_effect='s5_loss(alpha) - s5_loss(saved fit); positive = worse on S5; per choice',
                    fixed_nuisance='secondary: saved beta_CK, alpha_RL, beta_RL held fixed; not a profile likelihood',
                    final_test='never read: permitted_inputs exposes only training/initialization tables'),
        provenance=dict(raw_sha256=raw_hash, fold_results_sha256=sha256(args.fold_results.read_bytes()).hexdigest(),
                        protected_source_sha256=hashes, python=sys.version.split()[0], numpy=np.__version__,
                        scipy=scipy.__version__, platform=platform.platform()),
        verification=verification, cache=cache_info, summary=summary,
        representatives=cap.representatives(rows, 6), runtime_seconds=perf_counter()-start)
    out = args.output_dir; (out/'figures').mkdir(parents=True, exist_ok=True)
    (out/f'{args.prefix}_profile.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    write_csv(out/f'{args.prefix}_participants.csv', rows)
    from reward_pairs import ck_alpha_profile_report as report
    report.render(result, rows, out, args.prefix)
    o = summary['optimization']
    print(f'n={summary["analysed"]}/{summary["participants"]}; failed points {o["failed_points"]}; violations {len(o["violations"])}; '
          f'runtime {result["runtime_seconds"]:.0f}s; verification {verification["worst"]}')


if __name__ == '__main__':
    main()
