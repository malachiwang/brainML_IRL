#!/usr/bin/env python3
"""Experiment 2: evaluation-phase choice-kernel ablations of the authors' RL+CK.

Primary: saved S1-S4 RL+CK fits (docs/history_diagnostics_results.json) scored on
training session 5, which keeps reward feedback (Q updates normally).
Secondary (exploratory): saved five-session fits (docs/authors_baseline_results.json)
scored on the already-inspected final test. No parameter is refitted.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from reward_pairs.authors_audit import participant_inputs  # noqa: E402
from reward_pairs.authors_models import evaluate as authors_evaluate  # noqa: E402
from reward_pairs.ck_ablation import (  # noqa: E402
    CONDITIONS, HIGH_FREQUENCY_TRAINING_PAIRS, MODEL, NEUTRAL_K, RESET_SENSITIVITY, TRAINED_PAIRS, consistency,
    participant_summary, run_conditions, summarize,
)
from reward_pairs.data import load_rdata  # noqa: E402
from reward_pairs.history_diagnostics import make_fold, permitted_inputs, training_initialization  # noqa: E402
from reward_pairs.human_history_evaluation import replay_reference  # noqa: E402

VARIANT = 'released_code_intended'
_OBJECTS = None


def _objects(path):
    global _OBJECTS
    if _OBJECTS is None:
        _OBJECTS = load_rdata(path)
    return _OBJECTS


def validation_job(job):
    """Session-5 conditions for one participant from saved S1-S4 fits; verified first."""
    participant, data_path, ck_saved, rl_saved = job
    objects = _objects(data_path)
    permitted, _ = permitted_inputs(objects)
    frame = permitted['data_4_analysis_RP_training']
    fold = make_fold(frame.loc[frame.ID.astype(str).eq(participant)].copy(), participant, 5)
    q0, h0 = training_initialization(permitted, participant)
    theta = ck_saved['theta']
    fit = authors_evaluate(fold.fit_trials, MODEL, theta, q0, h0, scoring_pi=0.)
    checks = dict(
        fit_deviance=abs(fit.deviance-ck_saved['optimization']['optimizer_objective_deviance']),
        carried_q=float(np.max(abs(fit.q[-1]-ck_saved['initial_validation_q']))),
        carried_k=float(np.max(abs(fit.h[-1]-ck_saved['initial_validation_h']))))
    outputs = run_conditions(fold.validation_trials, theta, fit.q[-1], fit.h[-1])
    checks['c0_validation_nll'] = abs(float(outputs['C0'].log_loss.sum())-ck_saved['validation']['nll'])
    rl_fit = authors_evaluate(fold.fit_trials, 3, rl_saved['theta'], q0, h0, scoring_pi=0.)
    rl_val = authors_evaluate(fold.validation_trials, 3, rl_saved['theta'], rl_fit.q[-1], rl_fit.h[-1], scoring_pi=0.)
    checks['rl_validation_nll'] = abs(float(rl_val.log_loss.sum())-rl_saved['validation']['nll'])
    row = participant_summary(fold.validation_trials, theta, outputs, 'validation')
    row.update(participant=participant, checks=checks, rl_reference_loss=float(rl_val.log_loss.mean()),
               parameters=dict(zip(('alpha_ck', 'beta_ck', 'alpha_rl', 'beta_rl'), theta)),
               fit_pair_counts=_pair_counts(fold.fit_trials.pairs))
    return row


def test_job(job):
    """Final-test conditions from saved five-session fits; replay_reference verifies C0."""
    participant, data_path, ck_record, rl_record = job
    train, test, ratings, pretest, _ = participant_inputs(_objects(data_path), participant)
    from reward_pairs.authors_models import initial_values
    q0, h0 = initial_values(ratings, pretest, VARIANT)
    tr, te = replay_reference(train, test, q0, h0, ck_record)  # raises unless saved deviances reproduce
    theta = list(ck_record['free_parameters'].values())
    outputs = run_conditions(test, theta, tr.q[-1], tr.h[-1])
    np.testing.assert_array_equal(outputs['C0'].log_loss, te.log_loss)
    row = participant_summary(test, theta, outputs, 'test')
    row.update(participant=participant,
               checks=dict(c0_test_deviance=abs(2*float(outputs['C0'].log_loss.sum())-ck_record['test']['deviance'])),
               rl_reference_loss=rl_record['test']['nll']/rl_record['test']['n'],
               parameters=dict(zip(('alpha_ck', 'beta_ck', 'alpha_rl', 'beta_rl'), theta)))
    return row


def _pair_counts(pairs):
    p = np.sort(pairs+1, axis=1)
    return {f'{a}-{b}': int(np.all(p == [a, b], axis=1).sum()) for a, b in TRAINED_PAIRS}


def write_csv(path, validation, test):
    t = {r['participant']: r for r in test}
    rows = []
    for v in validation:
        p = v['participant']; x = t[p]
        row = dict(participant=p, s5_n=v['n'], test_n=x['n'])
        for tag, r in (('s5', v), ('test', x)):
            for c in CONDITIONS:
                row[f'{tag}_loss_{c}'] = r['conditions'][c]['loss']
            row[f'{tag}_loss_rl_reference'] = r['rl_reference_loss']
            for c in ('C1', 'C2', 'C3'):
                row[f'{tag}_effect_{c}_minus_C0'] = r['conditions'][c]['loss']-r['conditions']['C0']['loss']
            row[f'{tag}_interaction'] = (r['conditions']['C3']['loss']-r['conditions']['C1']['loss'])-(r['conditions']['C2']['loss']-r['conditions']['C0']['loss'])
            for j in ('early', 'middle', 'late'):
                row[f'{tag}_{j}_effect_C1'] = r['conditions']['C1']['phase_loss'][j]-r['conditions']['C0']['phase_loss'][j]
            c0 = r['conditions']['C0']['contributions']['all']
            row[f'{tag}_C0_median_abs_reward_logit'] = c0['median_abs_reward']
            row[f'{tag}_C0_median_abs_kernel_logit'] = c0['median_abs_kernel']
            row[f'{tag}_C0_kernel_flips_fraction'] = c0['kernel_flips_reward_direction_fraction']
            for k, val in r['parameters'].items():
                row[f'{tag}_fit_{k}'] = val
        for cl in ('trained_pair', 'novel_unequal', 'novel_equal'):
            row[f'test_n_{cl}'] = x['class_counts'].get(cl, 0)
            if x['conditions']['C0']['class_loss'].get(cl) is not None:
                for c in ('C1', 'C2', 'C3'):
                    row[f'test_{cl}_effect_{c}'] = x['conditions'][c]['class_loss'][cl]-x['conditions']['C0']['class_loss'][cl]
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: '' if r.get(k) is None else (f'{r[k]:.10g}' if isinstance(r.get(k), float) else r.get(k)) for k in keys})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/raw/02_comp_mod_RP_task_data_in.RData'))
    parser.add_argument('--fold-results', type=Path, default=Path('docs/history_diagnostics_results.json'))
    parser.add_argument('--authors-results', type=Path, default=Path('docs/authors_baseline_results.json'))
    parser.add_argument('--participant', nargs='+', help='Subset for smoke runs (report still generated).')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--output-dir', type=Path, default=Path('docs'))
    parser.add_argument('--prefix', default='experiment2_ck_ablation')
    args = parser.parse_args()
    start = perf_counter()
    raw_hash = sha256(args.data.read_bytes()).hexdigest()
    folds = json.loads(args.fold_results.read_text())
    authors = json.loads(args.authors_results.read_text())
    if raw_hash != folds['provenance']['raw_sha256'] or raw_hash != authors['audit']['source_sha256']:
        raise ValueError('Raw data differ from the data behind the saved fits; refusing to reuse them.')
    order = list(dict.fromkeys(r['participant'] for r in folds['rows'] if r['session'] == 5))
    participants = args.participant or order
    if any(p not in order for p in participants) or len(set(participants)) != len(participants):
        parser.error('Unknown or duplicate participants.')
    s5 = {r['participant']: r for r in folds['rows'] if r['session'] == 5}
    if len(s5) != len(order) or any(s5[p]['status'] != 'success' for p in participants):
        raise ValueError('Session-5 fold rows missing or failed; inspect, do not omit.')
    lookup = {(r['participant'], r['variant'], r['model']): r for r in authors['rows']}
    vjobs = [(p, args.data, s5[p]['models']['ck'], s5[p]['models']['rl']) for p in participants]
    tjobs = [(p, args.data, lookup[p, VARIANT, 19], lookup[p, VARIANT, 3]) for p in participants]
    with ProcessPoolExecutor(args.workers) as pool:
        validation = list(pool.map(validation_job, vjobs))
        test = list(pool.map(test_job, tjobs))
    tolerance = dict(fit_deviance=1e-7, carried_q=1e-12, carried_k=1e-12, c0_validation_nll=1e-9,
                     rl_validation_nll=1e-9, c0_test_deviance=1e-7)
    worst = {k: max(r['checks'][k] for r in rows) for rows in (validation, test) for k in rows[0]['checks']}
    failed = {k: v for k, v in worst.items() if v > tolerance[k]}
    if failed:
        raise RuntimeError(f'C0/benchmark reproduction failed {failed}; conditions NOT interpreted.')
    cohort_pairs = {k: sum(r['fit_pair_counts'][k] for r in validation) for k in validation[0]['fit_pair_counts']}
    hi = [f'{a}-{b}' for a, b in HIGH_FREQUENCY_TRAINING_PAIRS]
    if min(cohort_pairs[k] for k in hi) <= max(v for k, v in cohort_pairs.items() if k not in hi):
        raise ValueError('High/low training-frequency pair labels do not match the data.')
    kw = dict(seed=args.seed, draws=args.bootstrap_draws)
    result = dict(
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        design=dict(conditions={'C0': 'unchanged authors evaluate', 'C1': 'kernel frozen at carried-in state',
                                'C2': f'kernel reset to common {NEUTRAL_K} then normal updates', 'C3': 'beta_CK = 0 in logit only'},
                    factorial='state {kept, removed} x updating {on, off}; reset+frozen == C3 exactly',
                    reset_sensitivity=list(RESET_SENSITIVITY), effect_sign='loss(condition) - loss(C0); positive = ingredient helped',
                    primary='S1-S4 fits -> S5 (feedback present, Q updates)',
                    secondary='five-session fits -> final test (exploratory; already inspected)'),
        provenance=dict(raw_sha256=raw_hash, fold_results_sha256=sha256(args.fold_results.read_bytes()).hexdigest(),
                        authors_results_sha256=sha256(args.authors_results.read_bytes()).hexdigest()),
        verification=dict(worst=worst, tolerance=tolerance, cohort_fit_pair_counts=cohort_pairs),
        validation=summarize(validation, 'validation', **kw),
        test=summarize(test, 'test', **kw),
        consistency=consistency(validation, test),
        parameters=dict(validation={k: [r['parameters'][k] for r in validation] for k in validation[0]['parameters']},
                        test={k: [r['parameters'][k] for r in test] for k in test[0]['parameters']}),
        runtime_seconds=perf_counter()-start)
    out = args.output_dir
    (out/'figures').mkdir(parents=True, exist_ok=True)
    (out/f'{args.prefix}.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    write_csv(out/f'{args.prefix}_participants.csv', validation, test)
    from reward_pairs import ck_ablation_report as report
    report.render(result, validation, test, out, args.prefix)
    for tag in ('validation', 'test'):
        f = result[tag]['factorial']
        print(f'{tag}: n={result[tag]["participants"]}, C0 loss {result[tag]["losses"]["C0"]["mean"]:.5f}; '
              + '; '.join(f'{k} {f[k]["mean"]:+.5f} [{f[k]["ci95"][0]:+.5f},{f[k]["ci95"][1]:+.5f}] hurt/help {f[k]["hurts"]}/{f[k]["helps"]}'
                          for k in ('total_kernel_value', 'updating_given_carried_state', 'carried_state_given_updating', 'interaction')))
    print('verification worst:', {k: f'{v:.1e}' for k, v in worst.items()}, f'runtime {result["runtime_seconds"]:.1f}s')


if __name__ == '__main__':
    main()
