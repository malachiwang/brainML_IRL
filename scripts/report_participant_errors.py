#!/usr/bin/env python3
"""Experiment 1: participant-level prediction-error diagnostics from saved fits.

Reads docs/human_history_results.json (no refitting). Optionally replays the
selected example participants trial-by-trial from the local raw .RData using the
validated replay functions, verifying stored deviances.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from reward_pairs.participant_error_diagnostics import (  # noqa: E402
    PAIR_KEYS, PRIMARY_VARIANT, SENSITIVITY_VARIANT, analyze_variant, csv_rows, replay_participant,
    reproduction_checks, select_examples, write_csv,
)
from reward_pairs import participant_error_report as report  # noqa: E402


def trial_replay(args, data, selected):
    if not args.data.exists():
        return dict(available=False, reason=f'raw data not found at {args.data}'), {}
    from reward_pairs.authors_audit import participant_inputs
    from reward_pairs.data import load_rdata
    authors = json.loads(args.authors_results.read_text())
    raw_hash = sha256(args.data.read_bytes()).hexdigest()
    if raw_hash != authors['audit']['source_sha256'] or raw_hash != data['provenance']['raw_sha256']:
        return dict(available=False, reason='raw data hash differs from the validated benchmark'), {}
    objects = load_rdata(args.data)
    lookup = {(r['participant'], r['variant'], r['model']): r for r in authors['rows']}
    rows = {r['participant']: r for r in data['rows'] if r['variant'] == PRIMARY_VARIANT}
    replays, split, ck_err, cum_err = {}, {}, 0., 0.
    for sel in selected:
        pid = sel['participant']
        cum = rows[pid]['models']['cumulative']
        rep = replay_participant(participant_inputs(objects, pid), PRIMARY_VARIANT, lookup[pid, PRIMARY_VARIANT, 19], cum)
        ck_err = max(ck_err, abs(rep['deviance']['ck']-rows[pid]['models']['ck']['test']['deviance']))
        cum_err = max(cum_err, abs(rep['deviance']['cumulative']-cum['test']['deviance']))
        gap = rep['loss']['cumulative']-rep['loss']['ck']
        eq = np.zeros(len(gap), bool)
        for key in PAIR_KEYS:
            a, b = map(int, key.split('-'))
            eq |= (np.sort(rep['pairs'], 1) == [a, b]).all(1)
        h = len(gap)//2
        split[pid] = dict(first_half=float(gap[:h].sum()), second_half=float(gap[h:].sum()),
                          equal_pairs=float(gap[eq].sum()), total=float(gap.sum()))
        replays[pid] = rep
    return dict(available=True, max_ck_error=ck_err, max_cum_error=cum_err, split=split), replays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('docs/human_history_results.json'))
    parser.add_argument('--authors-results', type=Path, default=Path('docs/authors_baseline_results.json'))
    parser.add_argument('--data', type=Path, default=Path('data/raw/02_comp_mod_RP_task_data_in.RData'))
    parser.add_argument('--no-trial-replay', action='store_true', help='Skip optional raw-data replay of selected participants.')
    parser.add_argument('--no-sensitivity', action='store_true', help='Skip the separate paper_intended appendix.')
    parser.add_argument('--output-dir', type=Path, default=Path('docs'))
    args = parser.parse_args()

    raw = args.results.read_bytes()
    data = json.loads(raw)
    metrics, primary = analyze_variant(data, PRIMARY_VARIANT)
    if any(m['participant'] not in data['participant_order'] for m in metrics) or len(metrics) != len(data['participant_order']):
        raise RuntimeError('Participant alignment failed.')
    checks = reproduction_checks(data, PRIMARY_VARIANT, metrics, primary)
    failed = [c for c in checks if not c['passed']]
    selected = select_examples(metrics)
    sensitivity = None if args.no_sensitivity else analyze_variant(data, SENSITIVITY_VARIANT)[1]

    figdir = args.output_dir/'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    names = dict(a='e1_a_delta_distribution.png', b='e1_b_loss_scatter.png', c='e1_c_pair_errors.png',
                 d='e1_d_observed_vs_predicted.png', e='e1_e_overprediction_vs_gap.png', f='e1_f_example_trials.png')
    report.figure_delta(metrics, primary['delta_vs_ck'], figdir/names['a'])
    report.figure_scatter(metrics, figdir/names['b'])
    report.figure_pair_errors(metrics, primary, figdir/names['c'])
    report.figure_observed_predicted(metrics, primary, figdir/names['d'])
    report.figure_gap_link(metrics, primary['gap_link'], figdir/names['e'])
    if args.no_trial_replay:
        replay_info, replays = dict(available=False, reason='--no-trial-replay requested'), {}
    else:
        replay_info, replays = trial_replay(args, data, selected)
    if replay_info['available']:
        report.figure_examples(replays, selected, figdir/names['f'])
    elif (figdir/names['f']).exists():
        (figdir/names['f']).unlink()  # never leave a stale trial figure behind

    write_csv(csv_rows(metrics, selected), args.output_dir/'experiment1_participant_metrics.csv')
    summary = dict(source=str(args.results), source_sha256=sha256(raw).hexdigest(), primary_variant=PRIMARY_VARIANT,
                   sign_convention='delta = loss_cumulative - loss_reference (positive = cumulative worse); '
                                   'pair error = predicted - observed frequent rate (positive = overprediction)',
                   reproduction_checks=checks, selected_examples=selected, trial_replay=replay_info,
                   primary={k: v for k, v in primary.items()},
                   sensitivity=sensitivity)
    for block in (summary['primary'], summary['sensitivity'] or {}):
        for key in ('delta_vs_ck', 'delta_vs_rl'):
            if key in block:
                block[key] = {k: v for k, v in block[key].items() if k != 'cumulative_share'}
    (args.output_dir/'experiment1_participant_errors.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    ci = data['summary'][PRIMARY_VARIANT]['comparisons']['ck']['bootstrap']['ci95']
    report.write_report(args.output_dir/'experiment1_participant_errors.md', primary=primary, metrics=metrics, checks=checks,
                        selected=selected, sensitivity=sensitivity, replay_info=replay_info,
                        figures={k: f'figures/{v}' for k, v in names.items()},
                        provenance=dict(source=str(args.results), sha256=summary['source_sha256'], bootstrap_ci=ci))

    d = primary['delta_vs_ck']
    print(f'{primary["participants"]} participants ({PRIMARY_VARIANT}); reproduction checks {len(checks)-len(failed)}/{len(checks)} passed.')
    print(f'Delta (cum - CK): mean {d["mean"]:+.5f}, median {d["distribution"]["median"]:+.5f}, '
          f'worse/better/tied {d["worse"]}/{d["better"]}/{d["tied"]}, top-10% share {100*d["top_contributors"]["top_10pct"]["share_of_total"]:.1f}%.')
    for key in PAIR_KEYS:
        print(key, ' '.join(f'{m}: mean err {primary["pairs"][key][m]["error"]["mean"]:+.3f}, '
                            f'{100*primary["pairs"][key][m]["positive_fraction"]:.0f}% >0, direction '
                            f'{primary["pairs"][key][m]["direction_agreement"]}/{primary["pairs"][key][m]["direction_decided"]};'
                            for m in ('ck', 'cumulative')))
    g = primary['gap_link']
    print(f'Equal pairs: {100*g["equal_pair_share_of_test_choices"]:.1f}% of choices, {100*g["equal_pair_share_of_total_gap"]:.1f}% of NLL gap; '
          f'Spearman(err, delta) {g["spearman_error_vs_delta"]["rho"]:+.3f}.')
    print('Selected:', ', '.join(f'{s["role"]}={s["participant"]}' for s in selected))
    print('Trial replay:', replay_info if not replay_info['available'] else
          f'ok (max dev err CK {replay_info["max_ck_error"]:.1e}, cum {replay_info["max_cum_error"]:.1e})')
    if failed:
        raise SystemExit(f'Reproduction checks failed: {[c["check"] for c in failed]}')


if __name__ == '__main__':
    main()
