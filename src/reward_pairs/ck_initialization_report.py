"""Figures and Markdown for Experiment 4 (choice-kernel initialization)."""
from pathlib import Path

import numpy as np

from .ck_initialization import CONDITIONS, LABELS, PROFILE_GRID, SEGMENTS, THRESHOLD, usable
from .participant_error_report import GRID as GRIDCOLOR, INK, INK2, _plt, _save, table

COLOR = {'C0': '#5c5c58', 'C1': '#2a78d6', 'C2': '#eb6834'}
SESSION_COLORS = {'S1 early': '#eb6834', 'S1': '#f2a07f', 'S2': '#9ec5f0', 'S3': '#5a9be3', 'S4': '#2a78d6', 'S5': '#123f7a'}
MARKER = '<!-- Reviewed interpretation -->'


def _p(rows, c, key):
    return np.array([r['conditions'][c]['parameters'][key] for r in rows])


def _strip(ax, j, d, color, rng):
    ax.scatter(j+rng.uniform(-.18, .18, len(d)), d, s=8, color=color, alpha=.4, edgecolor='none')
    ax.boxplot(d, positions=[j], widths=.5, showfliers=False, medianprops=dict(color=INK, lw=1.5),
               boxprops=dict(color=INK2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))


def figure_alpha(result, rows, path):
    plt = _plt(); rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
    ax = axes[0]
    for j, c in enumerate(CONDITIONS):
        _strip(ax, j, np.clip(_p(rows, c, 'alpha_ck'), 1e-3, 1), COLOR[c], rng)
        m = result['summary']['alpha'][c]['alpha']
        ax.annotate(f'median {m["median"]:.3f}\nIQR {m["quantiles"]["p25"]:.3f}–{m["quantiles"]["p75"]:.3f}',
                    xy=(j, 1.0), xycoords=('data', 'axes fraction'), ha='center', va='bottom', fontsize=8)
    ax.axhspan(.05, .10, color='#1baf7a', alpha=.12, lw=0, label='0.05–0.10 (E3 S2–S5 range)')
    ax.set_yscale('log'); ax.set_ylim(8e-4, 1.3); ax.set_xticks(range(3), [f'{c}\n{LABELS[c]}' for c in CONDITIONS], fontsize=8.5)
    ax.set_ylabel('fitted α_CK (log; values < 0.001 drawn at 0.001)'); ax.legend(fontsize=8, frameon=False, loc='lower right')
    ax.set_title('Fitted α_CK by condition', pad=28, fontsize=10)
    for ax, key, name in ((axes[1], 'alpha_ck', 'α_CK'), (axes[2], 'alpha_rl', 'α_RL')):
        x = np.clip(_p(rows, 'C0', key), 1e-3, 1)
        for c in ('C1', 'C2'):
            ax.scatter(x, np.clip(_p(rows, c, key), 1e-3, 1), s=12, color=COLOR[c], alpha=.55, edgecolor='none', label=f'{c} ({LABELS[c]})')
        ax.plot([1e-3, 1], [1e-3, 1], color=INK, lw=1, ls='--', label='no change')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(8e-4, 1.3); ax.set_ylim(8e-4, 1.3)
        ax.set_xlabel(f'C0 fitted {name}'); ax.set_ylabel(f'alternative fitted {name}'); ax.grid(color=GRIDCOLOR, lw=.5)
        ax.legend(fontsize=8, frameon=False, loc='upper left')
        ax.set_title(f'Paired {name}: each point is one participant', fontsize=10)
    fig.suptitle('E4-A  Fitted learning rates under each kernel initialization (S1–S4 fits, 213 participants)', fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def figure_sessions(result, rows, path):
    plt = _plt(); rng = np.random.default_rng(1)
    S = result['summary']['sessions']
    keys = ('S1 early', 'S1', 'S2', 'S3', 'S4', 'S2-S4', 'total')
    fig, ax = plt.subplots(figsize=(13, 5))
    for k, c in enumerate(('C1', 'C2')):
        off = -.18 if c == 'C1' else .18
        m = np.array([S[c][s]['mean'] for s in keys]); ci = np.array([S[c][s]['ci95'] for s in keys])
        ax.errorbar(np.arange(len(keys))+off, m, yerr=[m-ci[:, 0], ci[:, 1]-m], fmt='o' if c == 'C1' else 's', color=COLOR[c],
                    capsize=3, ms=6, lw=1.4, label=f'{c} − C0 ({LABELS[c]}): mean, 95% CI')
        med = [S[c][s]['distribution']['median'] for s in keys]
        ax.scatter(np.arange(len(keys))+off, med, marker='_', s=160, color=COLOR[c], lw=2)
    ax.axhline(0, color=INK, lw=1); ax.axvline(4.5, color=INK2, lw=.8, ls=':')
    ax.set_xticks(range(len(keys)), ['S1 first third', 'S1', 'S2', 'S3', 'S4', 'S2–S4', 'all S1–S4'])
    ax.set_ylabel('Δ training log loss vs C0 [nats/choice]\n(negative = alternative fits better)')
    ax.scatter([], [], marker='_', s=160, color=INK2, lw=2, label='median')
    ax.legend(fontsize=8.5, frameon=False); ax.grid(axis='y', color=GRIDCOLOR, lw=.6)
    ax.set_title('E4-B  In-sample training loss by session (each condition refitted on S1–S4; descriptive, not held out)',
                 fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def figure_validation(result, rows, path):
    plt = _plt(); rng = np.random.default_rng(2)
    V = result['summary']['validation']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw=dict(width_ratios=[1.2, 1]))
    s5 = {c: np.array([r['conditions'][c]['s5_loss'] for r in rows]) for c in CONDITIONS}
    series = [('C1', s5['C1']-s5['C0'], 'C1 − C0'), ('C2', s5['C2']-s5['C0'], 'C2 − C0'), ('C0', s5['C2']-s5['C1'], 'C2 − C1')]
    for j, (c, d, lab) in enumerate(series):
        _strip(a1, j, d, COLOR[c], rng)
        key = {'C1': 'C1', 'C2': 'C2', 'C0': 'C2_minus_C1'}[c]
        s = V[key]
        a1.errorbar(j+.32, s['mean'], yerr=[[s['mean']-s['ci95'][0]], [s['ci95'][1]-s['mean']]], fmt='D', color=INK, ms=5, capsize=3)
        a1.annotate(f'mean {s["mean"]:+.4f}\nworse/better {s["hurts"]}/{s["helps"]}', xy=(j, 1), xycoords=('data', 'axes fraction'),
                    ha='center', va='bottom', fontsize=8)
    a1.axhline(0, color=INK, lw=1)
    for t in (THRESHOLD, -THRESHOLD):
        a1.axhline(t, color=INK2, lw=.8, ls=':')
    a1.set_xticks(range(3), [s[2] for s in series]); a1.set_ylabel('Δ session-5 log loss [nats/choice]\n(positive = first condition worse)')
    a1.set_title('Participant-level differences (◆ mean, 95% bootstrap CI; dotted ±0.005)', pad=30, fontsize=10)
    for c in ('C1', 'C2'):
        a2.hist(s5[c]-s5['C0'], bins=50, histtype='step', lw=2, color=COLOR[c], label=f'{c} − C0')
    a2.axvline(0, color=INK, lw=1); a2.set_xlabel('Δ session-5 log loss vs C0 [nats/choice]'); a2.set_ylabel('participants')
    a2.legend(fontsize=8.5, frameon=False); a2.set_title('Distribution', fontsize=10)
    fig.suptitle('E4-C  Session-5 prediction (previously inspected held-out session; no condition selected on it)', fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def figure_profile(result, path):
    P = result['summary']['profile']
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(PROFILE_GRID)); labels = [f'{a:g}' for a in PROFILE_GRID]
    ax = axes[0]
    for c in ('C0', 'C1'):
        d = P['conditions'][c]['delta_nll']
        ax.fill_between(x, d['q25'], d['q75'], color=COLOR[c], alpha=.18, lw=0)
        ax.plot(x, d['median'], color=COLOR[c], marker='o', lw=2, label=f'{c} ({LABELS[c]})')
    ax.axhline(1, color=INK2, ls='--', lw=.8, label='1 nat')
    ax.set_ylabel('Δ training NLL vs condition optimum [nats]\n(median, IQR)'); ax.set_title('(2) Fixed-α profiles, nuisance re-fit', fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for ax, c in zip(axes[1:], ('C0', 'C1')):
        per = P['conditions'][c]['session_relative_to_alpha015']
        for s in ('S1 early', 'S1', 'S2', 'S3', 'S4', 'S5'):
            m = np.array(per[s]['mean_minus_alpha015']); ci = np.array(per[s]['ci95'])
            ax.plot(x, m, color=SESSION_COLORS[s], marker='o', ms=3.5, lw=2 if s.startswith('S1') else 1.5,
                    ls='--' if s == 'S1 early' else '-', label=s + (' (held out)' if s == 'S5' else ''))
            ax.fill_between(x, ci[:, 0], ci[:, 1], color=SESSION_COLORS[s], alpha=.12, lw=0)
        ax.axhline(0, color=INK, lw=1)
        ax.set_ylabel('session log loss − loss at α = 0.15 [nats/choice]')
        ax.set_title(f'(3) Sessions within each profile fit: {c} ({LABELS[c]})', fontsize=10)
        ax.legend(fontsize=7.5, frameon=False, ncol=2)
    lo = min(min(min(P['conditions'][c]['session_relative_to_alpha015'][s]['mean_minus_alpha015']) for s in SESSION_COLORS) for c in ('C0', 'C1'))
    hi = max(max(max(P['conditions'][c]['session_relative_to_alpha015'][s]['mean_minus_alpha015']) for s in SESSION_COLORS) for c in ('C0', 'C1'))
    for ax in axes:
        ax.set_xticks(x, labels); ax.set_xlabel('fixed α_CK'); ax.grid(axis='y', color=GRIDCOLOR, lw=.6)
    for ax in axes[1:]:
        ax.set_ylim(lo-.005, hi+.005)
    fig.suptitle(f'E4-D  α sensitivity by initialization and session (full cohort, n = {P["participants"]}; participant means)',
                 fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def figure_contributions(result, path):
    C = result['summary']['contributions']
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(SEGMENTS))
    for ax, key, title in ((axes[0], 'median_abs_zk', 'Kernel term |z_K| = |β_CK (K_L − K_R)|'),
                           (axes[1], 'median_abs_zq', 'Reward term |z_Q| = |β_RL (Q_L − Q_R)|')):
        for c in CONDITIONS:
            med = [C[c][s][key]['median'] for s in SEGMENTS]
            q1 = [C[c][s][key]['quantiles']['p25'] for s in SEGMENTS]; q3 = [C[c][s][key]['quantiles']['p75'] for s in SEGMENTS]
            ax.fill_between(x, q1, q3, color=COLOR[c], alpha=.15, lw=0)
            ax.plot(x, med, color=COLOR[c], marker='o', lw=2, label=f'{c} ({LABELS[c]})')
        ax.set_xticks(x, ['S1 first\nthird', 'S1 rest', 'S2', 'S3', 'S4', 'S5\n(held out)'])
        ax.set_ylabel('per-participant median |logit term|\n(cohort median, IQR)'); ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, frameon=False); ax.grid(axis='y', color=GRIDCOLOR, lw=.6)
    ax = axes[2]
    for c in CONDITIONS:
        f = C[c]['first20']
        ax.bar(CONDITIONS.index(c)-.2, f['mean_abs_zk']['median'], .38, color=COLOR[c], label=None)
        ax.bar(CONDITIONS.index(c)+.2, f['mean_abs_zq']['median'], .38, color=COLOR[c], alpha=.45, hatch='//', edgecolor='white')
    ax.set_xticks(range(3), [f'{c}\n{LABELS[c]}' for c in CONDITIONS], fontsize=8.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=INK2, label='|z_K| kernel (solid)'),
                       Patch(facecolor=INK2, alpha=.45, hatch='//', edgecolor='white', label='|z_Q| reward (hatched)')],
              fontsize=8, frameon=False)
    ax.set_ylabel('median over participants of mean |term|'); ax.set_title('First 20 training trials', fontsize=10)
    ax.grid(axis='y', color=GRIDCOLOR, lw=.6)
    fig.suptitle('E4-E  Weighted reward and kernel contributions to the choice logit, by training segment', fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def render(result, rows, out, prefix):
    out = Path(out); fig = out/'figures'
    good, _ = usable(rows)
    names = dict(a='e4_a_fitted_alpha.png', b='e4_b_session_training_loss.png', c='e4_c_session5_validation.png',
                 d='e4_d_alpha_profile_by_session.png', e='e4_e_contributions.png')
    figure_alpha(result, good, fig/names['a'])
    figure_sessions(result, good, fig/names['b'])
    figure_validation(result, good, fig/names['c'])
    if result['summary']['profile']:
        figure_profile(result, fig/names['d'])
    figure_contributions(result, fig/names['e'])
    write_report(result, good, out/f'{prefix}.md', names)


def _ci(s, d=4):
    return f'{s["mean"]:+.{d}f} [{s["ci95"][0]:+.{d}f}, {s["ci95"][1]:+.{d}f}]'


def write_report(result, rows, path, figures):
    path = Path(path); S = result['summary']; A = S['alpha']; SS = S['sessions']; V = S['validation']
    C = S['contributions']; P = S['profile']; ver = result['verification']; O = S['optimization']
    n = S['analysed']
    lines = [
        '# Experiment 4: choice-kernel initialization and the fitted kernel learning rate',
        '',
        'Authors\' RL+CK (model 19, π = 0, bias fixed at 2.5), released-code reward initialization Q0 in every condition. '
        'The **only** modeling difference is the initial kernel vector K0. Fit on training sessions 1–4 and score training '
        'session 5 by causal replay. Session 5 has been inspected in E2 and E3, so it is a previously used validation session, '
        'not a prospective test. The final test is never read.',
        '',
        table(['condition', 'Q0', 'K0', 'role'], [
            ['C0', 'released pretraining values', 'released values (= Q0, up to ≈ 4.1)', 'primary reference'],
            ['C1', 'unchanged', '0.5 for all eight stimuli', 'primary intervention'],
            ['C2', 'unchanged', '0 for all eight stimuli', 'secondary robustness check']]),
        '',
        '## 1. Protocol and checks',
        '',
        f'- {S["participants"]} participants, {n} analysed (excluded: {S["excluded"] or "none"}).',
        f'- Optimizer: unchanged `authors_fitting.fit_authors` with the identical {result["configuration"]["starts"]} R-seeded starts per '
        'condition used for the saved C0 fits, plus warm starts from the other two conditions\' optima. K0 is passed externally to '
        'the unchanged objective.',
        '- C0 is refitted from scratch and compared with the saved S1–S4 fits:',
        '',
        table(['check', 'worst over participants', 'tolerance'],
              [[k, f'{v:.1e}', f'{ver["tolerance"][k]:.0e}'] for k, v in ver['worst'].items()]),
        '',
        '- Q0 is bit-identical across conditions for every participant, and the S1–S4 final Q and K are the S5 starting states (checked).',
        '',
        table(['condition', 'failed fits', 'failed starts', 'median starts reaching best (of 200)', 'won by cross-condition start', 'max cross-start gain (deviance)'],
              [[c, O[c]['failed_conditions'], O[c]['failed_starts'], f'{O[c]["reaching_best"]["median"]:.0f} (min {O[c]["reaching_best"]["minimum"]:.0f})',
                O[c]['won_by_cross_start'], f'{O[c]["max_cross_start_improvement"]:.2g}'] for c in CONDITIONS]),
        '',
        '- Compute: ' + '; '.join(f'{e["participants_computed"]} participants in {e["seconds"]:.0f} s with {e["workers"]} workers'
                                for e in result['compute_log']) + '.',
        '',
        '## 2. Analysis A: fitted α_CK',
        '',
        table(['condition', 'mean α_CK', 'median α_CK', 'IQR', 'share in 0.05–0.10', 'α_CK = 1', 'α_CK = 0',
               'α_RL at a bound', 'β_CK at a bound', 'median α_RL', 'median β_CK', 'median β_RL'],
              [[c, f'{A[c]["alpha"]["mean"]:.3f}', f'{A[c]["alpha"]["median"]:.3f}',
                f'{A[c]["alpha"]["quantiles"]["p25"]:.3f}–{A[c]["alpha"]["quantiles"]["p75"]:.3f}', f'{A[c]["share_in_005_010"]:.2f}',
                A[c]['alpha_ck_at_1'], A[c]['alpha_ck_at_0'], A[c]['boundary']['alpha_rl'], A[c]['boundary']['beta_ck'],
                f'{A[c]["parameters"]["alpha_rl"]["median"]:.3f}', f'{A[c]["parameters"]["beta_ck"]["median"]:.2f}',
                f'{A[c]["parameters"]["beta_rl"]["median"]:.2f}'] for c in CONDITIONS]),
        '',
        table(['paired change vs C0', 'mean Δα [95% CI]', 'median Δα [95% CI]', 'median ratio', 'decreases', 'increases', 'unchanged'],
              [[c, _ci(A[c]['paired_change']['difference']), f'{A[c]["paired_change"]["difference"]["distribution"]["median"]:+.4f} '
                f'[{A[c]["paired_change"]["median_difference_ci"][0]:+.4f}, {A[c]["paired_change"]["median_difference_ci"][1]:+.4f}]',
                f'{A[c]["paired_change"]["median_ratio"]:.2f}×', A[c]['paired_change']['decreases'], A[c]['paired_change']['increases'],
                A[c]['paired_change']['unchanged']] for c in ('C1', 'C2')]),
        '',
        f'![E4-A](figures/{figures["a"]})',
        '',
        '## 3. Analysis B: training fit by session (in-sample, descriptive)',
        '',
        'Each condition is refitted on all of S1–S4, so these are descriptive, in-sample diagnostics, not held-out tests of each session. '
        'The values are per-participant mean log loss for the alternative minus C0, in nats/choice, as a participant-weighted mean [95% CI]; '
        'worse/better counts compare each participant\'s alternative fit with their C0 fit.',
        '',
        table(['segment', 'C1 − C0', 'C1 worse / better', 'C2 − C0', 'C2 worse / better'],
              [[s, _ci(SS['C1'][s]), f'{SS["C1"][s]["hurts"]} / {SS["C1"][s]["helps"]}', _ci(SS['C2'][s]),
                f'{SS["C2"][s]["hurts"]} / {SS["C2"][s]["helps"]}'] for s in ('S1 early', 'S1', 'S2', 'S3', 'S4', 'S2-S4', 'total')]),
        '',
        '"S1 early" is the first third of each participant\'s session-1 trials.',
        '',
        table(['', 'C1', 'C2'], [
            ['total training NLL change, nats (mean [95% CI])', _ci(SS['C1']['total_nll'], 2), _ci(SS['C2']['total_nll'], 2)],
            ['participants: S1 better and S2–S4 worse', SS['C1']['offset']['s1_better_and_later_worse'], SS['C2']['offset']['s1_better_and_later_worse']],
            ['participants: S1 worse and S2–S4 better', SS['C1']['offset']['s1_worse_and_later_better'], SS['C2']['offset']['s1_worse_and_later_better']],
            ['corr(S1 NLL change, S2–S4 NLL change)', f'{SS["C1"]["offset"]["correlation"]:+.2f}', f'{SS["C2"]["offset"]["correlation"]:+.2f}'],
            ['share of summed |total NLL change| in top 10% of participants', f'{SS["C1"]["concentration"]["top10pct_share"]:.2f}',
             f'{SS["C2"]["concentration"]["top10pct_share"]:.2f}']]),
        '',
        f'![E4-B](figures/{figures["b"]})',
        '',
        '## 4. Analysis C: session-5 prediction',
        '',
        f'Session-5 loss (participant mean): ' + ', '.join(f'{c} {V["losses"][c]["mean"]:.5f}' for c in CONDITIONS) + ' nats/choice. '
        'Δ = alternative − C0; positive means the alternative predicts worse.',
        '',
        table(['comparison', 'mean [95% CI]', 'median', 'worse / better / tied', f'|Δ| > {THRESHOLD}: worse / better',
               'trial-weighted [95% CI]', 'top-10% share of |Δ|'],
              [[c + ' − C0', _ci(V[c]), f'{V[c]["median"]:+.4f}', f'{V[c]["hurts"]} / {V[c]["helps"]} / {V[c]["tied"]}',
                f'{V[c]["over_threshold_worse"]} / {V[c]["over_threshold_better"]}',
                f'{V[c]["trial_weighted"]:+.4f} [{V[c]["trial_weighted_ci"][0]:+.4f}, {V[c]["trial_weighted_ci"][1]:+.4f}]',
                f'{V[c]["concentration"]["top10pct_share"]:.2f}'] for c in ('C1', 'C2')]
              + [['C2 − C1', _ci(V['C2_minus_C1']), f'{V["C2_minus_C1"]["distribution"]["median"]:+.4f}',
                  f'{V["C2_minus_C1"]["hurts"]} / {V["C2_minus_C1"]["helps"]} / {V["C2_minus_C1"]["tied"]}', '', '', '']]),
        '',
        f'The {THRESHOLD} nats/choice threshold is descriptive.',
        '',
        f'![E4-C](figures/{figures["c"]})',
        '',
    ]
    if P:
        j05, j15 = PROFILE_GRID.index(.05), PROFILE_GRID.index(.15)
        pc = P['conditions']
        lines += [
            '## 5. Analysis D: α × initialization (fixed-α profiles)',
            '',
            'Three different comparisons answer different questions: (1) fully refitted C0 vs C1 (§2–4), (2) fixed-α profile fits '
            'with β_CK, α_RL and β_RL re-optimized (this section), and (3) sessions within the same fitted model (columns below). '
            f'Grid: {", ".join(f"{a:g}" for a in PROFILE_GRID)}. The E3 protocol was used (refit nuisance + 20 random starts + warm sweeps). '
            f'At about 2 minutes for the whole cohort, the cost allowed **all {P["participants"]} participants** rather than a subset, so these are cohort results.',
            '',
            table(['α_CK'] + [f'{a:g}' for a in PROFILE_GRID],
                  [[f'{c} median ΔNLL'] + [f'{v:.2f}' for v in pc[c]['delta_nll']['median']] for c in ('C0', 'C1')]
                  + [[f'{c} within 1 nat'] + list(pc[c]['within_1_nat']) for c in ('C0', 'C1')]
                  + [[f'{c} S1 loss − loss at 0.15'] + [f'{v:+.4f}' for v in pc[c]['session_relative_to_alpha015']['S1']['mean_minus_alpha015']] for c in ('C0', 'C1')]
                  + [[f'{c} S1-early − at 0.15'] + [f'{v:+.4f}' for v in pc[c]['session_relative_to_alpha015']['S1 early']['mean_minus_alpha015']] for c in ('C0', 'C1')]
                  + [[f'{c} S4 loss − loss at 0.15'] + [f'{v:+.4f}' for v in pc[c]['session_relative_to_alpha015']['S4']['mean_minus_alpha015']] for c in ('C0', 'C1')]
                  + [[f'{c} S5 loss − loss at 0.15'] + [f'{v:+.4f}' for v in pc[c]['session_relative_to_alpha015']['S5']['mean_minus_alpha015']] for c in ('C0', 'C1')]),
            '',
            'The E3 pattern, per participant, is the session-1 loss at α = 0.05 minus that at α = 0.15 (positive = slow kernel penalized in S1):',
            '',
            table(['quantity', 'C0', 'C1', 'C1 − C0 (paired)'], [
                ['S1 penalty of α = 0.05 vs 0.15', _ci(P['s1_penalty_005_vs_015']['C0']), _ci(P['s1_penalty_005_vs_015']['C1']),
                 _ci(P['s1_penalty_005_vs_015']['C1_minus_C0'])],
                ['mean S2–S4 penalty of α = 0.05 vs 0.15', _ci(P['s2_s4_penalty_005_vs_015']['C0']), _ci(P['s2_s4_penalty_005_vs_015']['C1']),
                 _ci(P['s2_s4_penalty_005_vs_015']['C1_minus_C0'])],
                ['grid α minimizing training NLL (median)', f'{pc["C0"]["argmin_alpha"]["median"]:g}', f'{pc["C1"]["argmin_alpha"]["median"]:g}', ''],
                ['profile points beating the full multistart fit', pc['C0']['beyond_multistart'], pc['C1']['beyond_multistart'], '']]),
            '',
            f'![E4-D](figures/{figures["d"]})',
            '',
        ]
    seg = ('S1 early', 'S1 late', 'S2', 'S4', 'S5')
    lines += [
        '## 6. Analysis E: weighted reward and kernel contributions',
        '',
        f'Released K0 values: median {C["released_k0"]["values"]["median"]:.2f}, maximum {C["released_k0"]["values"]["maximum"]:.2f}; '
        f'{C["released_k0"]["above_1"]} of {C["released_k0"]["cells"]} participant-stimulus values exceed 1, the largest value an update can target. '
        f'Median within-participant range {C["released_k0"]["range_per_participant"]["median"]:.2f}.',
        '',
        table(['segment'] + [f'{c} |z_K|' for c in CONDITIONS] + [f'{c} |z_Q|' for c in CONDITIONS],
              [[s] + [f'{C[c][s]["median_abs_zk"]["median"]:.2f}' for c in CONDITIONS] + [f'{C[c][s]["median_abs_zq"]["median"]:.2f}' for c in CONDITIONS]
               for s in seg]
              + [['first 20 trials (mean)'] + [f'{C[c]["first20"]["mean_abs_zk"]["median"]:.2f}' for c in CONDITIONS]
                 + [f'{C[c]["first20"]["mean_abs_zq"]["median"]:.2f}' for c in CONDITIONS]]),
        '',
        'Each cell is the cohort median of per-participant median |term|, in logit units. Paired change in the per-participant median, C1 − C0:',
        '',
        table(['segment', 'Δ|z_K| [95% CI]', 'Δ|z_Q| [95% CI]'],
              [[s, _ci(C['C1_minus_C0'][s]['zk'], 3), _ci(C['C1_minus_C0'][s]['zq'], 3)] for s in seg]),
        '',
        f'![E4-E](figures/{figures["e"]})',
        '',
        MARKER, '']
    old = path.read_text() if path.exists() else ''
    if MARKER in old:
        lines.append(old.split(MARKER, 1)[1].strip())
    path.write_text('\n'.join(lines)+'\n')
