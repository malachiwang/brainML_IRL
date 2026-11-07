"""Figures and Markdown for Experiment 2 (RL+CK evaluation-phase kernel ablations)."""
from pathlib import Path

import numpy as np

from .ck_ablation import CONDITION_LABELS, NEUTRAL_K, PHASES, RESET_SENSITIVITY
from .participant_error_report import GRID, INK, INK2, SURFACE, _plt, _save, table

COLORS = {'C1': '#2a78d6', 'C2': '#eb6834', 'C3': '#1baf7a', 'interaction': '#5c5c58'}
EFFECTS = (('C1', 'updating_given_carried_state', 'C1 − C0\nfreeze kernel'),
           ('C2', 'carried_state_given_updating', 'C2 − C0\nreset kernel'),
           ('C3', 'total_kernel_value', 'C3 − C0\nkernel weight 0'))
MARKER = '<!-- Reviewed interpretation -->'
CLASS_LABELS = {'high_freq_pair': 'high-frequency\ntrained pairs', 'low_freq_pair': 'low-frequency\ntrained pairs',
                'trained_pair': 'trained pairs', 'novel_unequal': 'novel unequal\npairs', 'novel_equal': 'novel equal-\nreward pairs'}
TITLES = {'validation': 'Session 5 (primary; S1–S4 fits)', 'test': 'Final test (exploratory; 5-session fits)'}


def _effect(rows, c, field=None, level=None):
    if field is None:
        return np.array([r['conditions'][c]['loss']-r['conditions']['C0']['loss'] for r in rows])
    return np.array([r['conditions'][c][field][level]-r['conditions']['C0'][field][level] for r in rows
                     if r['conditions']['C0'][field].get(level) is not None])


def figure_effects(result, validation, test, path):
    plt = _plt()
    rng = np.random.default_rng(0)  # jitter only
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, tag, rows in zip(axes, ('validation', 'test'), (validation, test)):
        f = result[tag]['factorial']
        series = [(c, key, label, _effect(rows, c)) for c, key, label in EFFECTS]
        inter = np.array([(r['conditions']['C3']['loss']-r['conditions']['C1']['loss'])-(r['conditions']['C2']['loss']-r['conditions']['C0']['loss']) for r in rows])
        series.append(('interaction', 'interaction', 'interaction\n(C3−C1)−(C2−C0)', inter))
        for j, (c, key, label, d) in enumerate(series):
            ax.scatter(j+rng.uniform(-.18, .18, len(d)), d, s=9, color=COLORS[c], alpha=.45, edgecolor='none')
            ax.boxplot(d, positions=[j], widths=.5, showfliers=False, medianprops=dict(color=INK, lw=1.5),
                       boxprops=dict(color=INK2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
            s = f[key]
            ax.errorbar(j+.32, s['mean'], yerr=[[s['mean']-s['ci95'][0]], [s['ci95'][1]-s['mean']]], fmt='D', ms=5,
                        color=INK, capsize=3, lw=1.2)
            ax.annotate(f'mean {s["mean"]:+.4f}\nworse/better {s["hurts"]}/{s["helps"]}', xy=(j, 1.0), xycoords=('data', 'axes fraction'),
                        ha='center', va='top', fontsize=8.3)
        ax.axhline(0, color=INK, lw=1.1)
        ax.set_xticks(range(len(series)), [s[2] for s in series], fontsize=8.8)
        ax.set_title(TITLES[tag], pad=34)
        ax.set_ylabel('Δ log loss vs C0 [nats/choice]\n(positive = removing it hurts)')
    fig.suptitle('E2-A  Per-participant effects of evaluation-phase kernel interventions (◆ = mean, 95% participant bootstrap CI)',
                 fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def figure_phases(result, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    x = np.arange(3)
    for ax, tag in zip(axes, ('validation', 'test')):
        for k, (c, _, label) in enumerate(EFFECTS):
            s = [result[tag]['phases'][p][c] for p in PHASES]
            m = np.array([v['mean'] for v in s]); lo = np.array([v['ci95'][0] for v in s]); hi = np.array([v['ci95'][1] for v in s])
            xx = x+(k-1)*.08
            ax.errorbar(xx, m, yerr=[m-lo, hi-m], color=COLORS[c], marker='o', ms=5, lw=2, capsize=3,
                        label=CONDITION_LABELS[c].replace('C2 kernel reset (0.5)', f'C2 kernel reset ({NEUTRAL_K})'))
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks(x, ['early third', 'middle third', 'late third'])
        ax.set_ylabel('Mean Δ log loss vs C0 [nats/choice]')
        ax.set_title(TITLES[tag])
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=9)
    fig.suptitle('E2-B  Intervention effects by evaluation phase (participant-weighted means, 95% CI)', fontweight='bold', fontsize=11)
    fig.tight_layout(rect=(0, .07, 1, 1)); _save(fig, path)


def figure_classes(result, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw=dict(width_ratios=[2, 3]))
    for ax, tag in zip(axes, ('validation', 'test')):
        classes = result[tag]['class_names']
        if tag == 'test':
            classes = [c for c in ('trained_pair', 'novel_unequal', 'novel_equal') if c in classes]
        x = np.arange(len(classes))
        for k, (c, _, _) in enumerate(EFFECTS):
            s = [result[tag]['classes'][cl][c] for cl in classes]
            m = np.array([v['mean'] for v in s]); lo = np.array([v['ci95'][0] for v in s]); hi = np.array([v['ci95'][1] for v in s])
            ax.errorbar(x+(k-1)*.12, m, yerr=[m-lo, hi-m], color=COLORS[c], fmt='o', ms=6, capsize=3, lw=1.8,
                        label=CONDITION_LABELS[c])
        ax.axhline(0, color=INK, lw=1)
        tw = result[tag]['class_trial_weighted']
        ax.set_xticks(x, [f'{CLASS_LABELS[cl]}\n({tw[cl]["choices"]:,} choices)' for cl in classes], fontsize=8.8)
        ax.set_ylabel('Mean Δ log loss vs C0 [nats/choice]')
        ax.set_title(TITLES[tag])
    axes[1].legend(loc='upper left', fontsize=8.8)
    fig.suptitle('E2-C  Intervention effects by pair type (participant-weighted means within type, 95% CI)', fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def figure_contributions(validation, test, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, tag, rows in zip(axes[:2], ('validation', 'test'), (validation, test)):
        r = np.array([x['conditions']['C0']['contributions']['all']['mean_abs_reward'] for x in rows])
        k = np.array([x['conditions']['C0']['contributions']['all']['mean_abs_kernel'] for x in rows])
        lim = (0, max(r.max(), k.max())*1.05)
        ax.plot(lim, lim, color=INK, lw=1, label='equal magnitude')
        ax.scatter(r, k, s=14, color=COLORS['C3'], alpha=.7, edgecolor='none', label='participant (C0)')
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
        ax.set_xlabel('Mean |β_RL·ΔQ|  reward logit'); ax.set_ylabel('Mean |β_CK·ΔK|  kernel logit')
        ax.set_title(TITLES[tag], fontsize=10)
        ax.legend(loc='upper left', fontsize=8.5)
    ax = axes[2]
    groups = [('S5 all', [x['conditions']['C0']['contributions']['all'] for x in validation]),
              ('test trained', [x['conditions']['C0']['contributions']['classes'].get('trained_pair') for x in test]),
              ('test novel\nunequal', [x['conditions']['C0']['contributions']['classes'].get('novel_unequal') for x in test]),
              ('test novel\nequal', [x['conditions']['C0']['contributions']['classes'].get('novel_equal') for x in test])]
    for j, (label, recs) in enumerate(groups):
        recs = [r for r in recs if r]
        for off, key, color, name in ((-.17, 'reward_points_to_choice_fraction', COLORS['C3'], 'reward term'),
                                      (.17, 'kernel_points_to_choice_fraction', COLORS['C1'], 'kernel term')):
            v = np.array([r[key] for r in recs])
            ax.boxplot(v, positions=[j+off], widths=.28, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=.55, color=INK2), medianprops=dict(color=INK, lw=1.5),
                       whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
            if j == 0:
                ax.plot([], [], color=color, lw=6, alpha=.55, label=name)
    ax.axhline(.5, color=INK2, lw=.8, ls=':')
    ax.annotate('reward term = 0\n(ΔQ = 0)', xy=(3-.17, 0), xytext=(3-.17, .17), ha='center', fontsize=8,
                arrowprops=dict(arrowstyle='-', color=INK2, lw=.8))
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8.5)
    ax.set_ylim(-.02, 1.02)
    ax.set_ylabel('Fraction of trials the term\npoints toward the observed choice')
    ax.set_title('Which term points to the choice?', fontsize=10)
    ax.legend(loc='lower left', fontsize=8.5)
    fig.suptitle('E2-D  Weighted reward vs kernel contributions in the original model (C0)', fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def figure_consistency(validation, test, result, path):
    plt = _plt()
    t = {r['participant']: r for r in test}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ax, (c, key, label) in zip(axes, EFFECTS):
        x = np.array([v['conditions'][c]['loss']-v['conditions']['C0']['loss'] for v in validation])
        y = np.array([t[v['participant']]['conditions'][c]['loss']-t[v['participant']]['conditions']['C0']['loss'] for v in validation])
        ax.scatter(x, y, s=12, color=COLORS[c], alpha=.65, edgecolor='none')
        ax.axhline(0, color=INK2, lw=.8); ax.axvline(0, color=INK2, lw=.8)
        rho = result['consistency'].get(key, {}).get('spearman')
        ax.set_title(f'{label.replace(chr(10), ": ")}   (Spearman ρ = {rho:+.2f})', fontsize=9.8)
        ax.set_xlabel('Session 5 effect [nats/choice]'); ax.set_ylabel('Final-test effect [nats/choice]')
    fig.suptitle('E2-E  Do participants rank similarly on session 5 and on the final test?', fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def f(v, d=4, sign=True):
    return 'NA' if v is None else (f'{v:+.{d}f}' if sign else f'{v:.{d}f}')


def ci(s, d=4):
    return f'[{s["ci95"][0]:+.{d}f}, {s["ci95"][1]:+.{d}f}]'


def _factorial_table(fac):
    names = [('total_kernel_value', 'Total kernel value: C3 − C0'),
             ('updating_given_carried_state', 'Updating, given carried-in state: C1 − C0'),
             ('carried_state_given_updating', 'Carried-in state, given updating: C2 − C0'),
             ('carried_state_without_updating', 'Carried-in state, no updating: C3 − C1'),
             ('updating_without_carried_state', 'Updating, no carried-in state: C3 − C2'),
             ('interaction', 'Interaction: (C3 − C1) − (C2 − C0)'),
             ('main_effect_carried_state', 'Main effect, carried-in state'),
             ('main_effect_updating', 'Main effect, updating')]
    rows = []
    for key, label in names:
        s = fac[key]
        share = f'{100*s["share_of_total"]:.1f}% [{100*s["share_ci95"][0]:.1f}, {100*s["share_ci95"][1]:.1f}]' if 'share_of_total' in s else ''
        rows.append([label, f(s['mean'], 5), ci(s, 5), f(s['distribution']['median'], 5),
                     f'{f(s["distribution"]["quantiles"]["p05"],3)} / {f(s["distribution"]["quantiles"]["p95"],3)}',
                     f'{s["hurts"]} / {s["helps"]} / {s["tied"]}', share])
    return table(['Contrast (Δ loss, nats/choice)', 'Mean', '95% CI', 'Median', 'P5 / P95', 'Removal worse / better / tied', 'Share of total'], rows)


def render(result, validation, test, out, prefix):
    out = Path(out); fig = out/'figures'
    names = dict(a='e2_a_condition_effects.png', b='e2_b_phase_effects.png', c='e2_c_pair_type_effects.png',
                 d='e2_d_contributions.png', e='e2_e_consistency.png')
    figure_effects(result, validation, test, fig/names['a'])
    figure_phases(result, fig/names['b'])
    figure_classes(result, fig/names['c'])
    figure_contributions(validation, test, fig/names['d'])
    figure_consistency(validation, test, result, fig/names['e'])
    write_report(result, validation, test, out/f'{prefix}.md', {k: f'figures/{v}' for k, v in names.items()})


def _contrib_rows(summary, tag):
    rows = []
    keys = ('median_abs_reward', 'median_abs_kernel', 'kernel_larger_fraction', 'kernel_flips_reward_direction_fraction',
            'reward_points_to_choice_fraction', 'kernel_points_to_choice_fraction')
    c = summary['contributions']['C0']
    groups = [('all trials', c['all'])] + [(p, c['phases'][p]) for p in PHASES] + [(k, v) for k, v in c['classes'].items()]
    for name, g in groups:
        rows.append([tag, name] + [f'{g[k]["median"]:.3f} ({g[k]["quantiles"]["p25"]:.2f}–{g[k]["quantiles"]["p75"]:.2f})' for k in keys])
    return rows


def write_report(result, validation, test, path, figures):
    path = Path(path)
    V, T = result['validation'], result['test']
    ver = result['verification']
    rl_v = np.mean([r['rl_reference_loss'] for r in validation]); rl_t = np.mean([r['rl_reference_loss'] for r in test])
    hl = lambda a: np.log(.5)/np.log1p(-np.clip(np.asarray(a), 1e-12, 1-1e-12))
    pv, pt = result['parameters']['validation'], result['parameters']['test']
    lines = [
        '# Experiment 2 — Where does RL+CK’s predictive power come from?', '',
        'Evaluation-phase ablations of the **unchanged authors’ RL + choice-kernel model** (authors model 19, '
        '`released_code_intended`, pi = 0). Fixed saved parameters only: **no parameter is refitted**, no participant '
        'is excluded, and `src/reward_pairs/authors_*.py` is untouched. Generated by `scripts/run_ck_ablation.py`; text '
        'after the marker near the end is hand-written interpretation and is preserved on regeneration.', '',
        '## Questions', '',
        '1. How much predictive performance comes from the kernel state carried into evaluation?',
        '2. How much comes from updating the kernel during evaluation?',
        '3. How much predictive information does the kernel add beyond the reward term?',
        '4. Are these contributions consistent across participants?',
        '5. What should we modify in the authors’ model next?', '',
        '## Design', '',
        '**Model.** `P(left) = σ(β_CK(K_L − K_R) + β_RL(Q_L − Q_R))`; both displayed Qs update from feedback with α_RL; '
        'K of each displayed stimulus moves toward 1 (chosen) or 0 (rejected) with α_CK. Released-code initialization '
        'sets K0 = Q0 = `pre_val`.', '',
        '**Conditions (applied from the first evaluation trial; everything else identical):**', '',
        table(['Condition', 'Kernel state at evaluation start', 'Kernel updates in evaluation', 'Kernel term in logit', 'Q'], [
            ['C0 original', 'carried in from fitted phase', 'yes (α_CK)', 'β_CK·ΔK', 'unchanged'],
            ['C1 frozen', 'carried in', '**no**', 'β_CK·ΔK (constant per pair)', 'unchanged'],
            [f'C2 reset', f'**{NEUTRAL_K} for every stimulus**', 'yes (α_CK)', 'β_CK·ΔK', 'unchanged'],
            ['C3 weight 0', 'irrelevant', 'irrelevant', '**0**', 'unchanged']]),
        f'**Why {NEUTRAL_K} for C2.** K tracks a 0/1 chosen indicator, so its fixed points lie in [0, 1]; {NEUTRAL_K} is the '
        'midpoint, i.e. "chosen half the time on appearance", the natural no-preference value in a two-alternative task. '
        'Because only K differences enter the logit, any common reset value gives identical first-trial predictions '
        '(ΔK = 0); the value only matters through how fast stimuli shown different numbers of times move away from it. '
        f'Sensitivity resets of {", ".join(str(c) for c in RESET_SENSITIVITY)} are reported. This is a **diagnostic '
        'intervention on fixed parameters, not a newly fitted model**, and the reset value was not tuned.', '',
        '**2 × 2 structure.** Reset + frozen leaves ΔK = 0 on every trial, which is exactly C3 (unit-tested). So C0–C3 '
        'form a factorial: carried-in state {kept, removed} × updating {on, off}. The kernel’s total value C3 − C0 '
        'decomposes in two orders, plus an interaction; shares are ratios of participant means, bootstrapped jointly.', '',
        '**Evaluation sets.**', '',
        '- **Primary — session 5 (S5).** Saved RL+CK fits to sessions 1–4 (`docs/history_diagnostics_results.json`, 200 '
        'starts each), replayed through S1–S4 and scored on S5. S5 keeps reward feedback, so Q updates normally in every '
        'condition. S5 contains only the eight trained pairs; they are split into high- and low-frequency training pairs '
        '(verified from cohort counts).',
        '- **Secondary — final test (exploratory).** Saved five-session fits (`docs/authors_baseline_results.json`), scored '
        'on the final test (no feedback, Q frozen). This test has already been inspected repeatedly; results here are '
        'descriptive and are **not** used to select any modification. Pairs are split into trained, novel '
        'unequal-reward, and novel equal-reward (2–3, 4–5, 6–7).', '',
        '**Effects and statistics.** `Δ = loss(condition) − loss(C0)` in nats/choice; positive means removing the '
        'ingredient hurts prediction. Participant-weighted means with 95% percentile bootstrap CIs (10,000 participant '
        'resamples, seed 20261015, whole participants resampled); trial-weighted values reported separately. Phases are '
        'thirds of each participant’s evaluation trials. Ties: |Δ| ≤ 1e-8.', '',
        '**Contributions.** Per trial, the weighted terms actually used: reward logit `β_RL·ΔQ`, kernel logit `β_CK·ΔK` '
        '(left minus right). Summaries: median magnitudes; fraction of trials where |kernel| > |reward|; fraction where '
        'the kernel reverses the reward-only direction; fraction where each term points toward the observed choice. '
        'Values are per-participant first, then summarized across participants.', '',
        '## Verification (before any interpretation)', '',
        table(['Check', 'Worst absolute discrepancy', 'Tolerance'],
              [[k.replace('_', ' '), f'{v:.2e}', f'{ver["tolerance"][k]:g}'] for k, v in ver['worst'].items()]),
        'C0 reproduces the saved S5 validation NLL and the saved final-test deviance for all 213 participants; the S1–S4 '
        'replay reproduces each saved optimizer objective and the saved carried-in Q and K exactly. Q trajectories were '
        'asserted identical across all conditions. Separately fitted RL-only references also reproduce their saved S5 '
        'losses.', '',
        f'Cohort S1–S4 training-pair counts: {ver["cohort_fit_pair_counts"]}.', '',
        '**Context: rewards are deterministic per stimulus.** Every training presentation of stimulus 1…8 pays '
        '1, 2, 2, 3, 3, 4, 4, 5 respectively, so after learning, ΔQ = ±1 on every trained pair and 0 on equal pairs. In '
        'S5 and the test the reward term is therefore close to a constant ±β_RL per trained pair; within-pair '
        'trial-to-trial variation in predictions comes almost entirely from the kernel.', '',
        '## Results', '',
        '### Losses by condition', '',
        table(['Set', 'Participants / choices', 'C0', 'C1 frozen', 'C2 reset', 'C3 weight 0', 'Separately fitted RL-only'],
              [[s, f'{R["participants"]} / {R["choices"]:,}'] +
               [f'{R["losses"][c]["mean"]:.5f}' for c in ('C0', 'C1', 'C2', 'C3')] + [f'{rl:.5f}']
               for s, R, rl in (('S5 (primary)', V, rl_v), ('Final test (exploratory)', T, rl_t))]),
        'Participant-weighted mean log loss (nats/choice). Trial-weighted: S5 ' +
        ', '.join(f'{c} {V["trial_weighted_loss"][c]:.5f}' for c in ('C0', 'C1', 'C2', 'C3')) + '; test ' +
        ', '.join(f'{c} {T["trial_weighted_loss"][c]:.5f}' for c in ('C0', 'C1', 'C2', 'C3')) + '.', '',
        f'![E2-A]({figures["a"]})', '',
        '### Session 5 (primary) — factorial decomposition', '', _factorial_table(V['factorial']),
        '### Final test (exploratory) — factorial decomposition', '', _factorial_table(T['factorial']),
        '**Reset-value sensitivity (C2 with other common values).** ' +
        '; '.join(f'{tag} {k.replace("C2_reset_", "reset ")}: {f(v["mean"], 5)} {ci(v, 5)}'
                  for tag, R in (('S5', V), ('test', T)) for k, v in R['reset_sensitivity'].items()) +
        f' (compare reset {NEUTRAL_K}: S5 {f(V["factorial"]["carried_state_given_updating"]["mean"], 5)}, test '
        f'{f(T["factorial"]["carried_state_given_updating"]["mean"], 5)}).', '',
        '### By evaluation phase', '', f'![E2-B]({figures["b"]})', '',
        table(['Set', 'Phase', 'C1 − C0 (worse/better)', 'C2 − C0 (worse/better)', 'C3 − C0 (worse/better)'],
              [[tag, p] + [f'{f(R["phases"][p][c]["mean"])} {ci(R["phases"][p][c], 3)} ({R["phases"][p][c]["hurts"]}/{R["phases"][p][c]["helps"]})'
                           for c in ('C1', 'C2', 'C3')] for tag, R in (('S5', V), ('test', T)) for p in PHASES]),
        '### By pair type', '', f'![E2-C]({figures["c"]})', '',
        table(['Set', 'Pair type', 'Choices', 'C1 − C0', 'C2 − C0', 'C3 − C0', 'Trial-weighted C1 / C2 / C3'],
              [[tag, cl.replace('_', ' '), f'{R["class_trial_weighted"][cl]["choices"]:,}'] +
               [f'{f(R["classes"][cl][c]["mean"])} {ci(R["classes"][cl][c], 3)} ({R["classes"][cl][c]["hurts"]}/{R["classes"][cl][c]["helps"]})' for c in ('C1', 'C2', 'C3')] +
               [' / '.join(f(R['class_trial_weighted'][cl][c]) for c in ('C1', 'C2', 'C3'))]
               for tag, R in (('S5', V), ('test', T)) for cl in R['class_names']]),
        '### Weighted reward and kernel contributions (C0)', '', f'![E2-D]({figures["d"]})', '',
        table(['Set', 'Subset', 'Median |reward logit|', 'Median |kernel logit|', 'Kernel larger', 'Kernel reverses reward direction',
               'Reward → choice', 'Kernel → choice'], _contrib_rows(V, 'S5') + _contrib_rows(T, 'test')),
        'Cells: median across participants (IQR) of each participant’s per-trial summary. On novel equal pairs ΔQ = 0, '
        'so the reward term is 0 and the kernel alone sets the prediction; “reverses direction” is undefined there and '
        'counted as 0.', '',
        '### Consistency across participants and evaluation sets', '', f'![E2-E]({figures["e"]})', '',
        'Spearman correlation of participant effects, S5 vs final test: ' +
        '; '.join(f'{k.replace("_", " ")} ρ = {v["spearman"]:+.2f} (p = {v["p"]:.2g})'
                  for k, v in result['consistency'].items() if isinstance(v, dict)) + '.', '',
        '### Parameter context (descriptive only)', '',
        table(['Fit', 'α_CK median (IQR)', 'Kernel half-life, appearances: median (IQR)', 'β_CK median', 'α_RL = 1 (upper bound)', 'β_RL median'],
              [[name, f'{np.median(p["alpha_ck"]):.3f} ({np.quantile(p["alpha_ck"], .25):.3f}–{np.quantile(p["alpha_ck"], .75):.3f})',
                f'{np.median(hl(p["alpha_ck"])):.1f} ({np.quantile(hl(p["alpha_ck"]), .25):.1f}–{np.quantile(hl(p["alpha_ck"]), .75):.1f})',
                f'{np.median(p["beta_ck"]):.2f}', f'{int(np.sum(np.asarray(p["alpha_rl"]) > 1-1e-6))} / {len(p["alpha_rl"])}',
                f'{np.median(p["beta_rl"]):.2f}'] for name, p in (('S1–S4 (primary)', pv), ('S1–S5 (test)', pt))]),
        '## Limitations', '',
        '- Interventions are counterfactual scorings of parameters fitted under the original model; a model *fitted* '
        'without updating or with a different reset could re-weight β_RL/β_CK and perform differently.',
        '- C3 removes the kernel but keeps the RL+CK-fitted β_RL, which was estimated jointly with the kernel; the '
        'separately fitted RL-only column is the appropriate “no kernel” model comparison.',
        '- S5 contains only trained pairs with feedback; the novel-pair and no-feedback questions can only be examined '
        'on the already-inspected final test, so those results are exploratory.',
        '- Bootstrap CIs are not multiplicity-adjusted across the many contrasts shown.',
        '- Nothing here identifies a psychological habit mechanism; the kernel is a predictive state variable.', '',
        MARKER, '']
    old = path.read_text() if path.exists() else ''
    if MARKER in old:
        lines.append(old.split(MARKER, 1)[1].strip())
    path.write_text('\n'.join(lines)+'\n')
