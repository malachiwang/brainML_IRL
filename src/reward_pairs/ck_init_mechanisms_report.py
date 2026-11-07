"""Figures and generated Markdown sections for Experiment 5 (K0 mechanisms + parameter recovery)."""
from pathlib import Path

import numpy as np

from .ck_init_mechanisms import FIT_INITS, GENERATORS, GRID_ALPHA, GRID_BETA_CK, VARIANTS
from .participant_error_report import GRID as GRIDCOLOR, INK, INK2, _plt, _save, table

MARKER = '<!-- Reviewed interpretation -->'
SHORT = {'released': 'released', 'level_shift': 'level→0.5', 'shrink_0.5': 'pattern×0.5', 'shrink_0.25': 'pattern×0.25',
         'uniform_0.5': 'uniform 0.5', 'uniform_mean': 'uniform v̄', 'permuted': 'permuted', 'minmax': 'min-max'}
VCOLOR = {'released': '#5c5c58', 'level_shift': '#8f8f89', 'shrink_0.5': '#5a9be3', 'shrink_0.25': '#9ec5f0',
          'uniform_0.5': '#2a78d6', 'uniform_mean': '#123f7a', 'permuted': '#eb6834', 'minmax': '#1baf7a'}
FCOLOR = {'released': '#5c5c58', 'uniform_0.5': '#2a78d6', 'minmax': '#1baf7a'}


def _p(rows, v, k):
    return np.array([r['variants'][v]['parameters'][k] for r in rows])


def _strip(ax, j, d, color, rng):
    ax.scatter(j+rng.uniform(-.2, .2, len(d)), d, s=6, color=color, alpha=.35, edgecolor='none')
    ax.boxplot(d, positions=[j], widths=.55, showfliers=False, medianprops=dict(color=INK, lw=1.5),
               boxprops=dict(color=INK2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))


def figure_variants(result, rows, path):
    plt = _plt(); rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, key, name, log in ((axes[0], 'alpha_ck', 'fitted α_CK', True), (axes[1], 'alpha_rl', 'fitted α_RL', True),
                               (axes[2], 'beta_ck', 'fitted β_CK', False)):
        for j, v in enumerate(VARIANTS):
            d = _p(rows, v, key)
            _strip(ax, j, np.clip(d, 1e-3, None) if log else d, VCOLOR[v], rng)
        if log:
            ax.set_yscale('log'); ax.set_ylim(8e-4, 1.3)
        ax.set_xticks(range(len(VARIANTS)), [SHORT[v] for v in VARIANTS], rotation=35, ha='right', fontsize=8.5)
        ax.set_ylabel(name + (' (log; <0.001 drawn at 0.001)' if log else '')); ax.grid(axis='y', color=GRIDCOLOR, lw=.6)
        ax.set_title(name, fontsize=10)
    axes[0].axhspan(.05, .10, color='#1baf7a', alpha=.1, lw=0)
    fig.suptitle('E5-A  Fitted parameters under eight K0 variants (Q0 unchanged; S1–S4 fits, 213 participants)', fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def figure_contrasts(result, path):
    C = result['real']['contrasts']; D = result['real']['spread_dial']
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    lam = D['lambdas']
    ax.plot(lam, D['alpha_ck_median'], color='#2a78d6', marker='o', lw=2, label='α_CK (median)')
    ax.plot(lam, D['alpha_rl_median'], color='#1baf7a', marker='s', lw=2, label='α_RL (median)')
    V = result['real']['variants']
    ax.axhline(V['released']['parameters']['alpha_ck']['median'], color=VCOLOR['released'], ls='--', lw=1, label='released α_CK')
    ax.axhline(V['permuted']['parameters']['alpha_ck']['median'], color=VCOLOR['permuted'], ls=':', lw=1.5, label='permuted α_CK')
    ax.set_xlabel('λ: K0 = 0.5 + λ·d (released pattern d, level fixed at 0.5)'); ax.set_ylabel('median fitted rate')
    ax.set_title('Pattern-strength dial', fontsize=10); ax.legend(fontsize=8, frameon=False); ax.grid(color=GRIDCOLOR, lw=.6)
    ax.set_xticks(lam)
    others = [v for v in VARIANTS if v != 'released']
    y = np.arange(len(others))
    for ax, key, title in ((axes[1], 'train_nll', 'Training NLL vs released [nats/participant]'),
                           (axes[2], 's5_loss', 'Session-5 log loss vs released [nats/choice]')):
        m = np.array([C[f'{v}-vs-released'][key]['mean'] for v in others]); ci = np.array([C[f'{v}-vs-released'][key]['ci95'] for v in others])
        ax.errorbar(m, y, xerr=[m-ci[:, 0], ci[:, 1]-m], fmt='o', color=INK, capsize=3)
        for i, v in enumerate(others):
            ax.scatter(m[i], y[i], color=VCOLOR[v], s=50, zorder=3)
        ax.axvline(0, color=INK, lw=1); ax.set_yticks(y, [SHORT[v] for v in others]); ax.invert_yaxis()
        ax.set_title(title + '\n(mean, 95% participant bootstrap CI; negative = better)', fontsize=10); ax.grid(axis='x', color=GRIDCOLOR, lw=.6)
    fig.suptitle('E5-B  What in K0 matters: level, pattern strength, and pattern identity', fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def figure_recovery(result, synth, path):
    plt = _plt()
    fig, axes = plt.subplots(len(GENERATORS), len(FIT_INITS), figsize=(14, 9), sharex=True, sharey=True)
    data = [d for r in synth for d in r['datasets']]
    for i, g in enumerate(GENERATORS):
        for j, f in enumerate(FIT_INITS):
            ax = axes[i, j]
            ds = [d for d in data if d['generator'] == g]
            t = np.clip([d['truth'][0] for d in ds], 1e-3, 1); e = np.clip([d['fits'][f]['theta'][0] for d in ds], 1e-3, 1)
            ax.scatter(t, e, s=8, alpha=.4, color=FCOLOR[f], edgecolor='none')
            ax.plot([1e-3, 1], [1e-3, 1], color=INK, lw=1, ls='--')
            ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(color=GRIDCOLOR, lw=.5)
            rec = result['recovery'][f'{g}->{f}']
            ax.set_title(f'generated {g}, fitted {f}\nmedian fitted/true α {rec["alpha_ck_ratio_median"]:.2f}×; '
                         f'Spearman {rec["spearman_alpha_ck"]:.2f}', fontsize=9, fontweight='bold' if g == f else 'normal')
            if i == len(GENERATORS)-1:
                ax.set_xlabel('true α_CK')
            if j == 0:
                ax.set_ylabel('fitted α_CK')
    fig.suptitle('E5-C  Parameter recovery from simulated choices (truth = each participant\'s own fit; 3 seeds each; diagonal = matched K0)',
                 fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def figure_grid(result, path):
    G = result['grid']
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, g in zip(axes, GENERATORS):
        for f, ls in (('released', '-'), ('uniform_0.5', '--')):
            for b, mk in zip(GRID_BETA_CK, ('o', 's')):
                cells = [c for c in G if c['generator'] == g and c['beta_ck'] == b]
                med = [c[f]['alpha_median'] for c in cells]; lo = [c[f]['alpha_q25'] for c in cells]; hi = [c[f]['alpha_q75'] for c in cells]
                x = np.array(GRID_ALPHA)*(1.04 if f == 'released' else .96)
                ax.errorbar(x, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], color=FCOLOR[f], ls=ls, marker=mk, ms=5, capsize=2,
                            label=f'fit {f}, true β_CK={b:g}')
        ax.plot([.015, .5], [.015, .5], color=INK, lw=1, ls=':', label='fitted = true')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('true α_CK'); ax.grid(color=GRIDCOLOR, lw=.5)
        ax.set_xticks(GRID_ALPHA, [f'{a:g}' for a in GRID_ALPHA]); ax.minorticks_off()
        ax.set_yticks([.01, .02, .05, .1, .2, .5], ['0.01', '0.02', '0.05', '0.1', '0.2', '0.5']); ax.set_ylim(1e-3, .7)
        ax.set_title(f'Generated with K0 = {g}', fontsize=10)
    axes[0].set_ylabel('fitted α_CK (median, IQR over sequences)'); axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle('E5-D  Controlled recovery grid (nuisance at cohort medians; 40 real sequences per cell)', fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def figure_contributions(result, path):
    V = result['real']['variants']
    plt = _plt()
    segs = ('S1 early', 'S1 late', 'S2', 'S4', 'S5')
    show = ('released', 'level_shift', 'uniform_0.5', 'permuted', 'minmax')
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, key, title in ((axes[0], 'zk', 'Kernel term |β_CK (K_L − K_R)|'), (axes[1], 'zq', 'Reward term |β_RL (Q_L − Q_R)|')):
        for v in show:
            ax.plot(range(len(segs)), [V[v]['segments'][s][key]['median'] for s in segs], marker='o', lw=2, color=VCOLOR[v], label=SHORT[v])
        ax.set_xticks(range(len(segs)), ['S1 first\nthird', 'S1 rest', 'S2', 'S4', 'S5\n(held out)'])
        ax.set_title(title, fontsize=10); ax.grid(axis='y', color=GRIDCOLOR, lw=.6)
    axes[0].set_ylabel('cohort median of per-participant median |term|'); axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('E5-E  Weighted contributions by training segment under selected K0 variants', fontweight='bold', fontsize=11)
    fig.tight_layout(); _save(fig, path)


def render(result, rows, synth, grid, out, prefix):
    out = Path(out); fig = out/'figures'
    names = dict(a='e5_a_variant_parameters.png', b='e5_b_mechanism_contrasts.png', c='e5_c_recovery.png',
                 d='e5_d_recovery_grid.png', e='e5_e_contributions.png')
    figure_variants(result, rows, fig/names['a'])
    figure_contrasts(result, fig/names['b'])
    figure_recovery(result, synth, fig/names['c'])
    figure_grid(result, fig/names['d'])
    figure_contributions(result, fig/names['e'])
    write_report(result, out/'experiment5_ck_init_mechanisms.md', names)


def _ci(s, d=4):
    return f'{s["mean"]:+.{d}f} [{s["ci95"][0]:+.{d}f}, {s["ci95"][1]:+.{d}f}]'


def write_report(result, path, figures):
    """Generated tables are placed in an appendix; hand-written analysis lives above the marker."""
    R = result['real']; V = R['variants']; C = R['contrasts']; REC = result['recovery']; G = result['grid']
    gen = [
        '## Appendix: generated results',
        '',
        f'Verification: released-K0 refit vs saved fits, worst deviance difference {result["verification"]["worst"]["released_vs_saved_deviance"]:.1e}; '
        f'worst S5 NLL difference {result["verification"]["worst"]["released_s5_nll"]:.1e}. '
        f'{R["analysed"]}/{R["participants"]} participants analysed (excluded: {R["excluded"] or "none"}). Compute: ' +
        '; '.join(f'{k} {sum(e["seconds"] for e in v):.0f} s' for k, v in result['compute'].items()) + ' (4 workers).',
        '',
        '### A1. Fitted parameters by K0 variant',
        '',
        table(['variant', 'K0', 'K0 level (median)', 'K0 spread (median)', 'median α_CK [IQR]', 'half-life', 'α_CK < 0.01',
               'median α_RL', 'α_RL at bound', 'median β_CK', 'median β_RL', 'failed starts', 'median starts reaching best'],
              [[v, V[v]['description'], f'{V[v]["k0"]["level"]["median"]:.2f}', f'{V[v]["k0"]["spread"]["median"]:.2f}',
                f'{V[v]["parameters"]["alpha_ck"]["median"]:.3f} [{V[v]["parameters"]["alpha_ck"]["quantiles"]["p25"]:.3f}–{V[v]["parameters"]["alpha_ck"]["quantiles"]["p75"]:.3f}]',
                f'{V[v]["half_life_median"]:.1f}', V[v]['alpha_below_001'], f'{V[v]["parameters"]["alpha_rl"]["median"]:.3f}',
                V[v]['boundary']['alpha_rl'], f'{V[v]["parameters"]["beta_ck"]["median"]:.2f}', f'{V[v]["parameters"]["beta_rl"]["median"]:.2f}',
                V[v]['failed_starts'], f'{V[v]["reaching_best"]["median"]:.0f}'] for v in VARIANTS]),
        '',
        '### A2. Paired contrasts (participant-weighted mean [95% bootstrap CI])',
        '',
        table(['contrast', 'log α_CK ratio', 'median ratio', 'α_CK ↓ / ↑', 'Δ α_RL', 'Δ β_CK', 'Δ train NLL (nats)', 'Δ S1 loss', 'Δ S2–S4 loss', 'Δ S5 loss'],
              [[k.replace('-vs-', ' vs '), _ci(c['log_alpha_ck'], 2), f'{c["log_alpha_ck"]["median_ratio"]:.2f}×',
                f'{c["alpha_ck_decreases"]} / {c["alpha_ck_increases"]}', _ci(c['alpha_rl'], 3), _ci(c['beta_ck'], 2), _ci(c['train_nll'], 2),
                _ci(c['s1_loss']), _ci(c['s2_s4_loss']), _ci(c['s5_loss'])] for k, c in C.items()]),
        '',
        f'Training-likelihood winner per participant (descriptive; equal parameter counts): {R["training_winner_counts"]}.',
        '',
        '### A3. Weighted contributions (cohort median of per-participant median |term|)',
        '',
        table(['variant', 'first 20 |z_K|', 'first 20 |z_Q|'] + [f'{s} |z_K|' for s in ('S1 early', 'S2', 'S5')] + [f'{s} |z_Q|' for s in ('S1 early', 'S5')],
              [[v, f'{V[v]["first20"]["zk"]["median"]:.2f}', f'{V[v]["first20"]["zq"]["median"]:.2f}'] +
               [f'{V[v]["segments"][s]["zk"]["median"]:.2f}' for s in ('S1 early', 'S2', 'S5')] +
               [f'{V[v]["segments"][s]["zq"]["median"]:.2f}' for s in ('S1 early', 'S5')] for v in VARIANTS]),
        '',
        '### A4. Parameter recovery, realistic regime',
        '',
        table(['generated → fitted', 'datasets', 'median fitted/true α_CK', 'α over / under (>10%)', 'Spearman(true, fitted)',
               'median error α_RL', 'median error β_CK', 'optimization failures', 'truth beats fit', 'true α within 1 nat',
               'median S5 loss − truth', 'median S5 |ΔP| vs truth'],
              [[k, r['n'], f'{r["alpha_ck_ratio_median"]:.2f}×', f'{r["alpha_ck_overestimated"]} / {r["alpha_ck_underestimated"]}',
                f'{r["spearman_alpha_ck"]:.2f}', f'{r["parameter_error"]["alpha_rl"]["median"]:+.3f}', f'{r["parameter_error"]["beta_ck"]["median"]:+.2f}',
                r['optimization_failures'], r['truth_beats_fit'], f'{r["true_alpha_within_1_nat"]}/{r["n"]}',
                f'{r["s5_loss_minus_truth"]["median"]:+.4f}', f'{r["s5_mean_abs_dp_vs_truth"]["median"]:.3f}'] for k, r in REC.items()]),
        '',
        table(['mismatch', 'Δ log α vs matched fit (participant mean [CI])', 'median S1-early loss − matched', 'median train loss − matched',
               'median S5 loss − matched', 'median S5 |ΔP| vs matched'],
              [[k, _ci(r['log_alpha_mismatch_minus_matched'], 2), f'{r["s1_early_minus_matched"]["median"]:+.4f}',
                f'{r["train_nll_minus_matched"]["median"]:+.4f}', f'{r["s5_loss_minus_matched"]["median"]:+.4f}',
                f'{r["s5_mean_abs_dp_vs_matched"]["median"]:.3f}'] for k, r in REC.items() if r.get('s1_early_minus_matched')]),
        '',
        '### A5. Controlled recovery grid (median fitted α_CK [IQR]; 40 sequences per cell)',
        '',
        table(['generator', 'true α', 'true β_CK', 'fit released', 'fit uniform 0.5', 'true α within 1 nat (rel / uni)',
               'opt. failures (rel / uni)', 'median α_RL (rel / uni)', 'train loss released − uniform'],
              [[c['generator'], f'{c["alpha"]:g}', f'{c["beta_ck"]:g}',
                f'{c["released"]["alpha_median"]:.3f} [{c["released"]["alpha_q25"]:.3f}–{c["released"]["alpha_q75"]:.3f}]',
                f'{c["uniform_0.5"]["alpha_median"]:.3f} [{c["uniform_0.5"]["alpha_q25"]:.3f}–{c["uniform_0.5"]["alpha_q75"]:.3f}]',
                f'{c["released"]["true_alpha_within_1_nat"]} / {c["uniform_0.5"]["true_alpha_within_1_nat"]}',
                f'{c["released"]["optimization_failures"]} / {c["uniform_0.5"]["optimization_failures"]}',
                f'{c["released"]["alpha_rl_median"]:.2f} / {c["uniform_0.5"]["alpha_rl_median"]:.2f}',
                f'{c["released_minus_uniform_train_loss"]:+.4f}'] for c in G]),
        '',
    ]
    figs = ['', f'![E5-A](figures/{figures["a"]})', '', f'![E5-B](figures/{figures["b"]})', '', f'![E5-C](figures/{figures["c"]})', '',
            f'![E5-D](figures/{figures["d"]})', '', f'![E5-E](figures/{figures["e"]})', '']
    old = path.read_text() if path.exists() else ''
    head = old.split(MARKER, 1)[0].rstrip() if MARKER in old else '# Experiment 5: choice-kernel initialization mechanisms and parameter recovery\n\n(Interpretation pending.)'
    path.write_text(head + '\n\n' + MARKER + '\n\n' + '\n'.join(gen + ['### Figures'] + figs))
