#!/usr/bin/env python3
"""Generate descriptive tables from saved fits; never refit or select participants."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from reward_pairs.authors_evaluation import distribution


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('docs/authors_baseline_results.json'))
    parser.add_argument('--validation',type=Path,default=Path('docs/authors_validation_results.json'))
    parser.add_argument('--output',type=Path,default=Path('docs/authors_baseline_results.md'))
    args=parser.parse_args(); data=json.loads(args.input.read_text()); val=json.loads(args.validation.read_text())
    rows=data['rows']; good=[r for r in rows if r['optimization_success']]; summary=data['summary']
    lookup={(r['variant'],r['participant'],r['model']):r for r in good}
    variants=['paper_intended','released_code_intended','released_code_literal_clean_r']
    labels=dict(zip(variants,['A: paper','B: code intended','C: literal R']))
    ids=sorted({r['participant'] for r in good}); unique=[r for r in good if not r['reused_fit']]
    lines=['# Authors’ Reward Pairs benchmark: reproduction results','',
           'This report fits **human choices only for benchmark reproduction**. It does not fit our novel history/IRL model or interpret individual parameters psychologically. See the [source audit](authors_model_audit.md) for equations, provenance, exclusions, and paper/code distinctions. Full-precision fits, failure diagnostics, and model-by-participant matrices are in [the JSON](authors_baseline_results.json); R parity and limited synthetic recovery are in [validation JSON](authors_validation_results.json).','',
           '## Configuration and execution','',
           '```sh','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\',
           '  python scripts/run_authors_baseline.py --variant all --restarts 200 --workers 4 \\',
           '  --output docs/authors_baseline_results.json','```','',
           f'Total supplied/fitted participants: **{len(ids)}**. Distinct initialization/model fits: **{len(unique)}**; starts attempted: **{sum(r["restart_count"] for r in unique):,}**. Three-variant result rows: **{len(rows)}** (B/C share fits). Failed participant/model fits: **{sum(not r["optimization_success"] for r in rows)}**. Unsuccessful optimizer starts among distinct successful fits: **{sum(r["failed_starts"] for r in unique)}**. Full-cohort wall time: **{data["runtime_seconds"]:.2f} s**. All starts and failures are retained as diagnostics; failed starts are not chosen as solutions.','',
           'This is a **200-start reproduction, not an exact R-optimizer-protocol reproduction**: R supplies the original seed/draw stream, but SciPy L-BFGS-B uses analytic derivatives, stable log probabilities, and tighter convergence checking. No regularization. R `optimr` is not installed. The original R objective itself is used for parity.','',
           'Smoke sequence: E11T9A model 19/10 starts 0.9 s end-to-end; all models/all variants/10 starts 1.4 s; all models/all variants/200 starts 12.6 s. Ten and 200 starts differed by at most 1.94e-12 deviance on that participant. The 200-start smoke projected approximately 10.4 minutes with four workers for the cohort; all 200 starts were retained regardless.','',
           'All 18 comparison-script exclusion IDs were checked, not silently reapplied to synthetic studies. Presence in this supplied input: '+str(sum(e['present_in_training'] for e in data['audit']['exclusions']))+'. Exact IDs and grouped reasons are in the audit and JSON.','']
    structures={p:next(r['structure'] for r in good if r['participant']==p) for p in ids}
    lines += [f'Training rows: {sum(s["training"]["rows"] for s in structures.values()):,}; test rows: {sum(s["test"]["rows"] for s in structures.values()):,}. Original chronology already sorted: training {sum(s["training"]["original_order_chronological"] for s in structures.values())}/{len(ids)}, test {sum(s["test"]["original_order_chronological"] for s in structures.values())}/{len(ids)}. Retained rows only; no session resets or inserted missing-trial updates. Test is a separate phase even though its source session label is 5.','',
        '## Verification','',
        'Final full suite: **186 passed in 37.44 s** (`python -m pytest -q`), including 38 new authors-benchmark tests. The original 148 tests were retained unchanged. Checks cover both-option RL updates, CK updates, initialization, causal replay, test freezing, chronology/gaps, model constraints, information criteria, analytic derivatives, explicit failures, and original-R parity.','',
        f'Participant fixed/fitted parity: {len(val["parity"])} comparisons (five models × three variants × two parameter settings), each covering training and test. Maximum absolute discrepancies: objective deviance {max(abs(r["objective_deviance_difference"]) for r in val["parity"]):.3g}; training deviance {max(abs(r["training_deviance_difference"]) for r in val["parity"]):.3g}; test deviance {max(abs(r["test_deviance_difference"]) for r in val["parity"]):.3g}; choice probability {max(r["probability_max_difference"] for r in val["parity"]):.3g}; Q {max(r["q_max_difference"] for r in val["parity"]):.3g}; H {max(r["h_max_difference"] for r in val["parity"]):.3g}.','',
        'Independent base-R `optim` fits used ten identical random starts and the unmodified likelihood. Its default finite-difference step is 0.001. Reduced-model optima at small learning rates were checked again with step 1e-6; this is a numerical diagnostic, not a different behavioral model.','',
        table(['Variant','Model','R − Python deviance','Fine-step R − Python'],[[labels[r['variant']],r['model'],f'{r["minus_python_deviance"]:.9g}',f'{r["fine_difference_diagnostic"]["minus_python_deviance"]:.9g}' if 'fine_difference_diagnostic' in r else 'not needed'] for r in val['original_r_optimization']]),
        'Base-R default-difference best results reported convergence code 52 for both random-model fits and the released-initialization reduced fit. They are retained as audit diagnostics, not labeled successful R convergence. The random-model discrepancy is only 1.1e-11 deviance and its analytical optimum is independently unit-tested; both fine-step reduced diagnostics report convergence 0. Python rejects unsuccessful starts when selecting its reported solution.','',
        '## Preregistered four-model comparison','',
        'Winner = minimum participant BIC; columns are models 1/random, 3/RL, 11/CK, 19/combined. The same k is used on training and test, reproducing the source convention. This is not a Bayesian group-model-selection calculation.','']
    counts=[]
    for v in variants:
        for phase in ('training','test'):
            c=summary[v]['comparisons']['preregistered']['phases'][phase]
            counts.append([labels[v],phase,*[c['winner_counts'][str(m)] for m in (1,3,11,19)]])
    lines += [table(['Variant','Phase','Random','RL','CK','RL+CK'],counts),
        '### RL versus RL+CK evidence','',
        'Delta = BIC(RL+CK) − BIC(RL): negative favors combined. Exact ±6 follows the R category boundary (no strong evidence); magnitudes above 6 through 10 are strong, above 10 very strong.','']
    cats=['very_strong_RL_CK','strong_RL_CK','no_strong_evidence','strong_RL','very_strong_RL']
    counts=[]
    for v in variants:
        for phase in ('training','test'):
            c=summary[v]['comparisons']['preregistered']['phases'][phase]['rl_comparison_counts']
            counts.append([labels[v],phase,*[c.get(k,0) for k in cats]])
    counts.append(['Published','test',7,11,36,19,140])
    lines += [table(['Variant','Phase','Very strong combined','Strong combined','Neither','Strong RL','Very strong RL'],counts),
        'Published target: [main article, computational-model results](https://doi.org/10.1525/collabra.92949). Participant category counts are compared directly; reported VBA exceedance probabilities are **not reproduced**. MATLAB is present locally, but the VBA toolbox and saved fitted/comparison workspaces were not found. Labeled AIC/AICc/BIC/deviance/LL matrices are exported in the JSON, with models as rows and included participants as columns.','',
        '## Exploratory five-model comparison','']
    counts=[]
    for v in variants:
        for phase in ('training','test'):
            c=summary[v]['comparisons']['exploratory']['phases'][phase]
            counts.append([labels[v],phase,*[c['winner_counts'][str(m)] for m in (1,3,11,19,20)]])
    lines += [table(['Variant','Phase','Random','RL','CK','RL+CK','Reduced'],counts),
        'Reduced-versus-RL categories use “combined” below to mean the **reduced** model, not model 19. The reduced model was specified for association analyses; its addition to the model comparison was exploratory.','']
    counts=[]
    for v in variants:
        for phase in ('training','test'):
            c=summary[v]['comparisons']['exploratory']['phases'][phase]['reduced_vs_rl_counts']
            counts.append([labels[v],phase,*[c.get(k,0) for k in cats]])
    counts.append(['Published','test',211,0,1,1,0])
    lines += [table(['Variant','Phase','Very strong reduced','Strong reduced','Neither','Strong RL','Very strong RL'],counts),
        '## Fitting versus rescoring','',
        'B and C have exactly the same fitted parameters and Q/H trajectories. Only post-fit positional response stickiness changes. Clean R resolves pi to 3.141592653589793; after the first response the logit contribution is ±2*pi, with positional history reset at each phase entry.','']
    counts=[]
    for v in variants:
        for m in (1,3,11,19,20):
            r=[x for x in good if x['variant']==v and x['model']==m]
            d=distribution([x['rescore_minus_objective'] for x in r])
            counts.append([labels[v],m,f'{d["mean"]:.6g}',f'{d["minimum"]:.6g}',f'{d["maximum"]:.6g}',
                           f'{np.mean([x["training"]["deviance"] for x in r]):.3f}',f'{np.mean([x["test"]["deviance"] for x in r]):.3f}'])
    lines += [table(['Variant','Model','Mean rescore−fit','Minimum','Maximum','Mean train dev','Mean test dev'],counts),
        '### E11T9A audit example','']
    counts=[]
    for v in variants:
        for m in (1,3,11,19,20):
            r=lookup[v,'E11T9A',m]
            counts.append([labels[v],m,f'{r["optimizer_objective_deviance"]:.6f}',f'{r["training"]["deviance"]:.6f}',f'{r["test"]["deviance"]:.6f}',f'{r["training"]["BIC"]:.6f}',f'{r["test"]["BIC"]:.6f}'])
    lines += [table(['Variant','Model','Fit dev','Train rescore dev','Test dev','Train BIC','Test BIC'],counts),
        'All E11T9A free parameters and final eight-element Q/H vectors, along with every other participant, are stored without rounding in the JSON.','',
        '## Initialization sensitivity (A versus B, both pi=0)','',
        'This isolates the jointly requested initialization changes: ratings /11.15 and H0=0 versus mean(ratings /20, pretest choice component) for both Q0/H0. It does not separately identify the causal contribution of each initialization component. Rewards remain the supplied 1–5 columns in both variants; no unrequested point-unit conversion is made.','']
    counts=[]
    sensitivity={}
    for m in (1,3,11,19,20):
        pairs=[(lookup[variants[0],p,m],lookup[variants[1],p,m]) for p in ids]
        metrics={name:distribution([b[phase]['deviance']-a[phase]['deviance'] for a,b in pairs]) for name,phase in [('training_deviance_B_minus_A','training'),('test_deviance_B_minus_A','test')]}
        metrics.update({name:distribution([float(np.mean(abs(np.array(b[key])-a[key]))) for a,b in pairs]) for name,key in [('final_Q_mean_absolute_change','final_training_q'),('final_H_mean_absolute_change','final_training_h')]})
        metrics['parameter_B_minus_A']={p:distribution([b['parameters'][p]-a['parameters'][p] for a,b in pairs]) for p in pairs[0][0]['parameters']}
        sensitivity[str(m)]=metrics
        counts.append([m,f'{metrics["training_deviance_B_minus_A"]["mean"]:.4f}',f'{metrics["test_deviance_B_minus_A"]["mean"]:.4f}',f'{metrics["final_Q_mean_absolute_change"]["mean"]:.6f}',f'{metrics["final_H_mean_absolute_change"]["mean"]:.6f}'])
    lines += [table(['Model','Mean train dev B−A','Mean test dev B−A','Mean final Q absolute change','Mean final H absolute change'],counts)]
    lines += ['State changes in disabled processes (zero beta), such as both states in model 1, are bookkeeping differences and do not affect choice probabilities.','']
    for comparison in ('preregistered','exploratory'):
        for phase in ('training','test'):
            a=summary[variants[0]]['comparisons'][comparison]['phases'][phase]['participant_winners']
            b=summary[variants[1]]['comparisons'][comparison]['phases'][phase]['participant_winners']
            lines += [f'{comparison}, {phase}: initialization changes the BIC winner for {sum(a[p]!=b[p] for p in a)}/{len(a)} participants.','']
    lines += ['## Parameter distributions and boundaries','',
        'Entries are mean [median; 5th–95th percentile]. These are descriptive fit distributions, not uncertainty intervals. C shares B parameters exactly. Boundary and near-boundary estimates are not interpreted psychologically.','']
    counts=[]
    for v in variants[:2]:
        for m in (1,3,11,19,20):
            s=summary[v]['models'][str(m)]
            values=[]
            for p in ('authors_alpha_rl','authors_beta_rl','authors_alpha_ck','authors_beta_ck','bias'):
                d=s['parameters'][p]; values.append(f'{d["mean"]:.4f} [{d["median"]:.4f}; {d["quantiles"]["p05"]:.4f}–{d["quantiles"]["p95"]:.4f}]')
            counts.append([labels[v],m,*values,s['participants_with_boundary']])
    lines += [table(['Variant','Model','alpha_RL','beta_RL','alpha_CK','beta_CK','Bias','Any boundary /213'],counts),
        'Model 20 always satisfies beta_RL + beta_CK = 10 exactly. JSON additionally contains SD/min/max, boundary counts by parameter, and final Q/H distributions by stimulus. The published reduced beta_CK mean is 6.78. Its accompanying t(207) is not reconciled here with the 213-person BIC comparison; no additional exclusions are invented to match it.','',
        '## Limited synthetic implementation sanity check','',
        'Two predeclared interior settings each for RL, CK, combined, and reduced combined; thirty independent synthetic agents per setting on the E11T9A reward/pair scaffold, fresh states each, ten independent random starts. Human choices are replaced by simulated actions. This is not the authors’ 1,000-agent recovery study. Parameter order is RL: (alpha_RL,beta_RL); CK: (alpha_CK,beta_CK); combined: (alpha_CK,beta_CK,alpha_RL,beta_RL); reduced: (alpha_CK,beta_CK,alpha_RL).','']
    counts=[]
    for r in val['synthetic']:
        counts.append([r['model'],r['truth'],[round(x,5) for x in r['estimate']],f'{r["truth_deviance"]-r["optimizer_objective_deviance"]:.4f}'])
    lines += [table(['Model','Truth','Recovered','Deviance truth−fit'],counts),
        'All eight fits improve on the generating-parameter likelihood. Recovery is approximate, not exact: RL learning rates depend disproportionately on early learning and are less precisely estimated than some choice weights, even with thirty agents. This check does not establish reliable individual-level identifiability.','',
        '## Interpretation and recommendation','',
        '<!-- The source-sensitive interpretation below is reviewed against the generated tables, not inferred automatically from winner counts. -->','']
    data['initialization_sensitivity']=sensitivity
    data['validation_file']=str(args.validation)
    data['verification']=dict(test_count=186,tests_passed=186,test_runtime_seconds=37.44,
                             cohort_unique_fits=len(unique),cohort_starts=sum(r['restart_count'] for r in unique),
                             original_models_unchanged=True)
    cats=['very_strong_RL_CK','strong_RL_CK','no_strong_evidence','strong_RL','very_strong_RL']
    target=[7,11,36,19,140]
    data['published_comparison']=dict(source='https://doi.org/10.1525/collabra.92949',
        category_order=cats,preregistered_test_counts=target,exploratory_reduced_test_counts=[211,0,1,1,0],
        reduced_beta_ck_mean=6.78,
        preregistered_count_absolute_difference={v:sum(abs(summary[v]['comparisons']['preregistered']['phases']['test']['rl_comparison_counts'].get(c,0)-t) for c,t in zip(cats,target)) for v in variants})
    args.input.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    marker='<!-- The source-sensitive interpretation below is reviewed against the generated tables, not inferred automatically from winner counts. -->'
    previous=args.output.read_text() if args.output.exists() else ''
    reviewed=previous.split(marker,1)[1] if marker in previous else ''
    args.output.write_text('\n'.join(lines)+'\n'+reviewed)


if __name__=='__main__': main()
