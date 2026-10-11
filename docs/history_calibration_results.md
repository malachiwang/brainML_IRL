# History uncertainty and planning distinguishability: synthetic calibration

This report tests two different claims: evidence for a history coefficient and
evidence that history-dependent behavior is forward-looking. The
[method explanation](history_calibration.md) specifies the unchanged mechanisms,
profile search, and held-out comparisons. Full-precision parameters, profile
evaluations, outcomes, failure records, distributions, and scaffold summaries
are stored in [the JSON artifact](history_calibration_results.json).

## Fixed design and reproduction

```sh
.venv/bin/python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
.venv/bin/python scripts/run_history_calibration.py \
  --beta 1 --gamma 0.95 --planning-horizon 3 \
  --alphas 0 0.5 1 --rollouts 1 5 20 \
  --seed-batches 0 1 2 3 4 \
  --uncertainty-seed-batches 0 1 2 3 4 5 6 7 8 9 \
  --fresh-eval-agents 20 --workers 4 \
  --output docs/history_calibration_results.json
```

The thread settings bound numerical-library parallelism inside each worker;
they do not change either likelihood. Primary h=0 versus h=3, beta=1,
gamma=0.95, true alpha values, rollout counts, windows, and seed counts were
fixed before looking at classification results. No batch expansion, subset
reduction, model modification, penalty, or participant exclusion was made.

The source remains `data_4_analysis_RP_training`, SHA-256
`037dde9d968dd8d0705891a31315930d70f475fd33a99c4743712eed89e9c5e1`.
Only ID/session/trial/stim1/stim2 are projected before normalization. Human
actions, rewards, test data, accuracy, and prior human histories are unused.
Raw and reference files remain Git-ignored.

The exact stored twelve-scaffold selection from the earlier replication is
reused and checked against current structural summaries:

| Scaffold | Retained trials | Sessions 1–5 |
|---|---:|---|
| r01b4b | 667 | 128, 138, 127, 141, 133 |
| m03e0b | 745 | 149, 142, 148, 152, 154 |
| s02l2z | 758 | 144, 150, 150, 156, 158 |
| m09t1z | 766 | 149, 155, 155, 152, 155 |
| h01m4k | 771 | 152, 156, 160, 147, 156 |
| e05w0s | 775 | 153, 153, 157, 156, 156 |
| s05t1r | 779 | 150, 156, 160, 155, 158 |
| E11T9A | 783 | 153, 157, 156, 159, 158 |
| k08c1o | 783 | 150, 158, 160, 156, 159 |
| m08m7h | 785 | 153, 157, 158, 159, 158 |
| l06s0m | 788 | 153, 159, 159, 159, 158 |
| s04t2z | 797 | 159, 160, 159, 159, 160 |

The JSON also retains their pair frequencies and gap counts. The structural
selection was not revised based on difficulty or any calibration result.

### Sizes, independence, and denominators

Uncertainty uses 12 × 3 alphas × 3 counts × 10 batches = **1,080 datasets**.
Each has one unchanged unrestricted myopic fit and a profile interval obtained
by repeatedly optimizing seven nuisance utilities at fixed alpha.

Model comparison uses 12 × 3 × 3 × 5 batches × 2 generators = **1,080 datasets**.
Each has both myopic and planning fits on all sessions and separately on sessions
1–4: **4,320 model fits**. Including unrestricted profile fits gives 5,400
unrestricted fitting calls, plus the conditional profile optimizations.

Training RNG roots match the prior replication, so its alpha point estimates
are intentionally reproduced rather than counted as new independent evidence.
New fresh-agent evaluation roots use a separate fixed RNG namespace. Each
generator/scaffold/alpha/batch has 20 fresh agents, reused across nested training
sizes. Both candidate models score identical evaluation actions. Alpha and
generator conditions use common random draws. No rollout is concatenated to
another history. The 1/5/20 counts mean independent agents with common true
parameters, not extra sessions of one agent.

Each uncertainty condition has 120 scaffold × batch datasets. Each model-selection
condition has 60. Summaries equally weight these configurations, not individual
trials; shorter scaffolds do not get less weight. Conditional estimates and
paired prediction summaries report successful denominators and failures explicitly.
Per-scaffold coverage uses only ten batches and is therefore coarse (10% steps).

### Pre-run verification and timing

The existing 131 tests passed before implementation. The expanded 145-test suite
passed before the full grid. A one-agent E11T9A/alpha=0/batch=0 smoke profile
took **0.040 s**, with 16 conditional utility fits. Each complete model-comparison
configuration took about **0.59 s** for its four fits and both held-out schemes.

A second timing smoke run covered 1/5/20 agents. Profile times were
0.040/0.083/0.283 s; paired comparison times were approximately
0.59/1.68/5.63 s per generated dataset. Its total wall time was 17.75 s.
The projected full serial study was about 50–55 minutes. This justified running
the complete grid with four workers, without changing its design. These timings
were used only for feasibility, not model/horizon selection.

## Profile interval specification

The cutoff is `chi2_1(.95)=3.841458820694124` for twice the profile log-likelihood
drop. At every alpha, seven utilities are re-fitted without regularization.
Geometrically expanding brackets are followed by Brent root-finding. Search
distance is capped at 64 from the unrestricted estimate on each side; missing
crossings are labeled open, never replaced by artificial finite endpoints.
The optional Wald comparison uses the full inverse observed-information matrix.
See the method document for tolerances and open-side handling.

The nominal cutoff's finite-sample coverage is an empirical question. Successful
optimization or an accurately computed LR endpoint is not, by itself, evidence
that an interval has 95% coverage.

## Completion and independent audit

The full run took **881.32 seconds (14.69 minutes)** including final serialization.
All **1,080 unrestricted profile fits, 1,080 intervals, 14,698 conditional
utility optimizations, and 4,320 model-comparison fits succeeded**. There were
zero numerical failures and zero open intervals. Thus 20,098 optimization calls
completed, counting unrestricted and conditional fitting separately. Conditional
profile evaluations were cached within each dataset; duplicate alpha queries
were not extra fits. Summed per-configuration wall times across workers were
157.09 s for profiles and 3,160.19 s for comparisons, not serial elapsed runtime.

The final suite has **148 passing tests**, including all 131 existing tests and
17 new calibration cases. Verification during the experiment took 41.70 s;
the final post-experiment rerun passed in **35.08 s**.
Tests cover fixed-alpha nuisance refitting, agreement with independent BT at
alpha=0, reference invariance, finite endpoint LR values, forced open search
sides, nonfinite/failed optimization, controlled coverage, shrinking widths,
aggregation/failure denominators, correct delta/confusion signs, null controls,
disjoint fresh streams, window partitions, and no refitting/leakage in scoring.

An independent post-run audit verified every requested configuration key,
recomputed all stored summaries, checked all source hashes, verified all w1
references equal zero, and checked window counts and NLL additivity. All 1,080
unrestricted alpha estimates exactly reproduce the previous replication's
corresponding point estimates. Paired alpha=0 generators have identical training
choices, fresh choices, fitted parameters, and scores. Maximum absolute endpoint
LR error from the chi-square cutoff was **9.56e-7**. None of the old simulator,
recovery, planning, history, or preprocessing modules changed.

## Part I: alpha uncertainty

Each row below uses 120 successful intervals. SD is sample SD across estimates,
not an estimated standard error. Alpha is unrestricted; true positive values
were not enforced. All quantities retain full precision in JSON.

| True alpha | Agents | Mean alpha | Median alpha | Bias | MAE | SD | Mean width | Median width | Coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | -1.0339 | -0.8241 | -1.0339 | 1.1215 | 1.0451 | 3.0133 | 2.9327 | 86/120 = 71.7% |
| 0 | 5 | -0.1034 | -0.0957 | -0.1034 | 0.2285 | 0.2806 | 1.0320 | 1.0301 | 111/120 = 92.5% |
| 0 | 20 | -0.0160 | 0.0050 | -0.0160 | 0.1036 | 0.1314 | 0.5030 | 0.5040 | 114/120 = 95.0% |
| 0.5 | 1 | -0.7017 | -0.5323 | -1.2017 | 1.2387 | 1.0657 | 2.9727 | 2.9095 | 78/120 = 65.0% |
| 0.5 | 5 | 0.3791 | 0.3876 | -0.1209 | 0.2260 | 0.2698 | 0.9420 | 0.9346 | 111/120 = 92.5% |
| 0.5 | 20 | 0.4859 | 0.5116 | -0.0141 | 0.0982 | 0.1245 | 0.4525 | 0.4515 | 112/120 = 93.3% |
| 1 | 1 | -0.2577 | -0.1141 | -1.2577 | 1.2909 | 1.0070 | 2.9394 | 2.9349 | 76/120 = 63.3% |
| 1 | 5 | 0.9126 | 0.9239 | -0.0874 | 0.1831 | 0.2207 | 0.8329 | 0.8279 | 113/120 = 94.2% |
| 1 | 20 | 0.9893 | 1.0029 | -0.0107 | 0.0817 | 0.1022 | 0.3967 | 0.3953 | 113/120 = 94.2% |

The principal uncertainty result is negative for individual agents:
**the nominal 95% profile intervals are not calibrated at one rollout**.
They are broad—about three alpha units wide—but systematically displaced by
the strong downward point-estimate bias. Numerical accuracy of the endpoints
does not repair this small-sample problem. Open/unbounded frequency is 0% in
every condition, so lack of finite endpoints does not explain undercoverage.

With 5–20 independent agents, coverage is much closer to nominal. Differences
of a few percentage points at 120 datasets should not be overinterpreted;
even an ideal 95% procedure has appreciable Monte Carlo variation at this size.
We did not retune the cutoff, add a penalty, discard bad scaffolds, or replace
these results with a more favorable interval method.

### Detection errors and power

All denominators are 120. "Positive" means the entire interval is above zero;
"negative" means entirely below zero. At zero truth either exclusion is false.
At positive truth only positive exclusion counts as correct-direction power.

| True alpha | Agents | Positive intervals | Negative intervals | Contains zero |
|---:|---:|---:|---:|---:|
| 0 | 1 | 0 (0.0%) | 34 (28.3%) | 86 (71.7%) |
| 0 | 5 | 1 (0.8%) | 8 (6.7%) | 111 (92.5%) |
| 0 | 20 | 1 (0.8%) | 5 (4.2%) | 114 (95.0%) |
| 0.5 | 1 | 0 (0.0%) | 18 (15.0%) | 102 (85.0%) |
| 0.5 | 5 | 49 (40.8%) | 0 (0.0%) | 71 (59.2%) |
| 0.5 | 20 | 113 (94.2%) | 0 (0.0%) | 7 (5.8%) |
| 1 | 1 | 8 (6.7%) | 10 (8.3%) | 102 (85.0%) |
| 1 | 5 | 115 (95.8%) | 0 (0.0%) | 5 (4.2%) |
| 1 | 20 | 120 (100.0%) | 0 (0.0%) | 0 (0.0%) |

Zero-truth false history detection is therefore **28.3%, 7.5%, 5.0%** at
1/5/20 agents. All 34 one-agent false detections were negative. This is lower
than the earlier descriptive 66.7% rate of `abs(alpha_hat)>=0.5`, but still far
above nominal 5%; those two rules are not equivalent tests. At one agent,
wrong-direction detections occur even with true positive history, and correct
detection power is very poor. Five agents are sufficient for strong alpha=1
in this design, not for alpha=0.5; twenty agents detect both reliably in this
finite simulation. These are shared-parameter synthetic agents, not evidence
that twenty heterogeneous humans can be pooled without modeling differences.

### Scaffold-by-scaffold coverage

Entries are numbers covered out of **10**, at agent counts **1 / 5 / 20**.
All scaffold × alpha × count summaries, including recovery, width, detection
directions and Wald comparison, are stored in `uncertainty_summary.scaffold_conditions`.

| Scaffold | Truth 0 | Truth 0.5 | Truth 1 | Mean width, averaged over truths: 1 / 5 / 20 |
|---|---|---|---|---|
| r01b4b | 8 / 9 / 10 | 7 / 9 / 10 | 7 / 10 / 10 | 2.862 / .936 / .462 |
| m03e0b | 6 / 9 / 8 | 7 / 9 / 10 | 6 / 9 / 9 | 3.216 / .958 / .450 |
| s02l2z | 8 / 9 / 9 | 8 / 8 / 8 | 8 / 8 / 9 | 2.759 / .945 / .451 |
| m09t1z | 9 / 9 / 8 | 8 / 9 / 9 | 8 / 10 / 9 | 3.033 / .926 / .451 |
| h01m4k | 7 / 9 / 9 | 7 / 10 / 8 | 7 / 10 / 9 | 2.689 / .936 / .460 |
| e05w0s | 7 / 10 / 10 | 6 / 8 / 10 | 5 / 9 / 9 | 3.353 / .953 / .452 |
| s05t1r | 7 / 10 / 10 | 7 / 9 / 9 | 6 / 9 / 9 | 2.757 / .914 / .446 |
| E11T9A | 7 / 8 / 10 | 5 / 10 / 9 | 6 / 9 / 9 | 2.961 / .941 / .441 |
| k08c1o | 8 / 9 / 10 | 6 / 9 / 10 | 6 / 10 / 10 | 3.070 / .953 / .455 |
| m08m7h | 8 / 10 / 10 | 8 / 10 / 10 | 8 / 10 / 10 | 2.668 / .900 / .452 |
| l06s0m | 3 / 9 / 10 | 3 / 10 / 10 | 3 / 10 / 10 | 3.250 / .934 / .450 |
| s04t2z | 8 / 10 / 10 | 6 / 10 / 9 | 6 / 9 / 10 | 3.083 / .931 / .439 |

The difficult l06s0m single-agent cases remain difficult: only 3/10 intervals
cover truth in each condition, despite broad intervals. All its twenty-agent
intervals cover truth. This echoes the prior recovery difficulty rather than
an interval-search failure. No scaffold was removed. Sequence-specific history
variation and finite-batch randomness remain possible contributors; these ten
batches do not identify a structural cause or establish an intrinsic ranking.

### Secondary Wald comparison

Profile and Wald mean widths are similar, not dramatically different:

| Truth | Agents | Profile width | Wald width | Profile coverage | Wald coverage |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 3.0133 | 2.9969 | 71.7% | 74.2% |
| 0 | 5 | 1.0320 | 1.0310 | 92.5% | 93.3% |
| 0 | 20 | .5030 | .5029 | 95.0% | 95.0% |
| 0.5 | 1 | 2.9727 | 2.9577 | 65.0% | 66.7% |
| 0.5 | 5 | .9420 | .9408 | 92.5% | 92.5% |
| 0.5 | 20 | .4525 | .4523 | 93.3% | 93.3% |
| 1 | 1 | 2.9394 | 2.9271 | 63.3% | 68.3% |
| 1 | 5 | .8329 | .8317 | 94.2% | 95.0% |
| 1 | 20 | .3967 | .3966 | 94.2% | 94.2% |

Wald intervals were finite and numerically stable, but also badly under-covered
at one agent. Their slightly better coverage in these batches is not a remedy
or a reason to switch methods after seeing the outcome. This is principally a
small-sample calibration problem, not simply a Hessian-versus-profile choice.

## Part II: myopic versus planning distinguishability

For within-trajectory holdout, fit sessions 1–4 and score session 5. For fresh
holdout, fit all sessions of the training agents and score 20 independent new
agents, including their early choices. Parameters remain fixed while their
observed synthetic past choices update history causally. Both models have eight
free parameters; neither fitter sees the true generator or evaluation agents.

### Confusion matrices, not just planning wins

Each cell below lists **selected myopic / selected planning**. There are 60
datasets for each true generator, alpha and training-agent count. Thus each row
is a complete 2×2 confusion matrix; no numerical ties or missing pairs occurred.
"Within S5" uses the training agents' session holdout, not the fresh agents.

| Alpha | Training agents | Evaluation window | True myopic: M / P | True planning: M / P |
|---:|---:|---|---:|---:|
| .5 | 1 | Fresh early | 24 / 36 | 19 / 41 |
| .5 | 1 | Fresh mid | 27 / 33 | 20 / 40 |
| .5 | 1 | Fresh late | 26 / 34 | 19 / 41 |
| .5 | 1 | Fresh all | 28 / 32 | 19 / 41 |
| .5 | 1 | Within S5 | 21 / 39 | 19 / 41 |
| .5 | 5 | Fresh early | 44 / 16 | 24 / 36 |
| .5 | 5 | Fresh mid | 45 / 15 | 30 / 30 |
| .5 | 5 | Fresh late | 36 / 24 | 30 / 30 |
| .5 | 5 | Fresh all | 43 / 17 | 26 / 34 |
| .5 | 5 | Within S5 | 28 / 32 | 32 / 28 |
| .5 | 20 | Fresh early | 36 / 24 | 17 / 43 |
| .5 | 20 | Fresh mid | 35 / 25 | 32 / 28 |
| .5 | 20 | Fresh late | 31 / 29 | 31 / 29 |
| .5 | 20 | Fresh all | 39 / 21 | 18 / 42 |
| .5 | 20 | Within S5 | 30 / 30 | 26 / 34 |
| 1 | 1 | Fresh early | 29 / 31 | 18 / 42 |
| 1 | 1 | Fresh mid | 25 / 35 | 19 / 41 |
| 1 | 1 | Fresh late | 23 / 37 | 19 / 41 |
| 1 | 1 | Fresh all | 27 / 33 | 18 / 42 |
| 1 | 1 | Within S5 | 27 / 33 | 28 / 32 |
| 1 | 5 | Fresh early | 46 / 14 | 9 / 51 |
| 1 | 5 | Fresh mid | 44 / 16 | 27 / 33 |
| 1 | 5 | Fresh late | 41 / 19 | 26 / 34 |
| 1 | 5 | Fresh all | 50 / 10 | 9 / 51 |
| 1 | 5 | Within S5 | 28 / 32 | 26 / 34 |
| 1 | 20 | Fresh early | 45 / 15 | 7 / 53 |
| 1 | 20 | Fresh mid | 37 / 23 | 24 / 36 |
| 1 | 20 | Fresh late | 28 / 32 | 26 / 34 |
| 1 | 20 | Fresh all | 50 / 10 | 6 / 54 |
| 1 | 20 | Within S5 | 32 / 28 | 29 / 31 |

Balanced accuracy is the average of the two generator-specific correct-selection
rates, not the fraction of all configurations selecting planning:

| Alpha | Training agents | Fresh early | Fresh mid | Fresh late | Fresh all | Within S5 |
|---:|---:|---:|---:|---:|---:|---:|
| .5 | 1 | 54.2% | 55.8% | 55.8% | 57.5% | 51.7% |
| .5 | 5 | 66.7% | 62.5% | 55.0% | 64.2% | 46.7% |
| .5 | 20 | 65.8% | 52.5% | 50.0% | 67.5% | 53.3% |
| 1 | 1 | 59.2% | 55.0% | 53.3% | 57.5% | 49.2% |
| 1 | 5 | 80.8% | 64.2% | 62.5% | 84.2% | 51.7% |
| 1 | 20 | 81.7% | 60.8% | 51.7% | 86.7% | 52.5% |

The strongest condition (alpha=1, twenty training agents) correctly selects
myopic in 50/60 cases and planning in 54/60 using fresh all-trial scores. Early
trials are much more informative per response; adding the other trials still
improves overall classification in this condition. There is no requirement or
claim that accuracy improves monotonically across every count/window/generator.
At alpha=.5, even twenty-agent accuracy remains modest. Within-trajectory
session-5 classification is close to chance throughout.

The fresh scheme **always uses twenty evaluation agents**, even when fitted to
one training agent. These numbers are not single-person classification accuracies.
Common random draws pair generator conditions; the 120 labels in a balanced
matrix are not 120 fully independent experimental replications. No naive
binomial significance claim or threshold tuning is attached to these rates.

### Full NLL-difference distributions and practical size

`delta = NLL_myopic − NLL_planning`, so positive favors planning. Every
generator × alpha × count × window has mean, median, SD, P5/P25/P75/P95,
minimum/maximum and raw per-dataset scores in JSON. The table below exposes
the distributions for alpha=1, twenty training agents, rather than merely
reporting the favorable classification percentage. Each row has 60 paired cases.

| Generator | Window | Mean delta | Median | SD | P5 | P95 | Mean delta per choice |
|---|---|---:|---:|---:|---:|---:|---:|
| Myopic | Fresh early | -1.79691 | -1.48296 | 2.14420 | -5.07868 | 1.07821 | -.00449228 |
| Myopic | Fresh mid | -.49305 | -.61805 | .99826 | -1.93115 | .88636 | -.00004151 |
| Myopic | Fresh late | .02214 | .01786 | .21531 | -.28102 | .41581 | .00000693 |
| Myopic | Fresh all | -2.26783 | -1.94732 | 2.37474 | -7.05426 | .76983 | -.00014859 |
| Myopic | Within S5 | -.04908 | -.02928 | .31717 | -.57565 | .35231 | -.00001637 |
| Planning | Fresh early | 2.06064 | 1.92008 | 2.32576 | -1.36907 | 6.27774 | .00515159 |
| Planning | Fresh mid | .21958 | .23936 | .87027 | -1.02094 | 1.73517 | .00001887 |
| Planning | Fresh late | .04159 | .03543 | .17733 | -.23323 | .29674 | .00001357 |
| Planning | Fresh all | 2.32181 | 2.13803 | 2.50616 | -1.05985 | 7.23981 | .00015433 |
| Planning | Within S5 | .01334 | .03051 | .23170 | -.45185 | .34298 | .00000406 |

The distributions overlap zero, including the strongest fresh-agent condition.
On planning-generated fresh agents, mean all-trial NLL is 7,980.588 myopic versus
7,978.266 planning, and mean log loss is .52060340 versus .52044906. On myopic
fresh agents these are 8,013.882 versus 8,016.150 and .52279316 versus .52294175.
Mean session-holdout NLLs are 1,604.119 versus 1,604.106 for planning-generated
data, and 1,607.019 versus 1,607.068 for myopic-generated data. The corresponding
mean losses are .51649386/.51648980 and .51744090/.51745728. Because scaffold
lengths differ, mean per-choice delta is averaged per configuration, not formed
by dividing mean total delta by an arbitrary shared trial count.

Probability differences between fitted models, same alpha=1/twenty-agent cases:

| Generator | Fresh window | Mean abs probability difference | Mean of per-dataset maximum | Fraction >.01 | Fraction >.05 | Different most-likely action |
|---|---|---:|---:|---:|---:|---:|
| Myopic | Early | .026838 | .130160 | 56.43% | 17.97% | 3.88% |
| Myopic | Mid | .001946 | .039824 | 1.93% | .012% | .018% |
| Myopic | Late | .001188 | .004639 | 0% | 0% | 0% |
| Myopic | All | .002445 | .130160 | 2.97% | .480% | .116% |
| Planning | Early | .026879 | .138712 | 55.19% | 18.79% | 3.25% |
| Planning | Mid | .001745 | .041091 | 1.78% | .016% | .018% |
| Planning | Late | .000928 | .004040 | 0% | 0% | 0% |
| Planning | All | .002238 | .138712 | 2.82% | .504% | .100% |

The largest single observed probability difference across these cases was
.23065 for myopic-generated and .24980 for planning-generated data; both occur
in early windows. These are fitted-model differences, not necessarily the
same-parameter oracle effect reported in the previous milestone. Over all trials
the average difference is about 0.2 percentage points, but over the first twenty
it is about 2.7 percentage points. Thus an aggregate claim of either "identical"
or "large planning effects everywhere" would be misleading.

### Scaffold-level selection

For alpha=1/twenty training agents, entries are correct selections out of five
batches for **true myopic / true planning**. Full scaffold-specific distributions
for all conditions/windows are in `selection_summary.scaffold_conditions`.

| Scaffold | Fresh early | Fresh all | Within S5 |
|---|---:|---:|---:|
| r01b4b | 3 / 5 | 3 / 5 | 4 / 1 |
| m03e0b | 4 / 5 | 4 / 5 | 3 / 3 |
| s02l2z | 3 / 5 | 3 / 4 | 4 / 2 |
| m09t1z | 4 / 5 | 5 / 5 | 1 / 4 |
| h01m4k | 4 / 5 | 4 / 5 | 2 / 1 |
| e05w0s | 4 / 4 | 5 / 4 | 4 / 3 |
| s05t1r | 5 / 4 | 5 / 4 | 2 / 2 |
| E11T9A | 4 / 4 | 4 / 4 | 2 / 4 |
| k08c1o | 2 / 4 | 4 / 5 | 2 / 3 |
| m08m7h | 5 / 3 | 4 / 4 | 3 / 3 |
| l06s0m | 3 / 4 | 4 / 4 | 2 / 3 |
| s04t2z | 4 / 5 | 5 / 5 | 3 / 2 |

Fresh all-trial balanced accuracy exceeds 50% on every scaffold in this strong
condition. This is not solely an E11T9A effect. Five batches per generator are
too few to establish precise scaffold rankings; no difficult scaffold was
removed, and the early window is not uniformly best for every scaffold.

### Alpha=0 is an equivalence control

Paired generators produced identical choices at zero truth, as expected. Their
fitted outputs and evaluation scores also match across generator labels.
Nevertheless, the two *candidate fitted families* estimate unrestricted alpha,
so their predictions need not coincide. The following counts are the same for
either generator label and must not be interpreted as identity-recovery accuracy:

| Training agents | Planning selected, fresh early | Fresh all | Within S5 | Mean fresh-all delta |
|---:|---:|---:|---:|---:|
| 1 | 39/60 | 39/60 | 30/60 | +14.97124 |
| 5 | 28/60 | 24/60 | 28/60 | -.31375 |
| 20 | 31/60 | 32/60 | 27/60 | -.02610 |

There were no exact numerical NLL ties, despite identical generating mechanisms.
This illustrates why selecting a model by a tiny held-out advantage is not
evidence that an underlying planning identity exists. The one-agent preference
for the planning candidate under **both** null labels accompanies severe
estimation error; it is not successful planning detection. Alpha=0 is excluded
from every classification confusion matrix above.

## Parameter distortion under the wrong family

These are all-session fits on the model-comparison batches 0–4, not the ten-batch
uncertainty grid. Truth is used only after fitting. Twenty-agent means:

| Generator | True alpha | Fitted family | Mean alpha | Alpha bias | Utility MAE |
|---|---:|---|---:|---:|---:|
| Myopic | .5 | Myopic (matched) | .49798 | -.00203 | .06341 |
| Myopic | .5 | Planning | .47192 | -.02808 | .06424 |
| Myopic | 1 | Myopic (matched) | 1.00090 | .00090 | .06226 |
| Myopic | 1 | Planning | .96168 | -.03832 | .06436 |
| Planning | .5 | Planning (matched) | .49433 | -.00567 | .06367 |
| Planning | .5 | Myopic | .48990 | -.01010 | .06499 |
| Planning | 1 | Planning (matched) | 1.00798 | .00798 | .06515 |
| Planning | 1 | Myopic | 1.00528 | .00528 | .06610 |

At alpha=1, wrong planning assumptions shift recovered alpha downward by about
.0392 on myopic-generated data. The reverse mismatch changes mean alpha by only
-.0027. Utility errors change modestly; the myopic family can approximate much
of planning-generated behavior without a dramatic distortion of the recovered
parameters. This does not make the families exactly equivalent.

At one training agent, sampling bias dominates mismatch: for true alpha=1,
myopic-generated data yield alpha means -.3864 (myopic fit) and -.3193 (planning
fit), with utility MAE .5976/.5703. Planning-generated data yield -.2904 (myopic)
and -.1396 (planning), with MAE .5651/.5157. None of these is a trustworthy
individual reward/history estimate. Full parameter distributions at every count,
including null controls and both fit scopes, remain in JSON.

## Scientific interpretation and exact next step

**Can history be detected reliably?** Not from one retained trajectory with
this nominal profile rule: coverage is poor, false history detection is 28.3%
under zero truth, and positive-effect power is weak. With common-parameter
synthetic agents, alpha=1 is detected well at five agents; alpha=.5 generally
needs twenty for high power here. Coverage at twenty agents is near nominal,
not proven perfect by these finite batches.

**Is h=3 planning reliably distinguishable from h=0?** The best description is
**only with stronger history, multiple training agents, and informative fresh
evaluation including early trials**. At alpha=1 with twenty training and twenty
fresh agents, 86.7% balanced all-trial accuracy shows the models are not wholly
indistinguishable. But errors remain; alpha=.5 is much less distinguishable, and
session-5-only accuracy stays near chance. Even in the strong condition, most
individual probabilities differ very little. These are synthetic model-selection
results, not evidence that any human planned ahead.

This does not justify transferring pooled synthetic performance to one person.
Scaffolds are deterministic structural selections, beta is known, utilities are
fixed, every pooled agent shares parameters, and Planning Bob knows the future
retained-pair schedule. Missing original trials remain missing. The planning
optimizer reports successful locally identified fits, not a global-optimality
certificate; the largest information condition across comparison fits was
331.74, so numerical conditioning alone did not diagnose the single-agent
inferential failure. No formal test of practical equivalence was performed.

**Next recommended step:** before any individual human alpha claim, predeclare
a separate synthetic **parametric-bootstrap calibration of the alpha likelihood
ratio**, re-fitting nuisance utilities in every bootstrap dataset and checking
coverage/type-I error on independent calibration batches across these scaffolds.
Use the present nominal-profile results as the unchanged comparison, not a
cutoff tuned to these same outcomes. Retain early/fresh-agent predictive checks
for planning, but use the simpler myopic mechanism as the conservative initial
human candidate unless new independent evidence justifies the planning claim.
Do not pool real people as if they were identical Fake Bobs. Human model
comparison remains conditional on resolving these calibration/model-mismatch
limitations; no human action fitting was done in this ticket.
