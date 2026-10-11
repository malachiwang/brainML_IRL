# Authors’ Reward Pairs benchmark: reproduction results

This report fits **human choices only for benchmark reproduction**. It does not fit our novel history/IRL model or interpret individual parameters psychologically. See the [source audit](authors_model_audit.md) for equations, provenance, exclusions, and paper/code distinctions. Full-precision fits, failure diagnostics, and model-by-participant matrices are in [the JSON](authors_baseline_results.json); R parity and limited synthetic recovery are in [validation JSON](authors_validation_results.json).

## Configuration and execution

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/run_authors_baseline.py --variant all --restarts 200 --workers 4 \
  --output docs/authors_baseline_results.json
```

Total supplied/fitted participants: **213**. Distinct initialization/model fits: **2130**; starts attempted: **426,000**. Three-variant result rows: **3195** (B/C share fits). Failed participant/model fits: **0**. Unsuccessful optimizer starts among distinct successful fits: **371**. Full-cohort wall time: **631.96 s**. All starts and failures are retained as diagnostics; failed starts are not chosen as solutions.

This is a **200-start reproduction, not an exact R-optimizer-protocol reproduction**: R supplies the original seed/draw stream, but SciPy L-BFGS-B uses analytic derivatives, stable log probabilities, and tighter convergence checking. No regularization. R `optimr` is not installed. The original R objective itself is used for parity.

Smoke sequence: E11T9A model 19/10 starts 0.9 s end-to-end; all models/all variants/10 starts 1.4 s; all models/all variants/200 starts 12.6 s. Ten and 200 starts differed by at most 1.94e-12 deviance on that participant. The 200-start smoke projected approximately 10.4 minutes with four workers for the cohort; all 200 starts were retained regardless.

All 18 comparison-script exclusion IDs were checked, not silently reapplied to synthetic studies. Presence in this supplied input: 0. Exact IDs and grouped reasons are in the audit and JSON.

Training rows: 164,087; test rows: 28,320. Original chronology already sorted: training 213/213, test 213/213. Retained rows only; no session resets or inserted missing-trial updates. Test is a separate phase even though its source session label is 5.

## Verification

Final full suite: **186 passed in 37.44 s** (`python -m pytest -q`), including 38 new authors-benchmark tests. The original 148 tests were retained unchanged. Checks cover both-option RL updates, CK updates, initialization, causal replay, test freezing, chronology/gaps, model constraints, information criteria, analytic derivatives, explicit failures, and original-R parity.

Participant fixed/fitted parity: 30 comparisons (five models × three variants × two parameter settings), each covering training and test. Maximum absolute discrepancies: objective deviance 1.08e-11; training deviance 2.41e-10; test deviance 6.12e-09; choice probability 9.44e-16; Q 1.11e-16; H 5.55e-17.

Independent base-R `optim` fits used ten identical random starts and the unmodified likelihood. Its default finite-difference step is 0.001. Reduced-model optima at small learning rates were checked again with step 1e-6; this is a numerical diagnostic, not a different behavioral model.

| Variant | Model | R − Python deviance | Fine-step R − Python |
| --- | --- | --- | --- |
| A: paper | 1 | -1.06865627e-11 | not needed |
| A: paper | 3 | 2.05091055e-10 | not needed |
| A: paper | 11 | 5.48249091e-08 | not needed |
| A: paper | 19 | 5.33971161e-08 | not needed |
| A: paper | 20 | 0.154208426 | 1.37561074e-11 |
| B: code intended | 1 | -1.06865627e-11 | not needed |
| B: code intended | 3 | 2.50111043e-12 | not needed |
| B: code intended | 11 | 4.08381311e-07 | not needed |
| B: code intended | 19 | 2.42401939e-08 | not needed |
| B: code intended | 20 | 0.169821857 | 2.27373675e-12 |

Base-R default-difference best results reported convergence code 52 for both random-model fits and the released-initialization reduced fit. They are retained as audit diagnostics, not labeled successful R convergence. The random-model discrepancy is only 1.1e-11 deviance and its analytical optimum is independently unit-tested; both fine-step reduced diagnostics report convergence 0. Python rejects unsuccessful starts when selecting its reported solution.

## Preregistered four-model comparison

Winner = minimum participant BIC; columns are models 1/random, 3/RL, 11/CK, 19/combined. The same k is used on training and test, reproducing the source convention. This is not a Bayesian group-model-selection calculation.

| Variant | Phase | Random | RL | CK | RL+CK |
| --- | --- | --- | --- | --- | --- |
| A: paper | training | 0 | 12 | 2 | 199 |
| A: paper | test | 0 | 151 | 15 | 47 |
| B: code intended | training | 0 | 15 | 1 | 197 |
| B: code intended | test | 0 | 141 | 17 | 55 |
| C: literal R | training | 0 | 2 | 0 | 211 |
| C: literal R | test | 0 | 177 | 0 | 36 |

### RL versus RL+CK evidence

Delta = BIC(RL+CK) − BIC(RL): negative favors combined. Exact ±6 follows the R category boundary (no strong evidence); magnitudes above 6 through 10 are strong, above 10 very strong.

| Variant | Phase | Very strong combined | Strong combined | Neither | Strong RL | Very strong RL |
| --- | --- | --- | --- | --- | --- | --- |
| A: paper | training | 192 | 4 | 15 | 2 | 0 |
| A: paper | test | 24 | 13 | 71 | 30 | 75 |
| B: code intended | training | 175 | 13 | 20 | 5 | 0 |
| B: code intended | test | 23 | 10 | 96 | 41 | 43 |
| C: literal R | training | 210 | 1 | 1 | 0 | 1 |
| C: literal R | test | 7 | 11 | 36 | 19 | 140 |
| Published | test | 7 | 11 | 36 | 19 | 140 |

Published target: [main article, computational-model results](https://doi.org/10.1525/collabra.92949). Participant category counts are compared directly; reported VBA exceedance probabilities are **not reproduced**. MATLAB is present locally, but the VBA toolbox and saved fitted/comparison workspaces were not found. Labeled AIC/AICc/BIC/deviance/LL matrices are exported in the JSON, with models as rows and included participants as columns.

## Exploratory five-model comparison

| Variant | Phase | Random | RL | CK | RL+CK | Reduced |
| --- | --- | --- | --- | --- | --- | --- |
| A: paper | training | 0 | 8 | 2 | 127 | 76 |
| A: paper | test | 0 | 148 | 14 | 34 | 17 |
| B: code intended | training | 0 | 15 | 1 | 197 | 0 |
| B: code intended | test | 0 | 140 | 17 | 54 | 2 |
| C: literal R | training | 0 | 0 | 0 | 0 | 213 |
| C: literal R | test | 0 | 2 | 0 | 0 | 211 |

Reduced-versus-RL categories use “combined” below to mean the **reduced** model, not model 19. The reduced model was specified for association analyses; its addition to the model comparison was exploratory.

| Variant | Phase | Very strong reduced | Strong reduced | Neither | Strong RL | Very strong RL |
| --- | --- | --- | --- | --- | --- | --- |
| A: paper | training | 195 | 5 | 12 | 1 | 0 |
| A: paper | test | 12 | 5 | 37 | 21 | 138 |
| B: code intended | training | 1 | 0 | 1 | 0 | 211 |
| B: code intended | test | 2 | 1 | 10 | 5 | 195 |
| C: literal R | training | 213 | 0 | 0 | 0 | 0 |
| C: literal R | test | 211 | 0 | 1 | 0 | 1 |
| Published | test | 211 | 0 | 1 | 1 | 0 |

## Fitting versus rescoring

B and C have exactly the same fitted parameters and Q/H trajectories. Only post-fit positional response stickiness changes. Clean R resolves pi to 3.141592653589793; after the first response the logit contribution is ±2*pi, with positional history reset at each phase entry.

| Variant | Model | Mean rescore−fit | Minimum | Maximum | Mean train dev | Mean test dev |
| --- | --- | --- | --- | --- | --- | --- |
| A: paper | 1 | 0 | 0 | 0 | 1064.596 | 184.270 |
| A: paper | 3 | 4.30462e-13 | 0 | 3.75167e-12 | 700.216 | 119.952 |
| A: paper | 11 | 8.96685e-14 | 0 | 3.41061e-13 | 750.501 | 145.958 |
| A: paper | 19 | 1.21426e-13 | -1.13687e-13 | 9.09495e-13 | 624.241 | 115.219 |
| A: paper | 20 | 2.41518e-13 | 0 | 1.13687e-12 | 637.828 | 134.152 |
| B: code intended | 1 | 0 | 0 | 0 | 1064.596 | 184.270 |
| B: code intended | 3 | 4.57416e-13 | 0 | 3.41061e-12 | 697.273 | 119.959 |
| B: code intended | 11 | 8.16624e-14 | 0 | 9.09495e-13 | 777.354 | 140.470 |
| B: code intended | 19 | 9.15366e-14 | -1.13687e-13 | 9.09495e-13 | 632.332 | 112.242 |
| B: code intended | 20 | 1.01731e-12 | 0 | 5.34328e-12 | 1084.287 | 175.292 |
| C: literal R | 1 | 4006.86 | 2859.11 | 4688.45 | 5071.453 | 862.585 |
| C: literal R | 3 | 3509.7 | 2514.42 | 4451.1 | 4206.974 | 635.463 |
| C: literal R | 11 | 3607.56 | 2638.23 | 4465.51 | 4384.916 | 784.612 |
| C: literal R | 19 | 3352.37 | 2391.06 | 4406.22 | 3984.699 | 642.725 |
| C: literal R | 20 | 1641.68 | 676.662 | 3441.98 | 2725.968 | 473.637 |

### E11T9A audit example

| Variant | Model | Fit dev | Train rescore dev | Test dev | Train BIC | Test BIC |
| --- | --- | --- | --- | --- | --- | --- |
| A: paper | 1 | 1084.240833 | 1084.240833 | 189.541617 | 1090.903966 | 194.454272 |
| A: paper | 3 | 799.104139 | 799.104139 | 120.001920 | 812.430404 | 129.827229 |
| A: paper | 11 | 819.001867 | 819.001867 | 150.006297 | 832.328133 | 159.831607 |
| A: paper | 19 | 716.644096 | 716.644096 | 118.932857 | 743.296627 | 138.583477 |
| A: paper | 20 | 736.720427 | 736.720427 | 131.384764 | 756.709825 | 146.122728 |
| B: code intended | 1 | 1084.240833 | 1084.240833 | 189.541617 | 1090.903966 | 194.454272 |
| B: code intended | 3 | 799.903155 | 799.903155 | 119.986403 | 813.229421 | 129.811713 |
| B: code intended | 11 | 820.835360 | 820.835360 | 152.024091 | 834.161626 | 161.849400 |
| B: code intended | 19 | 721.063812 | 721.063812 | 118.052851 | 747.716342 | 137.703470 |
| B: code intended | 20 | 895.721231 | 895.721231 | 165.660920 | 915.710629 | 180.398884 |
| C: literal R | 1 | 1084.240833 | 5030.932965 | 919.162263 | 5037.596098 | 924.074917 |
| C: literal R | 3 | 799.903155 | 4463.802266 | 732.577231 | 4477.128531 | 742.402541 |
| C: literal R | 11 | 820.835360 | 4377.150314 | 840.101404 | 4390.476580 | 849.926714 |
| C: literal R | 19 | 721.063812 | 4180.711715 | 732.666066 | 4207.364246 | 752.316685 |
| C: literal R | 20 | 895.721231 | 3674.398747 | 606.681848 | 3694.388146 | 621.419813 |

All E11T9A free parameters and final eight-element Q/H vectors, along with every other participant, are stored without rounding in the JSON.

## Initialization sensitivity (A versus B, both pi=0)

This isolates the jointly requested initialization changes: ratings /11.15 and H0=0 versus mean(ratings /20, pretest choice component) for both Q0/H0. It does not separately identify the causal contribution of each initialization component. Rewards remain the supplied 1–5 columns in both variants; no unrequested point-unit conversion is made.

| Model | Mean train dev B−A | Mean test dev B−A | Mean final Q absolute change | Mean final H absolute change |
| --- | --- | --- | --- | --- |
| 1 | 0.0000 | 0.0000 | 0.726877 | 1.070665 |
| 3 | -2.9430 | 0.0071 | 0.079104 | 1.070665 |
| 11 | 26.8530 | -5.4879 | 0.726877 | 0.031791 |
| 19 | 8.0906 | -2.9776 | 0.041844 | 0.059282 |
| 20 | 446.4588 | 41.1394 | 0.275712 | 0.341264 |

State changes in disabled processes (zero beta), such as both states in model 1, are bookkeeping differences and do not affect choice probabilities.

preregistered, training: initialization changes the BIC winner for 6/213 participants.

preregistered, test: initialization changes the BIC winner for 30/213 participants.

exploratory, training: initialization changes the BIC winner for 77/213 participants.

exploratory, test: initialization changes the BIC winner for 43/213 participants.

## Parameter distributions and boundaries

Entries are mean [median; 5th–95th percentile]. These are descriptive fit distributions, not uncertainty intervals. C shares B parameters exactly. Boundary and near-boundary estimates are not interpreted psychologically.

| Variant | Model | alpha_RL | beta_RL | alpha_CK | beta_CK | Bias | Any boundary /213 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A: paper | 1 | 0.0000 [0.0000; 0.0000–0.0000] | 0.0000 [0.0000; 0.0000–0.0000] | 0.0000 [0.0000; 0.0000–0.0000] | 0.0000 [0.0000; 0.0000–0.0000] | 2.5141 [2.5113; 2.4191–2.6163] | 0 |
| A: paper | 3 | 0.3533 [0.1770; 0.0091–1.0000] | 1.7918 [1.6830; 1.0504–2.6142] | 0.0000 [0.0000; 0.0000–0.0000] | 0.0000 [0.0000; 0.0000–0.0000] | 2.5000 [2.5000; 2.5000–2.5000] | 36 |
| A: paper | 11 | 0.0000 [0.0000; 0.0000–0.0000] | 0.0000 [0.0000; 0.0000–0.0000] | 0.0445 [0.0440; 0.0233–0.0678] | 3.2792 [3.2629; 2.8732–3.7074] | 2.5000 [2.5000; 2.5000–2.5000] | 0 |
| A: paper | 19 | 0.4432 [0.3166; 0.0175–1.0000] | 1.2637 [1.2138; 0.6585–2.1087] | 0.0771 [0.0507; 0.0088–0.2533] | 2.3236 [2.1444; 1.1517–3.5944] | 2.5000 [2.5000; 2.5000–2.5000] | 53 |
| A: paper | 20 | 0.4343 [0.2930; 0.0151–1.0000] | 1.5009 [1.2616; 0.6867–2.1705] | 0.0057 [0.0023; 0.0011–0.0036] | 8.4991 [8.7384; 7.8295–9.3133] | 2.5000 [2.5000; 2.5000–2.5000] | 47 |
| B: code intended | 1 | 0.0000 [0.0000; 0.0000–0.0000] | 0.0000 [0.0000; 0.0000–0.0000] | 0.0000 [0.0000; 0.0000–0.0000] | 0.0000 [0.0000; 0.0000–0.0000] | 2.5141 [2.5113; 2.4191–2.6163] | 0 |
| B: code intended | 3 | 0.3068 [0.1428; 0.0104–1.0000] | 1.7253 [1.6618; 1.0366–2.4704] | 0.0000 [0.0000; 0.0000–0.0000] | 0.0000 [0.0000; 0.0000–0.0000] | 2.5000 [2.5000; 2.5000–2.5000] | 28 |
| B: code intended | 11 | 0.0000 [0.0000; 0.0000–0.0000] | 0.0000 [0.0000; 0.0000–0.0000] | 0.0974 [0.0973; 0.0356–0.1606] | 2.6356 [2.6570; 2.1004–3.1816] | 2.5000 [2.5000; 2.5000–2.5000] | 1 |
| B: code intended | 19 | 0.6628 [0.7624; 0.0333–1.0000] | 1.2694 [1.2499; 0.6969–1.9899] | 0.1693 [0.1210; 0.0209–0.4846] | 1.5783 [1.6237; 0.6853–2.2605] | 2.5000 [2.5000; 2.5000–2.5000] | 86 |
| B: code intended | 20 | 0.6034 [0.7890; 0.0033–1.0000] | 3.2158 [3.2374; 1.6107–4.7970] | 0.1152 [0.0951; 0.0060–0.2978] | 6.7842 [6.7626; 5.2030–8.3893] | 2.5000 [2.5000; 2.5000–2.5000] | 88 |

Model 20 always satisfies beta_RL + beta_CK = 10 exactly. JSON additionally contains SD/min/max, boundary counts by parameter, and final Q/H distributions by stimulus. The published reduced beta_CK mean is 6.78. Its accompanying t(207) is not reconciled here with the 213-person BIC comparison; no additional exclusions are invented to match it.

## Limited synthetic implementation sanity check

Two predeclared interior settings each for RL, CK, combined, and reduced combined; thirty independent synthetic agents per setting on the E11T9A reward/pair scaffold, fresh states each, ten independent random starts. Human choices are replaced by simulated actions. This is not the authors’ 1,000-agent recovery study. Parameter order is RL: (alpha_RL,beta_RL); CK: (alpha_CK,beta_CK); combined: (alpha_CK,beta_CK,alpha_RL,beta_RL); reduced: (alpha_CK,beta_CK,alpha_RL).

| Model | Truth | Recovered | Deviance truth−fit |
| --- | --- | --- | --- |
| 3 | [0.15, 1.2] | [0.17248, 1.18751] | 1.2673 |
| 3 | [0.6, 0.8] | [0.50432, 0.78884] | 2.3281 |
| 11 | [0.15, 2.0] | [0.14663, 2.03455] | 1.0414 |
| 11 | [0.5, 1.2] | [0.50664, 1.20292] | 0.3695 |
| 19 | [0.15, 1.5, 0.25, 1.0] | [0.15016, 1.46085, 0.28461, 0.9912] | 2.3355 |
| 19 | [0.4, 2.0, 0.6, 0.7] | [0.40039, 2.03388, 0.53883, 0.71225] | 2.0228 |
| 20 | [0.15, 8.0, 0.25] | [0.14466, 7.9755, 0.24407] | 0.8292 |
| 20 | [0.5, 9.0, 0.6] | [0.50916, 8.88088, 0.49274] | 2.7381 |

All eight fits improve on the generating-parameter likelihood. Recovery is approximate, not exact: RL learning rates depend disproportionately on early learning and are less precisely estimated than some choice weights, even with thirty agents. This check does not establish reliable individual-level identifiability.

## Interpretation and recommendation

<!-- The source-sensitive interpretation below is reviewed against the generated tables, not inferred automatically from winner counts. -->



### Main finding

**Variant C, not either pi=0 variant, most closely reproduces the published
numerical comparison.** Its five preregistered test evidence counts match the
publication exactly: 7 / 11 / 36 / 19 / 140. It also reproduces 211 very-strong
reduced-model preferences and the released-initialization fitted beta_CK mean
6.78422, which rounds to the published 6.78. The exact category-count agreement
is substantially stronger evidence than merely reproducing “RL often wins.”

This is evidence **consistent with literal mathematical-pi post-fit scoring**.
It is not evidence that the authors must have supplied a global pi=0; the
observed comparisons point in the other direction. Without their saved fitted
workspace, environment, and exact optimizer versions, historical execution
cannot be proven. No claim about the validity of the paper's separate behavioral
analyses or its entirety follows from this benchmark audit.

### One unresolved published discrepancy

For reduced versus RL, C yields one indifferent participant (`b01m6z`, delta
BIC=+1.47680) and one **very strongly** RL-favoring participant (`s04t2z`, delta
BIC=+55.79066). The article describes the latter category as **strong**, not
very strong. This is not a small floating-point threshold crossing.

A post-hoc diagnostic on `s04t2z` rechecked every fixed/fitted model against the
original R scorers and independently fit RL/reduced using ten base-R starts.
The released-initialization R fits gave literal-scoring delta BIC=+56.66707,
also very strong. The RL R fit returned convergence code 52; the reduced fit
converged. This does not resolve the historical discrepancy, and we neither
relabel the result nor exclude the participant. Details and the 45.8-second
diagnostic are in [the exception validation JSON](authors_exception_validation.json).

### What changes under consistent scoring?

Both A and B reproduce a **qualitative** tendency for RL to win more individual
test BIC comparisons than combined RL+CK. Neither reproduces the reported
overwhelming reduced-model advantage. Under B, reduced wins the five-model test
comparison for only **2/213**, compared with **211/213** under C, despite
identical parameters and final states. Mean reduced test deviance changes from
175.292 to 473.637; the other models deteriorate even more under literal scoring.
Relative “winning” therefore must not be confused with better absolute prediction.

The mean training rescore-minus-fit discrepancy is effectively zero for A/B
(maximum absolute discrepancy 5.35e-12). For C it is **3,223.63** across model
rows, range **676.66–4,688.45**. In particular, C's reported training ICs are not
ML-based ICs for the post-fit scoring likelihood: the parameters were optimized
under a different policy. We preserve the source calculations for audit, not
endorse them as a consistent estimation procedure.

Initialization is consequential independently of pi. A-to-B changes 30/213
preregistered test winners and 43/213 exploratory test winners. Combined-model
mean alpha_CK changes .0771→.1693 and beta_CK 2.3236→1.5783; reduced beta_CK
changes 8.4991→6.7842. B matches the published rounded reduced weight, but A is
closer than B to the preregistered evidence-count distribution. Thus B is not
uniformly the closest numerical reproduction; C is the clearest overall match.

### Numerical and inferential limits

- 371/426,000 Python starts reported abnormal termination (0.0871%); every
  participant/model still had successful starts. No failed start had deviance
  more than 1e-5 better than its selected successful fit.
- The unrestricted combined fit was never worse than the nested RL, CK, or
  reduced fit by more than 1e-5. This is an additional optimization sanity check,
  not proof of global optimality for a nonconvex objective.
- Many fits touch bounds: B has boundary estimates for 28 RL, 1 CK, 86 combined,
  and 88 reduced fits. Learning rates and choice weights must not be read as
  precise individual psychological quantities.
- Exact full R-driver execution was not reproduced: `optimr` is unavailable,
  stable likelihoods replace its infinite-deviance fallback, and original saved
  fitted results are absent. R objective/scorer parity, not binary-identical
  optimizer trajectories, was established.
- VBA exceedance probabilities were not computed. Participant evidence counts
  and exported matrices do not substitute for that group-level analysis.
- The paper/source reward-unit and initialization ambiguities remain explicit.
  A retains the supplied 1–5 training rewards; it is not an unvalidated conversion
  back to original point units. The optional hybrid initialization diagnostic
  was not run, so Q-initialization and H-initialization effects are not separated.
- Model 20 is exploratory in the model-comparison set. No new exclusion, prior,
  penalty, behavioral feature, or change to our previous models was introduced.

### Concrete benchmark recommendation and next comparison

Use **`released_code_intended` (B)** as the likelihood-consistent authors-code
benchmark for future predictive comparisons, explicitly labeled as a
reconstruction with pi=0, **not** an exact reproduction of the published scores.
Keep A as a mandatory initialization sensitivity analysis and C as the
published-result forensic reference only. Request the original saved fitting
workspace and clarification of pi/initialization before claiming exact historical
reproduction.

The next modeling comparison should predeclare our **myopic cumulative-history
model versus authors' RL (3) and RL+CK (19)** on identical retained human training
and test rows, with causal test-state updates and held-out log loss as the primary
predictive measure; keep reduced model 20 exploratory and carry forward the
synthetic uncertainty warnings. Do not start with a claim about planning. Their
alpha_CK controls trace-update speed, whereas our old alpha weights a different,
cumulative feature. Neither a fitted CK weight nor a predictive win establishes
a human habit mechanism. This next comparison has **not** been performed here.



