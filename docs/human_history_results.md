# Human RL + cumulative-history predictive comparison

This is the first fit of our cumulative representation to real human choices. It is **not forward planning**, reward-function IRL, or an individual habit diagnostic. Authors RL and RL+CK are unchanged. The primary benchmark is released-code-intended with pi=0; literal mathematical-pi scoring is not used.

## Model, data, and fixed analysis design

`P(left) = sigmoid(beta_RL*(Q_L-Q_R) + beta_H*(H_L-H_R))`. Before each retained trial, `H=C/N-.5` for seen stimuli and zero otherwise. Counts start at zero, carry across all training sessions and into test, and update only after scoring the observed response. Both displayed Qs receive the authors’ same-alpha feedback update in training. Q freezes during test while counts continue to update. There is no parameter refitting during test.

Bounds: alpha_RL [0,1], beta_RL [0,10], beta_H [-10,10]. Three parameters versus two for RL and four for RL+CK. Q initialization is exactly shared with the benchmark variant: primary uses released pre_val; sensitivity uses the paper rating rule. No pretest choices initialize cumulative counts. The existing pretest component remains in Q0 because changing it would change the reward-learning benchmark.

Participants: **213**. Retained training rows: **164,087**; test rows: **28,320**. Already chronological: training 213/213, test 213/213. No missing rows are invented, no new exclusions, no synthetic replication of human participants.

New-model starts: 200 per participant/variant; NumPy SeedSequence([20261015, one-based source participant index]). The existing 200-start authors fits are replayed and checked against their saved training/test deviances and initial states on the same data, not fitted anew. Raw and reference-file hashes are stored. Wall time: **88.29 s**.

Optimization: bounded L-BFGS-B, analytic gradients, ftol=1e-12, gtol=1e-7, maxiter=1000, maxls=40, no regularization. RL replay and its exact filtered recurrence/derivative reuse validated authors code. The multistart wrapper is separate because the authors fitter has model-specific parameter packing; that implementation was not edited. Training AIC/AICc/BIC are secondary; held-out log loss is primary.

Paired differences are **reference loss minus cumulative loss**, so positive favors cumulative. `delta_test_LL = NLL_reference - NLL_cumulative`. Numerical ties use absolute per-choice difference <=1e-8 nats; this is not a practical-equivalence margin. Differences beyond +/-0.01 nats/choice are also counted descriptively, without claiming a validated meaningful-effect threshold. Percentile 95% bootstrap intervals use **10,000 participant resamples**, seed 20261015; entire paired participant differences are resampled, never individual trials. Cohort mean differences weight participants equally; trial-weighted totals are reported separately.

## Actual human cumulative histories, characterized before fitting

| Pair / expected direction | Expected count | Fraction | Mean final H difference | Median | P5 / P95 |
| --- | --- | --- | --- | --- | --- |
| 2-3 | 211 | 0.9906 | 0.441232 | 0.451814 | 0.177870 / 0.664030 |
| 4-5 | 212 | 0.9953 | 0.479374 | 0.491831 | 0.218270 / 0.666223 |
| 6-7 | 212 | 0.9953 | 0.461695 | 0.462824 | 0.277203 / 0.614414 |

Differences are H3−H2, H4−H5, H7−H6, respectively, measured **after final training**, ready for the first test trial. They are not substituted for earlier trial histories. Expected directions are diagnostic labels, not fitting constraints.

| Stimulus | Mean final H | Median | P5 | P95 |
| --- | --- | --- | --- | --- |
| 1 | -0.318505 | -0.331633 | -0.476472 | -0.132736 |
| 2 | -0.197841 | -0.213542 | -0.342809 | -0.016675 |
| 3 | 0.243391 | 0.262626 | 0.029155 | 0.394913 |
| 4 | 0.233873 | 0.246193 | 0.044974 | 0.364003 |
| 5 | -0.245501 | -0.251269 | -0.371725 | -0.096227 |
| 6 | -0.287022 | -0.288360 | -0.392218 | -0.178829 |
| 7 | 0.174673 | 0.178392 | 0.043752 | 0.285663 |
| 8 | 0.390736 | 0.404523 | 0.262231 | 0.470000 |

## released_code_intended

Complete paired participants: 213; failed participant fits: []. Failures are retained, not silently removed.

| Model | Total test NLL | Trial-weighted log loss | Participant mean log loss | Participant median log loss | Training total NLL |
| --- | --- | --- | --- | --- | --- |
| RL | 12775.685023 | 0.45111882 | 0.45111846 | 0.44561715 | 74259.6248 |
| RL+CK | 11953.731248 | 0.42209503 | 0.42216430 | 0.42177305 | 67343.3710 |
| RL+cumulative | 12860.440776 | 0.45411161 | 0.45421641 | 0.44634757 | 66956.3320 |

| Comparison | Better / worse / tied | Mean gain | Median gain | P5 / P95 across people | 95% bootstrap CI for mean | Gain >.01 / loss >.01 |
| --- | --- | --- | --- | --- | --- | --- |
| cumulative vs RL | 93 / 120 / 0 | -0.00309795 | -0.00410256 | -0.082565 / 0.095826 | [-0.01023811, 0.00450424] | 79 / 92 |
| cumulative vs RL+CK | 41 / 172 / 0 | -0.03205211 | -0.02251392 | -0.112802 / 0.018027 | [-0.03801354, -0.02637429] | 26 / 143 |

Against RL: total held-out log-likelihood gain -84.755753 nats; mean participant absolute prediction-probability difference 0.094673.

Against RL+CK: total held-out log-likelihood gain -906.709528 nats; mean participant absolute prediction-probability difference 0.068075.

### Descriptive parameters and fitting diagnostics

beta_H: mean 1.864916, median 1.948787, SD 0.465668, P5 0.899823, P25 1.567710, P75 2.211738, P95 2.559022, min 0.607836, max 2.752255. Signs (tolerance 1e-8): 213 positive, 0 negative, 0 near zero. These are not individual habit-strength measurements.

Unsuccessful starts: 11. Boundary counts within 1e-6: {'alpha_rl': {'lower': 0, 'upper': 42}, 'beta_rl': {'lower': 0, 'upper': 0}, 'beta_h': {'lower': 0, 'upper': 0}}. Full starts/failure diagnostics, training criteria, parameters, final Q/H, and counts are in JSON.

### Equal-value test comparisons

“Frequent” means stimulus 3, 4, or 7 for pairs 2–3, 4–5, 6–7. Observed and predicted rates below are trial-weighted over the identical retained comparisons. Predictions are one-step-ahead, using prior observed choices. They are not free-running simulated preferences.

| Pair | Model | Trials | Observed frequent rate | Predicted frequent rate | Log loss | Mean Qfrequent−Qrare |
| --- | --- | --- | --- | --- | --- | --- |
| 2-3 | RL | 2452 | 0.537520 | 0.504327 | 0.691473 | 0.007832 |
| 2-3 | RL+CK | 2452 | 0.537520 | 0.568449 | 0.638700 | 0.000741 |
| 2-3 | RL+cumulative | 2452 | 0.537520 | 0.679594 | 0.696895 | 0.000700 |
| 4-5 | RL | 2450 | 0.512245 | 0.502682 | 0.697337 | 0.003464 |
| 4-5 | RL+CK | 2450 | 0.512245 | 0.565892 | 0.657636 | -0.001069 |
| 4-5 | RL+cumulative | 2450 | 0.512245 | 0.693660 | 0.752421 | 0.000581 |
| 6-7 | RL | 2493 | 0.473325 | 0.502878 | 0.695818 | 0.006171 |
| 6-7 | RL+CK | 2493 | 0.473325 | 0.548132 | 0.663119 | 0.000357 |
| 6-7 | RL+cumulative | 2493 | 0.473325 | 0.687331 | 0.785071 | 0.001486 |

## paper_intended

Complete paired participants: 213; failed participant fits: []. Failures are retained, not silently removed.

| Model | Total test NLL | Trial-weighted log loss | Participant mean log loss | Participant median log loss | Training total NLL |
| --- | --- | --- | --- | --- | --- |
| RL | 12774.923581 | 0.45109193 | 0.45106019 | 0.44489800 | 74573.0491 |
| RL+CK | 12270.848960 | 0.43329269 | 0.43339165 | 0.42804310 | 66481.7179 |
| RL+cumulative | 12885.804606 | 0.45500722 | 0.45512205 | 0.44746313 | 66974.0694 |

| Comparison | Better / worse / tied | Mean gain | Median gain | P5 / P95 across people | 95% bootstrap CI for mean | Gain >.01 / loss >.01 |
| --- | --- | --- | --- | --- | --- | --- |
| cumulative vs RL | 100 / 113 / 0 | -0.00406186 | -0.00598293 | -0.078521 / 0.078573 | [-0.01103531, 0.00296213] | 81 / 97 |
| cumulative vs RL+CK | 55 / 158 / 0 | -0.02173040 | -0.01321305 | -0.085778 / 0.024087 | [-0.02756217, -0.01611605] | 27 / 123 |

Against RL: total held-out log-likelihood gain -110.881025 nats; mean participant absolute prediction-probability difference 0.094625.

Against RL+CK: total held-out log-likelihood gain -614.955646 nats; mean participant absolute prediction-probability difference 0.049463.

### Descriptive parameters and fitting diagnostics

beta_H: mean 1.895311, median 1.954087, SD 0.478464, P5 0.907446, P25 1.610798, P75 2.233045, P95 2.562025, min 0.644807, max 3.395255. Signs (tolerance 1e-8): 213 positive, 0 negative, 0 near zero. These are not individual habit-strength measurements.

Unsuccessful starts: 17. Boundary counts within 1e-6: {'alpha_rl': {'lower': 0, 'upper': 56}, 'beta_rl': {'lower': 0, 'upper': 0}, 'beta_h': {'lower': 0, 'upper': 0}}. Full starts/failure diagnostics, training criteria, parameters, final Q/H, and counts are in JSON.

### Equal-value test comparisons

“Frequent” means stimulus 3, 4, or 7 for pairs 2–3, 4–5, 6–7. Observed and predicted rates below are trial-weighted over the identical retained comparisons. Predictions are one-step-ahead, using prior observed choices. They are not free-running simulated preferences.

| Pair | Model | Trials | Observed frequent rate | Predicted frequent rate | Log loss | Mean Qfrequent−Qrare |
| --- | --- | --- | --- | --- | --- | --- |
| 2-3 | RL | 2452 | 0.537520 | 0.506391 | 0.694120 | 0.010399 |
| 2-3 | RL+CK | 2452 | 0.537520 | 0.630564 | 0.662559 | -0.001377 |
| 2-3 | RL+cumulative | 2452 | 0.537520 | 0.681526 | 0.697951 | -0.000522 |
| 4-5 | RL | 2450 | 0.512245 | 0.501599 | 0.694948 | 0.002556 |
| 4-5 | RL+CK | 2450 | 0.512245 | 0.628641 | 0.684230 | -0.008137 |
| 4-5 | RL+cumulative | 2450 | 0.512245 | 0.694432 | 0.751890 | -0.005868 |
| 6-7 | RL | 2493 | 0.473325 | 0.502457 | 0.694515 | 0.005545 |
| 6-7 | RL+CK | 2493 | 0.473325 | 0.615862 | 0.704765 | -0.002107 |
| 6-7 | RL+cumulative | 2493 | 0.473325 | 0.689557 | 0.786726 | -0.001169 |

## Limits and interpretation

The paired participant bootstrap describes uncertainty in this cohort-level mean predictive contrast, assuming independent participants. It is not a confidence interval for any person’s beta_H, not a causal test of habit, and not evidence of planning. Previous synthetic single-trajectory calibration was poor; moreover that estimator used fixed utilities, whereas this milestone uses learned Q. Its exact error rates do not transfer to this new parameterization.

Holding the RL rule and Q initialization constant isolates the reward-learning component, not every property of the history families: the specified CK initializes from pre_val, cumulative history starts at zero, and cumulative beta_H is signed whereas authors beta_CK is nonnegative. Parameter counts alone do not establish fairness. The optional nonnegative-weight and shared-beta analyses are not performed; joint shared-weight nuisance optimization is deferred rather than introduced into this first participant-specific comparison.

No planning was fitted. A cumulative predictive advantage would not identify forward-looking decisions; a disadvantage would not establish that humans never use cumulative history. Final scientific conclusions are reviewed below after inspecting the complete predeclared cohort results.

<!-- Reviewed interpretation -->

## Reviewed scientific conclusions

1. **Predictive beyond RL? Not demonstrated on held-out test.** Cumulative history
   improves training likelihood, but the primary mean test gain versus RL is
   −0.003098 nats/choice, with participant-bootstrap 95% CI
   [−0.010238, +0.004504]. This is not evidence of an improvement, nor a formal
   demonstration of equivalence. It wins for 93/213 people (43.7%), loses for
   120, and has no numerical ties. The median gain is −0.004103; individual
   differences are heterogeneous (5th–95th percentiles −0.082565 to +0.095826).

2. **Better than the authors' exponential CK? No.** Cumulative history has
   0.032052 more test nats/choice on average; the entire paired bootstrap interval
   favors RL+CK. RL+CK wins in 172/213 participants (80.8%). Cumulative wins in
   41 (19.2%), so this is not a universal per-person ordering. The paper-intended
   sensitivity gives the same qualitative result: no demonstrated gain over RL,
   and a disadvantage to RL+CK (mean gain −0.021730, CI entirely below zero).

3. **Practical magnitude? There is no established cumulative advantage to assess.**
   Its small cohort mean disadvantage to RL is uncertain. Its disadvantage to CK
   is not merely a numerical tie: 0.032052 nats/choice is approximately 7.6% of CK's
   mean loss, and 143 participants lose more than the predeclared descriptive
   0.01-nat threshold, versus 26 who gain more than that threshold. This threshold
   is not a validated psychological or application-specific importance criterion.
   Mean absolute probability differences are 0.0947 versus RL and 0.0681 versus CK;
   the models make meaningfully different numerical predictions even though
   cumulative history does not provide better prediction.

4. **Justification for human planning fits next? Not from these results.**
   The myopic cumulative representation has not beaten RL or the established CK
   on the primary held-out comparison. Adding planning now would not resolve this
   demonstrated generalization problem, and these data have not tested planning.
   There is no conclusion here about whether humans do or do not plan.

### Why strong training history does not establish predictive success

Nearly every participant finishes training with the expected frequent-member
history ordering, and **all 213 signed beta_H estimates are positive**, despite
allowing negative values. Nevertheless those training-derived weights overpredict
frequent-member choices on the equal-value test pairs:

| Preferred diagnostic member | Observed test frequency | RL prediction | RL+CK prediction | RL+cumulative prediction |
| --- | --- | --- | --- | --- |
| 3 rather than 2 | 53.75% | 50.43% | 56.84% | 67.96% |
| 4 rather than 5 | 51.22% | 50.27% | 56.59% | 69.37% |
| 7 rather than 6 | 47.33% | 50.29% | 54.81% | 68.73% |

RL's mean final Q differences for these pairs are small (0.00783, 0.00346,
0.00617), consistent with near-equal aggregate predictions. Both history models
separate the pairs, but cumulative history separates them too strongly in the
frequent-member direction. CK has the lowest conditional log loss on all three
pair types. A pooled predicted proportion alone is not a complete scoring metric:
individual probabilities and causal adaptation to earlier test choices also matter.

The cumulative model's slow adaptation after hundreds of accumulated training
presentations is a **plausible model-based explanation** for this pattern, not an
identified human mechanism. We have not manipulated update rules or estimated
phase-specific weights to establish that explanation. This benchmark comparison
also retains the specified history-initialization and sign-constraint differences.

The positive beta_H distribution is descriptive, not 213 individually significant
history effects. No individual intervals were fitted, and prior single-agent
synthetic identifiability warnings remain relevant. Every selected signed optimum
is already inside [0,10], so a nonnegative constraint would not exclude any of the
reported solutions; a separate nonnegative fitting run was unnecessary here.
The optional shared-weight analysis is deferred.

**Concrete next step:** keep released-code-intended RL+CK as the human predictive
benchmark. Predeclare a phase-resolved generalization check using held-out training
sessions and subsequent test-trial positions to assess where cumulative-history
predictions become miscalibrated. Treat this already-inspected final test as an
evaluation set, not a tuning set; any revised mechanism needs new held-out
validation. Do not add human planning fits until a simpler representation earns
that additional comparison.

## Verification, restart stability, and reproducibility

The final complete repository suite passed: **216 tests in 44.14 seconds**
(186 existing tests plus 30 new cases). New coverage includes causal count timing,
session/gap handling, participant isolation, training-to-test carryover, frozen
test Q, signed history effects, beta_H=0 RL equivalence within floating-point
precision, analytic-gradient checks, controlled matched recovery for negative/zero/
positive weights, explicit optimizer failure handling, benchmark replay guards,
test-independent fit inputs, and paired participant bootstrap aggregation.
The full suite also passed before human fitting (213 tests at that point; three
additional reference-corruption guards were added during final verification).

The deterministic restart subset was selected by retained training length, with
lexicographic participant-ID ties: shortest rank, middle rank (zero-based 106),
longest rank, plus E11T9A. It contained r01b4b (667), e05w0s (775), s04t2z (797),
and E11T9A (783). Selection did not inspect human choices or fitted performance.
On these eight participant × initialization fits, 10 and 200 starts differed by
at most 2.28e-13 deviance; this check took 3.80 seconds. The prior E11T9A ten-start,
two-initialization smoke test took 0.88 seconds.

We nevertheless retained **200 starts** for all final fits. In the full cohort,
10 starts missed the best 200-start result by more than 1e-6 for two participants
in each initialization variant (maximum excess: 2.36083 primary, 1.56938 paper).
Fifty starts matched 200 to within 3.42e-13 across all 426 fits. This is a
descriptive post-run stability check, not a reason to discard participants or
replace the actual 200-start protocol.

There were **426 new fits**, **85,200 starts**, **zero failed fits**, and **28
unsuccessful starts** (11 primary, 17 paper; reported L-BFGS-B `ABNORMAL` termination).
Failed starts were not eligible minima and remain in JSON. Boundary estimates
occurred only for alpha_RL=1 (42 primary, 56 paper); no beta_H estimate hit −10 or
+10, and no beta_RL estimate hit a bound. Runtime for the full fit/replay/aggregation
run was 88.29 seconds with four workers, excluding final file serialization.
The 852 saved RL/RL+CK fits across the two initializations were replayed, not
optimized anew.

Independent scalar replay agreed with the new fast objective within 1.37e-12
deviance. Every cumulative training fit improved on its nested RL-only benchmark.
All saved benchmark deviances and row counts passed the pipeline's exact-input
checks. All 213 participants have five retained training sessions; all training
and test rows were already chronological, so sorting changed no participant's
sequence. Omitted trials receive no hidden updates. Original authors source-file
hashes stayed unchanged, and raw/reference files remain Git-ignored.

Run commands (after installing the project dependencies):

```sh
.venv/bin/python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/run_human_history_comparison.py \
  --stability-check --restarts 200 --workers 4 --paper-sensitivity \
  --output docs/human_history_restart_stability.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/run_human_history_comparison.py \
  --restarts 200 --workers 4 --paper-sensitivity \
  --output docs/human_history_results.json
```

Full-precision results include all participant parameters, phase likelihoods and
information criteria, final states, paired differences, start diagnostics,
reference/source hashes, bootstrap replicate means, and equal-pair diagnostics in
[human_history_results.json](human_history_results.json). The structural restart
sample is retained separately in
[human_history_restart_stability.json](human_history_restart_stability.json).
JSON model key `ck` always denotes **RL+CK (authors model 19)**, not CK-only.

No prior scientific model was changed. No commit or push was performed.


