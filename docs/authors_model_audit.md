# Authors' Reward Pairs benchmark: source audit

This is a separate benchmark family. It intentionally scores human choices to
reproduce the authors' analysis, not to fit our novel history/IRL model. Existing
synthetic models and their results are unchanged. Numerical reproduction results
and the final benchmark recommendation are in [the results report](authors_baseline_results.md).

## Sources and provenance

The three unmodified R files were copied from Downloads to ignored
`data/reference/`. The locally named `collabra_2024_10_1_92949_194475.pdf` is
actually the 35-page **Supplementary Information**, not the main article. Its
pages 4-12 were extracted with macOS PDFKit; equations on pages 5-6 and Table S5
on page 10 were also rendered and visually checked. Poppler was unavailable.
The original files, copied references, and RData remain untracked/ignored.

Source abbreviations below: D = `02_comp_mod_RP_task.R`;
F = `02_functions4modelling.R`; C = `02_comp_mod_RP_task_model_comparisons.R`.
The [main article](https://doi.org/10.1525/collabra.92949), Figures 4-5 and
computational-model results, supplies published comparison targets. The local
input is `02_comp_mod_RP_task_data_in.RData`, not saved fitted results.

## What the quantities mean

Q is a learned expectation of stimulus reward. H is an exponentially updated
trace of selecting a stimulus when displayed. `authors_alpha_ck` determines
**how quickly H updates**; `authors_beta_ck` determines **how strongly H affects
choice**. Neither parameter by itself proves a psychological habit.

For a displayed stimulus i, after scoring the current response:

```text
Q_i <- Q_i + authors_alpha_rl * (reward_i - Q_i)       [training only]
H_i <- H_i + authors_alpha_ck * (chosen_indicator_i - H_i)
P(left) = sigmoid(beta_rl*(Q_L-Q_R) + beta_ck*(H_L-H_R)
                  + 2*bias-5 + pi*(previous_left-previous_right))
```

Both displayed Q values update, not just the chosen value. Both displayed H
values update: selected toward 1, unselected toward 0. Undisplayed values stay
unchanged. Test supplies no feedback: Q freezes, but observed test choices
continue updating H. Choices are always scored **before** their updates.

## Audit table (recorded before benchmark implementation)

| Feature | Supplement description | Released driver | Released helper | Reproduction decision |
|---|---|---|---|---|
| RL initialization source | Participant pre-training liking ratings, pp.4-5 | D:72-98 averages ratings and pretest choice proportions | F accepts supplied Q vector | A: ratings only; B/C: driver's mean |
| RL initialization scale | `(rating-min)/11.15`, pp.4-5 | D:81 uses `/20`; pretest `(prop-min)*5` | No rescaling inside F | Preserve both explicit alternatives |
| CK initialization | All zero, p.5 | D:125 `value=pre_val`, passed as BOTH CK and RL | F uses supplied H, not zeros | A: zero; B/C: pre_val, possibly above 1 |
| Reward units | Text pp.4-5 discusses 0-9; Figure S2 p.12 uses 1-5 | Comment D:76 says 0-5; supplied training rewards 1-5 | Uses reward columns literally | All primary variants retain supplied 1-5 rewards; A isolates requested initialization/rule interpretation, not a reconstruction of original point units |
| RL chosen update | Prediction error with alpha_q | Equal chosen/unchosen rates | F:300-307 updates both | Same alpha for both displayed stimuli |
| RL unchosen update | Feedback for both, p.5 | Same alpha_rl supplied twice | Same prediction-error rule | No chosen-only RL |
| CK chosen update | Toward 1, p.5 | Same alpha_ck supplied twice | F:297-305 | Update after scoring |
| CK unchosen update | Toward 0, p.5 | Same alpha_ck supplied twice | F:297-305 | Update after scoring |
| Session boundaries | Fits all five sessions; forgetting considered but not retained, p.4 | One continuous training table | Session argument unused by relevant objective/scorers | Carry Q/H and previous response across sessions |
| Missing rows | Already processed observations | Iterates retained rows, no reconstruction | `1:nr_trials` | No hidden updates, imputation, or new exclusions |
| Train/test transition | Training parameters used at test, pp.6-7 | Passes final training Q/H | Each scorer starts previous-response vector at `(0,0)` | Carry Q/H; reset positional previous response at phase boundary |
| Test RL | No feedback | Passes final training Q | F:391-440 makes no Q updates | Freeze Q; test reward columns not needed in Python |
| Test CK | Choice-dependent trace | Passes final training H | Continues H updates from test choices | Causal one-step test scoring, not free-running simulation |
| Side bias | Left b, right 5-b; b in [0,5], p.6 | Free only model 1 | Others b=2.5 | Logit contribution `2*b-5`; neutral cancels |
| Response stickiness pi | Not in retained four-model softmax, p.6 | No assignment in D/C | Local pi=0 in F:213/229/245/261/277; unresolved in F:345/405 | A/B score pi=0; C uses clean-R lookup; retain separate objective/rescore |
| Alpha bounds | [0,1], p.5 | L-BFGS-B bounds | Ties chosen/unchosen | Enforce exact bounds |
| Beta bounds | [0,10], p.6 | L-BFGS-B bounds | Models disable unused beta | Enforce exact bounds |
| Reduced constraint | beta_q=10-beta_h, p.7 | Actual expressions use 10; some comments say 1 | F:272 uses 10-theta[2] | Use 10, not stale comments or Table S5 label |
| Optimizer | Individual maximum likelihood | `optimr`, L-BFGS-B; no explicit tolerances | Returns deviance | Bounded SciPy L-BFGS-B, stable likelihood, analytic derivative; explicitly not identical optimizer software |
| Random starts | Not specified in relevant model section | 200; seed=1234+17*one-based participant index; five uniforms per start | No RNG in objective | Configurable 200; R-generated starts when R exists; report actual protocol |
| Failures | Not specified | Retries exceptions at unchanged start up to 1001 attempts; accepts non-null fits without convergence check | Infinite deviance replaced by 10000 | Python retains failed starts, rejects invalid fits; stable logs avoid artificial 10000 plateau |
| IC formulas | BIC selection, p.7 | Computes train AND test IC | F:491 AIC=dev+2k; AICc=dev+2k*n/(n-k-1); BIC=dev+k*log(n) | Reproduce all three, including same k on test |
| Free parameter counts | Bias; two rates/weights; four combined; three reduced | k=1,2,2,4,3 | Reduced constraint removes one beta | Preserve counts even at fitted boundaries |
| Participant exclusions | Clinical/age plus task-specific criteria | D fits IDs from pretest table | C:28-37 explicit lists | Fit all supplied training IDs; apply exactly listed comparison exclusions, report already absent IDs |
| Preregistered vs exploratory | Four-model comparison; reduced model specified for association analyses, p.7 | Fits five models | C first compares four, then five | Model 20 itself was prespecified for associations; adding it to model comparison is exploratory |
| Group comparison | VBA exceedance probabilities | C exports transposed model-by-participant matrices | No group Bayesian calculation in helper | Export equivalent labeled matrices; no substitute method called VBA |

### Three named variants

* `paper_intended`: Q0=(rating-min)/11.15; H0=0; pi=0 throughout.
* `released_code_intended`: Q0=H0=mean((rating-min)/20,(pretest_prop-min)*5);
  pi=0 throughout. This is an explicit consistency reconstruction, not a claim
  about the authors' historical workspace.
* `released_code_literal_clean_r`: same initialization and fitted objective as
  B; post-fit scoring uses the pi found by clean R. B/C share fitted parameters
  so their comparison isolates rescoring, not optimizer randomness.

Pretest-choice information enters initialization only in the released variants,
as the driver explicitly requires. Post-training ratings and test choices never
enter initialization or training optimization.

## pi namespace audit

R 4.4.1 with `Rscript --vanilla` loaded exactly five input objects:
`data_4_analysis_RP_pretest`, `data_4_analysis_RP_training`,
`data_4_analysis_RP_test`, `ratings_RP2`, `grouped_data_RP_pretest2_clean`.
There is **no pi object**. Searching all local reference R files found only the
five local fitting assignments above. Sourcing the unmodified helper in clean R
and evaluating `get("pi", environment(funky_train))` and the analogous test
lookup returned **3.141592653589793**. Neither scorer has a pi argument.

Previous-response vectors become `(1,-1)` or `(-1,1)`, so literal scoring adds
**+/-2*pi** to the left-right logit after the first response. It does not merely
add pi to a one-hot chosen side. Both scorers reset this vector at entry.
The fitting objective's local pi=0 does not establish a global binding for the
later scorers. Direct behavioral parity tests accompany this namespace check.

This demonstrates fresh-process behavior, **not** the historical environment
that produced published results. Saved fitted/comparison workspaces, if absent,
cannot be used to prove whether the authors had a global pi=0. No inference
that the paper is invalid follows from the source discrepancy alone.

## Exclusions (benchmark comparison only)

Clinical/recent disorder/therapy or age category, exactly C:28:
`b09h8s M09R7M o09b0u b06g1z m07s6z b09u8r s07s3m s11r8m k09z9z l05d0p k06v1t`.
The script groups ten clinical exclusions and one age exclusion; it does not
identify which individual belongs to which subreason.

Missingness, low accuracy, or insufficient explicit value knowledge, C:33:
`e06m9w K05f9m l02m7p m04o8z r02n9k s04s8h s10p5l`.
The script does not attach an individual subreason within this category.
Presence/absence in the supplied input is recorded in the machine-readable audit.
These lists are not applied retroactively to any synthetic study.

## Relation to our work

| Family | State/update | Parameters | Planning? |
|---|---|---|---|
| Authors' CK | Exponential H: chosen toward 1, unchosen toward 0 | alpha_ck update speed; beta_ck choice influence | No |
| Authors' RL | Learned Q: displayed feedback prediction errors | alpha_rl learning speed; beta_rl choice influence | No |
| Our cumulative history | C/N minus .5, unseen=0, exact retained counts | Older alpha is a history weight; freely estimated base utilities | No |
| Our planning history | Same cumulative counts plus finite-lookahead future consequences | Base utilities and history weight; fixed beta/gamma/h | Yes, bounded lookahead |

**Our old alpha != authors_alpha_ck.** Its role is closer to authors_beta_ck,
but the underlying history features still differ. Their Q values are learned
from actual feedback, whereas our previous synthetic base utilities were fixed.
No existing result is renamed as an authors' model result.

Further source cautions: Table S5 p.10 places the CK/RL single-process entries
under apparently swapped alpha/beta column groups, and labels the reduced model
with `1-` despite surrounding text's `10-`. We follow equations and executable
parameter mappings, not infer a new model from these table labels. The table's
reduced CK-rate recovery correlation is .576, so even the authors' diagnostics
do not warrant confident individual interpretation of this rate.

## Measured outcome of the audit

The [completed cohort reproduction](authors_baseline_results.md) fitted all 213
supplied participants with all 200 starts, five models, and the two distinct
initializations; C reuses B's fits. All 18 listed exclusion IDs were already
absent. No sorting changed any participant's training or test order.

The evidence does **not** support assuming a hidden global pi=0. Literal C
exactly reproduces the publication's preregistered test comparison counts
(7/11/36/19/140). It also yields 211 very-strong reduced-model preferences and
the released fit's mean beta_CK=6.78422. One remaining category differs: our
single RL-favoring reduced comparison is very strong, whereas the article calls
it strong. An independent R diagnostic did not explain away that difference.
Historical saved workspaces are missing, so this is compelling numerical
consistency with literal scoring, not proof of the original runtime environment.

Under pi=0, fitting and rescoring agree to 5.35e-12 deviance; under literal pi,
training rescoring adds an average 3,223.63 deviance. Consequently **use B for a
consistent future author-code benchmark, A for initialization sensitivity, and
C only for forensic reproduction of published comparisons**. Do not use C's
fit/score mismatch to make a new model appear predictively superior.

Original-R unit/participant likelihood and state parity succeeded; 186 tests
pass. SciPy uses exact recurrence derivatives evaluated with linear filters,
which change computation speed, not the statistical model. Base-R default
finite differences were inaccurate near small learning rates in the reduced
model; reducing that step restored E11T9A deviance agreement to 1.4e-11. Neither
the unavailable `optimr` optimizer protocol nor MATLAB VBA exceedance
probabilities is claimed as exactly reproduced. No original source file or
earlier research-model implementation was changed.
