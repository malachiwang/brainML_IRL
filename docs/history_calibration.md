# Uncertainty about history is not evidence about planning

The earlier synthetic milestones recovered known utilities and history strength
with enough independent agents. They also exposed a serious limitation: a
single agent often had a large, wrong-sign alpha estimate. A point estimate
alone does not tell us how strongly behavior supports a history effect.

This milestone asks two separate questions:

1. **History uncertainty:** does a nominal 95% interval contain the known alpha
   at approximately the advertised rate across synthetic datasets?
2. **Planning distinguishability:** can held-out behavior tell the existing
   myopic and h=3 planning mechanisms apart?

Neither question uses human actions. Both use the already validated simulators,
history rule, and likelihoods without changing their behavioral assumptions.
See the [actual results](history_calibration_results.md) and
[machine-readable output](history_calibration_results.json).

## What a profile-likelihood interval does

First fit the existing myopic history model, estimating seven free base
utilities and alpha, with stimulus 1 fixed to zero. Beta remains known and fixed.
Call the maximized log likelihood `ell_max` and the estimated history strength
`alpha_hat`.

For each proposed alpha, **hold only alpha fixed** and optimize all seven
utilities again. This gives `ell_profile(alpha)`. It is not the likelihood
obtained by freezing utilities at their unrestricted estimates. Re-optimization
allows utility/history tradeoffs and is essential to this calculation.

The nominal 95% interval contains alpha values satisfying:

```text
LR(alpha) = 2 * [ell_max − ell_profile(alpha)] <= chi2_1(.95)
chi2_1(.95) = 3.841458820694124
```

Equivalently, the tolerated log-likelihood drop is about 1.92073. Alpha is free
to be positive or negative. There is no prior, penalty, equality constraint,
truth-based initialization, or estimated beta.

### Numerical endpoint search

The original unrestricted fit is unchanged. Conditional fits reuse the existing
causal-history design, stable signed-logit likelihood, and analytic utility
gradient. BFGS reoptimizes seven utilities with gradient tolerance 1e-8 and a
1,000-iteration limit. Nearby conditional solutions provide warm starts; they
are not accepted without reoptimization.

Start on each side of alpha_hat with distance `max(0.1, observed-information SE)`.
Double the distance until the LR crosses the cutoff, then use Brent root-finding
with absolute alpha tolerance 1e-7. Verify LR=0 at the unrestricted estimate and
record the LR at both endpoints. The search extends at most 64 alpha units on
each side of the estimate. This is a **computation limit**, not a constraint on
estimated alpha.

If no crossing is found, the endpoint is JSON `null` with status
`open_search_limit`; the searched location and its LR are recorded separately.
It is not silently reported as a finite endpoint and is not proof that the
true interval is infinite. Descriptive containment treats an unresolved side
as extending outward; widths summarize closed intervals only, with missing/open
counts explicit. Any conditional optimization failure makes that profile fail
explicitly rather than fabricating an endpoint. Failed profiles remain in raw
results and attempted counts.

## Coverage, detection, and power

The chi-square cutoff is an asymptotic approximation, not a guarantee for a
short adaptive sequence. Here "95% coverage" means that if we repeatedly
generate a dataset at a fixed known alpha, about 95% of the resulting intervals
would include that fixed value. It does not mean a 95% posterior probability
that alpha lies in one observed interval.

The synthetic study measures that coverage rather than assuming it. A nominal
interval can under-cover because of small-sample bias or inadequate asymptotic
approximation even when optimization succeeds and endpoint calculations are
accurate. Wider intervals alone do not guarantee correct calibration.

For zero truth, any interval excluding zero is counted as false history
detection in this simulation design. For positive truth, a strictly positive
interval counts as detection in the correct direction; a strictly negative
interval is a wrong-direction detection. Intervals containing zero mean this
rule does not establish a nonzero effect. These statements concern synthetic
data under the specified model, not a population-level human false-positive rate.

As a secondary diagnostic, calculate Wald intervals from the inverse full
observed-information matrix. Taking the alpha entry of that inverse includes
nuisance-utility uncertainty; it is not the reciprocal of the alpha diagonal
alone. The profile interval remains primary. Neither interval is adjusted after
seeing calibration results.

## Two mechanisms, held-out comparison

Myopic History Bob values `w_i + alpha H_i` now. Planning History Bob additionally
values how today's chosen-count update changes future history-dependent rewards.
The existing planner uses h=3, gamma=0.95, beta=1, and knows upcoming displayed
pairs. It never knows future observed choices. The history definition and
cross-session carryover are identical for both models.

For each synthetic training dataset, both estimators receive exactly the same
choices. Neither optimizer receives the generator label, true alpha, true
utilities, or evaluation agents. Both have eight free parameters. Selection is
based on held-out NLL, **not** on training fit or closeness to the answer key:

```text
delta = NLL_myopic − NLL_planning
positive delta favors planning; negative delta favors myopic
absolute delta <= 1e-8 total nats is a numerical tie
```

Truth labels are attached only for post-fit recovery and confusion matrices.
At alpha=0 the two generating mechanisms are behaviorally equivalent, so their
nominal identity cannot be recovered. This condition is reported separately as
a null control. Fitted models may still differ because freely estimated alpha
is not constrained to zero, and mechanical held-out winners need not split 50/50.

### Two complementary evaluation schemes

**Within trajectory:** fit sessions 1–4, then score session 5 one step ahead.
Observed synthetic history carries across the split. After scoring each held-out
action, its observed response updates history for later predictions. Parameters
remain fixed. This scheme closely matches the previous milestone, but largely
misses the early decisions where planning can matter most.

**Fresh agents:** fit all five sessions of the training agents, then score 20
new independent agents generated by the same mechanism and true parameters.
They start with fresh history and are never used for fitting. Both fitted
models score the same new observations. Independent evaluation-agent RNG roots
and disjoint rollout IDs protect the train/evaluation separation.

For fresh agents, predeclare disjoint windows:

| Window | Retained observations per agent |
|---|---|
| Early | First 20 trials of the entire retained scaffold, not original labels 1–20 |
| Mid | Remaining trials in sessions 1–4 |
| Late | Session 5 |
| All | Early + mid + late |

Mid is also split into session-1 remainder and sessions 2–4 for description.
An early prediction uses only earlier responses, even though later responses
are available in the stored synthetic trajectory. Scoring never refits.

## Predeclared design and reproducibility

Reuse the exact ordered list of 12 scaffolds stored in the prior replication
JSON. Validate the file hash and every scaffold's structural summary. Do not
rerank by any new result. Project the R table to ID/session/trial/stim1/stim2
before normalization, so human actions/rewards do not enter this pipeline.

- Uncertainty: myopic generator, alpha=0/0.5/1, 1/5/20 agents, batches 0–9:
  **1,080 datasets**, with one unrestricted fit and one profile per dataset.
- Distinguishability: both generators, same alphas/counts, batches 0–4:
  **1,080 datasets**, each with four fits (both families × both fit scopes).
- Every comparison configuration uses 20 fresh evaluation agents. Within a
  scaffold/alpha/batch/generator, these agents are shared across the three nested
  training counts, not newly treated as independent for each count.

Training roots are exactly the previous replication's `batch_root(participant,
batch)`. Evaluation roots are derived by
`SeedSequence([training_root, 0x4556414C]).generate_state(1, dtype=uint64)`.
Existing simulators spawn independent child streams from these roots. Every
rollout starts fresh; 1/5/20 training agents are nested prefixes. Alpha/generator
conditions share random draws as paired controls, while training and evaluation
roots are separate. Null-generator data therefore coincide for common draws.

This reuses the previous synthetic training design for interval calibration;
it is not a new independent replication of its point estimates. It does add
independent fresh-agent evaluation. Each primary condition has 120 profile
datasets or 60 model-comparison datasets. Conditions/windows are paired, and
the scaffolds are structurally selected rather than a random sample of humans.

Summaries give unweighted means, sample SD, linear-interpolated quantiles,
attempted/successful/failed counts, and scaffold-level results. Open/failed
profiles and unavailable paired predictions stay visible. A tiny NLL advantage
is accompanied by probability differences and modal-action disagreements so
a mechanical "winner" is not confused with a practically large distinction.

## Implications for eventual human work

Detecting a history coefficient and detecting planning are different problems.
History might be well supported while the two history mechanisms predict almost
the same behavior. Conversely, poor interval coverage can make a nominal
single-person history claim unsafe even when the simulator is perfectly matched.

Synthetic calibration must therefore govern what is claimed next. Pooling
identical-parameter synthetic agents does not justify pooling heterogeneous
people. Known beta, the fixed running-rate feature, retained-trial gaps, and the
planner's knowledge of future pairs remain modeling assumptions. No human habit,
planning coefficient, or reward function is inferred in this milestone.
