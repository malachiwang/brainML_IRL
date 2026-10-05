# Fake Human with causal choice history

## Why add history?

The original Fake Human, or Fake Bob, values each stimulus the same way forever.
Its choices can be noisy, but earlier choices cannot change later preferences.
The explicit current-pair MDP confirmed this limitation: action-independent
pair transitions make soft-IRL reduce to the same immediate logistic rule.

This milestone adds one known history mechanism and asks whether it can be
recovered from synthetic behavior. It uses a sequential conditional likelihood,
not a larger Bellman model. The objective utilities remain fixed; Bob does not
learn the rewards or update beliefs about them.

## What history means here

Before trial t, for each stimulus i, count:

- `C_i(t)`: previous retained synthetic trials on which i was chosen.
- `N_i(t)`: previous retained synthetic trials on which i was displayed.
- `R_i(t) = C_i(t)/N_i(t)` if seen; otherwise `R_i(t)=0.5`.
- `H_i(t) = R_i(t)-0.5`, ranging from −0.5 to 0.5.

An unseen stimulus has centered history zero. This is a running **choice rate
conditional on presentation**, not total choices, recency, reaction time, or
an authors' computational choice kernel. Two stimuli with equal reward can
acquire different rates because they face different opponents.

For example:

| Stimulus | Previous choices/presentations | Centered history |
| --- | ---: | ---: |
| 2 | 2/6 | −1/6 |
| 3 | 5/6 | +1/3 |

Both have base utility 2. With alpha=1 their effective values are 1.8333 and
2.3333. At beta=1, the probability of choosing stimulus 3 when paired with 2
is `sigmoid(0.5)`, about 0.6225. With alpha=0 the probability is exactly 0.5;
with negative alpha the history preference reverses.

## Exact timing and carryover

On each trial:

1. Read counts from earlier retained synthetic trials only.
2. Calculate the two centered histories and current probability.
3. Sample the current choice (or score it during prediction).
4. Increment presentation counts for the displayed stimuli, then the chosen
   count for the selected stimulus.

The current response never contributes to its own predictors. Final training
choice proportions would leak later responses, including potentially the
response being predicted, into earlier predictors. We never use them that way.
Estimator histories are rebuilt from chronological synthetic actions; externally
supplied history fields are ignored.

History carries across all five sessions by default. Trial-number gaps make
no updates and do not reset history. Only the retained displayed trials are
available, so we do not reconstruct missing exposures or claim to know a real
person's complete history. The previous MDP's fragment termination convention
has no effect on this counter-based simulator. Each independent synthetic agent
starts with fresh counts. `--reset-history-each-session` is an optional,
consistently applied simulation/fitting/prediction sensitivity setting.

## Choice rule and the hidden answer key

```text
effective_value_i(t) = U_i + alpha * H_i(t)
P(left at t) = sigmoid(beta * [U_L-U_R + alpha*(H_L(t)-H_R(t))])
```

The default base utilities are `[1,2,2,3,3,4,4,5]`. Beta is fixed and known,
normally 1. Alpha=0 exactly reproduces the old simulator's probabilities and
choices when supplied the same random draws. Positive alpha favors previously
selected stimuli; negative alpha favors those with lower earlier choice rates.
Neither sign is imposed during estimation.

The generator knows U and alpha. The estimator receives only rollout IDs,
session/trial chronology, displayed stimuli, synthetic actions, and fixed beta.
Generator probabilities and final counters are separate diagnostic outputs,
not estimator inputs. Human actions, human rewards, human choice proportions,
and human test behavior are unused. True utilities/alpha and their equality
relationships appear only in generation and subsequent evaluation.

## What the estimator does

Given observed synthetic actions, their preceding histories are deterministic.
The sequential likelihood therefore factors as:

```text
product_t P(observed action_t | displayed pair_t, observed actions before t)
```

At each row, use seven left-minus-right stimulus indicators (stimulus 1 is the
zero reference) and one history contrast `H_L-H_R`. This is logistic regression
on causally reconstructed covariates. Fit eight unrestricted parameters:
`w_2,...,w_8,alpha`. The history is held fixed during optimization because it
comes from the observed earlier actions; it is not resimulated at each proposed
parameter. There are no reward tie constraints, priors, penalties, or sign bounds.

The code uses stable signed-logit likelihoods and an analytic gradient. It checks
design rank and complete/quasi-separation before BFGS optimization, rejects
failed or nonfinite fits, and reports the observed-information eigenvalue and
condition number. The separation diagnostic uses a bounded *direction* in a
linear program; those diagnostic bounds never constrain fitted alpha.

For separation background, see the [linear-programming existence test](https://academic.oup.com/biomet/article-abstract/73/3/755/250559)
and the [SciPy linear-programming interface](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html).

History can become difficult to distinguish from utilities: if H_i is nearly
constant over time and agents, `w_i + alpha*H_i` behaves like a static utility.
Variation within and between independent rollouts supplies information about
alpha. A full-rank, well-conditioned fit is not a guarantee of small estimation
error from one short trajectory.

## Recovery, prediction, and probes answer different questions

Full-session fits measure recovery of the hidden answer key. The static model
fixes alpha=0 and uses the existing BT estimator on identical synthetic choices.
The history model estimates alpha jointly with base utilities. A higher training
likelihood alone is not evidence for history: the larger model can overfit.

For prediction, a separate fit uses only sessions 1–4. Session 5 is scored one
trial at a time using the previous observed synthetic actions, including those
in sessions 1–4. After scoring a session-5 response, it updates history for the
next response. Parameters stay fixed. This is one-step-ahead conditional
prediction, not a free-running prediction of an entire unobserved session.

After training, freeze counts and query `(2,3)`, `(4,5)`, and `(6,7)`. These are
analytic synthetic probes, not additional simulated choices or actual test-phase
trials. Equal base utilities mean that any true probe preference comes from
the history term. The favored member is measured in each rollout. A separate,
evaluation-only reward-only oracle calculates expected selection rates from
the scaffold's opponent mix; its ordering provides a schedule-direction
comparison without hard-coding a winner. Neither calculation informs fitting.

## Why this is still a synthetic milestone

This running-rate feature is a deliberate toy mechanism, not the authors'
exact habit model, a validated psychological mechanism, or an inferred human
coefficient. The model changes choices through history but does not plan over
future histories. Successful synthetic recovery would support further model
development, not psychological conclusions. See [the actual results](history_baseline.md),
including unreliable single-rollout estimates and small held-out gains, before
deciding how to approach formal history-aware IRL or human model comparison.
