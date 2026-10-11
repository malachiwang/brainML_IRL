# From displayed pairs to an explicit soft-IRL baseline

## What we are trying to learn at this milestone

The long-term project concerns human decision making. This experiment asks a
more controlled question: can we recover a simulated agent's known rewards,
and does representing the same problem as an MDP add information? It does not
fit a reward function to human choices.

## The human data, the scaffold, and Fake Human

Each retained human trial records the participant, session, trial number,
stimuli displayed on the left and right, the human action, and other fields
such as reaction times, rewards, and inclusion flags. The authors already
processed these data. We keep all supplied rows and the existing loader's
explicit handling of the analysis-table name.

For one participant, the code sorts trials by session and trial number. Fake
Human borrows **only the displayed pairs in that order**. Human actions are not
copied, imitated, or used to generate its choices. Session/trial labels are used
only to reconstruct chronology and transition boundaries; they are not state
features. Reward columns and reaction times never enter either estimator.

Fake Human is a memoryless agent with fixed utilities:

| Stimulus | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Utility | 1 | 2 | 2 | 3 | 3 | 4 | 4 | 5 |
| Relative to stimulus 1 | 0 | 1 | 1 | 2 | 2 | 3 | 3 | 4 |

On pair `(L, R)` it chooses left with probability
`sigmoid(beta * (U_L - U_R))`. Higher utility makes a choice more likely, but
does not make it certain. Beta controls this sensitivity and is known and fixed
in both simulation and estimation. The agent has no learning rule, history,
habit, or belief updates.

```text
real displayed-pair sequence (human actions ignored)
                     |
                     v
       Fake Human with hidden known rewards
                     |
                     v
          one set of synthetic actions
                 /           \
                v             v
          BT recovery    soft/MCE-style IRL
                |             |
                +------+------+
                       |
                       v
           compare both to hidden truth
```

The known utilities are an **answer key**. They generate synthetic observations
and score the completed fits. This is not leakage: neither optimizer receives
them. Both start from zero, fit seven unrestricted parameters, and receive no
information saying which stimuli should tie. A test also recovers a different
hidden reward vector with negative values and no ties, guarding against a fit
that merely reproduces the default answer key.

## What Bradley–Terry estimates

BT finds one relative utility per stimulus that makes the observed synthetic
choices as probable as possible under the logistic rule. Its inputs are only
the displayed left/right stimuli and the synthetic action.

Adding the same constant to every utility does not change utility differences.
We therefore fix stimulus 1 to zero. We do not estimate beta together with the
weights: multiplying all weights by a constant and dividing beta by that
constant would leave the predictions unchanged. Fixing beta enables a clean
synthetic comparison, not a claim of an absolute psychological reward scale.

## What the MDP contains

An MDP describes the situation an agent observes, the actions it can take,
their rewards, and what can happen next. Here:

| Component | Definition |
| --- | --- |
| State | Exactly `(left_stimulus, right_stimulus)`; orientation matters |
| Action | `left` or `right` |
| Feature `phi(s,a)` | Eight-dimensional one-hot indicator of the selected stimulus |
| Reward | `r_w(s,a) = w^T phi(s,a)` |
| Transition | Empirical next-pair probabilities, identical for both actions |
| Termination | Zero continuation and no choices/entropy after an observed fragment ends |
| Discount | Gamma, default 0.95, determines how future value enters planning |

For example, `(4,6)` with action left activates feature 4, so its reward is
`w_4`; action right activates feature 6. `(6,4)` is a different state.

To estimate transitions we count next states only for consecutive retained
trial numbers within one participant-session. The last retained row of each
session goes to a terminal marker. So does a row immediately before a gap in
trial numbers: the next original trial is unavailable, and we do not pretend
the next retained row was its immediate successor. Every row still contributes
an action demonstration and one outgoing transition count. The count row is
normalized and copied to both actions. Choices never affect this construction.

Termination at a gap is a modeling convention for a censored observation, not
a claim that the real experiment ended there. Similarly, a stationary pair-only
state cannot reproduce deterministic session length or a counterbalanced
schedule. The resulting MDP is a stationary Markov approximation to retained
fragments. An episode can start from any fragment's initial pair; the action
likelihood conditions on observed states, so no starting-state distribution
needs to be fitted. A terminal marker is bookkeeping, not a new behavioral
state feature.

## What the soft-IRL estimator does

A soft-optimal agent balances expected discounted reward against predictable
behavior: an entropy term of weight `1/beta` allows stochastic choices. The
soft Bellman construction is the one used in maximum-causal-entropy work;
see [Ziebart's explanation](https://www.cs.cmu.edu/~bziebart/) and the
[original MCE paper](https://www.cs.cmu.edu/~bziebart/publications/maximum-causal-entropy.html).

For each proposed reward vector, the code solves:

```text
Q(s,a) = r_w(s,a) + gamma * sum_s' P(s'|s,a) V(s')
V(s)   = logsumexp_a(beta * Q(s,a)) / beta
pi(a|s)= exp(beta * Q(s,a)) / sum_b exp(beta * Q(s,b))
V(terminal) = 0
```

Q means immediate reward plus expected future value. V summarizes the available
actions using a smooth maximum. The policy converts Q values into probabilities.
The implementation actually iterates these equations and differentiates them;
it does not call BT or replace planning with a hard-coded logistic formula.

The outer optimizer minimizes `-sum_t log pi_w(a_t|s_t)` with `w_1=0`. It uses
stable log probabilities, analytic Bellman derivatives, and unregularized BFGS.
Each observed action has equal weight; gamma discounts planning, not the
likelihood of later observations. The transition model is fixed and its
likelihood has no reward-dependent contribution. We call this **discounted
soft-optimal policy-likelihood IRL / MCE-style IRL**; we are not implementing a
separate finite-horizon feature-occupancy matching algorithm or globally
normalized noncausal trajectory MaxEnt model.

Disconnected comparison graphs and separated choices are rejected, as in BT.
The estimator also rejects action-dependent kernels: its v0 identifiability
checks and scientific scope are specific to the action-independent case.
The Bellman solver itself handles such kernels in a small negative-control test
that verifies future consequences really can change its policy.

## Why the two fits coincide

For either action in the current state, the experimental next-state distribution
is the same. Define its continuation value as `C(s)`. Then:

```text
Q(s,left)  = w_L + C(s)
Q(s,right) = w_R + C(s)
Q(s,left) - Q(s,right) = w_L - w_R
```

The common term cancels in the softmax normalization. Hence
`pi(left|s) = sigmoid(beta * (w_L - w_R))` for any allowed gamma and any
action-independent kernel, including its termination probabilities. This is an
algebraic result for this model, verified independently against explicit planning.

The action likelihoods are therefore identical to BT's on identical
demonstrations. A connected comparison graph and nonseparated choices give the
same reference-constrained solution, up to numerical optimization tolerance.
State values can change substantially when gamma or transitions change, while
choice probabilities do not. Such values are model-dependent continuation
quantities; they are not additional reward information recovered from choices.

This equivalence is useful. It explains why current-pair v0 contributes no
action-dependent sequential structure. It is not evidence that IRL generally
reduces to BT or that the model describes human habits.

## The next milestone: causal running-rate history

The [history-sensitive baseline](history_model.md) now implements this sequence:

```text
fixed Fake Bob
    -> history-sensitive Fake Bob
    -> recover known history parameter
    -> later: real-human model comparison
```

It adds the centered running choice rate `H_i=C_i/N_i-0.5`, calculated before
each synthetic choice, to effective utility as `U_i+alpha*H_i`. Counts update
only afterward and carry across sessions by default. Each independent Fake
Human starts fresh. Alpha=0 reproduces the original agent exactly; the new
estimator fits signed alpha and base utilities with a sequential conditional
likelihood. It does not solve a history-augmented Bellman problem. The
[actual results](history_baseline.md) include noisy single-agent estimates,
the zero-effect control, held-out session-5 prediction, and frozen equal-value
probes. Human actions remain unused.

## Replication across experimental scaffolds

```text
history-sensitive Fake Bob on one scaffold
    -> replication across many experimental scaffolds
    -> later: formal history-aware IRL / human modeling
```

The [replication milestone](history_replication.md) repeats the same simulator,
history definition, and sequential likelihood on structurally selected displayed
pair sequences. Twelve scaffolds span retained trajectory lengths, with E11T9A
kept as a reference. Ten independent seed batches per scaffold measure both
small-sample spurious history effects and recovery with 1, 5, or 20 independent
synthetic agents. No human action enters selection, simulation, or fitting.
History-aware IRL and human model comparison were not part of that replication.

## Planning history Bob and sequential IRL

```text
fixed Fake Bob
    -> myopic history-sensitive Fake Bob
    -> replication across schedules
    -> planning history-sensitive Fake Bob
    -> history-aware sequential IRL recovery
    -> later: uncertainty calibration + real-human modeling
```

The [planning milestone](history_irl.md) explicitly models how choosing now
changes future count states and history-dependent rewards. State is `(t,C)`;
the known scaffold determines presentation counts and future displayed pairs.
An exact finite-lookahead soft planner supplies Q-based choice probabilities,
and a matched policy-likelihood estimator recovers utilities and signed alpha.
The old myopic simulator is unchanged. Eight-trial exact DP verifies the logic;
bounded lookahead on real scaffolds is not claimed to solve full-horizon IRL.
The [results](history_irl_results.md) compare recovery, runtime, policy changes,
and myopic/planning mismatch using synthetic choices only.

## Later work, beyond the current-pair IRL milestone

The [uncertainty and model-comparison milestone](history_calibration.md) separates
evidence for a nonzero history parameter from evidence for forward planning:

```text
history mechanism validated
    -> history replication across scaffolds
    -> history-aware planning validated
    -> uncertainty + model distinguishability calibration
    -> NEXT: real-human model comparison, if justified
```

It profiles alpha while re-fitting nuisance utilities, measures synthetic
interval coverage and detection errors, and compares h=0 with h=3 using both
session holdout and fresh independent evaluation agents. Early/mid/late windows
are predeclared. Neither nominal interval confidence nor a tiny predictive win
is treated as proof of a psychological mechanism. Read the
[calibration results](history_calibration_results.md) before deciding whether
individual human inference or planning claims are justified.

The original next step was to specify a minimal **causal history variable**,
its update after each action, and a synthetic mechanism that actually uses it;
the running-rate milestone above provides that first example.
For example, a previous-choice variable could change as a consequence of the
current action. History must be computed from past observations only; reward
features must make that history relevant before it can affect preferences.
That design should be tested for identifiability and recovery on held-out
synthetic trajectories before any human reward inference. Merely appending an
unused history label would not make the problem informative. The new bounded
planning milestone is a first explicit history-aware IRL example, not a validated
habit mechanism. The calibration milestone above now tests nominal uncertainty
and predictive distinguishability. Any further uncertainty corrections, broader
planning formulations, and human fitting remain separate future work.
