# Reward Pairs: synthetic utility recovery baseline

A small Python research starting point for inverse reinforcement learning and
human decision making. The original baseline tests **pairwise-choice utility
recovery**. An explicit current-pair MDP and soft-optimal IRL baseline now test
why action-independent dynamics give the same answer. These are synthetic
sanity checks, **not results about human rewards or generic sequential IRL**.

## Scientific question

In the Reward Pairs task, people choose between two displayed stimuli with
different reward values. Training manipulates how frequently stimuli are chosen;
later tests compare preferences, including between equally rewarded stimuli.
There are eight stimuli and five training sessions, nominally 160 trials each.
See [Characterizing Human Habits in the Lab](https://pmc.ncbi.nlm.nih.gov/articles/PMC7615722/).

Here we ask a narrower question: if an agent has known, fixed stimulus utilities,
can a simple choice model recover their relative ordering from sampled choices
on the experimental pair sequence? Synthetic recovery checks the data plumbing,
choice conventions, estimation, and sampling uncertainty before attempting to
interpret human reward functions. Success under this matched simulator and
estimator does not establish that the model explains humans.

## Setup and run

Requires Python 3.11+. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/run_baseline.py
```

Place the authors' already-processed file at
`data/raw/02_comp_mod_RP_task_data_in.RData`. The data are not distributed by
this project; `data/raw/` and RData files are ignored by Git. An absent file
produces an actionable error. Tests use synthetic fixtures and need no dataset.

Useful options:

```sh
python scripts/run_baseline.py --participant E11T9A --beta 1 --seeds 0 1 2 3 4 --repeats 1 5 20
python scripts/run_baseline.py --data /path/to/file.RData --show-participants
python scripts/run_baseline.py --object data_RP_training_clean
```

Compare the existing estimator with explicit soft-IRL on identical choices:

```sh
python scripts/run_irl_baseline.py --participant E11T9A --beta 1 --gamma 0.95 --seeds 0 1 2 3 4 --repeats 1 5 20
```

The new script prints both fitted vectors and likelihoods, evaluation metrics,
failed fits, and gamma/transition-kernel checks. Add `--output PATH.json` to save
synthetic results. Read the [plain-language method overview](docs/method_overview.md)
and [actual IRL comparison results](docs/irl_baseline.md).

A third milestone adds a causal running-choice-rate effect to Fake Human and
recovers its signed strength with a sequential likelihood (not history-aware IRL):

```sh
python scripts/run_history_baseline.py --participant E11T9A --beta 1 --alphas 0 0.5 1 --seeds 0 1 2 3 4 --rollouts 1 5 20 --output docs/history_baseline_results.json
```

Here `--rollouts` means independent agents with fresh history, not concatenated
copies. History carries across sessions by default; optional
`--reset-history-each-session` applies the same reset rule in simulation and
inference. See [the history mechanism](docs/history_model.md) and
[its recovery/prediction results](docs/history_baseline.md), including the
small-sample limitations. The original v0 baselines remain unchanged.

The default chooses the lexicographically first supplied participant and prints
the ID. This is deterministic selection for a demonstration, not selection by
accuracy, human choices, or recovery quality. All supplied rows are retained.

## Data and chronology

`data.py` isolates loading through [`rdata.read_rda`](https://rdata.readthedocs.io/en/stable/usage.html).
It returns all original objects. Normalization operates on a copy, retains
original named columns, and adds these aliases:

| Original | Python alias |
| --- | --- |
| `ID` | `participant` |
| `stim1` | `left_stimulus` |
| `stim2` | `right_stimulus` |
| `reward1` | `left_reward` |
| `reward2` | `right_reward` |

`participant` becomes a string; `session`, `trial`, and the stimulus aliases
become integers using factor labels, never factor codes. `action` becomes a
nullable string, retaining `left`/`right`. Session/trial/action dtypes can change
in the normalized copy; the original loaded table remains available untouched.
All other columns—including phase, choices, raw stimulus IDs, reaction times,
rewards, value differences, correctness, and inclusion flags—are retained.

**Observed schema difference:** the supplied local file contains
`data_4_analysis_RP_training`, not `data_RP_training_clean`. Automatic selection
prefers the expected clean object and otherwise uses this specific known
analysis variant, printing its actual name. An explicit `--object` must exist.
The authors' local R script distinguishes the analysis table by their inclusion
flags; all flags in the supplied training table are already 1. This code does
not reapply flags, RT cutoffs, accuracy rules, or any new exclusions.
See [the real-data validation report](docs/real_data_validation.md) for all
object names, counts, and the verified file hash.

Validation rejects missing/invalid stimulus labels, sessions outside 1–5,
unknown action encodings, missing chronology keys, nonpositive/noninteger trial
indices, duplicate participant/session/trial keys, and unexpected phase labels.
It reports missing actions, input ordering, per-participant rows and session
counts, per-session row counts, flags, and unordered pair frequencies. Gaps and
counts below 160 are allowed; labels above 160 and optional schema differences
are reported, without renumbering or deleting rows.

`build_trajectory(training_df, participant_id)` returns immutable `Trial`
records sorted numerically by session, then trial. Its state is exactly
`(left_stimulus, right_stimulus)`; action is `"left"`, `"right"`, or `None` for a
missing human response. Thus `(4, 6)` and `(6, 4)` remain distinct states. Counts
combine their orientations only for the unordered-pair summary. No reaction
times, ratings, history, previous states, or test-phase frequency columns enter
the state. Gaps remain gaps in the authors' cleaned sequence; missing original
trials cannot be reconstructed from this file.

## Fake Human and recovery

Fake Human knows the eight fixed utilities `[1, 2, 2, 3, 3, 4, 4, 5]`, including
equal-value pairs `(2, 3)`, `(4, 5)`, and `(6, 7)`. It does not learn. For each
displayed pair it samples:

```text
P(left) = sigmoid(beta * (U_left - U_right))
```

`scipy.special.expit` evaluates this stably. `numpy.random.default_rng(seed)`
makes simulation reproducible. Only the selected human's displayed state
sequence is borrowed; their actions and rewards are ignored, including any
missing human actions. The estimator receives only two stimulus columns and
`synthetic_action`; it has no access to the agent's utility vector.

Recovery fits eight utilities by unregularized Bradley–Terry maximum likelihood,
fixing stimulus 1 to zero (a different reference can be selected in the Python
API). Beta is known and held fixed to the simulation value. Adding a constant
does not affect choice probabilities; if beta were also unknown, beta and
utility scale would be confounded. With beta fixed to 1, estimates can instead
be read as relative log-odds scores. Absolute psychological utility scale is
not established here. Beta zero can simulate random choices, but is rejected
for utility recovery because it carries no utility information.

A disconnected comparison graph cannot identify all eight stimuli using one
reference. Even a connected design can yield separated choices and no finite
maximum-likelihood solution. Both cases raise clear errors. There is no hidden
regularization, tie constraint, or prior forcing the answer toward the truth.
Repeated-seed runs retain and report failed fits; summary statistics describe
successful fits and show their counts.

Evaluation subtracts the true reference utility and reports:

- Reference-aligned true and estimated utilities, and absolute errors.
- Strict order agreement over the 25 unequal-utility pairs.
- Spearman rank correlation (with tied ranks) and Pearson correlation.
- Mean absolute error (MAE), meaningful here because beta is fixed and matched.
- Mean estimated gap for the three truly tied pairs, without forcing ties.

Finite samples normally break exact ties: perfect strict order agreement can
therefore coexist with Spearman correlation around 0.982 instead of 1. More
trials generally improve precision, but not monotonically for every seed.
`--repeats` generates fresh choices on repeated copies of the same state
sequence; these are extra simulated exposures, not more observed humans or
learning. Larger runs within a seed share the smaller run's choice prefix.

## Scope and files

The pair is an externally supplied context, and choices have no modeled effect
on future states. The new MDP has explicit empirical transitions, discounting,
and soft Bellman planning. Its continuation value is identical for both actions,
so it cancels from their comparison: the policy still reduces to the same
contextual-bandit/pairwise logistic form. This equivalence is the finding, not
evidence of recovered human learning mechanisms. Session endings and gaps in
retained trial numbers end observed fragments; the transition model is a
stationary approximation, not a reconstruction of the exact experimental schedule.

`src/reward_pairs/` contains `data.py`, `trajectories.py`, `simulation.py`,
`recovery.py`, and `evaluation.py`. `scripts/run_baseline.py` connects them;
`tests/` checks data handling, simulation, identifiability, and recovery.
`mdp.py` adds transitions, one-hot rewards, soft value iteration, and Bellman
derivatives; `irl.py` fits the soft policy's demonstration likelihood.
`scripts/run_irl_baseline.py` compares the two estimators. No new dependencies
or preprocessing rules were added.

Later work could study history-dependent choices, richer states, learning or
habit mechanisms, human model comparison, and formal sequential IRL with an
action-dependent environment. History remains outside v0; the separate
running-rate milestone implements and evaluates one synthetic history mechanism.
Human fitting, reward learning, and formal history-aware IRL remain future work.
