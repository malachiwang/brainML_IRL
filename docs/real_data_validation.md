# Local-data validation and baseline run

Verified on 2026-10-01 using the existing local file. This report contains
aggregate summaries and synthetic estimates, not raw trial data.

```text
File: data/raw/02_comp_mod_RP_task_data_in.RData
SHA-256: 037dde9d968dd8d0705891a31315930d70f475fd33a99c4743712eed89e9c5e1
Python 3.12.4; numpy 2.5.3; pandas 3.0.6; scipy 1.18.1;
rdata 1.1.0; pytest 9.1.1
```

## Actual objects and provenance

| Actual object | Rows | Columns |
| --- | ---: | ---: |
| `data_4_analysis_RP_pretest` | 10,759 | 21 |
| `data_4_analysis_RP_training` | 164,087 | 20 |
| `data_4_analysis_RP_test` | 28,320 | 31 |
| `ratings_RP2` | 231 | 25 |
| `grouped_data_RP_pretest2_clean` | 1,704 | 7 |

The expected `data_RP_training_clean`, `data_RP_test_clean`, and
`data_RP_pretest_clean` objects are absent. The loader explicitly selects the
training analysis variant when the expected name is absent; it does not rename
the R objects or reconstruct excluded rows. Only training is used in the demo.
Its 20 source columns match the requested training columns exactly.

The existing `data/reference/01_analyses_RP_task.R`, lines 39–43, documents
`stim1` as left, `stim2` as right, and processed action factor labels as
`right`/`left`. Lines 62–63 create the RT-cleaned table, and lines 448–457
distinguish inclusion flags from the subsequent analysis tables. The local
file contains those analysis names, even though that script's save statement
lists clean names. This is a file-version/provenance difference; this project
uses the objects actually supplied and makes no additional exclusions.

`rdata` emits a warning that it falls back from R `data.table` to `data.frame`
conversion. This is expected for these objects: the result is a pandas
DataFrame, and the schema and factor labels were checked. The warning is not
suppressed. Original R-specific class attributes are not used by this baseline.

The local training reward columns use levels 1–5 and the stimulus mapping
`[1, 2, 2, 3, 3, 4, 4, 5]`. The [paper's task description](https://pmc.ncbi.nlm.nih.gov/articles/PMC7615722/)
uses points 1, 3, 5, 7, 9. We retain the supplied values; no rescaling of the
loaded rewards is performed. Fake Human uses the requested utility levels
independently, and rewards are never estimator inputs.

## Validation statistics

- 164,087 rows, 213 participants, 1,065 participant-sessions.
- Every participant has all five training sessions.
- Participant rows: minimum 667, median 775, mean 770.3615, maximum 797.
- Session rows: minimum 121, median 156, mean 154.0723, maximum 160.
- 61 participant-sessions have 160 rows; 1,004 have fewer. No rows were added.
- Stimuli cover 1–8; sessions cover 1–5; trial labels span 1–160.
- Input is already globally sorted by participant, session, trial.
- Zero duplicate chronology keys, zero missing actions, and zero identical-stimulus states.
- Actions: 83,200 left and 80,887 right. All chosen-stimulus labels agree with action and displayed side.
- All three inclusion flags equal 1 on all 164,087 supplied rows.
- Eight unordered pairs, represented in both orientations (16 ordered states).

| Session | Rows |
| --- | ---: |
| 1 | 31,815 |
| 2 | 32,739 |
| 3 | 33,052 |
| 4 | 33,160 |
| 5 | 33,321 |

| Unordered pair | Rows |
| --- | ---: |
| (1, 2) | 10,094 |
| (1, 3) | 30,739 |
| (2, 4) | 30,644 |
| (3, 5) | 9,858 |
| (4, 6) | 10,071 |
| (5, 7) | 30,787 |
| (6, 8) | 31,449 |
| (7, 8) | 10,445 |

These comparisons form a connected graph. The equal-utility pairs are not
directly compared during training, so their relative values are inferred
indirectly through the connected comparisons. No equality constraint is used.

## Synthetic results using the real displayed sequence

Command:

```sh
python scripts/run_baseline.py --repeats 1 5 20
```

Default participant `E11T9A`: 783 retained trials over five sessions. This is
lexicographic selection, with no selection based on human behavior. Beta is 1;
stimulus 1 is fixed to zero. For seed 0 and one trajectory:

| Stimulus | True, reference-aligned | Recovered |
| --- | ---: | ---: |
| 1 | 0 | 0.0000 |
| 2 | 1 | 0.8160 |
| 3 | 1 | 0.8639 |
| 4 | 2 | 1.7615 |
| 5 | 2 | 1.7219 |
| 6 | 3 | 2.4778 |
| 7 | 3 | 2.5796 |
| 8 | 4 | 3.4663 |

MAE 0.2891; strict order agreement 1.0000; Spearman 0.9820; Pearson 0.9992;
mean absolute estimated gap between truly tied stimuli 0.0631.

Across seeds 0–4 (standard deviations across seeds, not confidence intervals):

| Sequence repetitions | Simulated trials | MAE mean ± SD | Strict order agreement, mean | True-tie gap, mean |
| --- | ---: | ---: | ---: | ---: |
| 1 | 783 | 0.2234 ± 0.1278 | 1.0000 | 0.2980 |
| 5 | 3,915 | 0.1096 ± 0.0525 | 1.0000 | 0.1441 |
| 20 | 15,660 | 0.0497 ± 0.0224 | 1.0000 | 0.0699 |

All 15 fits converged with finite identified estimates. More data improved
average precision in this example; no monotonic-per-seed claim is made. Perfect
unequal-value ordering does not establish recovery of exact ties. These results
test a known stationary simulator under a matching model, not human preferences,
habit learning, or sequential IRL.
