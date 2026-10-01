import pandas as pd
import pytest

from reward_pairs.data import (
    load_rdata, normalize_training, select_training_object, validate_training,
)
from reward_pairs.trajectories import build_trajectory


@pytest.fixture
def source():
    return pd.DataFrame({
        "ID": pd.Categorical(["p", "p", "p", "q"]),
        "session": pd.Categorical(["2", "1", "1", "1"]),
        "trial": [1.0, 10.0, 2.0, 1.0],
        "phase": ["training"] * 4,
        "stim1": pd.Categorical(["4", "6", "4", "1"], categories=["6", "1", "4"]),
        "stim2": pd.Categorical(["6", "4", "6", "3"]),
        "action": pd.Categorical(["left", "right", None, "left"]),
        "reward1": [3.0, 4.0, 3.0, 1.0], "reward2": [4.0, 3.0, 4.0, 2.0],
        "flag_include": [0, 1, 1, 1], "rt1": [0.1, 0.2, None, 0.3],
    })


def test_normalization_preserves_labels_rows_and_source(source):
    original = source.copy(deep=True)
    frame = normalize_training(source)
    assert frame.left_stimulus.tolist() == [4, 6, 4, 1]  # labels, not factor codes
    assert frame.right_stimulus.tolist() == [6, 4, 6, 3]
    assert frame.participant.tolist() == ["p", "p", "p", "q"]
    assert frame.left_reward.tolist() == source.reward1.tolist()
    assert frame.right_reward.tolist() == source.reward2.tolist()
    assert pd.isna(frame.action.iloc[2])
    assert frame.flag_include.tolist() == [0, 1, 1, 1]  # no new exclusion
    pd.testing.assert_series_equal(frame.stim1, source.stim1)
    pd.testing.assert_frame_equal(source, original)


def test_trajectory_numeric_order_and_positional_states(source):
    trajectory = build_trajectory(normalize_training(source), "p")
    assert [(t.session, t.trial) for t in trajectory] == [(1, 2), (1, 10), (2, 1)]
    assert [t.state for t in trajectory] == [(4, 6), (6, 4), (4, 6)]
    assert [t.action for t in trajectory] == [None, "right", "left"]
    assert all(t.participant == "p" for t in trajectory)


def test_validation_counts_missing_and_incomplete_sessions(source):
    frame = normalize_training(source)
    report = validate_training(frame)
    assert report.row_count == 4 and report.missing_actions == 1
    assert not report.input_sorted
    assert report.participant_counts.loc["p"].to_dict() == {"rows": 3, "sessions": 2}
    pairs = report.pair_counts.set_index(["stimulus_a", "stimulus_b"])["rows"].to_dict()
    assert pairs == {(1, 3): 1, (4, 6): 3}


@pytest.mark.parametrize("column,value,match", [
    ("stim1", 9, "1..8"), ("session", 6, "1..5"),
    ("action", 1, "Actions"), ("trial", 1.5, "noninteger"),
    ("trial", None, "missing"), ("ID", None, "participant"),
    ("phase", "test", "phase"),
])
def test_unexpected_data_rejected_without_guessing(source, column, value, match):
    source[column] = source[column].astype(object)
    source.loc[0, column] = value
    with pytest.raises(ValueError, match=match):
        normalize_training(source)


def test_ambiguous_chronology_rejected(source):
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_training(pd.concat([source, source.iloc[[0]]], ignore_index=True))


def test_unknown_participant_and_missing_schema(source):
    with pytest.raises(ValueError, match="not found"):
        build_trajectory(normalize_training(source), "absent")
    with pytest.raises(ValueError, match="Missing training"):
        normalize_training(source.drop(columns="stim1"))


def test_absent_rdata_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="--data PATH"):
        load_rdata(tmp_path / "missing.RData")


def test_training_object_selection_is_explicit(source):
    objects = {"data_4_analysis_RP_training": source}
    name, selected = select_training_object(objects)
    assert name == "data_4_analysis_RP_training" and selected is source
    with pytest.raises(ValueError, match="available"):
        select_training_object(objects, "data_RP_training_clean")
    objects["data_RP_training_clean"] = source.copy()
    assert select_training_object(objects)[0] == "data_RP_training_clean"
