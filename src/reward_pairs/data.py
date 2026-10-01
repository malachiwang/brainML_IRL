"""Read the authors' processed RData without adding exclusions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rdata

DEFAULT_DATA_PATH = Path("data/raw/02_comp_mod_RP_task_data_in.RData")
COLUMN_MAP = {
    "ID": "participant",
    "stim1": "left_stimulus",
    "stim2": "right_stimulus",
    "reward1": "left_reward",
    "reward2": "right_reward",
}
EXPECTED_COLUMNS = {
    "ID", "session", "phase", "trial", "stim1", "stim2", "action",
    "stim_chosen", "stim_unchosen", "stim_rawID_chosen", "rt1", "reward1",
    "reward2", "reward_chosen", "reward_unchosen", "diff_val", "corr_choice",
    "flag_therapy1", "flag_accuracy", "flag_include",
}
KEYS = ["participant", "session", "trial"]
STATE_COLUMNS = ["left_stimulus", "right_stimulus"]


def load_rdata(path: str | Path = DEFAULT_DATA_PATH) -> dict[str, Any]:
    """Keep every R object available; do not mutate the returned source frames."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Reward Pairs RData not found: {path}. Place the authors' processed "
            "02_comp_mod_RP_task_data_in.RData in data/raw/ or pass --data PATH. "
            "Raw data are not distributed with this repository."
        )
    return rdata.read_rda(path)


def select_training_object(
    objects: dict[str, Any], object_name: str | None = None,
) -> tuple[str, pd.DataFrame]:
    """Prefer the requested clean object; disclose the known analysis variant.

    Return the actual name alongside the untouched table so callers can report
    provenance. An explicit name never silently falls back to a different one.
    """
    candidates = [object_name] if object_name else [
        "data_RP_training_clean", "data_4_analysis_RP_training",
    ]
    for name in candidates:
        if name in objects:
            if not isinstance(objects[name], pd.DataFrame):
                raise ValueError(f"R object {name!r} is not a data frame.")
            return name, objects[name]
    raise ValueError(f"Training object not found. Requested {candidates}; available: {sorted(objects)}")


def _integers(series: pd.Series, name: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise")
        values = numeric.to_numpy(dtype=float, na_value=np.nan)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must contain numeric integer labels.") from exc
    if not np.all(np.isfinite(values) & (values == np.floor(values))):
        raise ValueError(f"{name} contains missing, nonfinite, or noninteger values.")
    return numeric.astype("int64")


def normalize_training(source: pd.DataFrame) -> pd.DataFrame:
    """Add readable aliases, preserving all rows and original named columns.

    R factor *labels*, never category codes, are used. No numeric action coding
    is guessed: this processed object must already use left/right labels.
    The caller retains the untouched source for original dtypes/attributes.
    """
    required = {"ID", "session", "trial", "stim1", "stim2", "action"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")
    collisions = set(COLUMN_MAP.values()) & set(source.columns)
    if collisions or not source.columns.is_unique:
        raise ValueError(f"Ambiguous/duplicate schema; normalized aliases: {sorted(collisions)}")
    result = source.copy(deep=True)
    for original, normalized in COLUMN_MAP.items():
        if original in source:
            result[normalized] = source[original]
    result["participant"] = result["participant"].astype("string")
    for column in ["session", "trial", *STATE_COLUMNS]:
        result[column] = _integers(result[column], column)
    result["action"] = result["action"].astype("string")
    check_training_rows(result)
    return result


def check_training_rows(frame: pd.DataFrame) -> None:
    """Reject ambiguity; never drop, impute, relabel, or renumber trials."""
    required = {*KEYS, *STATE_COLUMNS, "action"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing normalized columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Training data are empty.")
    if frame["participant"].isna().any() or frame["participant"].astype("string").str.strip().eq("").any():
        raise ValueError("Missing/empty participant identifiers.")
    for column in STATE_COLUMNS:
        if not frame[column].isin(range(1, 9)).all():
            raise ValueError(f"{column} must be in 1..8 with no missing stimuli.")
    if not frame["session"].isin(range(1, 6)).all():
        raise ValueError("Training sessions must be in 1..5.")
    trials = _integers(frame["trial"], "trial")
    if (trials < 1).any():
        raise ValueError("Trial indices must be positive integers.")
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate participant/session/trial keys; chronology is ambiguous.")
    if not (frame["action"].isna() | frame["action"].isin(["left", "right"])).all():
        raise ValueError("Actions must be 'left', 'right', or missing; numeric codes are not inferred.")
    if "phase" in frame and not frame["phase"].eq("training").all():
        raise ValueError("Unexpected phase in training object; inspect the source schema.")


@dataclass
class ValidationReport:
    row_count: int
    missing_actions: int
    input_sorted: bool
    participant_counts: pd.DataFrame
    session_counts: pd.DataFrame
    pair_counts: pd.DataFrame
    flag_counts: dict[str, dict]
    notes: list[str]


def validate_training(frame: pd.DataFrame) -> ValidationReport:
    check_training_rows(frame)
    participants = frame.groupby("participant", observed=True).agg(
        rows=("trial", "size"), sessions=("session", "nunique")
    )
    sessions = frame.groupby(["participant", "session"], observed=True).agg(
        rows=("trial", "size"), first_trial=("trial", "min"), last_trial=("trial", "max")
    )
    pairs = np.sort(frame[STATE_COLUMNS].to_numpy(dtype=int), axis=1)
    pair_counts = (
        pd.DataFrame(pairs, columns=["stimulus_a", "stimulus_b"])
        .value_counts(sort=False).rename("rows").reset_index()
    )
    ordered = frame.sort_values(KEYS, kind="stable")
    input_sorted = frame[KEYS].reset_index(drop=True).equals(ordered[KEYS].reset_index(drop=True))
    notes = [
        "No trial or participant exclusions applied; flags are retained, not filtered.",
        f"{int(sessions.rows.ne(160).sum())} participant-sessions differ from nominal 160 rows (allowed).",
    ]
    if not input_sorted:
        notes.append("Input is not globally chronological; trajectories explicitly sort session then trial.")
    if (frame["trial"] > 160).any():
        notes.append("Trial labels above 160 observed; retained without renumbering.")
    if (frame["left_stimulus"] == frame["right_stimulus"]).any():
        notes.append("Identical stimuli on both sides observed; retained and uninformative for utility differences.")
    missing = EXPECTED_COLUMNS - set(frame.columns)
    extra = set(frame.columns) - EXPECTED_COLUMNS - set(COLUMN_MAP.values())
    if missing:
        notes.append(f"Expected source columns absent: {sorted(missing)}")
    if extra:
        notes.append(f"Additional source columns retained: {sorted(extra)}")
    flags = {
        name: frame[name].value_counts(dropna=False).to_dict()
        for name in sorted(EXPECTED_COLUMNS) if name.startswith("flag_") and name in frame
    }
    return ValidationReport(len(frame), int(frame.action.isna().sum()), input_sorted,
                            participants, sessions, pair_counts, flags, notes)
