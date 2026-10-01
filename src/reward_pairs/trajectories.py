"""Chronological observations; the v0 state contains only the displayed pair."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .data import check_training_rows

Action = Literal["left", "right"]


@dataclass(frozen=True)
class Trial:
    participant: str
    session: int
    trial: int
    left_stimulus: int
    right_stimulus: int
    action: Action | None

    @property
    def state(self) -> tuple[int, int]:
        return self.left_stimulus, self.right_stimulus


def build_trajectory(training_df: pd.DataFrame, participant_id: str) -> list[Trial]:
    selected = training_df.loc[training_df["participant"] == str(participant_id)]
    if selected.empty:
        raise ValueError(f"Participant {participant_id!r} not found.")
    check_training_rows(selected)
    return [
        Trial(str(row.participant), int(row.session), int(row.trial),
              int(row.left_stimulus), int(row.right_stimulus),
              None if pd.isna(row.action) else row.action)
        for row in selected.sort_values(["session", "trial"], kind="stable").itertuples()
    ]
