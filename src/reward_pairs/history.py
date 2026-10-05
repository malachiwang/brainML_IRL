"""Causal running choice rates, reconstructed from synthetic actions only."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .simulation import stimulus_pairs

CHOICE_COLUMNS = ["rollout", "session", "trial", "left_stimulus", "right_stimulus", "synthetic_action"]


@dataclass
class ChoiceHistory:
    presented: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=int))
    chosen: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=int))

    @property
    def rates(self) -> np.ndarray:
        return np.divide(self.chosen, self.presented, out=np.full(8, 0.5), where=self.presented > 0)

    @property
    def centered(self) -> np.ndarray:
        return self.rates - 0.5

    def update(self, left: int, right: int, action: str) -> None:
        """Call AFTER sampling/scoring. Count each presented stimulus once/trial."""
        if left not in range(1, 9) or right not in range(1, 9) or action not in ("left", "right"):
            raise ValueError("History update needs stimuli 1..8 and a left/right action.")
        for stimulus in {left, right}:
            self.presented[stimulus - 1] += 1
        self.chosen[(left if action == "left" else right) - 1] += 1


def causal_history(choices: pd.DataFrame, *, reset_history_each_session: bool = False) -> pd.DataFrame:
    """Rebuild pre-choice histories, ignoring any supplied history/reward fields.

    Each rollout begins at zero. Gaps do not reset or update counts. To score
    later sessions, pass the preceding sessions too, then select scoring rows.
    Returned rows have a fresh index and are sorted by rollout/session/trial.
    """
    if missing := set(CHOICE_COLUMNS) - set(choices.columns):
        raise ValueError(f"Missing synthetic chronology columns: {sorted(missing)}")
    frame = choices[CHOICE_COLUMNS].copy()
    if frame.empty or frame.isna().any().any():
        raise ValueError("Synthetic histories require nonempty data with no missing fields/actions.")
    stimulus_pairs(frame[["left_stimulus", "right_stimulus"]].to_numpy())
    for column in ("session", "trial"):
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all() or (values < 1).any():
            raise ValueError(f"{column} must contain positive integer chronology labels.")
        frame[column] = values.astype(int)
    if not frame.session.isin(range(1, 6)).all():
        raise ValueError("Training sessions must be in 1..5.")
    if not frame.synthetic_action.isin(["left", "right"]).all():
        raise ValueError("Synthetic actions must be left/right.")
    if frame.duplicated(["rollout", "session", "trial"]).any():
        raise ValueError("Duplicate rollout/session/trial keys; cannot reconstruct history.")
    frame = frame.sort_values(["rollout", "session", "trial"], kind="stable").reset_index(drop=True)
    features = np.zeros((len(frame), 2))
    previous_rollout = previous_session = None
    history = ChoiceHistory()
    for i, row in enumerate(frame.itertuples(index=False)):
        if row.rollout != previous_rollout or (reset_history_each_session and row.session != previous_session):
            history = ChoiceHistory()
        left, right = int(row.left_stimulus), int(row.right_stimulus)
        features[i] = history.centered[[left - 1, right - 1]]
        history.update(left, right, row.synthetic_action)
        previous_rollout, previous_session = row.rollout, row.session
    frame["left_history"] = features[:, 0]
    frame["right_history"] = features[:, 1]
    return frame
