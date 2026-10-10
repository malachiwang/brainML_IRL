"""Scaffold-specific deterministic history MDP. Time counts retained trials."""

from dataclasses import dataclass
from collections import Counter
from itertools import combinations
from math import prod

import numpy as np

from .history import ChoiceHistory
from .simulation import stimulus_pairs


@dataclass(frozen=True)
class HistoryState:
    t: int
    chosen: tuple[int, ...]


class HistoryMDP:
    """State=(t,C); pair and N(t) are deterministic from the known schedule.

    Session breaks and gaps do not terminate/reset. Only the end of the retained
    scaffold is terminal. initial_history supports explicitly labeled probes.
    """

    def __init__(self, pairs, *, chronology=None, initial_history=None):
        self.pairs = stimulus_pairs(pairs)
        self.chronology = (tuple((1, i + 1) for i in range(len(self.pairs)))
                           if chronology is None else tuple(chronology))
        if len(self.chronology) != len(self.pairs) or len(set(self.chronology)) != len(self.pairs):
            raise ValueError("Provide one unique chronology key per displayed pair.")
        if tuple(sorted(self.chronology)) != self.chronology:
            raise ValueError("Scaffold chronology must be sorted.")
        h = ChoiceHistory() if initial_history is None else initial_history
        for x in (h.chosen, h.presented):
            if np.asarray(x).shape != (8,) or not np.isfinite(x).all() or (np.asarray(x) < 0).any() or not np.equal(x, np.floor(x)).all():
                raise ValueError("Initial history must contain eight nonnegative integer counts.")
        if (h.chosen > h.presented).any():
            raise ValueError("Chosen counts cannot exceed presented counts.")
        self.initial = HistoryState(0, tuple(map(int, h.chosen)))
        self.presented = np.zeros((len(self.pairs) + 1, 8), dtype=int)
        self.presented[0] = h.presented
        for t, pair in enumerate(self.pairs):
            self.presented[t + 1] = self.presented[t]
            self.presented[t + 1, np.unique(pair) - 1] += 1
        self.pairs.setflags(write=False)
        self.presented.setflags(write=False)

    @classmethod
    def from_trials(cls, trajectory):
        if not trajectory or len({t.participant for t in trajectory}) != 1:
            raise ValueError("Provide a single nonempty participant scaffold.")
        ordered = sorted(trajectory, key=lambda t: (t.session, t.trial))
        return cls([t.state for t in ordered], chronology=[(t.session, t.trial) for t in ordered])

    def validate(self, state):
        c = np.asarray(state.chosen)
        if not isinstance(state.t, (int, np.integer)) or not 0 <= state.t <= len(self.pairs):
            raise ValueError("State time outside scaffold.")
        if c.shape != (8,) or not np.isfinite(c).all() or not np.equal(c, np.floor(c)).all():
            raise ValueError("Chosen counts must be eight finite integers.")
        if (c < self.initial.chosen).any() or (c > self.presented[state.t]).any() or c.sum() != sum(self.initial.chosen) + state.t:
            raise ValueError("Chosen counts inconsistent with scaffold time/presentations.")

    def history(self, state):
        self.validate(state)
        n = self.presented[state.t]
        return np.divide(state.chosen, n, out=np.full(8, .5), where=n > 0) - .5

    def transition(self, state, action: str):
        self.validate(state)
        if state.t == len(self.pairs) or action not in ("left", "right"):
            raise ValueError("Need a nonterminal state and left/right action.")
        chosen = list(state.chosen)
        selected = self.pairs[state.t, int(action == "right")]
        chosen[selected - 1] += 1
        return HistoryState(state.t + 1, tuple(chosen))


def reachable_state_audit(mdp, *, cap=50_000):
    """Bounded prefix enumeration plus a rigorous preterminal-state lower bound.

    Fix choices on all non-tree edges. Vary the number selecting either end of
    each spanning-tree edge independently. The tree incidence columns are
    independent, so their product of (occurrences+1) yields distinct C vectors.
    This bounds *decision states* just before the final trial, not terminal V=0.
    """
    counts = Counter(tuple(sorted(map(int, p))) for p in mdp.pairs[:-1] if p[0] != p[1])
    bound, best = 0, None
    for edges in combinations(counts, 7):
        reached = {1}
        for _ in range(8):
            for a, b in edges:
                if a in reached or b in reached:
                    reached.update((a, b))
        if len(reached) == 8:
            candidate = prod(counts[e] + 1 for e in edges)
            if candidate > bound:
                bound, best = candidate, edges
    states = {mdp.initial.chosen}
    layers = [1]
    for pair in mdp.pairs:
        following = set()
        for c in states:
            for i in pair:
                updated = list(c)
                updated[i - 1] += 1
                following.add(tuple(updated))
                if len(following) > cap:
                    return dict(exact_prefix_layer_counts=layers, stopped_next_layer_at=len(following),
                                cap=cap, preterminal_lower_bound=bound, spanning_tree=best,
                                tree_occurrences=[counts[e] for e in best] if best else None,
                                value_storage_bytes_lower_bound=8 * bound)
        states = following
        layers.append(len(states))
    return dict(exact_prefix_layer_counts=layers, stopped_next_layer_at=None, cap=cap,
                preterminal_lower_bound=bound, spanning_tree=best,
                tree_occurrences=[counts[e] for e in best] if best else None,
                value_storage_bytes_lower_bound=8 * bound)
