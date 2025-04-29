"""Metropolis Monte-Carlo optimiser operating on ESM2 embeddings."""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Callable, List, Optional

import torch

from .energy import EnergyCalculator, EnergyResult
from .sequence_ops import (
    get_embeddings,
    hamming_distance,
    mutate_sequence,
)


# ---------------------------------------------------------------------- #
#  State dataclass                                                       #
# ---------------------------------------------------------------------- #
@dataclasses.dataclass
class MonteCarloState:
    initial_sequence: str
    sequence: str
    energy: float
    delta_E_history: List[float]
    mutation_position_history: List[int]
    mutation_details_history: List[str]
    accepted_mutations: List[bool]
    acceptance_probability_history: List[float]

    @property
    def steps(self) -> int:
        return len(self.delta_E_history)


# ---------------------------------------------------------------------- #
#  Main class                                                            #
# ---------------------------------------------------------------------- #
class MonteCarlo:
    def __init__(
        self,
        embedding_model,  # HF pipeline
        beta: float,
        strategy: str = "embedding_only",
        *,
        energy_calculator: Optional[EnergyCalculator] = None,
    ):
        if strategy != "embedding_only":
            raise ValueError("Publication build only supports 'embedding_only'.")

        self.model = embedding_model
        self.beta = float(beta)
        self.energy_calculator = energy_calculator or EnergyCalculator()

    # ------------------------------------------------------------------ #
    def run(
        self,
        initial_sequence: str,
        num_steps: int,
        protected_positions: Optional[List[int]] = None,
        callback: Optional[
            Callable[[int, MonteCarloState, EnergyResult], None]
        ] = None,
    ) -> MonteCarloState:
        ref_emb = get_embeddings(initial_sequence, self.model)

        # starting energy is 0 by definition (Emut=Eprev)
        state = MonteCarloState(
            initial_sequence=initial_sequence,
            sequence=initial_sequence,
            energy=0.0,
            delta_E_history=[],
            mutation_position_history=[],
            mutation_details_history=[],
            accepted_mutations=[],
            acceptance_probability_history=[],
        )

        rng = torch.Generator().manual_seed(int(time.time()))

        for step in range(num_steps):
            # --- propose mutation ------------------------------------------------
            mut_seq, pos, details = mutate_sequence(
                state.sequence,
                protected_positions,
                return_details=True,
            )

            energy_res = self._compute_energy(
                state.sequence,
                mut_seq,
                ref_emb,
                protected_positions,
            )

            accept = self._metropolis_accept(energy_res.delta_E, rng)
            # --- record ----------------------------------------------------------
            state.delta_E_history.append(energy_res.delta_E)
            state.mutation_position_history.append(pos)
            state.mutation_details_history.append(details)
            state.accepted_mutations.append(accept)
            state.acceptance_probability_history.append(
                math.exp(-self.beta * max(energy_res.delta_E, 0.0))
            )

            # --- update ----------------------------------------------------------
            if accept:
                state.sequence = mut_seq
                state.energy += energy_res.delta_E

            # optional callback for progress UI
            if callback is not None:
                callback(step, state, energy_res)

        return state

    # ------------------------------------------------------------------ #
    #  internals                                                          #
    # ------------------------------------------------------------------ #
    def _compute_energy(
        self,
        prev_seq: str,
        mut_seq: str,
        ref_emb: torch.Tensor,
        protected_positions: Optional[List[int]],
    ) -> EnergyResult:
        prev_emb = get_embeddings(prev_seq, self.model)
        mut_emb = get_embeddings(mut_seq, self.model)

        return self.energy_calculator.calculate(
            prev_emb=prev_emb,
            mut_emb=mut_emb,
            ref_emb=ref_emb,
            protected_positions=protected_positions,
        )

    def _metropolis_accept(self, delta_E: float, rng: torch.Generator) -> bool:
        if delta_E <= 0:
            return True
        p = math.exp(-self.beta * delta_E)
        return torch.rand(1, generator=rng).item() < p
