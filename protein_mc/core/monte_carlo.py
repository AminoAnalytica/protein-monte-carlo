"""Pure-python Monte Carlo implementation for the publication build."""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Callable, List, Optional

import torch
import random

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
    sequence: str
    energy: float
    delta_E_history: List[float]
    acceptance_probability_history: List[float]
    accepted_mutations: List[tuple]
    initial_sequence: str  # Add to track original sequence
    sequence_history: List[str] = dataclasses.field(default_factory=list)  # Track sequence changes

    @property
    def steps(self) -> int:
        return len(self.delta_E_history)


# ---------------------------------------------------------------------- #
#  Main class                                                            #
# ---------------------------------------------------------------------- #
class MonteCarlo:
    def __init__(
        self,
        embedding_pipeline,
        beta: float = 1000.0,  # Changed from 1.0 to 1000.0 to match original
        random_seed: Optional[int] = None,
    ):
        self.pipeline = embedding_pipeline
        self.beta = float(beta)
        self.energy_calculator = EnergyCalculator()  # Remove alpha parameter
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)

    def run(
        self,
        sequence: str,
        num_steps: int,
        protected_positions: Optional[List[int]] = None,
        progress_callback = None,
    ) -> MonteCarloState:
        state = MonteCarloState(
            sequence=sequence,
            energy=0.0,  # Will be set after first energy calculation
            delta_E_history=[],
            acceptance_probability_history=[],
            accepted_mutations=[],
            initial_sequence=sequence,  # Store initial sequence
            sequence_history=[sequence],  # Initialize with initial sequence
        )

        # Get initial embeddings
        ref_embedding = self._get_embedding(sequence)
        prev_embedding = ref_embedding.clone()

        # Calculate initial energy
        energy_res = self.energy_calculator.calculate(
            prev_embedding, prev_embedding, ref_embedding, protected_positions
        )
        state.energy = energy_res.E_mut  # Store absolute energy

        # Report initial state
        if progress_callback:
            progress_callback(
                step=0,
                delta_E=0.0,
                accepted=True,
                acceptance_prob=1.0,
                sequence=sequence,
            )

        for step in range(num_steps):
            # Propose mutation (now aware of protected positions)
            mut_seq, mut_embedding = self._propose_mutation(
                state.sequence,
                protected_positions,
            )

            # Calculate energy difference
            energy_res = self.energy_calculator.calculate(
                prev_embedding, mut_embedding, ref_embedding, protected_positions
            )

            # Metropolis acceptance
            p = math.exp(-self.beta * energy_res.delta_E)
            if p > 1.0:
                p = 1.0
            accept = random.random() < p

            # Update state if accepted
            if accept:
                state.sequence = mut_seq
                state.energy = energy_res.E_mut  # Store absolute energy
                state.accepted_mutations.append((step, mut_seq))
                state.sequence_history.append(mut_seq)  # Track sequence changes
            else:
                state.sequence_history.append(state.sequence)  # Track unchanged sequence

            # Record history
            state.delta_E_history.append(energy_res.delta_E)
            state.acceptance_probability_history.append(p)

            # Update previous embedding
            prev_embedding = mut_embedding if accept else prev_embedding

            # Report progress
            if progress_callback:
                progress_callback(
                    step=step + 1,
                    delta_E=energy_res.delta_E,
                    accepted=accept,
                    acceptance_prob=p,
                    sequence=state.sequence,
                )

        return state

    def _get_embedding(self, sequence: str) -> torch.Tensor:
        """Get per-token embeddings for a sequence."""
        with torch.no_grad():
            # Get embeddings from pipeline and keep all internal residues
            embeddings = self.pipeline(sequence)[0]  # Shape: [L+2, D]
            # Convert to tensor and drop CLS/EOS tokens
            emb_tensor = torch.tensor(embeddings)[1:-1]  # Shape: [L, D]
            return emb_tensor

    # NEW: honour `protected_positions`
    def _propose_mutation(
        self,
        sequence: str,
        protected_positions: Optional[List[int]] = None,
    ) -> tuple[str, torch.Tensor]:
        """
        Propose a single-point mutation that **skips** any
        1-based positions listed in *protected_positions*.
        """
        mut_seq, _, _ = mutate_sequence(
            sequence,
            protected_positions,
            return_details=True,   # just to reuse the checked logic
        )
        mut_emb = self._get_embedding(mut_seq)
        return mut_seq, mut_emb

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
        prev_emb = get_embeddings(prev_seq, self.pipeline)
        mut_emb = get_embeddings(mut_seq, self.pipeline)

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
