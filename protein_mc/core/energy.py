"""Pure-python energy function for the publication build."""

from dataclasses import dataclass
from typing import Optional, List

import torch


@dataclass
class EnergyResult:
    delta_E: float
    E_mut: Optional[float] = None
    structural_component: Optional[float] = None


class EnergyCalculator:
    """
    ΔE = E_mut − E_prev
    where
        E = 1 − mean(dot(emb_ref, emb_seq))
    """

    def __init__(self):
        pass

    # ---------- public --------------------------------------------------

    def calculate(
        self,
        prev_emb: torch.Tensor,
        mut_emb: torch.Tensor,
        ref_emb: torch.Tensor,
        protected_positions: Optional[List[int]] = None,
    ) -> EnergyResult:
        return self._calc_embedding_energy(
            prev_emb, mut_emb, ref_emb, protected_positions
        )

    # ---------- internals -----------------------------------------------

    def _calc_embedding_energy(
        self,
        prev_emb: torch.Tensor,
        mut_emb: torch.Tensor,
        ref_emb: torch.Tensor,
        protected_positions: Optional[List[int]],
    ) -> EnergyResult:
        # normalize all rows
        prev_norm = self._normalize_rows(prev_emb)
        mut_norm = self._normalize_rows(mut_emb)
        ref_norm = self._normalize_rows(ref_emb)

        # if any positions are protected, use ONLY those positions
        if protected_positions:
            # convert 1-based to 0-based indices
            indices = [pos - 1 for pos in protected_positions]
            prev_norm = prev_norm[indices]
            mut_norm = mut_norm[indices]
            ref_norm = ref_norm[indices]

        # compute energies
        dot_prev = (prev_norm * ref_norm).sum(dim=1).mean().item()
        dot_mut = (mut_norm * ref_norm).sum(dim=1).mean().item()
        E_prev = 1.0 - dot_prev
        E_mut = 1.0 - dot_mut

        # Calculate raw energy difference without alpha scaling
        delta_E = E_mut - E_prev

        return EnergyResult(
            delta_E=delta_E,
            E_mut=E_mut,
            structural_component=delta_E
        )

    @staticmethod
    def _normalize_rows(emb: torch.Tensor) -> torch.Tensor:
        """
        Row-wise L2-normalization of a [N x D] embedding tensor.
        Asserts that no row has zero norm.
        """
        norms = torch.norm(emb, p=2, dim=1, keepdim=True)
        assert torch.all(norms != 0), "Zero norm encountered in embedding row(s). Cannot normalize."
        return emb / norms
