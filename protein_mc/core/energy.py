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
    ΔE = α · (E_mut − E_prev)
    where
        E = 1 − mean(dot(emb_ref, emb_seq))
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

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

        # if any positions are protected, EXCLUDE them
        if protected_positions:
            # convert 1-based to 0-based indices to drop
            drop = {p - 1 for p in protected_positions}
            keep_idx = [i for i in range(prev_norm.size(0)) if i not in drop]
            prev_norm = prev_norm[keep_idx]
            mut_norm = mut_norm[keep_idx]
            ref_norm = ref_norm[keep_idx]

        # compute energies
        dot_prev = (prev_norm * ref_norm).sum(dim=1).mean().item()
        dot_mut = (mut_norm * ref_norm).sum(dim=1).mean().item()
        E_prev = 1.0 - dot_prev
        E_mut = 1.0 - dot_mut

        delta_E = self.alpha * (E_mut - E_prev)
        return EnergyResult(
            delta_E=delta_E,
            E_mut=E_mut,
            structural_component=delta_E,
        )

    @staticmethod
    def _normalize_rows(emb: torch.Tensor) -> torch.Tensor:
        return emb / (emb.norm(p=2, dim=1, keepdim=True) + 1e-8)
