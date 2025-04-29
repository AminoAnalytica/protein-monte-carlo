"""Uniform progress-reporting wrapper."""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Optional

from tqdm.auto import tqdm
from ..core.sequence_ops import hamming_distance as fdistance


class ProgressBar:
    """
    Simple tqdm wrapper that prints an inline summary every
    *status_frequency* steps.
    """

    def __init__(self, total: int, start_seq: str, status_frequency: int = 10):
        self._bar = tqdm(
            total=total,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        self.status_frequency = status_frequency
        self.start_time = self._bar.start_t
        self.start_seq = start_seq

    # ------------------------------------------------------------------ #
    def update(
        self,
        step: int,
        delta_E: float,
        accepted: bool,
        acceptance_prob: float,
        sequence: str,
    ):
        self._bar.update(1)

        if (step + 1) % self.status_frequency == 0 or step + 1 == self._bar.total:
            fd = self._bar.format_dict
            # Safely grab elapsed/remaining, defaulting to 0 if missing
            elapsed_secs = fd.get("elapsed", 0)
            remaining_secs = fd.get("remaining", 0)
            try:
                elapsed = self._bar.format_interval(elapsed_secs)
            except Exception:
                elapsed = str(elapsed_secs)
            try:
                remaining = self._bar.format_interval(remaining_secs)
            except Exception:
                remaining = str(remaining_secs)
            
            hamming_distance = fdistance(self.start_seq, sequence)
            
            self._bar.write(
                f"step {step+1:>4d} | ΔE={delta_E:+.5f} | "
                f"{'✓' if accepted else '✗'} p={acceptance_prob:.3f} | "
                f"hamming={hamming_distance} | "
                f"seq={sequence[:20]}… | "
                f"{elapsed} elapsed, {remaining} left"
            )

    # ------------------------------------------------------------------ #
    def close(self):
        self._bar.close()
