"""Uniform progress-reporting wrapper."""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Optional

from tqdm.auto import tqdm


class ProgressBar:
    """
    Simple tqdm wrapper that prints an inline summary every
    *status_frequency* steps.
    """

    def __init__(self, total: int, status_frequency: int = 10):
        self._bar = tqdm(
            total=total,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        self.status_frequency = status_frequency
        self.start_time = self._bar.start_t

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
            elapsed = self._bar.format_interval(self._bar.format_dict["elapsed"])
            remaining = self._bar.format_interval(
                self._bar.format_dict["remaining"]
            )
            self._bar.write(
                f"step {step+1:>4d} | ΔE={delta_E:+.5f} | "
                f"{'✓' if accepted else '✗'} p={acceptance_prob:.3f} | "
                f"{elapsed} elapsed, {remaining} left"
            )

    # ------------------------------------------------------------------ #
    def close(self):
        self._bar.close()
