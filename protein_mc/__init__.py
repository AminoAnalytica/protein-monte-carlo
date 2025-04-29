"""
protein_mc – minimal Monte-Carlo optimisation for protein sequences.

The top-level namespace intentionally exports only the handful of
objects a downstream script or notebook ever needs.
"""

# algorithmic core
from .core.monte_carlo import MonteCarlo, MonteCarloState
from .core.energy import EnergyCalculator, EnergyResult
from .core.sequence_ops import (
    mutate_sequence,
    hamming_distance,
    get_embeddings,          # handy shortcut
)

# I/O helpers
from .io.config import Config, load_config
from .io.loaders import get_sequence_from_file

__all__ = [
    # core
    "MonteCarlo",
    "MonteCarloState",
    "EnergyCalculator",
    "EnergyResult",
    "mutate_sequence",
    "hamming_distance",
    "get_embeddings",
    # I/O
    "Config",
    "load_config",
    "get_sequence_from_file",
]
