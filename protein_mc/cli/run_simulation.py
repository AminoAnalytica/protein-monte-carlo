"""
Command-line entry-point:

    python -m protein_mc.cli.run_simulation --config my.yml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from transformers import logging as hf_logging
import warnings

warnings.filterwarnings("ignore", message="Failed to load image Python extension*")

hf_logging.set_verbosity_error()

from ..core.monte_carlo import MonteCarlo
from ..io.config import load_config
from ..io.loaders import get_sequence_from_file
from ..models.esm_wrapper import load_esm_pipeline
from ..utils.progress import ProgressBar
from ..core.sequence_ops import hamming_distance


# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Protein Monte-Carlo optimiser")
    parser.add_argument("--config", required=True, help="YAML or JSON config file")
    parser.add_argument(
        "--output", "-o", default=None, help="Optional CSV output path"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipe = load_esm_pipeline(device=cfg.model.device)

    mc = MonteCarlo(pipe, beta=cfg.sim.beta)

    seq = get_sequence_from_file(cfg.sim.sequence_file, cfg.sim.sequence_index)

    # Create progress bar with status frequency from config or default to 10
    status_frequency = getattr(cfg.sim, 'status_frequency', 10)
    bar = ProgressBar(cfg.sim.num_steps, seq, status_frequency)

    result = mc.run(
        seq,
        num_steps=cfg.sim.num_steps,
        protected_positions=cfg.sim.protected_positions,
        progress_callback=bar.update,  # Pass the progress callback
    )

    bar.close()

    # save results if requested
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        # Create DataFrame with consistent lengths
        df = pd.DataFrame({
            "step": range(len(result.sequence_history)),  # Start from 0 to match sequence history
            "delta_E": [0.0] + result.delta_E_history,  # Add initial energy difference of 0
            "acceptance_prob": [1.0] + result.acceptance_probability_history,  # Add initial acceptance prob of 1
        })

        # Add accepted mutations as a boolean column
        accepted = [True] + [False] * len(result.delta_E_history)  # Start with True for initial sequence
        for step, _ in result.accepted_mutations:
            if step < len(accepted):
                accepted[step + 1] = True  # +1 because we added initial sequence
        df["accepted"] = accepted

        # Add sequence information using sequence history
        df["sequence"] = result.sequence_history
        df["hamming"] = [hamming_distance(result.initial_sequence, seq) for seq in result.sequence_history]

        df.to_csv(out, index=False)
        print(f"Saved ΔE series to {out.absolute()}")


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    main()
