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

    bar = ProgressBar(cfg.sim.num_steps, seq, cfg.sim.status_frequency)

    def cb(step, state, energy_res):
        bar.update(
            step,
            delta_E=energy_res.delta_E,
            accepted=state.accepted_mutations[-1],
            acceptance_prob=state.acceptance_probability_history[-1],
            sequence=state.sequence,
        )

    result = mc.run(
        seq,
        num_steps=cfg.sim.num_steps,
        protected_positions=cfg.sim.protected_positions,
        callback=cb,
    )
    bar.close()

    # save results if requested
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        df = pd.DataFrame(
            {
                "step": range(1, len(result.delta_E_history) + 1),
                "delta_E": result.delta_E_history,
                "accepted": result.accepted_mutations,
                "sequence": result.mutation_details_history,  # details already has "X->Y" info
            }
        )

        # append live sequence each step
        seq_col = []
        seq = result.initial_sequence
        for acc, det in zip(result.accepted_mutations, result.mutation_details_history):
            if acc:
                pos = result.mutation_position_history[len(seq_col)] - 1
                aa_to = det[-1]  # "...->Y"
                seq = seq[:pos] + aa_to + seq[pos + 1 :]
            seq_col.append(seq)
        df["current_sequence"] = seq_col
        df["hamming"] = [
            hamming_distance(result.initial_sequence, s) for s in seq_col
        ]

        df.to_csv(out, index=False)
        print(f"Saved ΔE series to {out.absolute()}")


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    main()
