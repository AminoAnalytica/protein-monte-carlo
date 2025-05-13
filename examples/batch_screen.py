"""
Run N short MC optimisations in parallel to build a candidate pool.
Usage:  python examples/batch_screen.py  --n 10
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer
from protein_mc.core.monte_carlo import MonteCarlo

def worker(seed: int, steps: int, wt: str, beta: float):
    # Fixed model loading approach to avoid meta tensor issue
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer, device="cpu")
    
    mc = MonteCarlo(pipe, beta=beta)
    res = mc.run(wt, num_steps=steps)
    best = min(res.delta_E_history)
    return seed, best, res.sequence

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5, help="number of trajectories")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--beta", type=float, default=0.5)
    args = p.parse_args()

    wild_type = "M" * 40
    with ThreadPoolExecutor() as ex:
        futs = [ex.submit(worker, i, args.steps, wild_type, args.beta)
                for i in range(args.n)]
        for fut in as_completed(futs):
            seed, best, seq = fut.result()
            print(f"[run {seed}]  best ΔE={best:+.4f}  seq={seq[:10]}…")

if __name__ == "__main__":
    main()
