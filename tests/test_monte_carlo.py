from transformers import pipeline
from protein_mc.core.monte_carlo import MonteCarlo
from protein_mc.core.sequence_ops import hamming_distance

PIPE = pipeline(
    "feature-extraction",
    model="facebook/esm2_t6_8M_UR50D",
    device="cpu",
)

def test_mc_runs_and_mutates():
    wt = "M" * 30
    mc = MonteCarlo(PIPE, beta=0.1)
    res = mc.run(wt, num_steps=5)
    assert len(res.delta_E_history) == 5
    assert hamming_distance(wt, res.sequence) >= 1, "At least one mutation expected"
