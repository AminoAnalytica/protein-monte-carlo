import torch
from protein_mc.core.energy import EnergyCalculator

def test_delta_e_zero_for_identical():
    emb = torch.randn(10, 128)          # fake embeddings
    calc = EnergyCalculator()
    res = calc.calculate(emb, emb, emb)
    assert abs(res.delta_E) < 1e-9, "ΔE should be zero when mut==prev"

def test_protected_positions_effect():
    torch.manual_seed(0)
    ref = torch.randn(5, 64)
    prev = ref.clone()
    mut = ref.clone()
    mut[2] += 0.5                      # change only position 3 (1-based)
    calc = EnergyCalculator()
    res_no_mask = calc.calculate(prev, mut, ref)
    res_masked = calc.calculate(prev, mut, ref, protected_positions=[3])
    assert abs(res_masked.delta_E) < abs(res_no_mask.delta_E), \
        "Masking the mutated site should reduce |ΔE|"
