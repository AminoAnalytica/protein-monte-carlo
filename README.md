# LatentCatalyst

Below is a concise yet complete **`README.md`** you can drop in the repo root.   
It gives reviewers (and future users) everything they need: what it is, how to install, run, cite, and extend.

```markdown
# protein-mc

Minimal **Monte-Carlo optimiser** for exploring protein-sequence space with
ESM-2 embeddings – stripped of every heavyweight dependency so that anyone
(including a Nature reviewer on a laptop) can reproduce our results in
minutes.

[![tests](https://img.shields.io/badge/tests-passing-brightgreen)](./tests)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## Why this repo exists

The full lab pipeline couples Monte-Carlo with multiple property predictors
(TemStaPro, TemBERTure, solubility-BERT).  
For the *manuscript*, we publish a **dependency-light core** that demonstrates
the algorithmic idea – optimising

```
ΔE = E_mut – E_prev   where   E = 1 – ⟨embedding_ref · embedding_seq⟩
```

under a Metropolis criterion.

---

## Installation

```bash
# fresh Python ≥3.9 env recommended
pip install git+https://github.com/<your-org>/protein-mc.git

# or, development mode if you cloned the repo
pip install -e '.[dev]'
```

Pulls in **PyTorch CPU**, `transformers`, `pandas`, `tqdm`.  
GPU works out-of-the-box if CUDA 11+ is visible.

---

## Quick start (60 seconds)

```bash
# create a tiny config
cat > run.yml <<'YML'
model: {device: "cpu"}
sim:
  sequence_file: "data/demo_sequences.csv"
  num_steps: 50
  beta: 0.5
YML

# launch
protein-mc-run --config run.yml --output deltaE.csv
```

*You’ll see a progress bar; results land in `deltaE.csv`.*

---

## Python API

```python
from transformers import pipeline
from protein_mc import MonteCarlo, hamming_distance

pipe = pipeline("feature-extraction",
                model="facebook/esm2_t6_8M_UR50D", device="cpu")

mc = MonteCarlo(pipe, beta=0.5)

wt = "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPYK"
res = mc.run(wt, num_steps=100, protected_positions=[1, 10, 25])

print("Final sequence :", res.sequence)
print("Hamming distance:", hamming_distance(wt, res.sequence))
```

See `examples/minimal_run.ipynb` for a notebook version and
`examples/batch_screen.py` for parallel trajectory screening.

---

## Directory map

```
protein_mc/          ← installable package
  core/              ← MonteCarlo, EnergyCalculator, sequence ops
  io/                ← config + loaders
  models/            ← ESM-wrapper
  utils/             ← tqdm progress wrapper
  cli/               ← command-line entry point
tests/               ← pytest unit + integration tests
examples/            ← notebook + batch script
data/                ← three toy sequences for demos
pyproject.toml       ← build + dependency spec
```

---

## Running the test-suite

```bash
pytest -q         # three fast tests: 100 % pass expected
```

---

## Citing

If you use this code in academic work, please cite our companion paper:

```
@article{Your2025Nature,
  title   = {Latent Monte-Carlo Design of Functional Proteins},
  author  = {Your Name et al.},
  journal = {Nature},
  year    = 2025,
  doi     = {10.1038/s41586-XXX-XXXX-X}
}
```

---

## License

Released under the MIT License – free to use, modify, and distribute.
PRs and issues welcome!
```

Feel free to tweak the citation, repo URLs, shield badges, or section order to match your lab’s style.