"""YAML-friendly configuration objects."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


# ------------------------------------------------------------------ #
#  Dataclasses                                                        #
# ------------------------------------------------------------------ #
@dataclass
class ModelConfig:
    device: str = field(default="cpu")           # "cpu" or CUDA id (e.g. "cuda:0")


@dataclass
class SimulationConfig:
    sequence_file: str = "../data/demo_sequences.csv"
    sequence_index: int = 0
    beta: float = 0.5
    num_steps: int = 100
    strategy: str = "embedding_only"             # fixed in the publication build
    protected_positions: Optional[List[int]] = None
    status_frequency: int = 10                   # hint for progress bars


@dataclass
class PathConfig:
    results_dir: str = "./results"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # --------------- convenience ------------------------------------ #
    def dump(self, path: str | Path) -> None:
        """Write the config in YAML (if installed) else JSON."""
        path = Path(path)
        data = asdict(self)
        if path.suffix in {".yml", ".yaml"}:
            if yaml is None:
                raise RuntimeError("PyYAML not installed – cannot dump YAML.")
            path.write_text(yaml.safe_dump(data, sort_keys=False))
        else:
            path.write_text(json.dumps(data, indent=2))

    # --------------- factory ---------------------------------------- #
    @classmethod
    def from_dict(cls, dct: dict) -> "Config":
        return cls(
            model=ModelConfig(**dct.get("model", {})),
            sim=SimulationConfig(**dct.get("sim", {})),
            paths=PathConfig(**dct.get("paths", {})),
        )


# ------------------------------------------------------------------ #
#  helpers                                                            #
# ------------------------------------------------------------------ #
def load_config(path: str | Path) -> Config:
    """Auto-detect JSON vs. YAML and load into a Config instance."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not installed – cannot read YAML config.")
        data = yaml.safe_load(path.read_text())
    else:
        data = json.loads(path.read_text())

    return Config.from_dict(data)
