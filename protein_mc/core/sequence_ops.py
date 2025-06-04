"""Sequence utilities: mutation, Hamming distance, ESM2 embeddings (cached)."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple, Union

import torch

_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
_embedding_cache: Dict[str, torch.Tensor] = {}


# ---------------------------------------------------------------------- #
#  Embeddings                                                            #
# ---------------------------------------------------------------------- #
def get_embeddings(sequence: str, hf_pipeline) -> torch.Tensor:
    """
    Obtain (and cache) ESM2 per-token embeddings for *sequence*.
    `hf_pipeline` must be a Hugging-Face pipeline created with
    `task="feature-extraction"`.
    """
    if sequence in _embedding_cache:
        return _embedding_cache[sequence]

    emb = hf_pipeline(sequence)
    emb_t = torch.tensor(emb[0])[1:-1]  # [L, D]
    _embedding_cache[sequence] = emb_t
    return emb_t


# ---------------------------------------------------------------------- #
#  Mutation & metrics                                                    #
# ---------------------------------------------------------------------- #
def mutate_sequence(
    sequence: str,
    protected_positions: Optional[List[int]] = None,
    *,
    return_details: bool = False,
) -> Union[str, Tuple[str, int, str]]:
    """
    Single random point mutation obeying optional 1-based *protected_positions*.
    Returns either the new sequence or (seq, position, "X->Y") if
    `return_details=True`.
    """
    protected_idx = {p - 1 for p in protected_positions or []}
    mutable_idx = [i for i in range(len(sequence)) if i not in protected_idx]
    if not mutable_idx:
        raise ValueError("All positions are protected – cannot mutate.")

    pos = random.choice(mutable_idx)
    current = sequence[pos]
    new = current
    while new == current:
        new = random.choice(_amino_acids)

    mutated = sequence[:pos] + new + sequence[pos + 1 :]

    if return_details:
        return mutated, pos + 1, f"{current}->{new}"
    return mutated


def hamming_distance(seq1: str, seq2: str) -> int:
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be the same length.")
    return sum(a != b for a, b in zip(seq1, seq2))
