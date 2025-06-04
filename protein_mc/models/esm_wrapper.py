"""Tiny helper around Hugging-Face's ESM pipeline.

Having it separated makes mocking dead-easy in unit-tests.
"""

from transformers import pipeline


def load_esm_pipeline(model_name: str = "facebook/esm2_t33_650M_UR50D", device: str = "cpu"):
    """
    Returns a HF feature-extraction pipeline that outputs per-token embeddings.
    The *tiny* `t6_8M` checkpoint is default for speed in CI demos.
    """
    return pipeline("feature-extraction", model=model_name, device=device)
