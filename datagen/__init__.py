"""Synthetic arithmetic data generation for DBZD Phase 0."""

from .generator import (
    SPLIT_FAMILIES,
    ZONE_NAMES,
    GeneratedExample,
    compute_answer,
    generate_dataset,
    generate_examples,
)
from .tokenizer import SimpleTokenizer, load_tokenizer

__all__ = [
    "GeneratedExample",
    "SPLIT_FAMILIES",
    "SimpleTokenizer",
    "ZONE_NAMES",
    "compute_answer",
    "generate_dataset",
    "generate_examples",
    "load_tokenizer",
]

