from __future__ import annotations

import json
import re

from datagen.generator import (
    FAMILY_BUILDERS,
    SECTION_TRANSITIONS,
    SPLIT_FAMILIES,
    ZONE_MARKER_VARIANTS,
    compute_answer,
    generate_dataset,
    generate_examples,
    tokenize_example,
)
from datagen.tokenizer import SimpleTokenizer


def test_zone_alignment_order_and_shape() -> None:
    tokenizer = SimpleTokenizer()
    examples = generate_examples("train", 64, seed=7)
    for example in examples:
        record = tokenize_example(example, tokenizer, split="train")
        assert len(record["tokens"]) == len(record["zone_ids"])
        assert record["tokens"]
        assert set(record["zone_ids"]) == set(range(7))
        assert record["zone_ids"] == sorted(record["zone_ids"])
        assert 1 <= record["shape"]["context_sentences"] <= 2
        assert 1 <= record["shape"]["constraint_count"] <= 3
        assert 2 <= record["shape"]["reasoning_steps"] <= 4
        assert 0 < record["prompt_token_count"] < len(record["tokens"])


def test_answers_are_programmatically_correct() -> None:
    for split in ("train", "val", "test"):
        for example in generate_examples(split, 50, seed=11):
            assert example.answer == compute_answer(example.calculation)
            match = re.search(r"The answer is (-?\d+)\.", example.raw_text)
            assert match is not None
            assert int(match.group(1)) == example.answer
            assert 0 <= example.answer < 100
            numeric_values: list[int] = []
            for value in example.calculation.values():
                if isinstance(value, int):
                    numeric_values.append(value)
                elif isinstance(value, list):
                    numeric_values.extend(item for item in value if isinstance(item, int))
            assert all(0 <= value < 100 for value in numeric_values)


def test_zone_markers_have_nontrivial_variants_and_are_sampled() -> None:
    assert all(len(set(variants)) >= 3 for variants in ZONE_MARKER_VARIANTS)
    assert all(len(set(variants)) >= 3 for variants in SECTION_TRANSITIONS[1:])
    observed = [set() for _ in ZONE_MARKER_VARIANTS]
    for example in generate_examples("train", 256, seed=17):
        for zone_id, marker in enumerate(example.section_markers):
            observed[zone_id].add(marker)
    assert all(len(markers) >= 3 for markers in observed)


def test_template_families_are_disjoint() -> None:
    family_sets = [set(families) for families in SPLIT_FAMILIES.values()]
    assert family_sets[0].isdisjoint(family_sets[1])
    assert family_sets[0].isdisjoint(family_sets[2])
    assert family_sets[1].isdisjoint(family_sets[2])
    assert len({id(FAMILY_BUILDERS[name]) for name in FAMILY_BUILDERS}) == len(
        FAMILY_BUILDERS
    )
    assert len(SPLIT_FAMILIES["train"]) >= 24


def test_tokenizer_round_trips_100_examples_losslessly() -> None:
    tokenizer = SimpleTokenizer()
    examples = generate_examples("train", 100, seed=23)
    for example in examples:
        record = tokenize_example(example, tokenizer, split="train")
        content_tokens = record["tokens"][:-1]
        assert tokenizer.decode(content_tokens) == example.raw_text
        assert re.search(r"(?<=[A-Za-z])\d|\d(?=[A-Za-z])", example.raw_text) is None


def test_tiny_dataset_writes_expected_contract(tmp_path) -> None:
    tokenizer = SimpleTokenizer()
    metadata = generate_dataset(
        tmp_path,
        tokenizer,
        tokenizer_name="simple",
        n=20,
        seed=99,
    )
    assert sum(metadata["counts"].values()) == 20
    assert metadata["tokenizer_roundtrip_examples"] == 20
    assert (tmp_path / "simple_tokenizer.json").exists()
    assert (tmp_path / "inspection_samples.txt").exists()
    for split, expected_count in metadata["counts"].items():
        lines = (tmp_path / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == expected_count
        record = json.loads(lines[0])
        assert {
            "tokens",
            "zone_ids",
            "answer",
            "raw_text",
            "prompt_token_count",
        }.issubset(record)
