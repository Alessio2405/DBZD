from __future__ import annotations

from dbzd.diagnostics import (
    classify_answer_error,
    gold_completion_token_budget,
    parse_answer,
)


def test_answer_parser_matches_expected_generated_format() -> None:
    assert parse_answer("FINAL: The answer is 37.") == 37
    assert parse_answer("final response: the   answer is 8 .") == 8
    assert parse_answer("ANSWER: The answer is: 42.") == 42
    assert parse_answer("I computed 37 but omitted the required phrase.") is None
    assert parse_answer("FINAL: The answer is 9es.") is None
    assert parse_answer("FINAL: The answer is 2") is None


def test_gold_completion_budget_uses_nearest_rank_p99_plus_margin() -> None:
    records = [
        {"tokens": [10] * length + [2], "prompt_token_count": 0}
        for length in range(1, 101)
    ]
    budget = gold_completion_token_budget(
        records,
        eos_token_id=2,
        percentile=0.99,
        margin_tokens=20,
    )
    assert budget["percentile_tokens"] == 99
    assert budget["max_new_tokens"] == 119
    assert budget["maximum_gold_tokens"] == 100


def test_answer_error_taxonomy() -> None:
    calculation = {"kind": "add", "values": [12, 9]}

    category, expected, generated = classify_answer_error(
        "Step 1: Add 12 and 9 to get 20. The answer is 20.",
        calculation,
        predicted_answer=20,
        gold_answer=21,
    )
    assert category == "ARITHMETIC_ERROR"
    assert expected == [12, 9]
    assert generated == [12, 9, 20]

    category, _, _ = classify_answer_error(
        "Step 1: Add 12 and 8 to get 20. The answer is 20.",
        calculation,
        predicted_answer=20,
        gold_answer=21,
    )
    assert category == "WRONG_OPERANDS"

    category, _, _ = classify_answer_error(
        "Step 1: Add 12 and 9 to get 21.",
        calculation,
        predicted_answer=None,
        gold_answer=21,
    )
    assert category == "PARSE_FAIL"

    category, _, _ = classify_answer_error(
        "Step 1: Add 12 and 9 to get 21. The answer is 21.",
        calculation,
        predicted_answer=21,
        gold_answer=21,
    )
    assert category is None


def test_step_numbers_do_not_masquerade_as_operands() -> None:
    category, _, generated = classify_answer_error(
        "Step 1: Keep the 5. The answer is 7.",
        {"kind": "add", "values": [1, 5]},
        predicted_answer=7,
        gold_answer=6,
    )
    assert category == "WRONG_OPERANDS"
    assert generated == [5]
