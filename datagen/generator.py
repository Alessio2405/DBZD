from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .tokenizer import SimpleTokenizer, TokenizerLike, tokenizer_metadata

ZONE_NAMES = (
    "context",
    "problem",
    "constraint",
    "data",
    "reasoning",
    "intermediate_conclusion",
    "final_answer",
)

DATAGEN_SCHEMA_VERSION = 3

ZONE_MARKER_VARIANTS = (
    ("CONTEXT:", "SETTING:", "BACKGROUND:"),
    ("PROBLEM:", "TASK:", "QUESTION:"),
    ("CONSTRAINT:", "RULES:", "REQUIREMENTS:"),
    ("DATA:", "VALUES:", "GIVEN:"),
    ("REASONING:", "WORK:", "SOLUTION STEPS:"),
    ("CONCLUSION:", "RESULT:", "INTERMEDIATE RESULT:"),
    ("FINAL:", "ANSWER:", "FINAL RESPONSE:"),
)

# Kept as the canonical representative of each zone for backwards-compatible
# metadata consumers. Generation samples from all variants above.
ZONE_MARKERS = tuple(variants[0] for variants in ZONE_MARKER_VARIANTS)

SECTION_TRANSITIONS = (
    ("",),
    ("", "Next section -- ", "The task now is: "),
    ("", "Please follow these: ", "Before solving, note the "),
    ("", "The quantities follow. ", "For this instance, use the "),
    ("", "Now work it out. ", "Proceed with the "),
    ("", "From those steps, the ", "This gives the "),
    ("", "To finish, give the ", "The requested output is the "),
)


@dataclass(frozen=True)
class GeneratedExample:
    family: str
    zone_texts: tuple[str, ...]
    answer: int
    calculation: dict[str, Any]
    context_sentences: int
    constraint_count: int
    reasoning_steps: int
    section_markers: tuple[str, ...]

    @property
    def raw_text(self) -> str:
        return "\n".join(self.zone_texts)

    @property
    def prompt_text(self) -> str:
        return "\n".join(self.zone_texts[:4]) + "\n"


FamilyBuilder = Callable[[random.Random, str], GeneratedExample]


def compute_answer(calculation: dict[str, Any]) -> int:
    kind = calculation["kind"]
    if kind == "add":
        return int(sum(calculation["values"]))
    if kind == "subtract":
        return int(calculation["start"] - sum(calculation["subtract"]))
    if kind == "multiply":
        result = 1
        for value in calculation["factors"]:
            result *= value
        return int(result)
    if kind == "add_subtract":
        return int(
            calculation["start"]
            + sum(calculation["add"])
            - sum(calculation["subtract"])
        )
    if kind == "subtract_add":
        return int(
            calculation["start"]
            - sum(calculation["subtract"])
            + sum(calculation["add"])
        )
    if kind == "multiply_add":
        product = 1
        for value in calculation["factors"]:
            product *= value
        return int(product + sum(calculation["add"]))
    raise ValueError(f"Unknown calculation kind: {kind}")


def _context(rng: random.Random, subject: str) -> tuple[str, int]:
    options = [
        [f"We solve a small task about {subject}."],
        [
            f"We solve a small task about {subject}.",
            "All quantities are whole numbers.",
        ],
        [
            f"This is a short task about {subject}.",
            "Only the listed quantities are used.",
        ],
    ]
    sentences = rng.choice(options)
    return " ".join(sentences), len(sentences)


def _constraints(rng: random.Random, operation_hint: str) -> tuple[str, int]:
    pool = [
        "Use every listed value.",
        "Keep the order of the changes.",
        "Return one whole number.",
        f"Use {operation_hint} as described.",
    ]
    count = rng.randint(1, 3)
    chosen = rng.sample(pool, count)
    return " ".join(chosen), count


def _reasoning(steps: Iterable[str]) -> tuple[str, int]:
    step_list = list(steps)
    rendered = " ".join(
        f"Step {index}: {text}" for index, text in enumerate(step_list, start=1)
    )
    return rendered, len(step_list)


def _render_section(
    rng: random.Random,
    zone_id: int,
    marker: str,
    content: str,
) -> str:
    transition = rng.choice(SECTION_TRANSITIONS[zone_id])
    return f"{transition}{marker} {content}"


def _build_example(
    *,
    rng: random.Random,
    family: str,
    subject: str,
    problem: str,
    operation_hint: str,
    data: str,
    steps: list[str],
    conclusion: str,
    calculation: dict[str, Any],
) -> GeneratedExample:
    answer = compute_answer(calculation)
    context, context_count = _context(rng, subject)
    constraints, constraint_count = _constraints(rng, operation_hint)
    reasoning, reasoning_count = _reasoning(steps)
    section_markers = tuple(
        rng.choice(variants) for variants in ZONE_MARKER_VARIANTS
    )
    section_contents = (
        context,
        problem,
        constraints,
        data,
        reasoning,
        conclusion.format(answer=answer),
        f"The answer is {answer}.",
    )
    zone_texts = tuple(
        _render_section(rng, zone_id, section_markers[zone_id], content)
        for zone_id, content in enumerate(section_contents)
    )
    return GeneratedExample(
        family=family,
        zone_texts=zone_texts,
        answer=answer,
        calculation=calculation,
        context_sentences=context_count,
        constraint_count=constraint_count,
        reasoning_steps=reasoning_count,
        section_markers=section_markers,
    )


def _add_two(rng: random.Random, family: str) -> GeneratedExample:
    a, b = rng.randint(3, 35), rng.randint(2, 25)
    return _build_example(
        rng=rng,
        family=family,
        subject="red and blue counters",
        problem="Find the total number of counters.",
        operation_hint="addition",
        data=f"There are {a} red counters and {b} blue counters.",
        steps=[
            f"Start with {a} red counters.",
            f"Add {b} blue counters to get {a + b}.",
        ],
        conclusion="The total number of counters is {answer}.",
        calculation={"kind": "add", "values": [a, b]},
    )


def _add_three(rng: random.Random, family: str) -> GeneratedExample:
    a, b, c = (rng.randint(2, 20) for _ in range(3))
    first = a + b
    return _build_example(
        rng=rng,
        family=family,
        subject="three trays",
        problem="Find how many pieces are on all trays.",
        operation_hint="addition",
        data=f"Tray one has {a} pieces, tray two has {b}, and tray three has {c}.",
        steps=[
            f"Add {a} and {b} to get {first}.",
            f"Add {c} to {first} to get {first + c}.",
        ],
        conclusion="All trays hold {answer} pieces.",
        calculation={"kind": "add", "values": [a, b, c]},
    )


def _subtract(rng: random.Random, family: str) -> GeneratedExample:
    start = rng.randint(25, 70)
    removed = rng.randint(2, start - 5)
    answer = start - removed
    return _build_example(
        rng=rng,
        family=family,
        subject="books on a shelf",
        problem="Find how many books remain.",
        operation_hint="subtraction",
        data=f"The shelf starts with {start} books and {removed} books are removed.",
        steps=[
            f"Start with {start} books.",
            f"Subtract {removed} from {start} to get {answer}.",
        ],
        conclusion="{answer} books remain.",
        calculation={"kind": "subtract", "start": start, "subtract": [removed]},
    )


def _subtract_add(rng: random.Random, family: str) -> GeneratedExample:
    start = rng.randint(20, 60)
    removed = rng.randint(2, 12)
    added = rng.randint(2, 15)
    after_remove = start - removed
    return _build_example(
        rng=rng,
        family=family,
        subject="items in a box",
        problem="Find the final number of items.",
        operation_hint="subtraction then addition",
        data=f"The box has {start} items. {removed} leave and then {added} arrive.",
        steps=[
            f"Subtract {removed} from {start} to get {after_remove}.",
            f"Add {added} to {after_remove} to get {after_remove + added}.",
        ],
        conclusion="The box finishes with {answer} items.",
        calculation={
            "kind": "subtract_add",
            "start": start,
            "subtract": [removed],
            "add": [added],
        },
    )


def _multiply(rng: random.Random, family: str) -> GeneratedExample:
    groups, size = rng.randint(2, 9), rng.randint(2, 10)
    return _build_example(
        rng=rng,
        family=family,
        subject="equal groups",
        problem="Find the total number of objects.",
        operation_hint="multiplication",
        data=f"There are {groups} groups with {size} objects in each group.",
        steps=[
            f"Use {groups} equal groups.",
            f"Multiply {groups} by {size} to get {groups * size}.",
        ],
        conclusion="The groups contain {answer} objects in total.",
        calculation={"kind": "multiply", "factors": [groups, size]},
    )


def _multiply_add(rng: random.Random, family: str) -> GeneratedExample:
    groups, size = rng.randint(2, 8), rng.randint(2, 10)
    loose = rng.randint(1, min(9, 99 - groups * size))
    product = groups * size
    return _build_example(
        rng=rng,
        family=family,
        subject="full packs and loose pieces",
        problem="Find the total number of pieces.",
        operation_hint="multiplication then addition",
        data=f"There are {groups} full packs of {size} pieces and {loose} loose pieces.",
        steps=[
            f"Multiply {groups} by {size} to get {product} packed pieces.",
            f"Add {loose} loose pieces to {product}.",
            f"The sum is {product + loose}.",
        ],
        conclusion="There are {answer} pieces altogether.",
        calculation={
            "kind": "multiply_add",
            "factors": [groups, size],
            "add": [loose],
        },
    )


def _row_product(rng: random.Random, family: str) -> GeneratedExample:
    rows, per_row = rng.randint(2, 9), rng.randint(3, 10)
    return _build_example(
        rng=rng,
        family=family,
        subject="seats arranged in rows",
        problem="Find the number of seats in the rectangular arrangement.",
        operation_hint="multiplication",
        data=f"The arrangement has {rows} rows with {per_row} seats in every row.",
        steps=[
            f"Count {rows} equal rows.",
            f"Each row contributes {per_row} seats.",
            f"Multiply {rows} by {per_row} to get {rows * per_row}.",
        ],
        conclusion="The arrangement contains {answer} seats.",
        calculation={"kind": "multiply", "factors": [rows, per_row]},
    )


def _add_subtract(rng: random.Random, family: str) -> GeneratedExample:
    start = rng.randint(10, 35)
    added = rng.randint(8, 25)
    removed = rng.randint(2, min(15, start + added - 1))
    after_add = start + added
    return _build_example(
        rng=rng,
        family=family,
        subject="a changing collection",
        problem="Find the final size of the collection.",
        operation_hint="addition then subtraction",
        data=f"The collection starts at {start}. Add {added}, then remove {removed}.",
        steps=[
            f"Add {added} to {start} to get {after_add}.",
            f"Subtract {removed} from {after_add}.",
            f"The final value is {after_add - removed}.",
        ],
        conclusion="The collection ends at {answer}.",
        calculation={
            "kind": "add_subtract",
            "start": start,
            "add": [added],
            "subtract": [removed],
        },
    )


def _jar_sum(rng: random.Random, family: str) -> GeneratedExample:
    first, second = rng.randint(4, 28), rng.randint(3, 24)
    return _build_example(
        rng=rng,
        family=family,
        subject="marbles in sealed jars",
        problem="Determine the combined marble count without opening another jar.",
        operation_hint="addition",
        data=f"One labeled jar contains {first} marbles; another contains {second}.",
        steps=[
            f"Read the first label as {first}.",
            f"Combine it with the second label, {second}.",
            f"The combined count is {first + second}.",
        ],
        conclusion="Together the two jars contain {answer} marbles.",
        calculation={"kind": "add", "values": [first, second]},
    )


def _ticket_remainder(rng: random.Random, family: str) -> GeneratedExample:
    printed = rng.randint(30, 80)
    issued = rng.randint(5, printed - 8)
    return _build_example(
        rng=rng,
        family=family,
        subject="numbered event tickets",
        problem="Determine the number of unissued tickets.",
        operation_hint="subtraction",
        data=f"A booth printed {printed} tickets and issued {issued} of them.",
        steps=[
            f"Use {printed} as the printed supply.",
            f"Remove the {issued} issued tickets.",
            f"The unissued amount is {printed - issued}.",
        ],
        conclusion="{answer} tickets have not been issued.",
        calculation={"kind": "subtract", "start": printed, "subtract": [issued]},
    )


def _carton_total(rng: random.Random, family: str) -> GeneratedExample:
    cartons, units = rng.randint(2, 8), rng.randint(3, 11)
    samples = rng.randint(1, 9)
    boxed = cartons * units
    return _build_example(
        rng=rng,
        family=family,
        subject="cartons and sample units",
        problem="Determine the shipment's complete unit count.",
        operation_hint="multiplication followed by addition",
        data=(
            f"A shipment has {cartons} cartons, {units} units per carton, "
            f"and {samples} separate samples."
        ),
        steps=[
            f"The cartons account for {cartons} times {units}.",
            f"That multiplication gives {boxed} boxed units.",
            f"Include {samples} samples to reach {boxed + samples}.",
        ],
        conclusion="The complete shipment has {answer} units.",
        calculation={
            "kind": "multiply_add",
            "factors": [cartons, units],
            "add": [samples],
        },
    )


def _token_balance(rng: random.Random, family: str) -> GeneratedExample:
    opening = rng.randint(18, 55)
    spent = rng.randint(2, 12)
    reward = rng.randint(3, 16)
    after_spend = opening - spent
    return _build_example(
        rng=rng,
        family=family,
        subject="a game token balance",
        problem="Determine the player's closing token balance.",
        operation_hint="subtraction followed by addition",
        data=(
            f"The player opens with {opening} tokens, spends {spent}, "
            f"then earns a reward of {reward}."
        ),
        steps=[
            f"After spending, the balance is {opening} minus {spent}, or {after_spend}.",
            f"Add the reward of {reward}.",
            f"The closing balance becomes {after_spend + reward}.",
        ],
        conclusion="The player closes with {answer} tokens.",
        calculation={
            "kind": "subtract_add",
            "start": opening,
            "subtract": [spent],
            "add": [reward],
        },
    )


@dataclass(frozen=True)
class SurfaceFamilySpec:
    kind: str
    subject: str
    item: str
    problem: str


_SURFACE_FAMILY_SPECS: dict[str, SurfaceFamilySpec] = {
    "bead_bowls": SurfaceFamilySpec(
        "add_two", "beads in two bowls", "beads", "Find the combined bead count."
    ),
    "coin_purses": SurfaceFamilySpec(
        "add_two", "coins in two purses", "coins", "Work out how many coins the purses hold together."
    ),
    "fruit_baskets": SurfaceFamilySpec(
        "add_three", "fruit in three baskets", "pieces of fruit", "Total the fruit across all three baskets."
    ),
    "route_segments": SurfaceFamilySpec(
        "add_three", "three route segments", "kilometers", "Find the full route length."
    ),
    "plant_remainder": SurfaceFamilySpec(
        "subtract", "plants in a greenhouse", "plants", "Determine how many plants stay in the greenhouse."
    ),
    "snack_remainder": SurfaceFamilySpec(
        "subtract", "snacks in a pantry", "snacks", "Calculate the number of snacks left."
    ),
    "bus_flow": SurfaceFamilySpec(
        "subtract_add", "passengers on a bus", "passengers", "Find the passenger count after one stop."
    ),
    "tank_flow": SurfaceFamilySpec(
        "subtract_add", "liters in a tank", "liters", "Compute the final amount in the tank."
    ),
    "team_groups": SurfaceFamilySpec(
        "multiply", "players assigned to teams", "players", "Count all players in the equal teams."
    ),
    "stamp_sheets": SurfaceFamilySpec(
        "multiply", "stamps printed on sheets", "stamps", "Determine the total number of printed stamps."
    ),
    "bundle_spares": SurfaceFamilySpec(
        "multiply_add", "bundled cables and spare cables", "cables", "Find the complete cable count."
    ),
    "case_samples": SurfaceFamilySpec(
        "multiply_add", "cases and individual samples", "samples", "Count every sample, boxed or separate."
    ),
    "score_flow": SurfaceFamilySpec(
        "add_subtract", "points in a game round", "points", "Compute the score after a bonus and a penalty."
    ),
    "inventory_flow": SurfaceFamilySpec(
        "add_subtract", "units in an inventory", "units", "Find the inventory after a delivery and a withdrawal."
    ),
    "tile_array": SurfaceFamilySpec(
        "multiply", "tiles in a rectangular array", "tiles", "Work out how many tiles fill the array."
    ),
    "packet_extras": SurfaceFamilySpec(
        "multiply_add", "packets and extra cards", "cards", "Find the total number of cards."
    ),
}


def _build_surface_family(
    rng: random.Random,
    family: str,
    spec: SurfaceFamilySpec,
) -> GeneratedExample:
    if spec.kind == "add_two":
        first, second = rng.randint(2, 35), rng.randint(2, 35)
        calculation = {"kind": "add", "values": [first, second]}
        data = (
            f"The first group contains {first} {spec.item}; "
            f"the second contains {second} {spec.item}."
        )
        steps = [
            f"Read the two counts as {first} and {second}.",
            f"Their sum is {first + second} {spec.item}.",
        ]
    elif spec.kind == "add_three":
        first, second, third = (rng.randint(2, 20) for _ in range(3))
        subtotal = first + second
        calculation = {"kind": "add", "values": [first, second, third]}
        data = (
            f"The three listed amounts are {first}, {second}, and {third} "
            f"{spec.item}."
        )
        steps = [
            f"Combine {first} with {second} to make {subtotal}.",
            f"Include {third} to reach {subtotal + third} {spec.item}.",
        ]
    elif spec.kind == "subtract":
        start = rng.randint(20, 90)
        removed = rng.randint(2, start - 2)
        calculation = {"kind": "subtract", "start": start, "subtract": [removed]}
        data = f"The initial amount is {start} {spec.item}, and {removed} are taken away."
        steps = [
            f"Use {start} as the starting count.",
            f"Removing {removed} leaves {start - removed} {spec.item}.",
        ]
    elif spec.kind == "subtract_add":
        start = rng.randint(15, 65)
        removed = rng.randint(2, min(15, start - 1))
        after_remove = start - removed
        added = rng.randint(2, min(20, 98 - after_remove))
        calculation = {
            "kind": "subtract_add",
            "start": start,
            "subtract": [removed],
            "add": [added],
        }
        data = (
            f"Begin with {start} {spec.item}; {removed} leave, then {added} arrive."
        )
        steps = [
            f"Subtract {removed} from {start} to obtain {after_remove}.",
            f"Add the arriving {added} for a final count of {after_remove + added}.",
        ]
    elif spec.kind == "multiply":
        groups = rng.randint(2, 9)
        size = rng.randint(2, min(10, 99 // groups))
        calculation = {"kind": "multiply", "factors": [groups, size]}
        data = f"There are {groups} equal sets with {size} {spec.item} in each set."
        steps = [
            f"Identify {groups} sets of size {size}.",
            f"Multiplying {groups} by {size} gives {groups * size} {spec.item}.",
        ]
    elif spec.kind == "multiply_add":
        groups, size = rng.randint(2, 8), rng.randint(2, 10)
        product = groups * size
        extra = rng.randint(1, min(15, 99 - product))
        calculation = {
            "kind": "multiply_add",
            "factors": [groups, size],
            "add": [extra],
        }
        data = (
            f"There are {groups} full sets of {size} {spec.item} plus "
            f"{extra} separate {spec.item}."
        )
        steps = [
            f"The full sets contain {groups} times {size}, which is {product}.",
            f"Adding the {extra} separate items gives {product + extra} {spec.item}.",
        ]
    elif spec.kind == "add_subtract":
        start = rng.randint(10, 45)
        added = rng.randint(5, min(30, 98 - start))
        after_add = start + added
        removed = rng.randint(2, min(20, after_add - 1))
        calculation = {
            "kind": "add_subtract",
            "start": start,
            "add": [added],
            "subtract": [removed],
        }
        data = (
            f"Start at {start} {spec.item}, add {added}, and later remove {removed}."
        )
        steps = [
            f"The addition changes {start} by {added} to {after_add}.",
            f"Taking away {removed} produces {after_add - removed} {spec.item}.",
        ]
    else:
        raise ValueError(f"Unsupported surface family kind: {spec.kind}")

    return _build_example(
        rng=rng,
        family=family,
        subject=spec.subject,
        problem=spec.problem,
        operation_hint=spec.kind.replace("_", " "),
        data=data,
        steps=steps,
        conclusion=f"The resulting number of {spec.item} is {{answer}}.",
        calculation=calculation,
    )


def _make_surface_builder(spec: SurfaceFamilySpec) -> FamilyBuilder:
    def builder(rng: random.Random, family: str) -> GeneratedExample:
        return _build_surface_family(rng, family, spec)

    return builder


FAMILY_BUILDERS: dict[str, FamilyBuilder] = {
    "counter_sum": _add_two,
    "tray_sum": _add_three,
    "shelf_remainder": _subtract,
    "box_flow": _subtract_add,
    "equal_groups": _multiply,
    "pack_plus_loose": _multiply_add,
    "collection_flow": _add_subtract,
    "row_product": _row_product,
    "jar_sum": _jar_sum,
    "ticket_remainder": _ticket_remainder,
    "carton_total": _carton_total,
    "token_balance": _token_balance,
    **{
        family: _make_surface_builder(spec)
        for family, spec in _SURFACE_FAMILY_SPECS.items()
    },
}

SPLIT_FAMILIES: dict[str, tuple[str, ...]] = {
    "train": (
        "counter_sum",
        "tray_sum",
        "shelf_remainder",
        "box_flow",
        "equal_groups",
        "pack_plus_loose",
        "collection_flow",
        "row_product",
        "bead_bowls",
        "coin_purses",
        "fruit_baskets",
        "route_segments",
        "plant_remainder",
        "snack_remainder",
        "bus_flow",
        "tank_flow",
        "team_groups",
        "stamp_sheets",
        "bundle_spares",
        "case_samples",
        "score_flow",
        "inventory_flow",
        "tile_array",
        "packet_extras",
    ),
    "val": ("jar_sum", "ticket_remainder"),
    "test": ("carton_total", "token_balance"),
}


def generate_examples(split: str, count: int, seed: int) -> list[GeneratedExample]:
    if split not in SPLIT_FAMILIES:
        raise ValueError(f"Unknown split {split!r}; expected one of {tuple(SPLIT_FAMILIES)}")
    rng = random.Random(seed)
    families = list(SPLIT_FAMILIES[split])
    examples: list[GeneratedExample] = []
    for index in range(count):
        family = families[index % len(families)]
        examples.append(FAMILY_BUILDERS[family](rng, family))
    rng.shuffle(examples)
    return examples


def tokenize_example(
    example: GeneratedExample,
    tokenizer: TokenizerLike,
    *,
    split: str,
) -> dict[str, Any]:
    tokens: list[int] = []
    zone_ids: list[int] = []
    prompt_token_count = 0
    for zone_id, zone_text in enumerate(example.zone_texts):
        suffix = "\n" if zone_id < len(example.zone_texts) - 1 else ""
        zone_tokens = tokenizer.encode(zone_text + suffix, add_special_tokens=False)
        tokens.extend(zone_tokens)
        zone_ids.extend([zone_id] * len(zone_tokens))
        if zone_id <= 3:
            prompt_token_count += len(zone_tokens)

    if tokenizer.eos_token_id is not None:
        tokens.append(int(tokenizer.eos_token_id))
        zone_ids.append(6)

    return {
        "tokens": tokens,
        "zone_ids": zone_ids,
        "answer": example.answer,
        "raw_text": example.raw_text,
        "prompt_text": example.prompt_text,
        "prompt_token_count": prompt_token_count,
        "family": example.family,
        "split": split,
        "calculation": example.calculation,
        "section_markers": list(example.section_markers),
        "shape": {
            "context_sentences": example.context_sentences,
            "constraint_count": example.constraint_count,
            "reasoning_steps": example.reasoning_steps,
        },
    }


def _tiny_counts(total: int) -> dict[str, int]:
    if total < 3:
        raise ValueError("--n must be at least 3")
    val = max(1, round(total * 0.05))
    test = max(1, round(total * 0.05))
    train = total - val - test
    return {"train": train, "val": val, "test": test}


def generate_dataset(
    output_dir: str | Path,
    tokenizer: TokenizerLike,
    *,
    tokenizer_name: str,
    train_n: int = 40_000,
    val_n: int = 2_000,
    test_n: int = 2_000,
    n: int | None = None,
    seed: int = 1234,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    counts = (
        _tiny_counts(n)
        if n is not None
        else {"train": train_n, "val": val_n, "test": test_n}
    )
    split_seed_offsets = {"train": 0, "val": 10_000, "test": 20_000}
    inspection_records: list[str] = []
    roundtrip_checked = 0

    for split, count in counts.items():
        examples = generate_examples(split, count, seed + split_seed_offsets[split])
        output_file = output_path / f"{split}.jsonl"
        with output_file.open("w", encoding="utf-8") as handle:
            for index, example in enumerate(examples):
                record = tokenize_example(example, tokenizer, split=split)
                if re.search(r"(?<=[A-Za-z])\d|\d(?=[A-Za-z])", example.raw_text):
                    raise ValueError(
                        f"Datagen emitted a number without clean spacing in {example.family}"
                    )
                if roundtrip_checked < 100:
                    content_tokens = list(record["tokens"])
                    if (
                        tokenizer.eos_token_id is not None
                        and content_tokens
                        and content_tokens[-1] == int(tokenizer.eos_token_id)
                    ):
                        content_tokens.pop()
                    try:
                        decoded = tokenizer.decode(
                            content_tokens,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                    except TypeError:
                        decoded = tokenizer.decode(
                            content_tokens, skip_special_tokens=True
                        )
                    if decoded != example.raw_text:
                        raise ValueError(
                            "Tokenizer round-trip changed datagen text at "
                            f"example {roundtrip_checked} ({split}/{example.family}): "
                            f"expected {example.raw_text!r}, got {decoded!r}"
                        )
                    roundtrip_checked += 1
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if index < 3:
                    inspection_records.append(
                        f"===== {split} / {example.family} =====\n{example.raw_text}\n"
                    )

    if isinstance(tokenizer, SimpleTokenizer):
        tokenizer.save(output_path / "simple_tokenizer.json")

    metadata = {
        "generator_schema_version": DATAGEN_SCHEMA_VERSION,
        "seed": seed,
        "counts": counts,
        "tokenizer_roundtrip_examples": roundtrip_checked,
        "zones": list(ZONE_NAMES),
        "zone_markers": list(ZONE_MARKERS),
        "zone_marker_variants": [
            list(variants) for variants in ZONE_MARKER_VARIANTS
        ],
        "split_families": {key: list(value) for key, value in SPLIT_FAMILIES.items()},
        "tokenizer": tokenizer_metadata(tokenizer, tokenizer_name),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_path / "inspection_samples.txt").write_text(
        "\n".join(inspection_records),
        encoding="utf-8",
    )
    return metadata
