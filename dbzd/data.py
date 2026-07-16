from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class JSONLTokenDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: str | Path) -> None:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(
                f"Missing dataset split {source}. Run `python -m datagen` first."
            )
        self.records = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class CausalLMCollator:
    def __init__(self, pad_token_id: int, max_length: int) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        lengths = [min(len(record["tokens"]), self.max_length) for record in records]
        batch_length = max(lengths)
        input_ids = torch.full(
            (len(records), batch_length), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        zone_labels = torch.full_like(input_ids, -100)

        for row, (record, length) in enumerate(zip(records, lengths)):
            token_tensor = torch.tensor(record["tokens"][:length], dtype=torch.long)
            zone_tensor = torch.tensor(record["zone_ids"][:length], dtype=torch.long)
            input_ids[row, :length] = token_tensor
            attention_mask[row, :length] = 1
            labels[row, :length] = token_tensor
            zone_labels[row, :length] = zone_tensor

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "zone_labels": zone_labels,
            "records": records,
        }

