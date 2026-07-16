from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol


class TokenizerLike(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None
    bos_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str: ...


_TOKEN_PATTERN = re.compile(r"\d+|[A-Za-z]+|[^\w\s]", re.ASCII)


class SimpleTokenizer:
    """Small serializable tokenizer used only for offline smoke tests.

    It deliberately keeps tokenization transparent: words, integer strings, and
    punctuation are separate tokens. The full experiment uses the student model's
    Hugging Face tokenizer instead.
    """

    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = "<unk>"

    def __init__(self, vocab: dict[str, int] | None = None, frozen: bool = False) -> None:
        base = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
        }
        self.vocab = dict(vocab or base)
        self.inverse_vocab = {value: key for key, value in self.vocab.items()}
        self.frozen = frozen

    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.pad_token]

    @property
    def bos_token_id(self) -> int:
        return self.vocab[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.vocab[self.eos_token]

    @property
    def unk_token_id(self) -> int:
        return self.vocab[self.unk_token]

    def __len__(self) -> int:
        return len(self.vocab)

    def _id_for(self, token: str) -> int:
        if token in self.vocab:
            return self.vocab[token]
        if self.frozen:
            return self.unk_token_id
        token_id = len(self.vocab)
        self.vocab[token] = token_id
        self.inverse_vocab[token_id] = token
        return token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self._id_for(token) for token in _TOKEN_PATTERN.findall(text)]
        if add_special_tokens:
            return [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        special = {self.pad_token, self.bos_token, self.eos_token, self.unk_token}
        tokens: list[str] = []
        for token_id in token_ids:
            token = self.inverse_vocab.get(int(token_id), self.unk_token)
            if skip_special_tokens and token in special:
                continue
            tokens.append(token)
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,:;!?])", r"\1", text)
        return text

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"vocab": self.vocab}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, frozen: bool = True) -> "SimpleTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(vocab=payload["vocab"], frozen=frozen)


def load_tokenizer(
    name_or_path: str,
    *,
    data_dir: str | Path | None = None,
    frozen_simple: bool = False,
) -> TokenizerLike:
    if name_or_path == "simple":
        tokenizer_path = Path(data_dir or ".") / "simple_tokenizer.json"
        if tokenizer_path.exists():
            return SimpleTokenizer.load(tokenizer_path, frozen=frozen_simple)
        return SimpleTokenizer(frozen=frozen_simple)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        else:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenizer_metadata(tokenizer: TokenizerLike, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "vocab_size": len(tokenizer),  # type: ignore[arg-type]
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

