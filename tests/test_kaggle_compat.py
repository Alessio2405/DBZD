from __future__ import annotations

import pytest

from scripts.kaggle_torch_compat import (
    needs_p100_repair,
    parse_gpu_info,
    required_cuda_arch,
)


def test_parse_kaggle_gpu_info() -> None:
    assert parse_gpu_info("Tesla P100-PCIE-16GB, 6.0") == (
        "Tesla P100-PCIE-16GB",
        (6, 0),
    )
    assert parse_gpu_info("Tesla T4, 7.5") == ("Tesla T4", (7, 5))


def test_required_architecture_and_repair_decision() -> None:
    assert required_cuda_arch((6, 0)) == "sm_60"
    assert required_cuda_arch((7, 5)) == "sm_75"
    assert needs_p100_repair((6, 0), ["sm_70", "sm_75"])
    assert not needs_p100_repair((6, 0), ["sm_60", "sm_70"])
    assert not needs_p100_repair((7, 5), ["sm_70", "sm_75"])


def test_gpu_parser_rejects_unexpected_output() -> None:
    with pytest.raises(RuntimeError):
        parse_gpu_info("unknown accelerator")
