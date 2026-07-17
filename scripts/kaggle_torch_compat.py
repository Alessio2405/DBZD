from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
from typing import Any


P100_TORCH_VERSION = "2.8.0"
P100_TORCHVISION_VERSION = "0.23.0"
P100_TORCHAUDIO_VERSION = "2.8.0"
P100_INDEX_URL = "https://download.pytorch.org/whl/cu126"


def _run_output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def parse_gpu_info(first_line: str) -> tuple[str, tuple[int, int]]:
    match = re.match(r"(.+),\s*(\d+)\.(\d+)", first_line)
    if match is None:
        raise RuntimeError(f"Could not parse nvidia-smi output: {first_line!r}")
    return match.group(1).strip(), (int(match.group(2)), int(match.group(3)))


def _gpu_info() -> tuple[str, tuple[int, int]]:
    output = _run_output(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap",
            "--format=csv,noheader",
        ]
    )
    return parse_gpu_info(output.splitlines()[0])


def required_cuda_arch(capability: tuple[int, int]) -> str:
    return f"sm_{capability[0]}{capability[1]}"


def needs_p100_repair(
    capability: tuple[int, int],
    compiled_architectures: list[str],
) -> bool:
    return capability == (6, 0) and required_cuda_arch(capability) not in compiled_architectures


def _torch_info() -> dict[str, Any]:
    code = """
import json
import torch
print(json.dumps({
    "version": torch.__version__,
    "cuda": torch.version.cuda,
    "architectures": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
}))
"""
    output = _run_output([sys.executable, "-c", code])
    return json.loads(output.splitlines()[-1])


def _installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _install_p100_build() -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--force-reinstall",
        f"torch=={P100_TORCH_VERSION}",
        f"torchvision=={P100_TORCHVISION_VERSION}",
        f"torchaudio=={P100_TORCHAUDIO_VERSION}",
        "--index-url",
        P100_INDEX_URL,
    ]
    print(
        "Installing the official CUDA 12.6 PyTorch wheel with Pascal/sm_60 "
        "support. This is a one-time cost for this Kaggle session.",
        flush=True,
    )
    subprocess.run(command, check=True)


def _validate_cuda(expected_arch: str) -> dict[str, Any]:
    code = f"""
import json
import torch
expected = {expected_arch!r}
architectures = torch.cuda.get_arch_list()
if expected not in architectures:
    raise RuntimeError(
        f"PyTorch {{torch.__version__}} supports {{architectures}}, "
        f"but this GPU requires {{expected}}."
    )
x = torch.ones(16, device="cuda")
y = (x * 2).sum()
torch.cuda.synchronize()
print(json.dumps({{
    "version": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "architectures": architectures,
    "test_value": float(y.item()),
}}))
"""
    output = _run_output([sys.executable, "-c", code])
    return json.loads(output.splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Kaggle's PyTorch wheel against its assigned GPU."
    )
    parser.add_argument(
        "--auto-fix-p100",
        action="store_true",
        help="Install PyTorch 2.8.0/cu126 when a P100 lacks sm_60 support.",
    )
    args = parser.parse_args()

    gpu_name, capability = _gpu_info()
    required_arch = required_cuda_arch(capability)
    info = _torch_info()
    print(
        f"GPU: {gpu_name} (compute capability {capability[0]}.{capability[1]}); "
        f"PyTorch: {info['version']} / CUDA {info['cuda']}; "
        f"compiled architectures: {info['architectures']}",
        flush=True,
    )

    if required_arch not in info["architectures"]:
        if needs_p100_repair(capability, info["architectures"]) and args.auto_fix_p100:
            print(
                f"The current wheel lacks {required_arch}. "
                f"Installed package versions: torch={_installed_version('torch')}, "
                f"torchvision={_installed_version('torchvision')}, "
                f"torchaudio={_installed_version('torchaudio')}.",
                flush=True,
            )
            _install_p100_build()
        else:
            raise RuntimeError(
                f"The assigned {gpu_name} requires {required_arch}, but the installed "
                f"PyTorch wheel only contains {info['architectures']}. Select a Kaggle "
                "T4 accelerator or install a compatible CUDA wheel."
            )

    validated = _validate_cuda(required_arch)
    print(
        f"CUDA compatibility check passed: {validated['gpu']}, "
        f"torch {validated['version']}, CUDA {validated['cuda']}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
