"""Ensure CUDA 12 shared libraries are visible to CTranslate2 / faster-whisper.

The project venv may ship CUDA 13 nvidia wheels (libcublas.so.13) while
faster-whisper/CTranslate2 still looks for libcublas.so.12. Prefacing
LD_LIBRARY_PATH with known CUDA 12 locations fixes GPU Whisper loads.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

from loguru import logger

_CANDIDATE_DIRS = [
    Path.home() / ".local/ollama/lib/ollama/cuda_v12",
    Path.home() / "miniconda3/lib/python3.11/site-packages/nvidia/cublas/lib",
    Path.home() / "miniconda3/lib/python3.11/site-packages/nvidia/cuda_runtime/lib",
    Path("/usr/local/cuda-12/lib64"),
    Path("/usr/local/cuda/lib64"),
]


def ensure_cuda12_library_path() -> list[str]:
    """Prepend CUDA 12 lib dirs that contain libcublas.so.12. Returns added dirs."""
    added: list[str] = []
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in current.split(":") if p]
    for directory in _CANDIDATE_DIRS:
        if not any(directory.glob("libcublas.so.12*")):
            continue
        path = str(directory)
        if path not in parts:
            parts.insert(0, path)
            added.append(path)
    if added:
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
        logger.info("Added CUDA 12 library paths for Whisper: {}", added)

    # Mid-process LD_LIBRARY_PATH changes are unreliable; preload by absolute path.
    for directory in _CANDIDATE_DIRS:
        for candidate in sorted(directory.glob("libcublas.so.12*")):
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                logger.debug("Preloaded {}", candidate)
                return added
            except OSError:
                continue
    return added


def cublas12_available() -> bool:
    ensure_cuda12_library_path()
    try:
        ctypes.CDLL("libcublas.so.12")
        return True
    except OSError:
        for directory in _CANDIDATE_DIRS:
            for candidate in directory.glob("libcublas.so.12*"):
                try:
                    ctypes.CDLL(str(candidate))
                    return True
                except OSError:
                    continue
        return False


def resolve_whisper_device(preferred: str = "auto") -> tuple[str, str]:
    """Return (device, compute_type) for WhisperSTTService / faster-whisper."""
    preferred = (preferred or "auto").lower()
    if preferred == "cpu":
        return "cpu", "int8"
    if preferred in {"cuda", "auto"} and cublas12_available():
        return "cuda", "float16"
    if preferred == "cuda":
        logger.warning(
            "CUDA requested for Whisper but libcublas.so.12 is missing; using CPU"
        )
    return "cpu", "int8"
