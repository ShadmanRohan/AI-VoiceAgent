"""Hardware and dependency detection for performance profiling."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx
import psutil

from app.config import PROFILE_PRESETS, PerformanceProfile, Settings


@dataclass
class GPUInfo:
    available: bool = False
    name: str | None = None
    vram_total_mb: int | None = None
    vram_free_mb: int | None = None
    cuda_available: bool = False
    cuda_version: str | None = None


@dataclass
class OllamaStatus:
    installed: bool = False
    running: bool = False
    version: str | None = None
    models: list[str] = field(default_factory=list)
    base_url: str = "http://127.0.0.1:11434"
    error: str | None = None


@dataclass
class HardwareReport:
    os_name: str
    os_version: str
    architecture: str
    cpu_model: str
    cpu_count: int
    ram_total_gb: float
    ram_available_gb: float
    gpu: GPUInfo
    ollama: OllamaStatus
    missing_dependencies: list[str] = field(default_factory=list)
    recommended_profile: str = PerformanceProfile.BALANCED.value
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cpu_model() -> str:
    if platform.system() == "Linux":
        try:
            for line in open("/proc/cpuinfo", encoding="utf-8"):
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def detect_gpu() -> GPUInfo:
    info = GPUInfo()
    try:
        import torch

        info.cuda_available = bool(torch.cuda.is_available())
        if info.cuda_available:
            info.available = True
            info.name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info.vram_total_mb = int(props.total_memory / (1024 * 1024))
            info.vram_free_mb = int(
                torch.cuda.mem_get_info(0)[0] / (1024 * 1024)
            )
            info.cuda_version = getattr(torch.version, "cuda", None)
            return info
    except Exception:
        pass

    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            ).strip()
            if out:
                parts = [p.strip() for p in out.split(",")[0:4]]
                # Handle multi-line / first GPU only
                first = out.splitlines()[0]
                parts = [p.strip() for p in first.split(",")]
                info.available = True
                info.name = parts[0]
                info.vram_total_mb = int(float(parts[1])) if len(parts) > 1 else None
                info.vram_free_mb = int(float(parts[2])) if len(parts) > 2 else None
                info.cuda_available = True
        except (subprocess.SubprocessError, ValueError, OSError):
            pass
    return info


async def detect_ollama(base_url: str) -> OllamaStatus:
    status = OllamaStatus(base_url=base_url)
    binary = shutil.which("ollama")
    status.installed = binary is not None
    if binary:
        try:
            status.version = subprocess.check_output(
                [binary, "--version"], text=True, timeout=5
            ).strip()
        except (subprocess.SubprocessError, OSError):
            status.version = None

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            tags = await client.get(f"{base_url.rstrip('/')}/api/tags")
            if tags.status_code == 200:
                status.running = True
                payload = tags.json()
                status.models = [m.get("name", "") for m in payload.get("models", [])]
            else:
                status.error = f"Ollama returned HTTP {tags.status_code}"
    except Exception as exc:  # noqa: BLE001
        status.error = str(exc)
    return status


def recommend_profile(ram_gb: float, gpu: GPUInfo) -> str:
    if ram_gb < 8 and not gpu.available:
        return PerformanceProfile.LOW.value
    if gpu.available and (gpu.vram_total_mb or 0) >= 12000 and ram_gb >= 16:
        return PerformanceProfile.QUALITY.value
    if ram_gb >= 12 or gpu.available:
        return PerformanceProfile.BALANCED.value
    return PerformanceProfile.LOW.value


async def build_hardware_report(settings: Settings) -> HardwareReport:
    vm = psutil.virtual_memory()
    ram_total = vm.total / (1024**3)
    ram_avail = vm.available / (1024**3)
    gpu = detect_gpu()
    ollama = await detect_ollama(settings.ollama_base_url)

    missing: list[str] = []
    notes: list[str] = []

    if not ollama.installed:
        missing.append("ollama (binary not found on PATH)")
    if not ollama.running:
        missing.append("ollama server (not reachable)")
    else:
        needed = {settings.ollama_model}
        if settings.performance_profile == PerformanceProfile.QUALITY:
            needed.add(settings.ollama_model_quality)
        for model in needed:
            if not any(model in m or m.startswith(model.split(":")[0]) for m in ollama.models):
                # Exact tag check
                if model not in ollama.models and not any(
                    m == model or m.startswith(f"{model}-") for m in ollama.models
                ):
                    if model not in ollama.models:
                        notes.append(f"Ollama model not downloaded yet: {model}")

    if not shutil.which("ffmpeg"):
        notes.append("ffmpeg not found (optional for some audio utilities)")

    if ram_avail < 2.0:
        notes.append("Low available RAM; prefer the LOW performance profile.")

    recommended = recommend_profile(ram_total, gpu)

    return HardwareReport(
        os_name=platform.system(),
        os_version=platform.release(),
        architecture=platform.machine(),
        cpu_model=_cpu_model(),
        cpu_count=os.cpu_count() or 1,
        ram_total_gb=round(ram_total, 2),
        ram_available_gb=round(ram_avail, 2),
        gpu=gpu,
        ollama=ollama,
        missing_dependencies=missing,
        recommended_profile=recommended,
        notes=notes,
    )


def resolve_profile(settings: Settings, report: HardwareReport) -> dict[str, Any]:
    profile = settings.performance_profile.value
    if profile == PerformanceProfile.AUTO.value:
        profile = report.recommended_profile
    presets = dict(
        PROFILE_PRESETS.get(profile, PROFILE_PRESETS[PerformanceProfile.BALANCED.value])
    )
    # If the profile wants a model that is not downloaded, fall back safely.
    wanted = presets.get("ollama_model")
    if wanted and report.ollama.running and report.ollama.models:
        if wanted not in report.ollama.models and not any(
            m == wanted or m.startswith(f"{wanted}-") for m in report.ollama.models
        ):
            if settings.ollama_model in report.ollama.models:
                presets["ollama_model"] = settings.ollama_model
            elif "qwen3:1.7b" in report.ollama.models:
                presets["ollama_model"] = "qwen3:1.7b"
            else:
                presets["ollama_model"] = report.ollama.models[0]
            report.notes.append(
                f"Requested model {wanted} not downloaded; using {presets['ollama_model']}"
            )
    return {"profile": profile, **presets}
