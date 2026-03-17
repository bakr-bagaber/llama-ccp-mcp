"""Helpers for recording and querying benchmark data."""

from __future__ import annotations

import json
import subprocess

from .catalog import CatalogStore
from .models import Backend, BenchmarkRecord, PlacementKind
from .settings import AppSettings
from .state import StateStore


class BenchmarkService:
    """Small service layer around the state store.

    Keeping benchmark logic in one place makes it easier to swap manual
    records for external `llama-bench` integration later.
    """

    def __init__(self, settings: AppSettings, catalog: CatalogStore, state: StateStore) -> None:
        self.settings = settings
        self.catalog = catalog
        self.state = state

    def record_manual_benchmark(
        self,
        *,
        alias_id: str,
        backend: Backend,
        placement: PlacementKind,
        prompt_tps: float,
        generation_tps: float,
        load_seconds: float = 0.0,
        peak_ram_bytes: int | None = None,
        peak_vram_bytes: int | None = None,
        metadata: dict | None = None,
    ) -> BenchmarkRecord:
        record = BenchmarkRecord(
            alias_id=alias_id,
            backend=backend,
            placement=placement,
            prompt_tps=prompt_tps,
            generation_tps=generation_tps,
            load_seconds=load_seconds,
            peak_ram_bytes=peak_ram_bytes,
            peak_vram_bytes=peak_vram_bytes,
            metadata=metadata or {"source": "manual"},
        )
        self.state.add_benchmark(record)
        return record

    def run_llama_bench(
        self,
        *,
        alias_id: str,
        backend: Backend,
        n_gpu_layers: int | None = None,
    ) -> BenchmarkRecord:
        """Run a real llama-bench process and store the parsed result."""
        executable = self.settings.bench_executable_for_backend(backend)
        if not executable:
            raise RuntimeError(f"No llama-bench executable is available for backend '{backend.value}'.")

        _alias, model, profile, _preset = self.catalog.resolve_alias(alias_id)
        if not model.local_path:
            raise RuntimeError(f"Alias '{alias_id}' does not point to a local model file.")

        command = [
            executable,
            "--model",
            str(model.local_path),
            "--output",
            "json",
        ]
        if n_gpu_layers is not None:
            command.extend(["--n-gpu-layers", str(n_gpu_layers)])
        elif backend is not Backend.CPU and profile.gpu_layers is not None:
            command.extend(["--n-gpu-layers", str(profile.gpu_layers)])

        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
        payload = json.loads(result.stdout)
        entries = payload if isinstance(payload, list) else [payload]
        prompt_entry = next((item for item in entries if int(item.get("n_prompt", 0)) > 0), {})
        generation_entry = next((item for item in entries if int(item.get("n_gen", 0)) > 0), {})

        placement = PlacementKind.CPU_ONLY
        if backend is Backend.CUDA:
            placement = PlacementKind.CPU_DGPU_HYBRID if (n_gpu_layers or profile.gpu_layers or 0) > 0 else PlacementKind.DGPU_ONLY
        elif backend is Backend.VULKAN:
            placement = PlacementKind.IGPU_ONLY if "intel" in (model.family or "").lower() else PlacementKind.DGPU_ONLY

        prompt_tps = float(prompt_entry.get("avg_ts") or 0.0)
        generation_tps = float(generation_entry.get("avg_ts") or 0.0)
        load_seconds = 0.0

        record = BenchmarkRecord(
            alias_id=alias_id,
            backend=backend,
            placement=placement,
            prompt_tps=prompt_tps,
            generation_tps=generation_tps,
            load_seconds=load_seconds,
            metadata={"source": "llama-bench", "raw": entries},
        )
        self.state.add_benchmark(record)
        return record
