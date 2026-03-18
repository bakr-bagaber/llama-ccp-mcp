"""Application settings."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field

from .models import Backend, MemoryPolicy, StrictModel


class AppSettings(StrictModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    api_key: str | None = None
    catalog_path: Path = Path("catalog/catalog.yaml")
    state_path: Path = Path("state/mcp.db")
    idle_scan_interval_seconds: int = Field(default=30, ge=1)
    runtime_start_timeout_seconds: int = Field(default=20, ge=1)
    http_timeout_seconds: float = Field(default=120.0, gt=0)
    default_idle_unload_seconds: int = Field(default=900, ge=1)
    cpu_executable: str | None = None
    cuda_executable: str | None = None
    vulkan_executable: str | None = None
    sycl_executable: str | None = None
    cpu_bench_executable: str | None = None
    cuda_bench_executable: str | None = None
    vulkan_bench_executable: str | None = None
    sycl_bench_executable: str | None = None
    policy: MemoryPolicy = Field(default_factory=MemoryPolicy)

    @classmethod
    def load(cls) -> "AppSettings":
        return cls(
            host=os.getenv("LLAMA_MCP_HOST", "127.0.0.1"),
            port=int(os.getenv("LLAMA_MCP_PORT", "8080")),
            api_key=os.getenv("LLAMA_MCP_API_KEY") or None,
            catalog_path=Path(os.getenv("LLAMA_MCP_CATALOG_PATH", "catalog/catalog.yaml")),
            state_path=Path(os.getenv("LLAMA_MCP_STATE_PATH", "state/mcp.db")),
            idle_scan_interval_seconds=int(os.getenv("LLAMA_MCP_IDLE_SCAN_SECONDS", "30")),
            runtime_start_timeout_seconds=int(os.getenv("LLAMA_MCP_RUNTIME_START_TIMEOUT", "20")),
            http_timeout_seconds=float(os.getenv("LLAMA_MCP_HTTP_TIMEOUT", "120")),
            default_idle_unload_seconds=int(os.getenv("LLAMA_MCP_DEFAULT_IDLE_UNLOAD", "900")),
            cpu_executable=os.getenv("LLAMA_SERVER_CPU") or None,
            cuda_executable=os.getenv("LLAMA_SERVER_CUDA") or None,
            vulkan_executable=os.getenv("LLAMA_SERVER_VULKAN") or None,
            sycl_executable=os.getenv("LLAMA_SERVER_SYCL") or None,
            cpu_bench_executable=os.getenv("LLAMA_BENCH_CPU") or None,
            cuda_bench_executable=os.getenv("LLAMA_BENCH_CUDA") or None,
            vulkan_bench_executable=os.getenv("LLAMA_BENCH_VULKAN") or None,
            sycl_bench_executable=os.getenv("LLAMA_BENCH_SYCL") or None,
            policy=MemoryPolicy(
                min_free_system_ram_bytes=int(os.getenv("LLAMA_MCP_MIN_FREE_RAM", str(4 * 1024**3))),
                min_free_dgpu_vram_bytes=int(os.getenv("LLAMA_MCP_MIN_FREE_DGPU_VRAM", str(1 * 1024**3))),
                min_free_igpu_shared_ram_bytes=int(os.getenv("LLAMA_MCP_MIN_FREE_IGPU_RAM", str(2 * 1024**3))),
                max_loaded_instances=int(os.getenv("LLAMA_MCP_MAX_LOADED", "4")),
                max_concurrent_requests_per_runtime=int(os.getenv("LLAMA_MCP_MAX_CONCURRENCY", "4")),
                allow_experimental_igpu=os.getenv("LLAMA_MCP_ALLOW_EXPERIMENTAL_IGPU", "").lower() in {"1", "true", "yes"},
                allow_experimental_mixed_gpu=os.getenv("LLAMA_MCP_ALLOW_EXPERIMENTAL_MIXED", "").lower() in {"1", "true", "yes"},
            ),
        )

    def ensure_directories(self) -> None:
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def executable_for_backend(self, backend: Backend) -> str | None:
        """Resolve a backend executable.

        We prefer explicit environment overrides first. If none are set,
        we fall back to common local Windows llama.cpp install locations
        so the project works out of the box on this machine.
        """
        mapping = {
            Backend.CPU: self.cpu_executable,
            Backend.CUDA: self.cuda_executable,
            Backend.VULKAN: self.vulkan_executable,
            Backend.SYCL: self.sycl_executable,
        }
        explicit = mapping[backend]
        if explicit:
            return explicit

        defaults = {
            Backend.CPU: [Path(r"C:\llama.cpp\cpu\llama-server.exe")],
            Backend.CUDA: [Path(r"C:\llama.cpp\cuda13\llama-server.exe"), Path(r"C:\llama.cpp\cuda\llama-server.exe")],
            Backend.VULKAN: [Path(r"C:\llama.cpp\vulkan\llama-server.exe")],
            Backend.SYCL: [Path(r"C:\llama.cpp\sycl\llama-server.exe")],
        }
        for candidate in defaults[backend]:
            if candidate.exists():
                return str(candidate)
        return None

    def bench_executable_for_backend(self, backend: Backend) -> str | None:
        mapping = {
            Backend.CPU: self.cpu_bench_executable,
            Backend.CUDA: self.cuda_bench_executable,
            Backend.VULKAN: self.vulkan_bench_executable,
            Backend.SYCL: self.sycl_bench_executable,
        }
        explicit = mapping[backend]
        if explicit:
            return explicit

        defaults = {
            Backend.CPU: [Path(r"C:\llama.cpp\cpu\llama-bench.exe")],
            Backend.CUDA: [Path(r"C:\llama.cpp\cuda13\llama-bench.exe"), Path(r"C:\llama.cpp\cuda\llama-bench.exe")],
            Backend.VULKAN: [Path(r"C:\llama.cpp\vulkan\llama-bench.exe")],
            Backend.SYCL: [Path(r"C:\llama.cpp\sycl\llama-bench.exe")],
        }
        for candidate in defaults[backend]:
            if candidate.exists():
                return str(candidate)
        return None
