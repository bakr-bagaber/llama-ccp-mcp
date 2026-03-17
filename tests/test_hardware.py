from __future__ import annotations

from llama_orchestrator.hardware import HardwareProbe


def test_normalize_vulkan_name_strips_memory_suffix() -> None:
    name = "Intel(R) Arc(TM) Graphics (18361 MiB, 17593 MiB free)"

    normalized = HardwareProbe._normalize_vulkan_name(name)

    assert normalized == "Intel(R) Arc(TM) Graphics"
