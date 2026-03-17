from __future__ import annotations

from pathlib import Path

import pytest

from llama_orchestrator.catalog import CatalogError
from llama_orchestrator.catalog import CatalogStore
from llama_orchestrator.models import AliasDefinition, BaseModelDefinition, GenerationPreset, LoadProfile


def test_catalog_upsert_and_resolve_alias(sandbox_path: Path) -> None:
    catalog_path = sandbox_path / "catalog.yaml"
    store = CatalogStore(catalog_path)
    store.load()

    store.upsert_model(
        BaseModelDefinition(
            id="qwen-coder-7b",
            display_name="Qwen Coder 7B",
            local_path=sandbox_path / "qwen-coder.gguf",
        )
    )
    store.upsert_profile(LoadProfile(id="balanced", context_size=8192))
    store.upsert_preset(GenerationPreset(id="precise", temperature=0.1, max_tokens=512))
    store.upsert_alias(
        AliasDefinition(
            id="qwen-coder/precise",
            base_model_id="qwen-coder-7b",
            load_profile_id="balanced",
            preset_id="precise",
        )
    )

    alias, model, profile, preset = store.resolve_alias("qwen-coder/precise")

    assert alias.id == "qwen-coder/precise"
    assert model.id == "qwen-coder-7b"
    assert profile.id == "balanced"
    assert preset.id == "precise"


def test_catalog_prevents_deleting_model_in_use(sandbox_path: Path) -> None:
    catalog_path = sandbox_path / "catalog.yaml"
    store = CatalogStore(catalog_path)
    store.load()
    store.upsert_model(
        BaseModelDefinition(
            id="demo-model",
            display_name="Demo",
            local_path=sandbox_path / "demo.gguf",
        )
    )
    store.upsert_profile(LoadProfile(id="balanced"))
    store.upsert_preset(GenerationPreset(id="default"))
    store.upsert_alias(
        AliasDefinition(
            id="demo/alias",
            base_model_id="demo-model",
            load_profile_id="balanced",
            preset_id="default",
        )
    )

    with pytest.raises(CatalogError):
        store.delete_model("demo-model")
