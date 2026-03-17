from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llama_orchestrator.catalog import CatalogStore
from llama_orchestrator.hardware import HardwareProbe
from llama_orchestrator.http_api import (
    _anthropic_messages_to_openai_messages,
    _anthropic_tool_choice_to_openai,
    _apply_preset_defaults,
    _chat_completion_to_anthropic,
    _openai_stream_to_anthropic_events,
    _openai_stream_to_responses_events,
    create_app,
)
from llama_orchestrator.models import AliasDefinition, BaseModelDefinition, GenerationPreset, LoadProfile, ReasoningMode
from llama_orchestrator.router import Router
from llama_orchestrator.runtime import RuntimeManager
from llama_orchestrator.settings import AppSettings
from llama_orchestrator.state import StateStore


def _build_app(sandbox_path: Path) -> TestClient:
    settings = AppSettings(
        catalog_path=sandbox_path / "catalog.yaml",
        state_path=sandbox_path / "orchestrator.db",
    )
    settings.ensure_directories()
    catalog = CatalogStore(settings.catalog_path)
    catalog.load()
    catalog.upsert_model(
        BaseModelDefinition(
            id="demo-model",
            display_name="Demo Model",
            local_path=sandbox_path / "demo.gguf",
        )
    )
    catalog.upsert_profile(LoadProfile(id="balanced"))
    catalog.upsert_preset(GenerationPreset(id="default"))
    catalog.upsert_alias(
        AliasDefinition(
            id="demo/alias",
            base_model_id="demo-model",
            load_profile_id="balanced",
            preset_id="default",
        )
    )
    state = StateStore(settings.state_path)
    hardware_probe = HardwareProbe(settings)
    runtime_manager = RuntimeManager(settings, catalog, state, Router(settings))
    app = create_app(settings, catalog, hardware_probe, runtime_manager, state)
    return TestClient(app)


def test_list_models_returns_catalog_aliases(sandbox_path: Path) -> None:
    client = _build_app(sandbox_path)
    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"][0]["id"] == "demo/alias"


def test_legacy_completions_reject_tool_fields(sandbox_path: Path) -> None:
    client = _build_app(sandbox_path)
    response = client.post(
        "/v1/completions",
        json={
            "model": "demo/alias",
            "prompt": "hello",
            "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
        },
    )

    assert response.status_code == 400
    assert "does not support tool use" in response.json()["detail"]


def test_anthropic_message_translation_preserves_tool_results() -> None:
    translated = _anthropic_messages_to_openai_messages(
        [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tool_1", "name": "lookup", "input": {"q": "abc"}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "done"}],
            },
        ]
    )

    assert translated[0]["tool_calls"][0]["function"]["name"] == "lookup"
    assert translated[1]["role"] == "tool"
    assert translated[1]["tool_call_id"] == "tool_1"


def test_chat_completion_to_anthropic_maps_tool_calls() -> None:
    response = _chat_completion_to_anthropic(
        {
            "id": "chatcmpl_1",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "lookup", "arguments": "{\"q\": \"abc\"}"},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
        {"model": "demo/alias"},
    )

    assert response["stop_reason"] == "tool_use"
    assert response["content"][0]["type"] == "tool_use"
    assert response["content"][0]["input"]["q"] == "abc"


def test_anthropic_tool_choice_any_maps_to_openai_required() -> None:
    assert _anthropic_tool_choice_to_openai("any") == "required"


def test_preset_defaults_add_qwen_no_think_directive(sandbox_path: Path) -> None:
    settings = AppSettings(
        catalog_path=sandbox_path / "catalog.yaml",
        state_path=sandbox_path / "orchestrator.db",
    )
    settings.ensure_directories()
    catalog = CatalogStore(settings.catalog_path)
    catalog.load()
    catalog.upsert_model(
        BaseModelDefinition(
            id="qwen",
            display_name="Qwen",
            local_path=sandbox_path / "demo.gguf",
            family="qwen",
        )
    )
    catalog.upsert_profile(LoadProfile(id="balanced"))
    catalog.upsert_preset(GenerationPreset(id="precise", reasoning_mode=ReasoningMode.OFF, temperature=0.1))
    catalog.upsert_alias(
        AliasDefinition(
            id="qwen/alias",
            base_model_id="qwen",
            load_profile_id="balanced",
            preset_id="precise",
        )
    )

    payload = _apply_preset_defaults(
        catalog,
        "qwen/alias",
        {"model": "qwen/alias", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert payload["temperature"] == 0.1
    assert payload["messages"][0]["role"] == "system"
    assert "/no_think" in payload["messages"][0]["content"]


def test_preset_defaults_use_enable_thinking_for_qwen35(sandbox_path: Path) -> None:
    settings = AppSettings(
        catalog_path=sandbox_path / "catalog.yaml",
        state_path=sandbox_path / "orchestrator.db",
    )
    settings.ensure_directories()
    catalog = CatalogStore(settings.catalog_path)
    catalog.load()
    catalog.upsert_model(
        BaseModelDefinition(
            id="qwen35",
            display_name="Qwen3.5",
            local_path=sandbox_path / "demo.gguf",
            family="qwen3.5",
        )
    )
    catalog.upsert_profile(LoadProfile(id="balanced"))
    catalog.upsert_preset(GenerationPreset(id="precise", reasoning_mode=ReasoningMode.OFF))
    catalog.upsert_alias(
        AliasDefinition(
            id="qwen35/alias",
            base_model_id="qwen35",
            load_profile_id="balanced",
            preset_id="precise",
        )
    )

    payload = _apply_preset_defaults(
        catalog,
        "qwen35/alias",
        {"model": "qwen35/alias", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert payload["enable_thinking"] is False
    assert payload["chat_template_kwargs"]["enable_thinking"] is False
    assert payload["messages"][0]["role"] == "user"


@pytest.mark.anyio
async def test_openai_stream_is_translated_to_anthropic_events() -> None:
    async def fake_lines():
        yield 'data: {"id":"chatcmpl_1","choices":[{"delta":{"content":"Hello"}}]}'
        yield 'data: {"id":"chatcmpl_1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"lookup","arguments":"{\\"q\\": "}},{"index":0,"function":{"arguments":"\\"abc\\"}"}}]}}],"usage":{"prompt_tokens":3,"completion_tokens":2}}'
        yield 'data: {"id":"chatcmpl_1","choices":[{"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}'
        yield "data: [DONE]"

    events = [event async for event in _openai_stream_to_anthropic_events(fake_lines(), {"model": "demo/alias"})]

    assert events[0].startswith("event: message_start")
    assert any("event: content_block_start" in event and '"type": "text"' in event for event in events)
    assert any("event: content_block_delta" in event and '"text_delta"' in event for event in events)
    assert any("event: content_block_delta" in event and '"input_json_delta"' in event for event in events)
    assert any("event: message_delta" in event and '"stop_reason": "tool_use"' in event for event in events)
    assert events[-1].startswith("event: message_stop")


def test_chat_completion_to_response_includes_completed_status() -> None:
    from llama_orchestrator.http_api import _chat_completion_to_response

    response = _chat_completion_to_response(
        {
            "id": "chatcmpl_1",
            "model": "demo/alias",
            "choices": [{"message": {"content": "hello", "tool_calls": []}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )

    assert response["status"] == "completed"
    assert response["output"][0]["type"] == "output_text"


@pytest.mark.anyio
async def test_openai_stream_is_translated_to_responses_events() -> None:
    async def fake_lines():
        yield 'data: {"id":"chatcmpl_1","choices":[{"delta":{"content":"Hello"}}]}'
        yield 'data: {"id":"chatcmpl_1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"lookup","arguments":"{\\"q\\": "}},{"index":0,"function":{"arguments":"\\"abc\\"}"}}]}}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}'
        yield 'data: {"id":"chatcmpl_1","choices":[{"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}'
        yield "data: [DONE]"

    events = [event async for event in _openai_stream_to_responses_events(fake_lines(), {"model": "demo/alias"})]

    assert events[0].startswith("event: response.created")
    assert any("event: response.output_item.added" in event and '"type": "message"' in event for event in events)
    assert any("event: response.output_text.delta" in event and '"delta": "Hello"' in event for event in events)
    assert any("event: response.function_call_arguments.delta" in event for event in events)
    assert any("event: response.function_call_arguments.done" in event for event in events)
    assert events[-1].startswith("event: response.completed")
