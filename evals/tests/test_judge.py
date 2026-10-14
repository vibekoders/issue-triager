"""The judge's model addresses: which pydantic-ai model and settings each one gets (no network)."""

from __future__ import annotations

import pytest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openai import OpenAIResponsesModel

from triager_evals.judge import judge_model_settings, make_judge_model


@pytest.fixture(autouse=True)
def placeholder_keys(monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "AZURE_OPENAI_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-placeholder")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-placeholder")
    monkeypatch.setenv("AZURE_OPENAI_RESOURCE", "myresource")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-placeholder")


@pytest.mark.parametrize(
    ("address", "model_class", "name"),
    [
        ("openai:gpt-6-sol", OpenAIResponsesModel, "gpt-6-sol"),
        ("azure:gpt-6-sol", OpenAIResponsesModel, "gpt-6-sol"),
        ("anthropic:claude-sonnet-5", AnthropicModel, "claude-sonnet-5"),
        ("gemini:gemini-3.1-pro-preview", GoogleModel, "gemini-3.1-pro-preview"),
        ("ollama:qwen3:4b", OllamaModel, "qwen3:4b"),
    ],
)
def test_judge_model_mapping(address, model_class, name):
    model = make_judge_model(address)
    assert model.address == address
    assert type(model.wrapped) is model_class
    assert model.wrapped.model_name == name


def test_azure_uses_the_v1_api_of_the_resource():
    model = make_judge_model("azure:gpt-6-luna").wrapped
    assert str(model.client.base_url).rstrip("/") == "https://myresource.openai.azure.com/openai/v1"


def test_openai_goes_to_openai():
    model = make_judge_model("openai:gpt-6-luna").wrapped
    assert str(model.client.base_url).startswith("https://api.openai.com/")


def test_gemini_reads_the_service_key_first(monkeypatch):
    """The Rust service reads GEMINI_API_KEY before GOOGLE_API_KEY; pydantic-ai the other way round."""
    monkeypatch.setenv("GOOGLE_API_KEY", "google-placeholder")
    model = make_judge_model("gemini:gemini-3.5-flash-lite").wrapped
    assert model.client._api_client.api_key == "gemini-placeholder"
    monkeypatch.delenv("GEMINI_API_KEY")
    model = make_judge_model("gemini:gemini-3.5-flash-lite").wrapped
    assert model.client._api_client.api_key == "google-placeholder"


def test_other_prefixes_go_to_pydantic_ai():
    from pydantic_ai.models.openai import OpenAIChatModel

    model = make_judge_model("openai-chat:gpt-4.1-mini")
    assert type(model.wrapped) is OpenAIChatModel and model.wrapped.model_name == "gpt-4.1-mini"


@pytest.mark.parametrize(
    ("address", "settings"),
    [
        ("openai:gpt-6-sol", {}),
        ("azure:gpt-6-sol", {}),
        ("gemini:gemini-3.1-pro-preview", {}),
        ("anthropic:claude-sonnet-5", {"temperature": 0.0}),
        ("ollama:qwen3:4b", {"temperature": 0.0, "openai_reasoning_effort": "none"}),
    ],
)
def test_judge_model_settings(address, settings):
    assert judge_model_settings(address) == settings
