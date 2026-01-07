from __future__ import annotations

from google.genai import types as genai_types

from app.config import RuntimeConfig
from services import llm as llm_service


def test_build_google_llm_applies_thinking_level(monkeypatch) -> None:
    class DummyGoogleLLMService:
        class InputParams:
            def __init__(self, max_tokens=None, temperature=None, extra=None):
                self.max_tokens = max_tokens
                self.temperature = temperature
                self.extra = extra or {}

        def __init__(self, *, api_key, model, params, system_instruction, http_options):
            self.api_key = api_key
            self.model = model
            self.params = params
            self.system_instruction = system_instruction
            self.http_options = http_options

    monkeypatch.setattr(llm_service, "GoogleLLMService", DummyGoogleLLMService)
    monkeypatch.setattr(llm_service, "_build_google_http_options", lambda _timeout: None)

    config = RuntimeConfig()
    config.llm.thinking_level = "MINIMAL"

    service = llm_service.build_google_llm(config, api_key="fake-key")

    fields = getattr(genai_types.ThinkingConfig, "model_fields", {}) or {}
    if "thinking_level" in fields:
        expected_extra = {"thinking_config": {"thinking_level": "MINIMAL"}}
    elif "thinking_budget" in fields:
        expected_extra = {"thinking_config": {"thinking_budget": 0}}
    else:
        expected_extra = {}

    assert service.params.extra == expected_extra
