from __future__ import annotations

from types import SimpleNamespace

from openai import NOT_GIVEN

from app.config import RuntimeConfig
from app import pipeline as pipeline_module
from services import llm as llm_service


def test_build_openai_llm_uses_max_completion_tokens(monkeypatch) -> None:
    class DummyOpenAILLMService:
        class InputParams:
            def __init__(
                self,
                *,
                max_tokens="unset",
                max_completion_tokens="unset",
                temperature=None,
            ):
                self.max_tokens = max_tokens
                self.max_completion_tokens = max_completion_tokens
                self.temperature = temperature

        def __init__(self, *, api_key, model, params):
            self.api_key = api_key
            self.model = model
            self.params = params

    monkeypatch.setattr(llm_service, "OpenAILLMService", DummyOpenAILLMService)

    config = RuntimeConfig()
    config.llm.max_tokens = 777
    config.llm.temperature = 0.42

    service = llm_service.build_openai_llm(config, api_key="fake-key", model="gpt-5.2")

    assert service.params.max_completion_tokens == 777
    assert service.params.max_tokens == "unset"
    assert service.params.temperature == 0.42


class DummyHistory:
    def set_system_message(self, _text) -> None:
        return None


def _make_controller(model: str):
    controller = pipeline_module.VoicePipelineController.__new__(pipeline_module.VoicePipelineController)
    controller._config_manager = SimpleNamespace(config=SimpleNamespace(llm=SimpleNamespace(model=model)))
    controller._llm_service = SimpleNamespace(
        _settings={
            "max_tokens": NOT_GIVEN,
            "max_completion_tokens": NOT_GIVEN,
            "temperature": 0.5,
        }
    )
    controller._event_logger = SimpleNamespace(emit=lambda *_args, **_kwargs: None)
    controller._assistant_aggregator = None
    controller._user_aggregator = None
    controller._history = DummyHistory()
    controller._stt_service = None
    return controller


def test_apply_param_updates_openai_sets_max_completion_tokens() -> None:
    controller = _make_controller("openai-gpt-5.2-chat-latest")

    controller._apply_param_updates({"llm": {"max_tokens": 222}})

    params = controller._llm_service._settings
    assert params["max_completion_tokens"] == 222
    assert params["max_tokens"] is NOT_GIVEN


def test_apply_param_updates_ollama_keeps_max_tokens() -> None:
    controller = _make_controller("ollama-gemma3:4b")

    controller._apply_param_updates({"llm": {"max_tokens": 222}})

    params = controller._llm_service._settings
    assert params["max_tokens"] == 222
    assert params["max_completion_tokens"] is NOT_GIVEN
