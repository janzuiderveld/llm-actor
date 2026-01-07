from __future__ import annotations

from typing import Literal, Optional, Tuple, Union, cast

from pipecat.services.google.llm import (
    GoogleContextAggregatorPair,
    GoogleLLMContext,
    GoogleLLMService,
)
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.openai.llm import OpenAIContextAggregatorPair, OpenAILLMContext, OpenAILLMService

from app.config import RuntimeConfig

LLMProvider = Literal["google", "ollama", "openai"]
LLMService = Union[GoogleLLMService, OpenAILLMService]
LLMContextPair = Union[GoogleContextAggregatorPair, OpenAIContextAggregatorPair]

OLLAMA_MODEL_PREFIX = "ollama-"
OPENAI_MODEL_PREFIX = "openai-"


def parse_llm_model_spec(model: str) -> Tuple[LLMProvider, str]:
    model = (model or "").strip()
    lowered = model.lower()
    if lowered.startswith(OLLAMA_MODEL_PREFIX):
        return "ollama", model[len(OLLAMA_MODEL_PREFIX) :].strip()
    if lowered.startswith(OPENAI_MODEL_PREFIX):
        return "openai", model[len(OPENAI_MODEL_PREFIX) :].strip()
    return "google", model


def _coerce_request_timeout(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


def _build_google_http_options(timeout_s: Optional[float]) -> Optional[object]:
    timeout = _coerce_request_timeout(timeout_s)
    if timeout is None:
        return None
    from google.genai.types import HttpOptions

    timeout_ms = max(1, int(timeout * 1000))
    return HttpOptions(timeout=timeout_ms)


def _build_thinking_config(thinking_level: Optional[str]) -> Optional[dict]:
    if not isinstance(thinking_level, str):
        return None
    normalized = thinking_level.strip()
    if not normalized:
        return None
    normalized_upper = normalized.upper()
    try:
        from google.genai import types as genai_types
    except Exception:
        return {"thinking_level": normalized_upper}

    fields = getattr(genai_types.ThinkingConfig, "model_fields", {}) or {}
    if "thinking_level" in fields:
        return {"thinking_level": normalized_upper}
    if "thinking_budget" in fields:
        if normalized_upper in {"MINIMAL", "NONE", "DISABLED"}:
            return {"thinking_budget": 0}
        if normalized_upper in {"AUTO", "AUTOMATIC"}:
            return {"thinking_budget": -1}
        if normalized_upper.lstrip("-").isdigit():
            return {"thinking_budget": int(normalized_upper)}
    return None


def build_google_llm(config: RuntimeConfig, api_key: str) -> GoogleLLMService:
    extra: dict = {}
    thinking_config = _build_thinking_config(config.llm.thinking_level)
    if thinking_config:
        extra["thinking_config"] = thinking_config
    params = GoogleLLMService.InputParams(
        max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature,
        extra=extra or None,
    )
    http_options = _build_google_http_options(config.llm.request_timeout_s)
    return GoogleLLMService(
        api_key=api_key,
        model=config.llm.model,
        params=params,
        system_instruction=config.llm.system_prompt,
        http_options=http_options,
    )


def create_google_context(llm_service: GoogleLLMService, history_messages: list[dict]) -> GoogleContextAggregatorPair:
    context = GoogleLLMContext()
    if history_messages:
        context.set_messages(history_messages)
    system_instruction = getattr(llm_service, "_system_instruction", None)
    if system_instruction:
        context.system_message = system_instruction  # type: ignore[attr-defined]

    create_context = getattr(llm_service, "create_context_aggregator", None)
    if callable(create_context):
        return create_context(context)

    legacy_create_context = getattr(llm_service, "create_context_aggregators", None)
    if callable(legacy_create_context):
        return legacy_create_context(context)

    raise AttributeError("GoogleLLMService does not support context aggregator creation")


def build_ollama_llm(config: RuntimeConfig, *, model: str, base_url: str = "http://localhost:11434/v1") -> OLLamaLLMService:
    params = OpenAILLMService.InputParams(
        max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature,
    )
    return OLLamaLLMService(
        model=model,
        base_url=base_url,
        params=params,
    )


def create_openai_context(llm_service: OpenAILLMService, history_messages: list[dict]) -> OpenAIContextAggregatorPair:
    context = OpenAILLMContext(messages=[])
    if history_messages:
        context.set_messages(cast(list, history_messages))

    create_context = getattr(llm_service, "create_context_aggregator", None)
    if callable(create_context):
        return create_context(context)

    legacy_create_context = getattr(llm_service, "create_context_aggregators", None)
    if callable(legacy_create_context):
        return legacy_create_context(context)

    raise AttributeError("OpenAILLMService does not support context aggregator creation")


def build_openai_llm(config: RuntimeConfig, *, api_key: str, model: str) -> OpenAILLMService:
    params = OpenAILLMService.InputParams(
        max_completion_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature,
    )
    return OpenAILLMService(
        api_key=api_key,
        model=model,
        params=params,
    )


def build_llm_service(
    config: RuntimeConfig,
    *,
    google_api_key: Optional[str],
    openai_api_key: Optional[str],
) -> Tuple[LLMService, LLMProvider]:
    provider, model_name = parse_llm_model_spec(config.llm.model)
    if provider == "ollama":
        if not model_name:
            raise ValueError("Ollama model id missing; set `llm.model` to `ollama-{MODEL_ID}`.")
        return build_ollama_llm(config, model=model_name), provider
    if provider == "openai":
        if not model_name:
            raise ValueError("OpenAI model id missing; set `llm.model` to `openai-{MODEL_ID}`.")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY must be set when using OpenAI models.")
        return build_openai_llm(config, api_key=openai_api_key, model=model_name), provider

    if not google_api_key:
        raise RuntimeError("GOOGLE_API_KEY must be set when using Gemini models.")
    return build_google_llm(config, google_api_key), provider


def create_llm_context(llm_service: LLMService, history_messages: list[dict]) -> LLMContextPair:
    if isinstance(llm_service, GoogleLLMService):
        return create_google_context(llm_service, history_messages)

    return create_openai_context(llm_service, history_messages)


def normalize_model_for_service(model: str, llm_service: LLMService) -> Optional[str]:
    provider, model_name = parse_llm_model_spec(model)
    if isinstance(llm_service, OpenAILLMService):
        if provider in {"ollama", "openai"}:
            return model_name
        return model
    if provider != "google":
        return None
    return model
