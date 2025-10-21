from __future__ import annotations

from pipecat.services.google.llm import (
    GoogleContextAggregatorPair,
    GoogleLLMContext,
    GoogleLLMService,
)

from app.config import RuntimeConfig


def build_google_llm(config: RuntimeConfig, api_key: str) -> GoogleLLMService:
    params = GoogleLLMService.InputParams(
        max_tokens=config.llm.max_tokens,
        temperature=config.llm.temperature,
    )
    return GoogleLLMService(
        api_key=api_key,
        model=config.llm.model,
        params=params,
        system_instruction=config.llm.system_prompt,
    )


def create_google_context(llm_service: GoogleLLMService, history_messages: list[dict]) -> GoogleContextAggregatorPair:
    context = GoogleLLMContext(messages=history_messages)
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
