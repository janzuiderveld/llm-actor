from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService

from app.config import RuntimeConfig
from services.macos_hear_stt import HearSTTOptions, MacosHearSTTService

# Backwards compatibility for legacy model identifiers.
_DEEPGRAM_MODEL_ALIASES = {
    "deepgram-flux": "flux-general-en",
    "flux-general": "flux-general-en",
}

STTProvider = Literal["deepgram_flux", "macos_hear"]
STTService = Union[DeepgramFluxSTTService, MacosHearSTTService]

_HEAR_MODEL_ALIASES = {
    "macos_hear",
    "macos-hear",
    "hear",
}


def parse_stt_model_spec(model: str) -> Tuple[STTProvider, str]:
    model = (model or "").strip()
    if model.lower() in _HEAR_MODEL_ALIASES:
        return "macos_hear", model
    return "deepgram_flux", model


def build_deepgram_flux_stt(config: RuntimeConfig, api_key: str) -> DeepgramFluxSTTService:
    params = DeepgramFluxSTTService.InputParams(
        eager_eot_threshold=config.stt.eager_eot_threshold,
        eot_threshold=config.stt.eot_threshold,
        eot_timeout_ms=config.stt.eot_timeout_ms,
    )
    model = _DEEPGRAM_MODEL_ALIASES.get(config.stt.model, config.stt.model)
    return DeepgramFluxSTTService(
        api_key=api_key,
        model=model,
        params=params,
    )


def build_macos_hear_stt(config: RuntimeConfig) -> MacosHearSTTService:
    keep_mic_open = bool(config.stt.hear_keep_mic_open)
    options = HearSTTOptions(
        locale=config.stt.language,
        on_device=config.stt.hear_on_device,
        punctuation=config.stt.hear_punctuation,
        input_device_id=config.stt.hear_input_device_id,
        final_silence_sec=config.stt.hear_final_silence_sec,
        restart_on_final=config.stt.hear_restart_on_final and not keep_mic_open,
        keep_mic_open=keep_mic_open,
    )
    return MacosHearSTTService(options=options)


def build_stt_service(config: RuntimeConfig, *, deepgram_api_key: Optional[str]) -> Tuple[STTService, STTProvider]:
    provider, _model = parse_stt_model_spec(config.stt.model)
    if provider == "macos_hear":
        return build_macos_hear_stt(config), provider

    if not deepgram_api_key:
        raise RuntimeError("DEEPGRAM_API_KEY must be set when using Deepgram Flux STT.")
    return build_deepgram_flux_stt(config, deepgram_api_key), provider
