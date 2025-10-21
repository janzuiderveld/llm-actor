from __future__ import annotations

from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService

from app.config import RuntimeConfig

# Backwards compatibility for legacy model identifiers.
_DEEPGRAM_MODEL_ALIASES = {
    "deepgram-flux": "flux-general-en",
    "flux-general": "flux-general-en",
}


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
