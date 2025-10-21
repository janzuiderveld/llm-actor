from __future__ import annotations

from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService

from app.config import RuntimeConfig


def build_deepgram_flux_stt(config: RuntimeConfig, api_key: str) -> DeepgramFluxSTTService:
    params = DeepgramFluxSTTService.InputParams(
        eager_eot_threshold=config.stt.eager_eot_threshold,
        eot_threshold=config.stt.eot_threshold,
        eot_timeout_ms=config.stt.eot_timeout_ms,
    )
    return DeepgramFluxSTTService(
        api_key=api_key,
        model=config.stt.model,
        params=params,
    )
