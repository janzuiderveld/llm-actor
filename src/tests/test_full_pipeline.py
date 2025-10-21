from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.config import ConfigManager
from app.history import ConversationHistory
from app.logging_io import EventLogger
from app.metrics import MetricsTracker

try:
    from app.pipeline import VoicePipelineController
    IMPORT_ERROR = None
except Exception as exc:
    VoicePipelineController = None
    IMPORT_ERROR = exc
from app.session import new_session


@pytest.mark.asyncio
async def test_pipeline_construction_smoke():
    if IMPORT_ERROR is not None:
        pytest.skip(f"Local audio transport unavailable: {IMPORT_ERROR}")
    if not (os.getenv("DEEPGRAM_API_KEY") and os.getenv("GOOGLE_API_KEY")):
        pytest.skip("API keys not configured")
    if os.getenv("RUN_PIPELINE_TESTS") != "1":
        pytest.skip("Set RUN_PIPELINE_TESTS=1 to run integration pipeline smoke test")

    config_manager = ConfigManager()
    config_manager.set_audio_devices(0, 0)
    session_paths = new_session(config_manager.config, session_name="test-pipeline")
    event_logger = EventLogger(session_paths.event_log)
    metrics = MetricsTracker(event_logger)
    history = ConversationHistory(
        session_paths.transcript,
        clean_transcript_path=session_paths.llm_transcript,
        context_path=session_paths.llm_context,
    )

    controller = VoicePipelineController(
        config_manager,
        session_paths,
        history,
        event_logger,
        metrics,
        actions_path=session_paths.session_dir.parent.parent / "actions.txt",
        inbox_path=session_paths.session_dir.parent.parent / "inbox.txt",
        params_path=session_paths.session_dir.parent.parent / "params_inbox.ndjson",
    )

    run_task = asyncio.create_task(controller.start())
    await asyncio.sleep(1)
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task
