from __future__ import annotations

import asyncio
import platform
import shutil
import signal
from dataclasses import dataclass
from typing import Optional, Sequence

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601


@dataclass(frozen=True)
class HearSTTOptions:
    locale: str = "en-US"
    on_device: bool = True
    punctuation: bool = True
    input_device_id: Optional[int] = None
    final_silence_sec: float = 1.2
    restart_on_final: bool = True
    keep_mic_open: bool = False


class MacosHearSTTService(FrameProcessor):
    """macOS `hear`-backed STT service.

    `hear` continuously prints a full hypothesis each time the recognizer updates.
    We treat the final transcript as the last hypothesis after `final_silence_sec`
    of no output.
    """

    def __init__(
        self,
        *,
        options: HearSTTOptions | None = None,
        command: Optional[Sequence[str]] = None,
        extra_args: Optional[Sequence[str]] = None,
        user_id: str = "hear",
        check_platform: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._options = options or HearSTTOptions()
        if self._options.final_silence_sec <= 0:
            raise ValueError("final_silence_sec must be > 0")

        self._command = list(command) if command else ["hear"]
        self._extra_args = list(extra_args or [])
        self._user_id = user_id
        self._check_platform = check_platform

        self._process: Optional[asyncio.subprocess.Process] = None
        self._process_lock = asyncio.Lock()
        self._restart_lock = asyncio.Lock()
        self._stdout_task: Optional[asyncio.Task[None]] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._monitor_task: Optional[asyncio.Task[None]] = None
        self._finalize_task: Optional[asyncio.Task[None]] = None
        self._begin_task: Optional[asyncio.Task[None]] = None

        self._current_text: str = ""
        self._in_utterance = False
        self._generation = 0
        self._stopping = False
        self._language = self._parse_language(self._options.locale)

        self._ensure_supported()

    def _ensure_supported(self) -> None:
        if self._check_platform and platform.system() != "Darwin":
            raise RuntimeError("MacosHearSTTService is only supported on macOS (Darwin).")
        if self._command and self._command[0] == "hear" and shutil.which("hear") is None:
            raise RuntimeError("macOS `hear` binary not found on PATH.")

    @staticmethod
    def _parse_language(locale: str) -> Optional[Language]:
        locale = (locale or "").strip()
        if not locale:
            return None
        try:
            return Language(locale)
        except Exception:
            return None

    async def process_frame(self, frame: Frame, direction: FrameDirection):  # type: ignore[override]
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._start_process()
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop()
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def _stop(self) -> None:
        self._stopping = True
        await self._cancel_task(self._finalize_task)
        await self._cancel_task(self._begin_task)
        await self._cancel_task(self._monitor_task)
        await self._cancel_task(self._stdout_task)
        await self._cancel_task(self._stderr_task)
        await self._stop_process()
        self._finalize_task = None
        self._begin_task = None
        self._stdout_task = None
        self._stderr_task = None
        self._monitor_task = None
        self._in_utterance = False
        self._current_text = ""
        self._generation += 1  # invalidate pending finalize tasks

    async def _start_process(self) -> None:
        async with self._process_lock:
            if self._process is not None:
                return
            self._stopping = False
            args = self._build_command()
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            self._process = process

        self._stdout_task = self.create_task(self._stdout_reader(process), name="hear_stdout")
        self._stderr_task = self.create_task(self._stderr_reader(process), name="hear_stderr")
        self._monitor_task = self.create_task(self._monitor_process(process), name="hear_monitor")

    def _build_command(self) -> list[str]:
        opts = self._options
        args = list(self._command)
        if opts.on_device:
            args.append("-d")
        if opts.punctuation:
            args.append("-p")
        if opts.locale:
            args.extend(["-l", opts.locale])
        if opts.input_device_id is not None:
            args.extend(["-n", str(opts.input_device_id)])
        args.extend(self._extra_args)
        return args

    async def _stop_process(self) -> None:
        async with self._process_lock:
            process = self._process
            self._process = None
        if not process or process.returncode is not None:
            return

        for sig, timeout_s in (
            (signal.SIGINT, 0.15),
            (signal.SIGTERM, 0.15),
            (signal.SIGKILL, 0.15),
        ):
            if process.returncode is not None:
                return
            try:
                process.send_signal(sig)
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(process.wait(), timeout=timeout_s)
                return
            except asyncio.TimeoutError:
                continue

    async def _restart_process(self) -> None:
        async with self._restart_lock:
            if self._stopping:
                return
            await self._cancel_task(self._monitor_task)
            await self._cancel_task(self._stdout_task)
            await self._cancel_task(self._stderr_task)
            self._monitor_task = None
            self._stdout_task = None
            self._stderr_task = None
            await self._stop_process()
            await self._start_process()

    async def _stdout_reader(self, process: asyncio.subprocess.Process) -> None:
        if not process.stdout:
            return
        while not self._stopping and process.returncode is None:
            try:
                raw = await process.stdout.readline()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"{self}: hear stdout read failed: {exc!r}")
                break
            if not raw:
                break
            text = raw.decode("utf-8", errors="ignore").strip()
            if not text:
                continue
            await self._handle_hypothesis(text)

    async def _stderr_reader(self, process: asyncio.subprocess.Process) -> None:
        if not process.stderr:
            return
        while not self._stopping and process.returncode is None:
            try:
                raw = await process.stderr.readline()
            except Exception:  # pragma: no cover - defensive
                break
            if not raw:
                break
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            logger.debug(f"{self}: hear stderr: {line}")

    async def _monitor_process(self, process: asyncio.subprocess.Process) -> None:
        try:
            await process.wait()
        except asyncio.CancelledError:
            return

        if self._stopping:
            return

        async with self._process_lock:
            still_current = self._process is process

        if still_current:
            self.create_task(self._restart_process(), name="hear_restart")

    async def _handle_hypothesis(self, text: str) -> None:
        self._generation += 1
        generation = self._generation
        self._current_text = text

        if not self._in_utterance:
            self._in_utterance = True
            self._begin_task = self.create_task(self._emit_start_of_turn(), name="hear_begin_turn")

        await self._reschedule_finalize(generation)

    async def _emit_start_of_turn(self) -> None:
        try:
            await asyncio.wait_for(self.push_interruption_task_frame_and_wait(), timeout=0.6)
        except asyncio.TimeoutError:  # pragma: no cover - defensive
            logger.debug(f"{self}: interruption wait timed out")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"{self}: interruption request failed: {exc!r}")

        await self.push_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await self.push_frame(UserStartedSpeakingFrame(), FrameDirection.UPSTREAM)

    async def _reschedule_finalize(self, generation: int) -> None:
        if self._finalize_task:
            self._finalize_task.cancel()
        self._finalize_task = self.create_task(self._finalize_after_silence(generation), name="hear_finalize")

    async def _finalize_after_silence(self, generation: int) -> None:
        try:
            await asyncio.sleep(self._options.final_silence_sec)
        except asyncio.CancelledError:
            return

        if self._stopping or generation != self._generation:
            return

        text = (self._current_text or "").strip()
        if not text:
            self._in_utterance = False
            return

        if self._begin_task:
            try:
                await asyncio.wait_for(self._begin_task, timeout=0.6)
            except Exception:
                pass
            finally:
                self._begin_task = None

        await self.push_frame(
            TranscriptionFrame(
                text=text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=self._language,
                result={"provider": "macos_hear", "locale": self._options.locale},
            )
        )
        await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

        self._in_utterance = False
        self._current_text = ""

        if self._options.restart_on_final and not self._options.keep_mic_open:
            await self._restart_process()

    async def _cancel_task(self, task: Optional[asyncio.Task[None]]) -> None:
        if not task:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return
        except Exception:  # pragma: no cover - defensive
            return

    async def cleanup(self) -> None:
        await self._stop()
        await super().cleanup()
