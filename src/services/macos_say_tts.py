from __future__ import annotations

import asyncio
import os
import platform
import re
import shutil
import signal
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Optional, Sequence

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    TTSTextFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import TTSService
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator


_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


@dataclass(frozen=True)
class SayInteractiveRender:
    text: str
    highlight: Optional[tuple[int, int]]


def _parse_say_interactive_render(render: str) -> SayInteractiveRender:
    """Parse one `say --interactive` render line into plain text and the highlighted span.

    The output includes ANSI escape sequences, plus cursor controls. We only care
    about the reverse-video SGR (`ESC[7m`) span.
    """

    highlight_start: Optional[int] = None
    highlight_end: Optional[int] = None
    highlighted = False
    plain_chars: list[str] = []

    i = 0
    length = len(render)
    while i < length:
        ch = render[i]
        if ch == "\x1b" and i + 1 < length:
            nxt = render[i + 1]
            if nxt == "[":
                m = _CSI_RE.match(render, i)
                if m:
                    seq = m.group(0)
                    final = seq[-1]
                    if final == "m":
                        params = seq[2:-1]
                        if not params:
                            highlighted = False
                            if highlight_start is not None and highlight_end is None:
                                highlight_end = len(plain_chars)
                        else:
                            parts = [p for p in params.split(";") if p]
                            if "0" in parts:
                                highlighted = False
                                if highlight_start is not None and highlight_end is None:
                                    highlight_end = len(plain_chars)
                            if "7" in parts:
                                highlighted = True
                                if highlight_start is None:
                                    highlight_start = len(plain_chars)
                                    highlight_end = None
                    i = m.end()
                    continue
            if nxt == "(" and i + 2 < length:
                i += 3
                continue
            if nxt == "]":
                # OSC sequence: consume until BEL or ST (ESC \).
                j = i + 2
                while j < length:
                    if render[j] == "\x07":
                        j += 1
                        break
                    if render[j] == "\x1b" and j + 1 < length and render[j + 1] == "\\":
                        j += 2
                        break
                    j += 1
                i = j
                continue
            # Other escape sequence; skip the ESC and continue.
            i += 2
            continue

        if highlighted and highlight_start is None:
            highlight_start = len(plain_chars)
        plain_chars.append(ch)
        i += 1

    if highlight_start is not None and highlight_end is None and highlighted:
        highlight_end = len(plain_chars)

    highlight: Optional[tuple[int, int]] = None
    if highlight_start is not None and highlight_end is not None and highlight_end >= highlight_start:
        highlight = (highlight_start, highlight_end)

    return SayInteractiveRender(text="".join(plain_chars), highlight=highlight)


class SaySubsentenceTextAggregator(BaseTextAggregator):
    """Release TTS chunks at punctuation boundaries (low-latency)."""

    def __init__(self, *, max_chars: int = 180):
        self._text = ""
        self._max_chars = max_chars

    @property
    def text(self) -> str:
        return self._text

    async def aggregate(self, text: str) -> Optional[str]:
        self._text += text
        if not self._text:
            return None

        # Always flush on newlines.
        for idx, ch in enumerate(self._text):
            if ch in ("\n", "\r"):
                result = self._text[: idx + 1]
                self._text = self._text[idx + 1 :]
                return result

        # Flush on punctuation / subsentence boundaries (include trailing spaces).
        subsentence_delims = {",", ";", ":", "?", "!", "\u2013", "\u2014"}
        ellipsis_char = "\u2026"
        i = 0
        while i < len(self._text):
            ch = self._text[i]

            if self._text.startswith("...", i):
                j = i + 3
                while j < len(self._text) and self._text[j] == ".":
                    j += 1
                while j < len(self._text) and self._text[j].isspace() and self._text[j] not in ("\n", "\r"):
                    j += 1
                result = self._text[:j]
                self._text = self._text[j:]
                return result

            if ch == ".":
                next_char = self._text[i + 1] if i + 1 < len(self._text) else ""
                if next_char == "." or next_char.isdigit():
                    i += 1
                    continue
                j = i + 1
                while j < len(self._text) and self._text[j].isspace() and self._text[j] not in ("\n", "\r"):
                    j += 1
                result = self._text[:j]
                self._text = self._text[j:]
                return result

            if ch in subsentence_delims or ch == ellipsis_char:
                j = i + 1
                while j < len(self._text) and self._text[j].isspace() and self._text[j] not in ("\n", "\r"):
                    j += 1
                result = self._text[:j]
                self._text = self._text[j:]
                return result

            i += 1

        # Fallback: flush when long enough, but only at whitespace to avoid breaking words.
        if len(self._text) >= self._max_chars:
            cut = self._text.rfind(" ", 0, self._max_chars + 1)
            if cut != -1:
                result = self._text[: cut + 1]
                self._text = self._text[cut + 1 :]
                return result

        return None

    async def handle_interruption(self):
        self._text = ""

    async def reset(self):
        self._text = ""


class MacosSayTTSService(TTSService):
    """macOS `say` TTS backend with interruption-safe subprocess control.

    This service does not emit audio frames (macOS plays audio directly). It emits:
    - `TTSStartedFrame`
    - `TTSTextFrame` containing only the text known to have finished speaking
    - `TTSStoppedFrame` (with `metadata["tts_interrupted"]=True` when interrupted)
    """

    def __init__(
        self,
        *,
        voice: Optional[str] = None,
        rate_wpm: Optional[int] = None,
        audio_device: Optional[str] = None,
        interactive: bool = True,
        text_aggregator: Optional[BaseTextAggregator] = None,
        extra_args: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        super().__init__(
            aggregate_sentences=True,
            push_text_frames=False,
            push_stop_frames=False,
            pause_frame_processing=False,
            text_aggregator=text_aggregator or SaySubsentenceTextAggregator(),
            **kwargs,
        )
        self._ensure_supported()
        self._voice = voice
        self._rate_wpm = rate_wpm
        self._audio_device = audio_device
        self._interactive = interactive
        self._extra_args = list(extra_args or [])
        self._process: Optional[asyncio.subprocess.Process] = None
        self._process_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._interrupted = False
        self._last_first_highlight_ms: Optional[float] = None
        self._speaking = False
        self._last_plain_text: Optional[str] = None
        self._current_highlight: Optional[tuple[int, int]] = None
        self._spoken_end = 0
        self._emitted_end = 0

    @staticmethod
    def _ensure_supported() -> None:
        if platform.system() != "Darwin":
            raise RuntimeError("MacosSayTTSService is only supported on macOS (Darwin).")
        if shutil.which("say") is None:
            raise RuntimeError("macOS `say` binary not found on PATH.")

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        self._interrupted = True
        await self._stop_process()
        await self._flush_spoken_delta()
        await self._maybe_push_stopped_frame()

    async def _stop_process(self) -> None:
        async with self._process_lock:
            process = self._process
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

    async def _flush_spoken_delta(self) -> None:
        async with self._state_lock:
            last_plain = self._last_plain_text
            spoken_end = self._spoken_end
            emitted_end = self._emitted_end

        if not last_plain or spoken_end <= emitted_end:
            return

        delta = last_plain[emitted_end:spoken_end]
        if delta.strip():
            await self.push_frame(TTSTextFrame(delta))

        async with self._state_lock:
            self._emitted_end = max(self._emitted_end, spoken_end)

    async def _maybe_push_stopped_frame(self) -> None:
        if not self._speaking:
            return
        self._speaking = False
        stopped = TTSStoppedFrame()
        stopped.metadata["tts_interrupted"] = bool(self._interrupted)
        if self._last_first_highlight_ms is not None:
            stopped.metadata["tts_first_highlight_ms"] = self._last_first_highlight_ms
        await self.push_frame(stopped)

    async def process_frame(self, frame: Frame, direction: FrameDirection):  # type: ignore[override]
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseEndFrame) and not frame.skip_tts:
            await self.push_frame(frame, direction)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        self._interrupted = False
        self._last_first_highlight_ms = None
        self._speaking = True
        async with self._state_lock:
            self._last_plain_text = None
            self._current_highlight = None
            self._spoken_end = 0
            self._emitted_end = 0

        args: list[str] = ["say"]
        if self._voice:
            args.extend(["-v", self._voice])
        if self._rate_wpm is not None:
            args.extend(["-r", str(self._rate_wpm)])
        if self._audio_device:
            args.extend(["-a", self._audio_device])
        if self._interactive:
            args.append("--interactive")
        args.extend(self._extra_args)
        args.append(text)

        env = os.environ.copy()
        env.setdefault("TERM", "xterm-256color")

        start_ns = time.monotonic_ns()
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )

        async with self._process_lock:
            self._process = process

        yield TTSStartedFrame()

        spoken_end = 0
        current_highlight: Optional[tuple[int, int]] = None
        last_plain: Optional[str] = None
        emitted_end = 0
        buffer = bytearray()

        if process.stdout:
            while process.returncode is None:
                try:
                    chunk = await asyncio.wait_for(process.stdout.read(512), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                if not chunk:
                    break
                buffer.extend(chunk)

                while True:
                    try:
                        idx = buffer.index(ord("\r"))
                    except ValueError:
                        break
                    part = bytes(buffer[:idx])
                    del buffer[: idx + 1]
                    render = part.decode("utf-8", errors="ignore")
                    parsed = _parse_say_interactive_render(render)
                    if not parsed.text.strip():
                        continue
                    last_plain = parsed.text
                    if parsed.highlight:
                        if self._last_first_highlight_ms is None:
                            self._last_first_highlight_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0
                        if current_highlight is not None and parsed.highlight != current_highlight:
                            spoken_end = current_highlight[1]
                            if spoken_end > emitted_end and last_plain:
                                delta = last_plain[emitted_end:spoken_end]
                                emitted_end = spoken_end
                                if delta.strip():
                                    yield TTSTextFrame(delta)
                        current_highlight = parsed.highlight
                    else:
                        if current_highlight is not None:
                            spoken_end = current_highlight[1]
                            if spoken_end > emitted_end and last_plain:
                                delta = last_plain[emitted_end:spoken_end]
                                emitted_end = spoken_end
                                if delta.strip():
                                    yield TTSTextFrame(delta)
                        current_highlight = None

                    async with self._state_lock:
                        self._last_plain_text = last_plain
                        self._current_highlight = current_highlight
                        self._spoken_end = emitted_end
                        self._emitted_end = emitted_end

        if buffer:
            tail = buffer.decode("utf-8", errors="ignore")
            parsed = _parse_say_interactive_render(tail)
            if parsed.text.strip():
                last_plain = parsed.text
                if parsed.highlight and current_highlight is not None and parsed.highlight != current_highlight:
                    spoken_end = current_highlight[1]
                    if spoken_end > emitted_end and last_plain:
                        delta = last_plain[emitted_end:spoken_end]
                        emitted_end = spoken_end
                        if delta.strip():
                            yield TTSTextFrame(delta)
                    current_highlight = parsed.highlight
                elif current_highlight is not None and not parsed.highlight:
                    spoken_end = current_highlight[1]
                    if spoken_end > emitted_end and last_plain:
                        delta = last_plain[emitted_end:spoken_end]
                        emitted_end = spoken_end
                        if delta.strip():
                            yield TTSTextFrame(delta)
                    current_highlight = None

                async with self._state_lock:
                    self._last_plain_text = last_plain
                    self._current_highlight = current_highlight
                    self._spoken_end = emitted_end
                    self._emitted_end = emitted_end

        await process.wait()

        async with self._process_lock:
            if self._process is process:
                self._process = None

        async with self._state_lock:
            self._last_plain_text = last_plain
            self._current_highlight = current_highlight
            self._spoken_end = emitted_end
            self._emitted_end = emitted_end

        if not self._interrupted and current_highlight is not None and last_plain:
            final_end = current_highlight[1]
            if final_end > emitted_end:
                delta = last_plain[emitted_end:final_end]
                emitted_end = final_end
                if delta.strip():
                    yield TTSTextFrame(delta)
                async with self._state_lock:
                    self._spoken_end = emitted_end
                    self._emitted_end = emitted_end

        if self._speaking:
            self._speaking = False
            stopped = TTSStoppedFrame()
            stopped.metadata["tts_interrupted"] = bool(self._interrupted)
            if self._last_first_highlight_ms is not None:
                stopped.metadata["tts_first_highlight_ms"] = self._last_first_highlight_ms
            yield stopped
