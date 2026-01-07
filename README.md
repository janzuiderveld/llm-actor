# llm-actor

This repository installs as the Python package `pipecat-thin` (distribution name), even though the repo is named `llm-actor`.

This project packages a thin Python CLI around [Pipecat](https://docs.pipecat.ai/) to deliver a real-time audio loop using Deepgram Flux *or macOS `hear`* speech-to-text, Gemini 2.5 Flash, OpenAI, *or local Ollama* streaming text generation, and Deepgram Aura-2 *or macOS `say`* text-to-speech. External automation hooks are exposed via append-only files under `runtime/`.

## Features

* Deepgram Flux or macOS `hear` STT → LLM → Deepgram Aura-2 or macOS `say` TTS pipeline, fully streaming.
* Optional OpenAI LLM via `llm.model` set to `openai-{MODEL_ID}` (for example `openai-gpt-5.2-chat-latest`).
* Optional local LLM via Ollama by setting `llm.model` to `ollama-{MODEL_ID}` (for example `ollama-gemma3:4b`).
  * Models that emit hidden reasoning traces like `<think>...</think>` have those traces stripped before action parsing, TTS, and history logging.
* File-based automation using append-only `runtime/inbox.txt`, `runtime/actions.txt`, and `runtime/params_inbox.ndjson`.
* Action directives like `<Make_Coffee>` are stripped from spoken audio; adjacent stray punctuation is suppressed while transcripts/history retain the raw text for context.
* Turn-level transcript and event logging per session.
* Runtime parameter application between turns (LLM, STT, TTS, history operations).

## Prerequisites

* [Python 3.10](https://www.python.org/downloads/) or newer (3.11 recommended as it's the version that this is tested on).
* [Deepgram](https://developers.deepgram.com/reference/deepgram-api-overview) API credentials (required when using Deepgram STT/TTS).
* For Gemini models: [Google API](https://ai.google.dev/gemini-api/docs/api-key) credentials with access to Gemini 2.5 Flash.
* For OpenAI models: an `OPENAI_API_KEY` with access to your chosen model.
* For Ollama models: a local Ollama server running on `http://localhost:11434` with the model pulled.
* For macOS `hear`: the `hear` binary on PATH plus Speech Recognition permissions granted to your terminal.
* System audio devices accessible to PortAudio (used by Pipecat's local audio transport).
  * macOS: `brew install portaudio`
  * Ubuntu/Debian: `sudo apt install portaudio19-dev`
  * Windows: the bundled `sounddevice` wheel ships PortAudio automatically. However, Windows users need to install [ffmpeg](https://phoenixnap.com/kb/ffmpeg-windows) for audio playback.

## Quickstart

Projects provide self-contained recipes that wrap the core audio loop, refresh runtime state, and launch any helper daemons you need alongside the agent. The repository ships with an **Exclusive Door** experience that demonstrates how to run the full pipeline with custom automation.

Follow these steps to run the door project end-to-end:

1. **Set up the environment**

   ```bash
   git clone https://github.com/janzuiderveld/llm-actor
   cd llm_actor
   python -m venv .venv # make sure to use python3.10+ (use python -V to check)
   # if you get "command not found: python" type python3 instead of python
   source .venv/bin/activate # for Mac or Linux
   # .venv\Scripts\activate # for Windows
   python -m pip install --upgrade pip 
   pip install -e .
   ```

   Reactivate the virtual environment with `source .venv/bin/activate` (Mac/Linux) or `.venv\Scripts\activate` (Windows) whenever you open a new terminal for this project.
   Direct dependencies are pinned in `pyproject.toml` (including `pyserial` for Arduino support). For a full environment replica, install from `requirements.lock` instead.

2. **Add credentials and defaults**

   (Mac)
   ```bash
   cp .env.example .env
   ```
   
   (Windows)
   ```bash
   copy .env.example .env
   ```

   Open `.env` in your editor and fill in `DEEPGRAM_API_KEY` (plus `GOOGLE_API_KEY` for Gemini or `OPENAI_API_KEY` for OpenAI), along with any optional defaults (LLM, STT, voice) you want to preload. Save the file before continuing.

4. **Launch the example project**

   ```bash
   python EXAMPLE_PROJECT/boot.py
   ```

   The boot script resets `runtime/`, seeds the persona, and starts the main audio loop together with the helper daemons. Stop everything with `Ctrl+C` when you are done.
   On first run (or whenever you delete `runtime/audio_device_preferences.json`), you'll be prompted to select an input (mic) and output (speaker). The selected **device names** are saved so they can be re-resolved to indices on the next boot.

When the project is running, a terminal window running `inbox_writer.py` prompts you to press Enter to simulate “someone approaches the door.” The action watcher listens for `<UNLOCK>` / `<LOCK>` directives given by the persona and keeps the persona aligned with the door state.

### Door Project Files

- `EXAMPLE_PROJECT/boot.py`: Coordinates startup, clears prior transcripts, seeds the locked persona, and spins up the helper processes. Adjust the `locked_config` and `unlocked_config` blocks to experiment with different prompts or voices.
- `EXAMPLE_PROJECT/action_watcher.py`: Monitors `runtime/actions.txt` for `<UNLOCK>` / `<LOCK>` cues, keeps `runtime/inbox.txt` tidy, and flips between the locked and unlocked personalities.
- `EXAMPLE_PROJECT/inbox_writer.py`: Provides a simple keyboard loop that appends “someone approaches the door” to `runtime/inbox.txt`, triggering a new turn whenever a visitor arrives.
- `runtime/actions.txt`: Append-only instruction log produced by the agent. External automations (such as the watcher) listen here for directives like `<UNLOCK>`.
- `runtime/inbox.txt`: Entry point for out-of-band events and user speech. The watcher uses it to queue prompts that walk the agent through the door state transitions.
- `runtime/params_inbox.ndjson`: Applies configuration tweaks between turns. The watcher writes persona swaps into this file when the door locks or unlocks.
- `runtime/example_project_action_watcher.lock`: Prevents multiple action watchers from running at once; remove it if a crashed session left it behind.
- `runtime/audio_device_preferences.json`: Stores the chosen input/output device names. Delete it to re-run device selection on the next boot.
- `runtime/conversations/<timestamp>/`: Holds transcripts and event logs for each session. Use it to inspect how the persona responded while you iterate on prompts.
- `EXAMPLE_PROJECT/assets/*.mp3`: The short door-open and door-close cues that play alongside persona changes. Swap them with your own audio to customize the ambience.

### Coffee Machine Project

The repository also includes a minimal **COFFEE_MACHINE** project that reacts to file-driven button presses.

1. Launch it:

   ```bash
   python COFFEE_MACHINE/boot.py
   ```

   This project pins `llm.model` to `gemini-3-flash-preview`, sets Gemini `llm.thinking_level` to `MINIMAL`, and pins `tts.provider` to `macos_say` (adjust `COFFEE_MACHINE/boot.py` if you use different devices/models). It also disables idle shutdown, pauses STT when idle so the loop stays alive between button presses, and resets the conversation history after each idle timeout so each press starts fresh.

   If an Arduino is connected, the boot process also starts `COFFEE_MACHINE/arduino_bridge.py`. Install `pyserial` in your virtual environment to enable this (`.venv/bin/pip install pyserial`). The bridge auto-detects `/dev/tty*` ports containing `usbmodem` or `usbserial`, appends mapped button presses into `runtime/coffee_machine_buttons.txt`, and watches `runtime/coffee_machine_commands.txt` to send the make-coffee serial sequence when `<Make_Coffee>` appears.

   To swap the venue-specific system prompt, copy `COFFEE_MACHINE/prompts/venues/default.json`, edit the fields (and optionally `prompt_append`, `prompt_template`, or `template_name`), then launch with either:

   ```bash
   COFFEE_MACHINE_VENUE=my_venue python COFFEE_MACHINE/boot.py
   ```

   or

   ```bash
   python COFFEE_MACHINE/boot.py --prompt-file COFFEE_MACHINE/prompts/venues/my_venue.json
   ```

   You can also set `COFFEE_MACHINE_PROMPT_FILE` to point at a profile path if you prefer an environment variable. `prompt_template` points at a specific template path (relative to the profile), while `template_name` looks for `COFFEE_MACHINE/prompts/templates/<name>.txt`.

   Templates can include `{clean_time}`, which is rendered right before each LLM request so the assistant sees the current date/time.

   To select templates independently from venues, pass `--template-name`/`--template-file` or set `COFFEE_MACHINE_TEMPLATE_NAME`/`COFFEE_MACHINE_TEMPLATE_FILE`. These override any template fields in the venue profile.

   Example copies: `COFFEE_MACHINE/prompts/venues/HNI.json` (New Institute) and `COFFEE_MACHINE/prompts/templates/no_output.txt`.

   If you want the boot process auto-restarted when it exits or when there has been no audio output for 10 minutes, run:

   ```bash
   python COFFEE_MACHINE/keepalive.py
   ```

   `Ctrl+C` stops the supervisor and cleans up the running `boot.py` process. You can also type `q` and press Enter to quit cleanly. When macOS `say` is used (no `tts_audio_buffer` events), keepalive also treats assistant transcript activity as output so it doesn't restart mid-session.

   For long-running stability testing (launches `keepalive.py`, simulates button presses + speech every 1-20 minutes at random, and logs missing audio responses):

   ```bash
   python COFFEE_MACHINE/keepalive_stability.py
   ```

   To summarize stability failures for a specific keepalive session (prompts for the PID, or pass `--pid`):

   ```bash
   python COFFEE_MACHINE/keepalive_stability_report.py
   ```

   The harness writes to `runtime/coffee_machine_keepalive_stability.log`. Defaults: randomized 1-20 minute interval between cycles (override with `--interval-seconds` for a fixed cadence or `--min-interval-seconds`/`--max-interval-seconds` for a custom range), 10-second minimum speech delay after button press, 2-second reply delay, 1-second assistant settle window before follow-up audio, 10-second response timeout, a 60-second startup wait for runtime files, a 5-second initial delay before the first cycle, 2 turns per cycle (button press + talkback), and a 15-turn conversation every 10th cycle (override with flags). Playback defaults to macOS `say`, preferring `MacBook Pro Speakers` when available and otherwise the first non-Krisp device (override with `--say-audio-device` or `--allow-krisp`, or switch to file playback with `--playback-mode file` or Deepgram synthesis with `--playback-mode deepgram` and `DEEPGRAM_API_KEY`). Deepgram playback uses the system output device, so set macOS output to `MacBook Pro Speakers` when you want the sample routed there. If `python-dotenv` is available, the harness also loads `.env` in the repo root for Deepgram credentials. This only affects the injected speech sample; assistant responses follow `tts.say_audio_device`, which COFFEE_MACHINE boot auto-derives from `runtime/audio_device_preferences.json` when it can. For direct injection that ignores ambient audio (e.g. loud music), use `--input-mode inbox` to push the speech text straight into `runtime/inbox.txt`. It detects responses via `tts_audio_buffer` events or assistant transcript lines (for macOS `say`, which does not emit audio buffers). On response timeouts the log includes diagnostics (active event log paths, last audio/transcript timestamps, and runtime file stats) to pinpoint where the handoff stalled. Make sure your system routes the playback audio into the mic input (loopback device) so the prerecorded speech can be transcribed.
   If boot fails with an `AudioConfig` unexpected keyword error, remove the stale key from `runtime/config.json` (or delete the file to regenerate defaults).

2. In another terminal, append a button press (each new line triggers a turn):

   ```bash
   echo "BREW" >> runtime/coffee_machine_buttons.txt
   ```

3. Watch for brew commands:

   ```bash
   tail -f runtime/coffee_machine_commands.txt
   ```

Project files:

- `COFFEE_MACHINE/boot.py`: Boots the pipeline, sets a single coffee persona, and clears the coffee machine I/O files in `runtime/`.
- `COFFEE_MACHINE/prompts/template.txt`: Base system prompt template with placeholders like `{event_name}` and `{make_cmd}`.
- `COFFEE_MACHINE/prompts/templates/`: Optional home for versioned templates referenced by `template_name`.
- `COFFEE_MACHINE/prompts/venues/default.json`: Default venue profile; copy it and edit to customize prompts per location.
- `COFFEE_MACHINE/action_watcher.py`: Tails `runtime/coffee_machine_buttons.txt`, forwards presses into `runtime/inbox.txt`, and mirrors `<Make_Coffee>` into `runtime/coffee_machine_commands.txt`.
- `COFFEE_MACHINE/arduino_bridge.py`: Connects to the Arduino over serial, writes mapped button presses into `runtime/coffee_machine_buttons.txt`, and dispatches `<Make_Coffee>` commands to the hardware.
- `COFFEE_MACHINE/keepalive.py`: Supervises `boot.py`, restarting on process exit or when no audio output is detected for a configurable timeout.
- `COFFEE_MACHINE/keepalive_stability.py`: Runs a long-lived keepalive session, periodically injects button presses plus speech (audio playback or direct inbox injection), and logs missing audio responses.
- `COFFEE_MACHINE/keepalive_stability_report.py`: Summarizes recent keepalive stability failures from `runtime/coffee_machine_keepalive_stability.log`.
- `runtime/coffee_machine_action_watcher.lock`: Prevents multiple coffee watchers from duplicating button presses; delete it if you need to recover from a crash.
- `runtime/coffee_machine_action_watcher.log`: Logs button press forwarding, action mirroring, and file truncation recovery for the watcher.
- `runtime/coffee_machine_arduino_bridge.lock`: Prevents multiple Arduino bridges from running at once.
- `runtime/coffee_machine_arduino_bridge.log`: Logs serial connections, button press forwarding, and hardware command dispatches.


## Runtime Automation Reference

**`runtime/inbox.txt`**
- `P: <text>` — push a full user turn immediately.
- `A: <text>` — buffer supplemental text; multiple lines are joined with newlines and appended to the next user entry.
- Consecutive `P:` lines are queued and processed in order; each triggers a response.
- New `P:` entries interrupt any in-progress assistant speech before the latest response is generated.
- Inbox pushes reset any stuck LLM busy state so a new response is always scheduled.
- Gemini requests honor `llm.request_timeout_s` (default 30s) and `llm.thinking_level` when set (on older `google-genai` versions, `MINIMAL` is mapped to `thinking_budget=0`). Timeouts emit `llm_completion_timeout`, empty responses emit `llm_empty_response`, and the inbox item is retried once if no text was returned.
- Each inbox run has a watchdog (`llm.request_timeout_s + 5s`, default 45s) that emits `inbox_run_timeout` and releases the queue if the response never completes.

**`runtime/actions.txt`**
- Append-only log emitted by the agent (everything between <...> is added to this file. for example `<LOCK>` > `LOCK\n`). External services should tail this file and react to directives as they appear.

**`runtime/params_inbox.ndjson`**
- Processed between turns to adjust runtime behavior. Append any of the following operations one JSON object per line:

```bash
{"op":"history.reset"}
{"op":"history.append","role":"user","content":"Remember my name is Kai."}
{"op":"llm.set","model":"gemini-2.5-flash","temperature":0.6,"max_tokens":1024,"thinking_level":"MINIMAL"}
{"op":"llm.set","model":"openai-gpt-5.2-chat-latest","temperature":0.6,"max_tokens":1024}
{"op":"llm.set","model":"ollama-gemma3:4b","temperature":0.6,"max_tokens":1024}
{"op":"llm.system","text":"You are a concise assistant."}
{"op":"stt.flux","eager_eot_threshold":0.5,"eot_threshold":0.85,"eot_timeout_ms":1500}
{"op":"tts.set","voice":"aura-2-thalia-en","encoding":"linear16","sample_rate":24000}
```

- OpenAI requests map `max_tokens` to `max_completion_tokens` for newer chat models.

## Pipeline Idle Behavior

The pipeline uses Pipecat's idle monitor (default 300 seconds). Configure it in `runtime/config.json`:

```json
{
  "pipeline": {
    "idle_timeout_secs": 300,
    "cancel_on_idle_timeout": true,
    "pause_stt_on_idle": false,
    "history_on_idle": "keep",
    "max_history_messages": 50
  }
}
```

- Set `idle_timeout_secs` to `null` (or `0`) to disable idle monitoring entirely.
- Set `cancel_on_idle_timeout` to `false` to keep the script running indefinitely.
- Set `pause_stt_on_idle` to `true` to mute STT when idle; STT resumes when a new line is pushed into `runtime/inbox.txt` (which is what the coffee machine watcher writes).
- Set `history_on_idle` to `"reset"` to clear the conversation history on idle (system prompt is retained), including the in-memory LLM context. Use `"keep"` to preserve history.
- Set `max_history_messages` to cap in-memory conversation history (older messages drop first).
- Idle timing resets on STT/TTS activity (user speaking events, transcripts, and TTS audio frames).
- For Deepgram Flux, inbound audio is not gated while the assistant speaks so barge-in remains available.

## STT Providers

The STT backend is configured in `runtime/config.json` under `stt.model`.

### Deepgram Flux (default)

- Set `stt.model` to `"deepgram-flux"` (or a Flux model id like `"flux-general-en"`).
- Uses `stt.eager_eot_threshold`, `stt.eot_threshold`, `stt.eot_timeout_ms`.

### macOS `hear`

- Set `stt.model` to `"macos-hear"`.
- Runs `hear -d -p -l <stt.language>` and treats the last line as final after `stt.hear_final_silence_sec` seconds of no new output.
- Uses `stt.hear_on_device`, `stt.hear_punctuation`, `stt.hear_input_device_id`.
- Restarts the `hear` process after each final transcript to keep recognition state bounded (`stt.hear_restart_on_final`).
- Set `stt.hear_keep_mic_open` to `true` to keep the `hear` process alive between turns (useful for AEC); this disables restarts even if `stt.hear_restart_on_final` is true.
- Interim hypotheses are only used for interruption; only the final transcript is added to conversation history.

## TTS Providers

The TTS backend is configured in `runtime/config.json` under `tts`.

### macOS `say` (default on macOS)

- Set `tts.provider` to `"macos_say"`.
- Uses `tts.say_voice`, `tts.say_rate_wpm`, `tts.say_audio_device`, `tts.say_interactive`.
- Interruption behavior:
  - When STT detects the user speaking, any ongoing `say` utterance is terminated.
  - The transcript logs assistant speech incrementally, preserving ordering when interruptions happen.
  - The transcript only includes words that were actually spoken; if cut off, `tts.cutoff_marker` is appended.

To run the optional local integration test (plays audio):

```bash
RUN_SAY_TESTS=1 SAY_TEST_VOICE=Alex pytest tests/test_macos_say_tts.py -v
```

### Deepgram (default on non-macOS)

- Set `tts.provider` to `"deepgram"`.
- Uses `tts.voice`, `tts.encoding`, `tts.sample_rate`.

## Acoustic Echo Cancellation

- Install [Krisp](https://krisp.ai/download/) for system-level acoustic echo cancellation. This makes sure persona output does not leak into the microphone input.
- Set **Krisp Microphone** and **Krisp Speaker** as your input/output devices in `boot.py` or use the auto flag in the audio config.
NOTE: It is possible that sound output is crackling when using Krisp. If this happens, for now continue with headphones. Will do a universal fix in future update.

## Testing

The test harness expects valid API keys in your environment. Once configured, run:

```bash
pytest -q
```

Tests perform the following real API checks:

1. `tests/test_api_readiness.py`
   * Streams a short utterance through Deepgram Flux STT and validates a transcript.
   * Requests a 1-second sample from Deepgram Aura-2 TTS.
   * Streams a small prompt through Gemini 2.5 Flash and verifies tokens arrive.
2. `tests/test_full_pipeline.py`
   * Synthesizes test audio via Deepgram TTS and feeds it into the Pipecat pipeline.
   * Logs turn-level latency markers including TTFRAP (time to first response audio packet).

If credentials are missing the tests skip with a clear message.

## Troubleshooting

* **Audio devices missing**: Ensure PortAudio is installed (see prerequisites) and run `python -m sounddevice` inside your virtualenv to confirm the runtime can enumerate devices.
* **Wrong mic/speaker**: Delete `runtime/audio_device_preferences.json` and reboot the project to re-select devices.
* **Duplicate button presses / actions**: Make sure only one watcher is running; remove any stale `runtime/*_action_watcher.lock` files left behind after a crash.
* **429 / rate limits**: The tests and pipeline perform real API calls; adjust usage or upgrade account tiers if needed.
* **High latency**: Tune Flux EOT thresholds via `runtime/params_inbox.ndjson` to improve responsiveness.

## License

MIT
