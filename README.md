# LLM_Embodiment

This project packages a thin Python CLI around [Pipecat](https://docs.pipecat.ai/) to deliver a real-time audio loop using Deepgram Flux speech-to-text, Gemini 2.5 Flash streaming text generation, and Deepgram Aura-2 text-to-speech. External automation hooks are exposed via append-only files under `runtime/`.

## Features

* Deepgram Flux STT → Gemini 2.5 Flash → Deepgram Aura-2 TTS pipeline, fully streaming.
* Local audio transport using Pipecat's device indices; first run prompts for input/output selection.
* File-based automation using append-only `runtime/inbox.txt`, `runtime/actions.txt`, and `runtime/params_inbox.ndjson`.
* Turn-level transcript and event logging per session with latency instrumentation (TTFRAP and more).
* Runtime parameter application between turns (LLM, STT, TTS, history operations).
* Test harness hitting the real APIs to validate connectivity and measure latency.

## Prerequisites

* Python 3.10 or newer (3.11+ recommended).
* Deepgram and Google API credentials with access to Flux STT, Aura-2 TTS, and Gemini 2.5 Flash.
* System audio devices accessible to PortAudio (used by Pipecat's local audio transport).
  * macOS: `brew install portaudio`
  * Ubuntu/Debian: `sudo apt install portaudio19-dev`
  * Windows: the bundled `sounddevice` wheel ships PortAudio automatically.

## Setup

1. Clone this repository and create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .
   ```

2. Copy `.env.example` to `.env` and fill in your API keys:

   ```bash
   cp .env.example .env
   ```

3. Run the CLI once to capture your preferred audio input/output devices:

   ```bash
   python -m app.cli
   ```

   If you see `RuntimeError: No audio devices detected`, double-check that PortAudio is installed (see prerequisites) and that your virtual environment includes the `sounddevice` package (installed automatically via `pip install -e .`). You can verify discovery with:

   ```bash
   python -m sounddevice
   ```

   Follow the prompts to pick the indices that match your microphone and speakers. The selections are stored in `runtime/config.json` for future runs.

## Changing Audio Devices

The input/output indices are persisted under the `audio` section of `runtime/config.json`. To switch devices after the initial run:

1. Enumerate available hardware with `python -m sounddevice` (or another PortAudio-aware tool) and note the desired indices.
2. Edit `runtime/config.json`, updating `audio.input_device_index` and `audio.output_device_index` to the new values. Setting either field to `null` (or deleting it) will cause the CLI to prompt for a fresh selection on the next launch.
3. Save the file and restart the pipeline via `python -m app.cli` to use the new devices.

## Runtime Files

* `runtime/actions.txt`: assistant `<...>` directives are appended here (one per line). Use an external watcher to execute them.
* `runtime/inbox.txt`: external systems append lines to inject user input.
  * `P: <text>` starts a new user turn immediately.
  * `A: <text>` appends to the pending user message before it is sent to the LLM; if no pending message exists the text is treated as `P:`.
* `runtime/params_inbox.ndjson`: NDJSON entries processed **between turns** to adjust runtime parameters. Supported operations:
  * `{"op":"llm.set","model":"gemini-2.5-flash","temperature":0.6,"max_tokens":1024}`
  * `{"op":"llm.system","text":"You are a concise assistant."}`
  * `{"op":"history.reset"}`
  * `{"op":"history.append","role":"user","content":"Remember my name is Kai."}`
  * `{"op":"stt.flux","eager_eot_threshold":0.5,"eot_threshold":0.85,"eot_timeout_ms":1500}`
  * `{"op":"tts.set","voice":"aura-2-thalia-en","encoding":"linear16","sample_rate":24000}`

## Running the CLI

```bash
python -m app.cli --session-name demo-session
```

The CLI prints the session directory under `runtime/conversations/` and streams audio until you exit with `Ctrl+C`. All transcripts (including `<...>` actions) are appended to `transcript.jsonl`, while operational events and latency metrics flow into `event_log.ndjson`.

## Examples

* `python -m examples.minimal_run` — minimal bootstrapping example that starts a pipeline using saved config.
* `python -m examples.inject_from_inbox` — helper script that writes sample lines into `runtime/inbox.txt`.

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

* **Audio devices missing**: Ensure PortAudio is installed (see prerequisites) and run `python -m sounddevice` inside your virtualenv to confirm the PortAudio runtime can enumerate devices.
* **429 / rate limits**: The tests and pipeline perform real API calls; adjust usage or upgrade account tiers if needed.
* **High latency**: Tune Flux EOT thresholds or Gemini temperature via `runtime/params_inbox.ndjson` to improve responsiveness.

## License

MIT
