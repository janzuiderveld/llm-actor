# Pipecat Thin Voice Pipeline

This project packages a thin Python CLI around [Pipecat](https://docs.pipecat.ai/) to deliver a real-time audio loop using Deepgram Flux speech-to-text, Gemini 2.5 Flash streaming text generation, and Deepgram Aura-2 text-to-speech. All external automation hooks are exposed via append-only files under `runtime/`.

## Features

* Deepgram Flux STT → Gemini 2.5 Flash → Deepgram Aura-2 TTS pipeline, fully streaming.
* File-based automation using append-only `runtime/inbox.txt`, `runtime/actions.txt`, and `runtime/params_inbox.ndjson`.
* Turn-level transcript and event logging per session.
* Runtime parameter application between turns (LLM, STT, TTS, history operations).
* Seeds `runtime/config.json` with system-default audio devices and prefers Krisp virtual endpoints for echo cancellation.

## Prerequisites

* Python 3.10 or newer (3.11 recommended as it's the only version that this is tested on).
* Deepgram and Google API credentials with access to Flux STT, Aura-2 TTS, and Gemini 2.5 Flash.
* System audio devices accessible to PortAudio (used by Pipecat's local audio transport).
  * macOS: `brew install portaudio`
  * Ubuntu/Debian: `sudo apt install portaudio19-dev`
  * Windows: the bundled `sounddevice` wheel ships PortAudio automatically.

## Setup

Follow these steps the first time you install the pipeline on a new machine:

1. Clone the repository (or pull the latest changes) and create a virtual environment:

   ```bash
   git clone <repo-url>
   cd llm_actor
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -e .
   ```

2. Copy `.env.example` to `.env` and supply your credentials and optional defaults:

   ```bash
   cp .env.example .env
   ```

   | Variable | What it does |
   | --- | --- |
   | `GOOGLE_API_KEY` | Gemini 2.5 Flash realtime key. |
   | `DEEPGRAM_API_KEY` | Deepgram Flux STT & Aura-2 TTS key. |
   | `PIPECAT_DEFAULT_LLM_MODEL` | (Optional) LLM used for new agents and fresh configs. |
   | `PIPECAT_DEFAULT_STT_MODEL` | (Optional) STT model used on first run. |
   | `PIPECAT_DEFAULT_VOICE` | (Optional) Default Deepgram voice for new agents. |

   The optional defaults are read when `runtime/config.json` is created; they are a convenient way to seed new agent personas (see [Create a New Agent](#create-a-new-agent)).

3. Create the runtime configuration file (runtime/config.json):
   ```bash
   python -m app.bootstrap
   ```

   This writes `runtime/config.json` so you can tweak it before the first run. When Krisp virtual devices are present, they are selected automatically; otherwise the script falls back to the system defaults reported by PortAudio. If you see `RuntimeError: Unable to detect system default audio devices automatically`, install PortAudio (see prerequisites) or set the indices manually in `runtime/config.json`. Run `python -m sounddevice` to double-check discovery.

4. Launch the CLI to verify audio and credentials:

   ```bash
   python -m app.cli
   ```

## Acoustic Echo Cancellation

- Install [Krisp](https://krisp.ai/download/) for system-level acoustic echo cancellation.
- Set **Krisp Microphone** and **Krisp Speaker** as your default input/output devices in the OS audio settings.
- Run `python -m app.bootstrap` again to refresh `runtime/config.json` with the Krisp device indices (the bootstrapper prefers them automatically).
- Start the pipeline with `python -m app.cli` and speak into the Krisp microphone; the agent should no longer hear its own replies.

## Create a New Agent

An agent is the combination of LLM, STT, and TTS settings plus the system prompt stored in `runtime/config.json`. The bootstrap step above creates that file before the first run so you can adjust it straight away. Use any of the following approaches to define a new persona:

1. **Set defaults before bootstrapping** — populate the optional `PIPECAT_DEFAULT_*` variables in `.env`, then run `python -m app.bootstrap`. The generated `runtime/config.json` will inherit those defaults automatically.
2. **Edit the saved runtime config** — adjust the `llm` and `tts` keys directly, for example:

   ```json
   {
     "llm": {
       "model": "gemini-2.5-flash",
       "temperature": 0.3,
       "max_tokens": 1024,
       "system_prompt": "You are Ava, a robotics lab assistant. Speak precisely and surface safety warnings."
     },
     "tts": {
       "voice": "aura-2-carter-en"
     }
   }
   ```

   Leave the other sections (such as `audio` and `stt`) untouched unless you intend to change them as well. Restart the CLI to pick up the changes.

3. **Queue a runtime preset** — to swap agents between turns without restarting, append NDJSON lines to `runtime/params_inbox.ndjson`. A typical preset looks like:

   ```bash
   cat <<'EOF' >> runtime/params_inbox.ndjson
   {"op":"history.reset"}
   {"op":"llm.set","model":"gemini-2.5-flash","temperature":0.4,"system_prompt":"You are Orbit, a mission-control specialist who prioritizes checklists and short callouts."}
   {"op":"tts.set","voice":"aura-2-damon-en"}
   {"op":"stt.flux","eager_eot_threshold":0.6,"eot_threshold":0.9,"eot_timeout_ms":1200}
   EOF
   ```

   The watcher applies each entry between turns, writes the new settings back to `runtime/config.json`, and the optional `history.reset` entry gives you a clean context when swapping personas. Keep separate files with your favorite presets and append them as needed to pivot quickly.

## Changing Audio Devices

- Run `python -m app.bootstrap` to re-detect the current system defaults (or newly installed Krisp devices) and rewrite `runtime/config.json`.
- Alternatively, edit `runtime/config.json` manually and update `audio.input_device_index` / `audio.output_device_index` with the indices you want (check `python -m sounddevice` for a listing).
- Restart the pipeline with `python -m app.cli` after any changes.

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
