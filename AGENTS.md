# AGENTS.md

## Testing

**Always write tests for new features.** If you implement something, write a corresponding test.

**Always use the repo's `.venv` when running commands or tests.** Activate it or call `.venv/bin/python` / `.venv/bin/pytest` directly.

**Run specific tests, not the full suite.** Target only the tests relevant to your changes:

```bash
# Run a single test file
pytest tests/test_action_extractor_filter.py -v

# Run a specific test function
pytest tests/test_history_logging.py::test_function_name -v
```

**If you find untested features, write tests for them.**

Always update the README.md and AGENTS.md files to reflect the changes you make.

## Dependencies

- `pyproject.toml` pins the full reference environment; refresh with `pip freeze` when the venv changes.

Action extraction updates that affect transcript/TTs behavior should be covered in `tests/test_action_extractor_filter.py`, `tests/test_reasoning_trace_filter.py`, and `tests/test_macos_say_tts.py`.
Action extraction suppresses action-adjacent commas/colons/semicolons in spoken output to avoid TTS reading stray punctuation.

## Test Locations

- `tests/` — Unit tests (action extractor, streaming, history)
- `src/tests/` — Integration tests (API readiness, full pipeline)

## Runtime Defaults

- `COFFEE_MACHINE/keepalive_stability.py` randomizes cycle intervals between 1 and 20 minutes unless `--interval-seconds` is provided.
- `COFFEE_MACHINE/boot.py` sets `pipeline.history_on_idle` to `reset` so idle timeouts clear conversation history.
- `COFFEE_MACHINE/boot.py` sets `llm.thinking_level` to `MINIMAL` for Gemini models.
- Idle history resets clear the active LLM context so the next turn starts fresh.

## Arduino Integration

- `COFFEE_MACHINE/boot.py` launches `COFFEE_MACHINE/arduino_bridge.py` to connect to the Arduino.
- The bridge appends mapped button presses to `runtime/coffee_machine_buttons.txt`.
- The bridge watches `runtime/coffee_machine_commands.txt` for `<Make_Coffee>` and sends the `0/7/5/*` serial sequence.
- `pyserial` is required for hardware IO.

## Prompt Profiles

- Coffee machine prompts live in `COFFEE_MACHINE/prompts/` with the default profile at `COFFEE_MACHINE/prompts/venues/default.json`.
- Select a venue with `COFFEE_MACHINE_VENUE`/`COFFEE_MACHINE_PROMPT_FILE` or `COFFEE_MACHINE/boot.py --venue/--prompt-file`.
- Use `template_name` for `COFFEE_MACHINE/prompts/templates/<name>.txt` or `prompt_template` for an explicit template path.
- Override templates separately with `COFFEE_MACHINE_TEMPLATE_NAME`/`COFFEE_MACHINE_TEMPLATE_FILE` or `COFFEE_MACHINE/boot.py --template-name/--template-file`.
- Example copies: `COFFEE_MACHINE/prompts/venues/HNI.json` (New Institute) and `COFFEE_MACHINE/prompts/templates/no_output.txt`.
- `{clean_time}` in templates is refreshed right before each LLM request.

## LLM Providers

- OpenAI models use `llm.model` prefixed with `openai-` (for example `openai-gpt-5.2-chat-latest`) and require `OPENAI_API_KEY`.
- OpenAI requests map `llm.max_tokens` to `max_completion_tokens` for newer chat models.
- Gemini `llm.thinking_level` is applied when supported by `google-genai`; `MINIMAL` maps to `thinking_budget=0` on older versions.
