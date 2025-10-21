import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from app.history import ConversationHistory


def test_history_logs_clean_and_context(tmp_path):
    case1 = tmp_path / "case1"
    case1.mkdir()

    transcript_path = case1 / "transcript.jsonl"
    clean_path = case1 / "transcript.llm.jsonl"
    context_path = case1 / "LLM_context.jsonl"

    history = ConversationHistory(
        transcript_path,
        clean_transcript_path=clean_path,
        context_path=context_path,
    )

    history.add("user", "hello")
    history.add_partial("assistant", "Hi")
    history.add_partial("assistant", " there")
    history.add("assistant", "Hi there", replace_last=True)

    transcript_lines = [json.loads(line) for line in transcript_path.read_text().splitlines()]
    assert transcript_lines[0]["role"] == "user"
    assert "partial" not in transcript_lines[0]
    assert transcript_lines[1]["partial"] is True
    assert transcript_lines[-1]["replace"] is True
    assert transcript_lines[-1]["content"] == "Hi there"

    clean_lines = [json.loads(line) for line in clean_path.read_text().splitlines()]
    assert clean_lines == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    context_lines = [json.loads(line) for line in context_path.read_text().splitlines()]
    assert [entry["role"] for entry in context_lines[0]["context"]] == ["user"]
    assert context_lines[1]["context"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    # Replicate multi-chunk assistant response with action tag
    case2 = tmp_path / "case2"
    case2.mkdir()

    transcript_path2 = case2 / "transcript.jsonl"
    clean_path2 = case2 / "transcript.llm.jsonl"
    context_path2 = case2 / "LLM_context.jsonl"

    history2 = ConversationHistory(
        transcript_path2,
        clean_transcript_path=clean_path2,
        context_path=context_path2,
    )

    history2.add("user", "Okay. Nice. Try it again. And now say a sentence while you also say the command.")
    history2.add_partial("assistant", "Sure.")
    history2.add("assistant", "Sure.", replace_last=True)
    history2.add_partial("assistant", " I'm turning on the light for you.")
    history2.add("assistant", "I'm turning on the light for you.", replace_last=True)
    history2.add_partial("assistant", " <turn_on_light>")
    history2.add("assistant", "<turn_on_light>", replace_last=True)

    clean_lines2 = [json.loads(line) for line in clean_path2.read_text().splitlines()]
    assert clean_lines2 == [
        {
            "role": "user",
            "content": "Okay. Nice. Try it again. And now say a sentence while you also say the command.",
        },
        {
            "role": "assistant",
            "content": "Sure. I'm turning on the light for you. <turn_on_light>",
        },
    ]

    context_lines2 = [json.loads(line) for line in context_path2.read_text().splitlines()]
    assert context_lines2[-1]["context"][-1]["content"] == "Sure. I'm turning on the light for you. <turn_on_light>"
