import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from app.history import ConversationHistory


def test_history_logs_transcripts_without_context(tmp_path):
    case1 = tmp_path / "case1"
    case1.mkdir()

    transcript_path = case1 / "transcript.jsonl"
    clean_path = case1 / "transcript.llm.jsonl"
    context_path = case1 / "LLM_context.jsonl"

    history = ConversationHistory(
        transcript_path,
        clean_transcript_path=clean_path,
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

    assert not context_path.exists()

    # Replicate multi-chunk assistant response with action tag
    case2 = tmp_path / "case2"
    case2.mkdir()

    transcript_path2 = case2 / "transcript.jsonl"
    clean_path2 = case2 / "transcript.llm.jsonl"
    context_path2 = case2 / "LLM_context.jsonl"

    history2 = ConversationHistory(
        transcript_path2,
        clean_transcript_path=clean_path2,
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

    assert not context_path2.exists()


def test_system_prompt_reflects_in_clean_transcript(tmp_path):
    transcript_path = tmp_path / "transcript.jsonl"
    clean_path = tmp_path / "transcript.llm.jsonl"
    context_path = tmp_path / "LLM_context.jsonl"

    history = ConversationHistory(
        transcript_path,
        clean_transcript_path=clean_path,
    )

    locked_prompt = "You guard the Velvet Room."
    unlocked_prompt = "You are the Open Door."
    reset_prompt = "System reset to neutral."

    history.set_system_message(locked_prompt)
    history.add("user", "hello")

    clean_lines = [json.loads(line) for line in clean_path.read_text().splitlines()]
    assert clean_lines[0] == {"role": "system", "content": locked_prompt}
    assert clean_lines[1] == {"role": "user", "content": "hello"}

    history.set_system_message(unlocked_prompt)
    clean_lines = [json.loads(line) for line in clean_path.read_text().splitlines()]
    assert clean_lines[0] == {"role": "system", "content": unlocked_prompt}

    history.reset(system_prompt=reset_prompt)
    clean_lines = [json.loads(line) for line in clean_path.read_text().splitlines()]
    assert clean_lines == [{"role": "system", "content": reset_prompt}]

    history.set_system_message(None)
    assert clean_path.read_text() == ""
    assert not context_path.exists()


def test_assistant_replacement_dedupes_logged_content(tmp_path):
    transcript_path = tmp_path / "transcript.jsonl"
    history = ConversationHistory(transcript_path)

    history.add_partial("assistant", "I am doing wonderfully, thank you for asking!")
    repeated = (
        "I am doing wonderfully, thank you for asking! It's always a pleasure to greet new friends. "
        "I am doing wonderfully , thank you for asking! It's always a pleasure to greet new friends . "
        "How may I help you today?"
    )
    expected = (
        "I am doing wonderfully, thank you for asking! It's always a pleasure to greet new friends. "
        "How may I help you today?"
    )

    history.add("assistant", repeated, replace_last=True)

    transcript_lines = [json.loads(line) for line in transcript_path.read_text().splitlines()]
    assert transcript_lines[-1]["replace"] is True
    assert transcript_lines[-1]["content"] == expected
