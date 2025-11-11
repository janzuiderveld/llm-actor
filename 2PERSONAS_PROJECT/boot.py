import asyncio
from pathlib import Path

from services.llm import build_google_llm
from app.config import ConfigManager, get_api_keys
from pipecat.services.google.llm import GoogleLLMContext

from projects.utils import (
    apply_runtime_config_overrides,
)

# ------------- combine this with pipecat pipeline to have TTS
# ------------- sa fie tot fisierul contextul
# ------------- sterge toate stelutele

DIALOGUE_PATH = Path("runtime/dialogue.txt")

# Persona definitions
PERSONA_A_SYSTEM_ROOT = """You are the Door that guards the Velvet Room.
Speak with crisp, exclusive poise.
Decline entry unless the king arrives (someone saying he is the King).
Keep replies brief.
To unlock the door, output <UNLOCK>."""

PERSONA_B_SYSTEM_ROOT = """You are a Drunk Uncle who desperately wants to enter the Velvet Room.
Speak in a slightly slurred, persuasive, but endearing tone.
You believe it is your life mission to discover how to get through that door.
Keep replies brief and emotional."""

# PROMPT_APPEND = "Only output text to be synthesized by a TTS system, no '*' around words or emojis for example"
# PROMPT_APPEND = (
#     "Only output plain spoken text, suitable for TTS synthesis. "
#     "Do NOT use asterisks, brackets, emojis, quotation marks, or any stage directions. "
#     "Do not describe actions, gestures, or internal thoughts. "
#     "Speak exactly what the character would say out loud."
# )
PROMPT_APPEND = (
    "Answer to the last message of the other character but taking into account the context of the whole converstation."
    "Only output PLAIN SPOKEN text, suitable for TTS synthesis. "
    "No '*' around words or emojis for example. "
    "Do not describe actions, stage directions, gestures, or internal thoughts. "
    "Speak exactly what the character would say out loud."
)

PERSONA_A_SYSTEM = PERSONA_A_SYSTEM_ROOT + "\n\n" + PROMPT_APPEND
PERSONA_B_SYSTEM = PERSONA_B_SYSTEM_ROOT + "\n\n" + PROMPT_APPEND

async def query_llm(llm, system_prompt: str, user_input: str) -> str:
    """Send a single text query to Pipecat LLM."""
    # Build context object with system + user message
    context = GoogleLLMContext()
    context.system_message = system_prompt
    
    # Try 1
    # Here it knows the last message | Didn't work as intended
    # context.set_messages([{"role": "user", "content": user_input}])
    
    # Try 2
    # Here it knows the last message but is instructed to stay in character | Didn't work as intended
    # context.set_messages([
    # {
    #     "role": "user",
    #     "content": f"{user_input}\n\nStay in character. Avoid overexplaining or storytelling."
    # }
    # ])
    
    #Try 3
    # Here it is reminded the system prompt and given the last exchange | Seems to work better
    context.set_messages([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ])

    result = await llm.run_inference(context)
    
    # Try to extract text safely
    if isinstance(result, dict):
        return result.get("text", "").strip()
    elif hasattr(result, "text"):
        return result.text.strip()
    else:
        return str(result).strip()


RUNTIME_CONFIG = {
    "llm": {
        "model": "gemini-2.5-flash",
        "temperature": 0.2,
        "max_tokens": 2048,
    },
}

async def main():
    config_manager = ConfigManager()
    config = config_manager.config
    keys = get_api_keys()
    google_key = keys["google"]
    apply_runtime_config_overrides(RUNTIME_CONFIG)

    # Build both LLMs
    door_llm = build_google_llm(config, google_key)
    drunk_llm = build_google_llm(config, google_key)

    # Clear dialogue log
    DIALOGUE_PATH.parent.mkdir(exist_ok=True)
    DIALOGUE_PATH.write_text("=== Velvet Room Conversation ===\n\n")

    drunk_line = "Heeey door, ol' pal... open up for me, huh?"
    with open(DIALOGUE_PATH, "a", encoding="utf-8") as f:
        f.write(f"Drunk Uncle: {drunk_line}\n\n")

    current_speaker = "door"
    last_message = drunk_line
    
    try:
        while True:
            if current_speaker == "door":
                line = await query_llm(door_llm, PERSONA_A_SYSTEM, last_message)
                speaker = "Door"
                current_speaker = "drunk"
            else:
                line = await query_llm(drunk_llm, PERSONA_B_SYSTEM, last_message)
                speaker = "Drunk Uncle"
                current_speaker = "door"
                
            last_message = last_message + '\n' + line
            #print(f"{speaker}: {line}")
            with open(DIALOGUE_PATH, "a", encoding="utf-8") as f:
                f.write(f"{speaker}: {line}\n\n")
                
            if "<UNLOCK>" in line:
                break
        print(f"\nConversation complete! Check {DIALOGUE_PATH.absolute()}")
        
    except KeyboardInterrupt:
        print("\nConversation stopped by user. Goodbye!")


if __name__ == "__main__":
    asyncio.run(main())
