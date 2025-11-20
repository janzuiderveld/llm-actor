from pathlib import Path
import configparser

parser = configparser.ConfigParser()
files = parser.read(Path(__file__).parent / "settings.ini")
print("Loaded settings from:", files)

input_device_index = parser.getint("AUDIO", "input_device_index", fallback=1)
output_device_index = parser.getint("AUDIO", "output_device_index", fallback=2)
if parser.getboolean("AUDIO", "mute_microphone_while_tts", fallback=True):
    aec_setting = "mute_while_tts"
elif parser.getboolean("AUDIO", "echo_cancellation", fallback=False):
    aec_setting = "pyaec"
else:
    aec_setting = "off"

pipeline = parser.get("LLM", "pipeline", fallback="ollama")  # options: "google", "groq", "ollama"
model = parser.get("LLM", "model", fallback="gpt-oss:20b")
# options: GOOGLE "gemini-2.5-flash", 
#          GROQ "openai/gpt-oss-20b", ...
#          OLLAMA "deepseek-r1:1.5b", "deepseek-r1:32b", "gpt-oss:20b"
temperature = parser.getfloat("LLM", "temperature", fallback=0.2)
mode = parser.get("LLM", "mode", fallback="2personas")  # options: "1to1", "2personas", "NARRATOR"

sys_prompt = parser.get("SYSTEM", "prompt")
sys_voice = parser.get("SYSTEM", "voice", fallback="aura-2-thalia-en")

p1_name = parser.get("PERSONA_1", "name", fallback="UNCLE")
p1_opening = parser.get("PERSONA_1", "opening")
p1_prompt = parser.get("PERSONA_1", "prompt")
p1_voice = parser.get("PERSONA_1", "voice")

p2_name = parser.get("PERSONA_2", "name", fallback="DOOR")
p2_opening = parser.get("PERSONA_2", "opening")
p2_prompt = parser.get("PERSONA_2", "prompt")
p2_voice = parser.get("PERSONA_2", "voice")

n_name = parser.get("NARRATOR", "name", fallback="NARRATOR")
n_prompt = parser.get("NARRATOR", "prompt")
n_voice = parser.get("NARRATOR", "voice")