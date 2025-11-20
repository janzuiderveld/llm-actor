from pathlib import Path
import configparser

parser = configparser.ConfigParser()
files = parser.read(Path(__file__).parent / "settings.ini")
print("Loaded settings from:", files)

input_device_index = parser.getint("audio", "input_device_index", fallback=1)
output_device_index = parser.getint("audio", "output_device_index", fallback=2)
if parser.getboolean("audio", "mute_microphone_while_tts", fallback=True):
    aec_setting = "mute_while_tts"
elif parser.getboolean("audio", "echo_cancellation", fallback=False):
    aec_setting = "pyaec"
else:
    aec_setting = "off"

pipeline = parser.get("llm", "pipeline", fallback="ollama")  # options: "google", "groq", "ollama"
model = parser.get("llm", "model", fallback="gpt-oss:20b")
# options: GOOGLE "gemini-2.5-flash", 
#          GROQ "openai/gpt-oss-20b", ...
#          OLLAMA "deepseek-r1:1.5b", "deepseek-r1:32b", "gpt-oss:20b"
temperature = parser.getfloat("llm", "temperature", fallback=0.2)
mode = parser.get("llm", "mode", fallback="2personas")  # options: "1to1", "2personas", "narrator"

sys_prompt = parser.get("system", "prompt")
sys_voice = parser.get("system", "voice", fallback="aura-2-thalia-en")

p1_name = parser.get("persona1", "name", fallback="UNCLE")
p1_opening = parser.get("persona1", "opening")
p1_prompt = parser.get("persona1", "prompt")
p1_voice = parser.get("persona1", "voice")

p2_name = parser.get("persona2", "name", fallback="DOOR")
p2_opening = parser.get("persona2", "opening")
p2_prompt = parser.get("persona2", "prompt")
p2_voice = parser.get("persona2", "voice")

n_name = parser.get("narrator", "name", fallback="NARRATOR")
n_prompt = parser.get("narrator", "prompt")
n_voice = parser.get("narrator", "voice")