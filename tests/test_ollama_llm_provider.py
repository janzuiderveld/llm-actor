from app.config import RuntimeConfig
from services.llm import build_llm_service, normalize_model_for_service, parse_llm_model_spec


def test_parse_llm_model_spec():
    assert parse_llm_model_spec("ollama-gemma3:4b") == ("ollama", "gemma3:4b")
    assert parse_llm_model_spec("OLLAMA-qwen3-vl") == ("ollama", "qwen3-vl")
    assert parse_llm_model_spec("openai-gpt-5.2-chat-latest") == ("openai", "gpt-5.2-chat-latest")
    assert parse_llm_model_spec("gemini-2.5-flash") == ("google", "gemini-2.5-flash")


def test_build_llm_service_ollama():
    config = RuntimeConfig()
    config.llm.model = "ollama-gemma3:4b"

    service, provider = build_llm_service(config, google_api_key=None, openai_api_key=None)
    assert provider == "ollama"
    assert service.__class__.__name__ == "OLLamaLLMService"


def test_build_llm_service_openai():
    config = RuntimeConfig()
    config.llm.model = "openai-gpt-5.2-chat-latest"

    service, provider = build_llm_service(config, google_api_key=None, openai_api_key="test-key")
    assert provider == "openai"
    assert service.__class__.__name__ == "OpenAILLMService"


def test_build_llm_service_google():
    config = RuntimeConfig()
    config.llm.model = "gemini-2.5-flash"

    service, provider = build_llm_service(config, google_api_key="test-key", openai_api_key=None)
    assert provider == "google"
    assert service.__class__.__name__ == "GoogleLLMService"


def test_normalize_model_for_service():
    config = RuntimeConfig()
    config.llm.model = "ollama-gemma3:4b"
    ollama_service, _ = build_llm_service(config, google_api_key=None, openai_api_key=None)

    assert normalize_model_for_service("ollama-qwen3-vl", ollama_service) == "qwen3-vl"
    assert normalize_model_for_service("gemma3:4b", ollama_service) == "gemma3:4b"

    config.llm.model = "openai-gpt-5.2-chat-latest"
    openai_service, _ = build_llm_service(config, google_api_key=None, openai_api_key="test-key")
    assert normalize_model_for_service("openai-gpt-5.2-chat-latest", openai_service) == "gpt-5.2-chat-latest"

    config.llm.model = "gemini-2.5-flash"
    google_service, _ = build_llm_service(config, google_api_key="test-key", openai_api_key=None)
    assert normalize_model_for_service("ollama-gemma3:4b", google_service) is None
