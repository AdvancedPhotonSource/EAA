from eaa_core.api.llm_config import OpenAIConfig
from eaa_core.llm.model import build_chat_model


def test_anthropic_model_enables_prompt_caching_by_default() -> None:
    config = OpenAIConfig.from_dict(
        {
            "model": "anthropic/claude-sonnet-4",
            "base_url": "https://gateway.example/v1",
            "api_key": "test-key",
        }
    )
    model = build_chat_model(config)

    assert model.extra_body == {"cache_control": {"type": "ephemeral"}}
    assert "prompt_cache_key" not in model.model_kwargs


def test_openai_model_uses_session_prompt_cache_key_by_default() -> None:
    config = OpenAIConfig(model="openai/gpt-4.1", api_key="test-key")

    model = build_chat_model(config, session_name="cache-test-session")

    assert model.model_kwargs["prompt_cache_key"] == "eaa:098493d37d5a7132"
    assert model.extra_body is None


def test_dashless_openai_model_alias_uses_prompt_cache_key() -> None:
    for model_name in ("gpt55", "provider/gpt55"):
        config = OpenAIConfig(model=model_name, api_key="test-key")

        model = build_chat_model(config, session_name="cache-test-session")

        assert model.model_kwargs["prompt_cache_key"] == "eaa:098493d37d5a7132"
        assert model.extra_body is None


def test_unknown_model_does_not_receive_provider_cache_fields() -> None:
    config = OpenAIConfig(model="provider/custom-model", api_key="test-key")

    model = build_chat_model(config)

    assert model.extra_body is None
    assert "prompt_cache_key" not in model.model_kwargs


def test_empty_model_kwargs_disable_default_prompt_caching() -> None:
    for model_name in ("anthropic/claude-sonnet-4", "openai/gpt-4.1"):
        config = OpenAIConfig(
            model=model_name,
            api_key="test-key",
            model_kwargs={},
        )

        model = build_chat_model(config, session_name="cache-test-session")

        assert model.extra_body is None
        assert model.model_kwargs == {}
