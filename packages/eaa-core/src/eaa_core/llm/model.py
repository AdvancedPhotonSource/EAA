import re
from hashlib import sha256
from typing import Any, Dict, List

from langchain_core.messages import AIMessage
from langchain_core.messages.utils import convert_to_messages
from langchain_openai import ChatOpenAI

from eaa_core.api.llm_config import AskSageConfig, LLMConfig, OpenAIConfig
from eaa_core.message_proc import ai_message_to_openai_dict


def build_chat_model(
    llm_config: LLMConfig,
    session_name: str = "default",
) -> ChatOpenAI:
    """Build a LangChain chat model from an EAA LLM config.

    Parameters
    ----------
    llm_config : LLMConfig
        Model and endpoint configuration.
    session_name : str, default="default"
        Stable session identifier used to namespace OpenAI prompt caching.
    """
    if isinstance(llm_config, OpenAIConfig):
        model_kwargs = (
            _prompt_cache_kwargs(llm_config.model, session_name)
            if llm_config.model_kwargs is None
            else llm_config.model_kwargs
        )
        return ChatOpenAI(
            model=llm_config.model,
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
            **model_kwargs,
        )
    if isinstance(llm_config, AskSageConfig):
        if llm_config.server_base_url is None:
            raise ValueError("AskSageConfig.server_base_url must be provided.")
        return ChatOpenAI(
            model=llm_config.model,
            base_url=llm_config.server_base_url,
            api_key=llm_config.api_key,
        )
    raise TypeError(f"Unsupported llm config type: {type(llm_config)}")


def _prompt_cache_kwargs(
    model_name: str | None,
    session_name: str,
) -> dict[str, Any]:
    """Return provider-specific arguments that enable prompt caching."""
    normalized_name = (model_name or "").lower()
    if "claude" in normalized_name or "anthropic" in normalized_name:
        return {"extra_body": {"cache_control": {"type": "ephemeral"}}}
    if _is_openai_model(normalized_name):
        cache_key = sha256(session_name.encode()).hexdigest()[:16]
        return {"model_kwargs": {"prompt_cache_key": f"eaa:{cache_key}"}}
    return {}


def _is_openai_model(model_name: str) -> bool:
    """Return whether a model ID identifies an OpenAI chat model."""
    model_id = model_name.rsplit("/", 1)[-1]
    return model_name.startswith("openai/") or bool(
        re.match(r"^(?:(?:chatgpt|gpt)(?:-|\d)|o\d(?:-|$))", model_id)
    )


def invoke_chat_model(
    llm: ChatOpenAI,
    messages: List[Dict[str, Any]],
    tool_schemas: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Invoke the chat model and normalize the response."""
    bound_llm = llm.bind_tools(tool_schemas) if tool_schemas else llm
    response = bound_llm.invoke(convert_to_messages(messages))
    if not isinstance(response, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(response)}")
    return ai_message_to_openai_dict(response)
