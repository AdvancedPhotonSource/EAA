from typing import Any, Dict, List

from langchain_core.messages import AIMessage
from langchain_core.messages.utils import convert_to_messages
from langchain_openai import ChatOpenAI

from eaa_core.api.llm_config import ArgoConfig, AskSageConfig, LLMConfig, OpenAIConfig
from eaa_core.message_proc import ai_message_to_openai_dict


def build_chat_model(llm_config: LLMConfig) -> ChatOpenAI:
    """Build a LangChain chat model from an EAA LLM config."""
    if isinstance(llm_config, (OpenAIConfig, ArgoConfig)):
        return ChatOpenAI(
            model=llm_config.model,
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
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
