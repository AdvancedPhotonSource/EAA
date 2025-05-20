import autogen


def register_hooks(agent: autogen.ConversableAgent, *args, **kwargs) -> None:
    model_name = agent.llm_config.config_list[0].model
    if model_name == "Qwen/Qwen2.5-VL-72B-Instruct":
        agent.register_hook("process_message_before_send", qwen_25_vl_presend_hook)
        

def register_tool_executor_hook(agent: autogen.ConversableAgent, *args, **kwargs) -> None:
    agent.register_hook("process_message_before_send", tool_executor_hook)


def qwen_25_vl_presend_hook(
    sender: autogen.ConversableAgent, 
    message: list[dict],
    recipient: autogen.Agent,
    silent: bool,
) -> None:
    if (
        message["content"] is None 
        and "reasoning_content" in message.keys() 
        and message["reasoning_content"] is not None
    ):
        message["content"] = message["reasoning_content"]
    return message


def tool_executor_hook(
    sender: autogen.ConversableAgent, 
    message: list[dict],
    recipient: autogen.Agent,
    silent: bool,
) -> None:
    if isinstance(message, dict) and "tool_responses" in message.keys():
        message["tool_call_id"] = message["tool_responses"][-1]["tool_call_id"]
    return message