"""WeUI based on Chainlit. To use this WebUI, you need to make some slight
modifications to your run script.

Example: 

If you originally have
```
task_manager.run_fov_search(
    feature_description="a camera or optical device",
    y_range=(0, 800),
    x_range=(0, 800),
    fov_size=(200, 200),
    step_size=(200, 200)
)
```

Change it to:
```
from eaa.gui.chat import *
set_task_manager(task_manager)
set_function_to_run(
    task_manager.run_fov_search, 
    kwargs={
        "feature_description": "a camera or optical device",
        "y_range": (0, 800),
        "x_range": (0, 800),
        "fov_size": (200, 200),
        "step_size": (200, 200)
    }
)
```
Then instead of running the script with `python`, run it with
```
chainlit run myscript.py
```
"""

import chainlit as cl
import asyncio
import threading
import re
import base64

from eaa.agents.openai import print_message


_task_manager = None
_function_to_run = None
_kwargs = None


def set_task_manager(task_manager):
    global _task_manager
    _task_manager = task_manager


def get_task_manager():
    global _task_manager
    return _task_manager


def set_function_to_run(function_to_run, kwargs=None):
    global _function_to_run
    global _kwargs
    if kwargs is None:
        kwargs = {}
    _function_to_run = function_to_run
    _kwargs = kwargs
    

def compose_chainlit_message(message: dict) -> cl.Message:
    elements = []
    content = ""
    role = message["role"]
    if isinstance(message["content"], list) and "type" in message["content"][0]:
        for item in message["content"]:
            if item["type"] == "image_url":
                img_base64 = item["image_url"]["url"]
                base64_data = re.sub("^data:image/.+;base64,", "", img_base64)
                image_bytes = base64.b64decode(base64_data)
                image = cl.Image(name="image", display="inline", content=image_bytes)
                elements.append(image)
        content += print_message(message, return_string=True)
    elif (
        "tool_calls" in message 
        and isinstance(message["tool_calls"], list) 
        and len(message["tool_calls"]) > 0
    ):
        tool_call_message = print_message(message, return_string=True)
        content += tool_call_message
    elif role == "tool":
        tool_response_message = print_message(message, return_string=True)
        content += tool_response_message
    else:
        if message["content"] is None:
            message["content"] = ""
        content += message["content"]
    return cl.Message(
        content=content,
        author=role,
        elements=elements
    )


@cl.on_chat_start
async def on_chat_start():    
    await cl.Message("Chat started. Listening for external messages...").send()

    # Start background thread only once
    if not hasattr(cl.user_session, "thread_started"):
        threading.Thread(target=_function_to_run, kwargs=_kwargs, daemon=True).start()
        cl.user_session.thread_started = True

    # Async polling loop to check for new messages
    async def poll_new_messages():
        last_index = 0
        while True:
            messages = _task_manager.full_history
            if last_index < len(messages):
                for i_msg in range(last_index, len(messages)):
                    cl_message = compose_chainlit_message(messages[i_msg])
                    await cl_message.send()
                last_index = len(messages)
            await asyncio.sleep(1)

    # Start the polling task
    cl.user_session.set("polling_task", asyncio.create_task(poll_new_messages()))

@cl.on_message
async def on_message(message: cl.Message):
    await cl.Message(content=f"You said: {message.content}").send()

