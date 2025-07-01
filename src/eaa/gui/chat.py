"""WeUI based on Chainlit. To use this WebUI, create a Python
script named `start_webui.py` that contains the following:

```
from eaa.gui.chat import *
set_message_db_path("messages.db")
```
Replace `messages.db` with the path to the SQLite database that stores 
the chat history. To make the WebUI show messages in real time, this
DB path must match  `message_db_path` given to the task manager.

To launch the WebUI, run the following command:
```
chainlit run start_webui.py
```
"""

import chainlit as cl
import sqlite3
import asyncio
import re
import base64

from eaa.util import get_timestamp


_message_db_path = None
_message_db_conn = None


def set_message_db_path(path: str):
    """Set the path to the SQLite database that stores the chat history.

    Parameters
    ----------
    path : str
        _description_
    """
    global _message_db_path
    _message_db_path = path
    

def get_message_db_path():
    global _message_db_path
    return _message_db_path


def get_message_db_conn():
    global _message_db_conn
    return _message_db_conn


def create_message_db_conn(path: str):
    global _message_db_conn
    _message_db_conn = sqlite3.connect(path)


def set_function_to_run(function_to_run, kwargs=None):
    global _function_to_run
    global _kwargs
    if kwargs is None:
        kwargs = {}
    _function_to_run = function_to_run
    _kwargs = kwargs
    
    
def get_messages():
    global _message_db_conn
    cursor = _message_db_conn.cursor()
    cursor.execute("SELECT timestamp, role, content, tool_calls, image FROM messages ORDER BY rowid")
    return cursor.fetchall()
    

def compose_chainlit_message(
    role: str = None, 
    content: str = None, 
    tool_calls: list = None, 
    image_base64: str = None
) -> cl.Message:
    message_content = "" if content is None else content
    content = f"[Role] {role}\n"
    content += "[Content]\n"
    content += message_content
    
    if tool_calls is not None:
        content += f"[Tool calls]\n{tool_calls}\n"
        
    elements = []
    if image_base64 is not None:
        image_base64 = base64.b64decode(image_base64)
        image = cl.Image(name="image", display="inline", content=image_base64)
        elements.append(image)
    
    return cl.Message(
        content=content,
        author=role,
        elements=elements
    )


@cl.on_chat_start
async def on_chat_start():
    create_message_db_conn(_message_db_path)
    
    await cl.Message("Chat started. Listening for external messages...").send()

    # Async polling loop to check for new messages
    async def poll_new_messages():
        last_index = 0
        while True:
            messages = get_messages()
            if last_index < len(messages):
                for i_msg in range(last_index, len(messages)):
                    image_base64 = messages[i_msg][4]
                    if image_base64 is not None:
                        image_base64 = re.sub("^data:image/.+;base64,", "", image_base64)
                    cl_message = compose_chainlit_message(
                        role=messages[i_msg][1],
                        content=messages[i_msg][2],
                        tool_calls=messages[i_msg][3],
                        image_base64=image_base64
                    )
                    await cl_message.send()
                last_index = len(messages)
            await asyncio.sleep(1)

    # Start the polling task
    cl.user_session.set("polling_task", asyncio.create_task(poll_new_messages()))


@cl.on_message
async def on_message(message: cl.Message):
    _message_db_conn.execute(
        "INSERT INTO messages (timestamp, role, content, tool_calls, image) VALUES (?, ?, ?, ?, ?)",
        (str(get_timestamp(as_int=True)), "user_webui", message.content, None, None)
    )
    _message_db_conn.commit()
    await message.send()
