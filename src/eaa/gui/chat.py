"""WebUI based on Gradio.

To use the WebUI, import `launch_gui` from `eaa.gui.chat` and call it after
creating the task manager.

Example:
```python
from eaa.gui.chat import launch_gui

task_manager = TaskManager(...)
launch_gui(task_manager)
```
"""

import threading
from typing import Optional
import re

import gradio as gr

from eaa.task_managers.base import BaseTaskManager
from eaa.util import decode_image_base64
from eaa.agents.openai import print_message


class ChatUI:
    def __init__(self, task_manager: BaseTaskManager):
        self.task_manager = task_manager
        self.chatbot: Optional[gr.Chatbot] = None
        self.blocks: Optional[gr.Blocks] = None
        self._setup_ui()
        
    def _setup_ui(self):
        with gr.Blocks() as self.blocks:
            self.chatbot = gr.Chatbot(type="messages", height="80vh", autoscroll=True)
            timer = gr.Timer(1.0)
            
            # Create a function to update the chat
            def update_chat():
                context = self.task_manager.full_history
                context_processed = []
                for i in range(len(context)):
                    if isinstance(context[i]["content"], list) and "type" in context[i]["content"][0]:
                        for item in context[i]["content"]:
                            if item["type"] == "image_url":
                                img_base64 = item["image_url"]["url"]
                                base64_data = re.sub('^data:image/.+;base64,', '', img_base64)
                                pil_image = decode_image_base64(base64_data, return_type="pil")
                                if pil_image.mode != "RGB":
                                    pil_image = pil_image.convert("RGB")
                                gradio_im = gr.Image(pil_image)
                                context_processed.append({"content": gradio_im, "role": context[i]["role"]})
                            elif item["type"] == "text":
                                context_processed.append({"content": item["text"], "role": context[i]["role"]})
                    elif (
                        "tool_calls" in context[i] 
                        and isinstance(context[i]["tool_calls"], list) 
                        and len(context[i]["tool_calls"]) > 0
                    ):
                        tool_call_message = print_message(context[i], return_string=True)
                        context_processed.append({"content": tool_call_message, "role": context[i]["role"]})
                    elif context[i]["role"] == "tool":
                        tool_response_message = print_message(context[i], return_string=True)
                        context_processed.append({"content": tool_response_message, "role": "user"})
                    else:
                        if context[i]["content"] is None:
                            context[i]["content"] = ""
                        context_processed.append(context[i])
                return context_processed
            
            # Set up periodic updates
            timer.tick(update_chat, None, self.chatbot)
    
    def launch(self, **kwargs):
        """Launch the UI in a non-blocking way"""
        def run_server():
            self.blocks.launch(**kwargs)
            
        # Start the server in a separate thread
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True  # Make thread daemon so it exits when main program exits
        server_thread.start()
        
        return server_thread


def launch_gui(task_manager: BaseTaskManager, **kwargs) -> threading.Thread:
    """
    Launch the GUI in a non-blocking way.
    
    Parameters
    ----------
    task_manager : BaseTaskManager
        The task manager instance to use
    **kwargs : dict
        Additional arguments to pass to gr.Blocks.launch()
        
    Returns
    -------
    threading.Thread
        The thread running the Gradio server
    """
    ui = ChatUI(task_manager)
    return ui.launch(**kwargs)
