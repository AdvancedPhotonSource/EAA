WebUI
=====

Purpose
-------

The EAA WebUI is a lightweight standalone interface for watching agent progress
and sending user input from a browser. It does not run inside the task-manager
process. Instead, it communicates through a shared SQLite database.

Launch pattern
--------------

Configure the task manager with the same SQLite file you want the WebUI to use:

.. code-block:: python

   task_manager = BaseTaskManager(
       ...,
       session_db_path="session.sqlite",
       use_webui=True,
   )

Then start the WebUI process:

.. code-block:: python

   from eaa.gui.chat import run_webui, set_message_db_path

   set_message_db_path("session.sqlite")
   run_webui(host="127.0.0.1", port=8008)

Communication mechanism
-----------------------

The browser-facing FastAPI server reads and writes a shared SQLite file. The
main tables involved are:

``webui_messages``
   Explicit display messages pushed by the task manager.

``webui_inputs``
   User messages submitted from the browser and consumed by the agent process.

``status``
   WebUI status flags, including whether the agent is currently waiting for
   user input.

``checkpoints`` and ``writes``
   LangGraph checkpoint tables used when checkpointing is enabled.

The WebUI first tries to read explicit display messages. If none are present, it
can reconstruct the transcript from the latest checkpointed state.

Brief design introduction
-------------------------

The current WebUI has a deliberately small design:

- FastAPI backend in ``eaa.gui.chat``
- static frontend assets in ``src/eaa/gui/webui_static/``
- polling-based message/status updates
- clipboard image upload support
- no tight coupling to a specific task-manager subclass

This design keeps the UI process simple and lets the task manager remain the
source of truth for workflow execution.
