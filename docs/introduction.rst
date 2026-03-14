Introduction
============

Overview
--------

The current EAA runtime is organized around a single base task manager that
coordinates five concerns:

- model construction and invocation
- tool registration and serialized tool execution
- conversation state and checkpoint persistence
- optional long-term memory
- optional WebUI integration

At a high level the pieces connect like this:

.. code-block:: text

   user / browser
        |
        v
   BaseTaskManager
        |
        +--> chat_graph / feedback_loop_graph / custom workflow
        |
        +--> chat model built from an LLMConfig
        |
        +--> SerialToolExecutor --> BaseTool instances / MCPTool wrappers
        |
        +--> MemoryManager --> vector store (Chroma in the current built-in path)
        |
        +--> SQLite session DB --> checkpoints + WebUI relay tables
                                      |
                                      v
                                   FastAPI WebUI

Key components
--------------

``BaseTaskManager``
   Owns runtime state, graphs, persistence, model invocation, and tool
   execution.

``LLMConfig``
   Declares how the chat model is created. The shipped subclasses are
   ``OpenAIConfig``, ``AskSageConfig``, and ``ArgoConfig``.

``BaseTool``
   Declares stateful tool objects. Methods decorated with ``@tool`` become
   model-callable tools.

``MemoryManager``
   Adds optional retrieval and saving of user memories on chat turns.

``WebUI``
   Runs as a separate process. It reads from and writes to the same SQLite file
   used by the task manager.

Current workflow boundary
-------------------------

The reusable built-in graphs shipped by the base runtime are:

- the chat graph
- the feedback-loop graph

The repository currently does not ship task-manager subclasses with their own
custom ``task_graph`` implementation. Instead, concrete managers mostly either
reuse the base graphs or implement analytical workflows directly in Python while
still updating the shared transcript and WebUI state through task-manager
helpers.
