Task Manager
============

BaseTaskManager
---------------

``BaseTaskManager`` is the main integration point in EAA. It owns:

- the model object
- the registered tools
- long-term memory
- SQLite persistence
- graph compilation and execution
- active transcript state

Construction options that matter most in practice are:

- ``llm_config``
- ``memory_config``
- ``tools``
- ``skill_dirs``
- ``session_db_path``
- ``use_webui``
- ``use_coding_tools``
- ``run_codes_in_sandbox``
- ``prune_checkpoints``

Built-in graphs
---------------

Chat graph
~~~~~~~~~~

``build_chat_graph()`` creates the reusable conversation graph. It:

- accepts bootstrap input or waits for user input
- calls the model
- executes any tool calls
- returns to user input when the assistant produces a plain response

This is what backs ``run_conversation()`` and
``run_conversation_from_checkpoint()``.

Feedback-loop graph
~~~~~~~~~~~~~~~~~~~

``build_feedback_loop_graph()`` creates the iterative tool-driving graph. It:

- starts from an ``initial_prompt``
- expects tool calls on each model turn
- optionally reprompts the model if the turn is invalid
- executes tool calls and injects follow-up messages
- supports human gates via ``NEED HUMAN`` and termination via ``TERMINATE``

This is what backs ``run_feedback_loop()`` and
``run_feedback_loop_from_checkpoint()``.

Customization
-------------

Custom graph
~~~~~~~~~~~~

``build_task_graph()`` is the hook for a task-manager-specific LangGraph
workflow. The base class returns ``None``; if you want a custom graph, override
it and implement ``run()`` or ``run_from_checkpoint()`` around it.

.. code-block:: python

   from langgraph.graph import END, START, StateGraph

   from eaa.core.task_manager.base import BaseTaskManager
   from eaa.core.task_manager.state import TaskManagerState


   class CustomState(TaskManagerState):
       pass


   class CustomTaskManager(BaseTaskManager):
       def build_task_graph(self, checkpointer=None):
           builder = StateGraph(CustomState)
           builder.add_node("call_model", self.node_factory.call_model, input_schema=CustomState)
           builder.add_edge(START, "call_model")
           builder.add_edge("call_model", END)
           return builder.compile(checkpointer=checkpointer)

       def run(self):
           initial_state = CustomState(
               messages=list(self.context),
               full_history=list(self.full_history),
           )
           graph = self.task_graph
           graph_kwargs = {}
           if self.session_db_path is not None:
               graph, checkpoint_config = self.get_checkpointed_graph("task_graph")
               self.task_graph = graph
               graph_kwargs["config"] = checkpoint_config
           final_state = graph.invoke(initial_state, **graph_kwargs)
           self.state = CustomState.model_validate(final_state)

Status note:
   The repository currently does not ship built-in task-manager subclasses that
   define their own ``task_graph``. The custom graph hook is present for
   extension work.

Custom workflow without a graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several analytical task managers in the repository do not rely on a custom
LangGraph workflow. Instead, they orchestrate the experiment directly in Python
and keep the shared transcript/WebUI updated through task-manager helpers.

The relevant helpers are:

- ``record_system_message()`` for narrative progress updates
- ``update_message_history()`` for explicit transcript mutations
- ``add_webui_message_to_db()`` for display-only WebUI messages
- ``run_conversation()`` if control should fall back to free-form chat after
  the analytical workflow finishes

This pattern is used heavily in
``AnalyticalScanningMicroscopeFocusingTaskManager``: the code performs explicit
steps, emits status messages as it goes, and can still enter chat mode at the
end.

Checkpointing and resume
------------------------

Checkpointing uses the same SQLite database referenced by ``session_db_path``.
That file can hold:

- LangGraph checkpoints
- explicit WebUI display messages
- queued WebUI input
- WebUI status flags

The base task manager exposes three resume paths:

- ``run_conversation_from_checkpoint()``
- ``run_feedback_loop_from_checkpoint()``
- ``run_from_checkpoint()`` for managers that implement ``task_graph``

Important details:

- ``prune_checkpoints=True`` keeps only the newest checkpoint per graph thread
- chat, feedback loop, and task graph each get their own checkpoint thread id
- the WebUI can read the transcript from explicit WebUI messages or, if needed,
  directly from the latest checkpointed state
