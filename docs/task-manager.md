# Task Manager

## BaseTaskManager

`BaseTaskManager` is the main integration point in EAA. It owns:

- the model object
- the registered tools
- long-term memory
- SQLite persistence
- graph compilation and execution
- active transcript state

Construction options that matter most in practice are:

- `llm_config`
- `name`
- `memory_config`
- `tools`
- `skill_dirs`
- `checkpoint_db_path`
- `session_id`
- `transcript_db_path`
- `transcript_table_name`
- `use_webui`
- `webui_runtime_host`
- `webui_runtime_port`
- `webui_upload_dir`
- `prune_checkpoints`
- `build`
- `is_subagent`

`session_db_path` is no longer supported. Use `checkpoint_db_path` for
LangGraph checkpoints and `transcript_db_path` for durable transcript display.
Built-in coding, workspace, image-captioning, uv, and subagent tools are
configured after construction through `task_manager.tool_manager`; they are not
constructor flags.

## Built-in graphs

### Chat graph

`build_chat_graph()` creates the reusable conversation graph. It:

- accepts bootstrap input or waits for user input
- calls the model
- executes any tool calls
- returns to user input when the assistant produces a plain response

This is what backs `run_conversation()` and
`run_conversation_from_checkpoint()`.

## Customization

### Custom graph

`build_task_graph()` is the hook for a task-manager-specific LangGraph
workflow. The base class returns `None`; if you want a custom graph, override it
and implement `run()` or `run_from_checkpoint()` around it.

```python
from langgraph.graph import END, START, StateGraph

from eaa_core.task_manager.base import BaseTaskManager
from eaa_core.task_manager.state import TaskManagerState


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
        if self.checkpoint_db_path is not None:
            graph, checkpoint_config, _ = self.get_checkpointed_graph(
                "task_graph",
                load_state=False,
            )
            self.task_graph = graph
            graph_kwargs["config"] = checkpoint_config
        final_state = graph.invoke(initial_state, **graph_kwargs)
        self.set_active_state(CustomState.model_validate(final_state), "task_graph")
```

Built-in graph-based examples include `MonitoringTaskManager`,
`ScanningMicroscopeFocusingTaskManager`, and `MultiAgentROISearchTaskManager`.
Other task managers use the base chat graph or direct Python orchestration.

### Custom workflow without a graph

Several analytical task managers in the repository do not rely on a custom
LangGraph workflow. Instead, they orchestrate the experiment directly in Python
and keep the agent-owned transcript updated through task-manager helpers.

The relevant helpers are:

- `record_system_message()` for narrative progress updates
- `update_message_history()` for explicit transcript mutations
- `publish_webui_message()` for live WebUI display messages
- `record_transcript_message()` for durable transcript writes
- `run_conversation()` if control should fall back to free-form chat after the
  analytical workflow finishes

This pattern is used heavily in `AnalyticalScanningMicroscopeFocusingTaskManager`:
the code performs explicit steps, emits status messages as it goes, and can
still enter chat mode at the end.

## Checkpointing and resume

Checkpointing uses the same SQLite database referenced by `checkpoint_db_path`
by default. The `run_*_from_checkpoint()` methods also accept a
`checkpoint_db_path` override when you need to resume from a different SQLite
file. The checkpoint database stores LangGraph checkpoints.

WebUI display messages live in `transcript_db_path`. Browser input and status
live in the task-manager-owned runtime controller.

The base task manager exposes two resume paths:

- `run_conversation_from_checkpoint()`
- `run_from_checkpoint()` for managers that implement `task_graph`

Important details:

- `prune_checkpoints=True` keeps only the newest checkpoint per graph thread
- chat and task graph each get their own checkpoint thread id
- live WebUI messages, approvals, interrupts, and input state are owned by the
  runtime controller
- `transcript_db_path` is a durable transcript log written by task-manager
  history helpers
