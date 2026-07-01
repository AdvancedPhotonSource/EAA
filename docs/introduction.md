# Introduction

## Overview

The current EAA runtime is organized around a single base task manager that
coordinates five concerns:

- model construction and invocation
- tool registration and serialized tool execution
- conversation state and checkpoint persistence
- optional long-term memory
- optional WebUI integration

At a high level the pieces connect like this:

```text
user / browser
     |
     v
BaseTaskManager
     |
     +--> chat_graph / task_graph / custom workflow
     |
     +--> chat model built from an LLMConfig
     |
     +--> SerialToolExecutor --> BaseTool instances / MCPTool wrappers
     |
     +--> MemoryManager --> vector store (Chroma in the current built-in path)
     |
     +--> checkpoint SQLite DB + transcript SQLite DB
     |
     +--> WebUI runtime API <--> FastAPI WebUI
```

## Key components

- `BaseTaskManager`: owns runtime state, graphs, persistence, model invocation,
  and tool execution.
- `LLMConfig`: declares how the chat model is created. The shipped subclasses
  are `OpenAIConfig`, `AskSageConfig`, and `ArgoConfig`.
- `BaseTool`: declares stateful tool objects. Methods decorated with `@tool`
  become model-callable tools.
- `MemoryManager`: adds optional retrieval and saving of user memories on chat
  turns.
- `WebUI`: runs as a separate browser-facing process that proxies API calls to
  the task-manager-owned WebUI runtime server. Checkpoints and durable
  transcript display use separate SQLite paths by default.

## Current workflow boundary

The base runtime ships the reusable chat graph.

Some concrete managers define custom `task_graph` implementations, such as the
monitoring loop, graph-based focusing workflow, and multi-agent ROI-search
workflow. Other managers reuse the base chat graph or implement analytical
workflows directly in Python while still updating the shared transcript and
WebUI state through task-manager helpers.
