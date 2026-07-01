# eaa-core

`eaa-core` contains the shared runtime for EAA:

- `BaseTaskManager`, graph construction, checkpointing, transcripts, and WebUI
  runtime integration
- `BaseTool`, serialized tool execution, built-in workspace/coding/subagent
  tools, and MCP client/server helpers
- LLM and memory configuration models
- reusable skills and workflow prompt templates

Application packages such as `eaa-imaging` and `eaa-spectroscopy` build on this
package instead of duplicating task-manager and tool infrastructure.
