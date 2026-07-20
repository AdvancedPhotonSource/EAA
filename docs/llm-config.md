# LLM Config

## Purpose

EAA separates model construction from task-manager logic through small config
objects in `eaa_core.api.llm_config`. The task manager passes the selected
config to `build_chat_model()`, which returns a LangChain chat model.

## Available config classes

- `LLMConfig`: empty base class used for typing and shared config helpers.
- `OpenAIConfig`: configuration for OpenAI-compatible chat endpoints. Fields:
  `model`, `base_url`, `api_key`, and `model_kwargs`. Entries in `model_kwargs`
  are passed as additional `ChatOpenAI` constructor arguments.
- `AskSageConfig`: configuration for AskSage endpoints. Fields: `model`,
  `server_base_url`, `user_base_url`, `api_key`, `email`, and `cacert_path`.
- `ArgoConfig`: configuration for Argo endpoints. Fields: `model`,
  `base_url`, `api_key`, `model_kwargs`, and `user`, which is deprecated,
  accepted temporarily, and ignored.

## How the config is used

`BaseTaskManager.build_model()` calls `build_chat_model(self.llm_config)`. In
the current implementation:

- `OpenAIConfig` and `ArgoConfig` are treated as OpenAI-compatible
  configurations
- `AskSageConfig` uses `server_base_url` as the model endpoint

## Example

```python
from eaa_core.api.llm_config import OpenAIConfig

llm_config = OpenAIConfig(
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    api_key="YOUR_API_KEY",
)
```

Prompt caching is configured automatically from `model`. OpenAI model IDs
receive a session-specific `prompt_cache_key`; model IDs containing `claude` or
`anthropic` receive Anthropic's top-level ephemeral `cache_control`. Compatible
gateways must forward these fields to the underlying provider. `model_kwargs`
defaults to `None`, which enables this automatic configuration. Supplying a
mapping replaces the automatic fields, so explicitly setting `model_kwargs={}`
disables prompt caching.

## Relationship to memory

`MemoryManagerConfig` can optionally carry its own `llm_config` override. If
that override is not supplied, the memory manager reuses the task manager's main
`llm_config` for embedding calls.
