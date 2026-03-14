LLM Config
==========

Purpose
-------

EAA separates model construction from task-manager logic through small config
objects in ``eaa.api.llm_config``. The task manager passes the selected config
to ``build_chat_model()``, which returns a LangChain chat model.

Available config classes
------------------------

``LLMConfig``
   Empty base class used for typing and shared config helpers.

``OpenAIConfig``
   Configuration for OpenAI-compatible chat endpoints. Fields:

   - ``model``
   - ``base_url``
   - ``api_key``

``AskSageConfig``
   Configuration for AskSage endpoints. Fields:

   - ``model``
   - ``server_base_url``
   - ``user_base_url``
   - ``api_key``
   - ``email``
   - ``cacert_path``

``ArgoConfig``
   Configuration for Argo endpoints. Fields:

   - ``model``
   - ``base_url``
   - ``api_key``
   - ``user``

How the config is used
----------------------

``BaseTaskManager.build_model()`` calls ``build_chat_model(self.llm_config)``.
In the current implementation:

- ``OpenAIConfig`` and ``ArgoConfig`` are treated as OpenAI-compatible
  configurations
- ``AskSageConfig`` uses ``server_base_url`` as the model endpoint

Example
-------

.. code-block:: python

   from eaa.api.llm_config import OpenAIConfig

   llm_config = OpenAIConfig(
       model="gpt-4o-mini",
       base_url="https://api.openai.com/v1",
       api_key="YOUR_API_KEY",
   )

Relationship to memory
----------------------

``MemoryManagerConfig`` can optionally carry its own ``llm_config`` override.
If that override is not supplied, the memory manager reuses the task manager's
main ``llm_config`` for embedding calls.
