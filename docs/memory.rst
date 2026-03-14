Memory
======

MemoryManagerConfig
-------------------

Long-term memory is optional and is configured through
``eaa.api.memory.MemoryManagerConfig``.

Important fields:

- ``enabled``: turn long-term memory on or off
- ``save_enabled``: allow memory saving
- ``retrieval_enabled``: allow memory retrieval
- ``top_k``: number of retrieved candidates
- ``score_threshold``: minimum relevance score kept
- ``embedding_model``: embedding model name
- ``llm_config``: optional embedding-only config override
- ``persist_directory``: on-disk storage directory
- ``collection_name``: Chroma collection name
- ``namespace``: logical partition for one session or manager
- ``trigger_phrases``: phrases that cause a user message to be saved

Current built-in behavior
-------------------------

In the current repository, ``MemoryManager.build_store()`` creates a
Chroma-backed vector store. When memory is enabled:

- chat turns can retrieve relevant user memories
- retrieved memories are injected into the model context as a system message
- user messages that match a trigger phrase can be saved as memories

Memory retrieval is only wired into the chat graph path, not every possible
custom workflow.

Saving memory with trigger words
--------------------------------

By default, a user message is considered a memory-saving request when it
contains one of the built-in trigger phrases, such as:

- ``remember this``
- ``remember that``
- ``note that``
- ``keep in mind``
- ``please remember``
- ``remember:``

Example:

.. code-block:: text

   remember this: the sample drifted after 3 pm when the enclosure fan was on

The memory manager strips the trigger phrase and saves the remaining text when
possible.

Example configuration
---------------------

.. code-block:: python

   from eaa.api.llm_config import OpenAIConfig
   from eaa.api.memory import MemoryManagerConfig

   memory_config = MemoryManagerConfig(
       enabled=True,
       persist_directory=".eaa_memory",
       namespace="beamtime-session-a",
       embedding_model="text-embedding-3-small",
       llm_config=OpenAIConfig(
           model="unused-for-embeddings",
           base_url="https://api.openai.com/v1",
           api_key="YOUR_API_KEY",
       ),
   )

Status note
-----------

The repository defines a ``postgresql_vector_store`` optional dependency set,
but the built-in memory manager code path documented here currently wires up
Chroma rather than a PostgreSQL-backed store.
