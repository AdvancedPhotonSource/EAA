Installation
============

Recommended: ``uv``
-------------------

At the repository root:

.. code-block:: bash

   uv sync
   source .venv/bin/activate
   which python

After activation, ``which python`` should resolve to ``.venv/bin/python``.

Optional extras are declared in ``pyproject.toml``. The most relevant ones are:

- ``docs`` for the Sphinx documentation toolchain
- ``asksage`` for AskSage-backed model access
- ``aps_mic`` for APS microscopy integrations
- ``postgresql_vector_store`` for external PostgreSQL/pgvector client
  dependencies

Install an extra set with:

.. code-block:: bash

   uv sync --extra docs

Editable install with ``pip``
-----------------------------

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -e .

With extras:

.. code-block:: bash

   pip install -e .[docs]

Documentation build
-------------------

.. code-block:: bash

   uv sync --extra docs
   source .venv/bin/activate
   cd docs
   make html

The built HTML site is written to ``docs/_build/html/``.

Read the Docs
-------------

This repository includes ``.readthedocs.yaml`` and ``docs/requirements.txt``.
Read the Docs builds the Sphinx site from ``docs/conf.py`` and installs the
documentation dependencies declared in ``docs/requirements.txt``.

Status note on memory dependencies
----------------------------------

The project defines a ``postgresql_vector_store`` extra, but the built-in
``MemoryManager`` path in the current repository creates a Chroma-backed store.
Use the PostgreSQL extra only if you are extending the memory stack yourself or
maintaining local integrations around it.
