Skills
======

What skills are
---------------

In EAA, a skill is a reusable, markdown-first task package that the agent can
discover and use at runtime. Skills are not Python subclasses. Instead, they
are directories of documentation that the agent can inspect through the
``SkillLibraryTool``.

At load time, EAA:

- makes the skill library tool available to the agent
- scans the configured skill directories for ``SKILL.md``
- extracts basic metadata such as the skill name and description
- exposes a catalog tool plus a loader tool for fetching skill docs on demand

This lets the agent pull in focused instructions for a workflow without baking
every workflow into one prompt or one task-manager class.

Skills folder
-------------

The repository ships bundled skills under ``src/eaa/skills/``. Each skill lives
in its own directory and must contain a ``SKILL.md`` file.

A typical structure is:

.. code-block:: text

   my-skill/
     SKILL.md
     references/
       api_reference.md
       figure.png

The current loader behavior is:

- if a configured directory itself contains ``SKILL.md``, it is treated as one
  skill
- otherwise, EAA recursively searches below that directory for ``SKILL.md``
- markdown files are collected and returned to the agent as documentation
- image references inside markdown are resolved relative to the markdown file

Only markdown files are collected as skill docs by the built-in loader.

How to use skills in EAA
------------------------

1. Point the task manager at one or more skill directories with ``skill_dirs``.
2. Build the task manager normally.
3. Use the interactive chat commands or tool-calling flows to inspect skills.

Example:

.. code-block:: python

   from eaa_core.task_manager.base import BaseTaskManager

   task_manager = BaseTaskManager(
       llm_config=llm_config,
       tools=[acquisition_tool],
       skill_dirs=["./src/eaa/skills", "~/.eaa_skills"],
   )

Once loaded, the base chat loop supports:

- ``/skill`` to display the discovered skills

Skills are currently best understood as documented agent playbooks that can be
loaded on demand through the skill library tool.

Installing bundled skills elsewhere
-----------------------------------

If you want to copy the packaged skills to a user-controlled directory, use the
CLI helper:

.. code-block:: bash

   python -m eaa_core.cli install-skills --destination ~/.eaa_skills

You can then add that directory to ``skill_dirs``.
