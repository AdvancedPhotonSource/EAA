Skills
======

What skills are
---------------

In EAA, a skill is a reusable, markdown-first task package that the agent can
discover and use at runtime. Skills are not Python subclasses. Instead, they
are directories of documentation that the task manager exposes through generated
``SkillTool`` wrappers.

At load time, EAA:

- scans the configured skill directories for ``SKILL.md``
- extracts basic metadata such as the skill name and description
- assigns each skill a tool name like ``skill-feature-tracking-task-manager``
- exposes a tool that returns the skill documentation files to the agent

This lets the agent pull in focused instructions for a subtask without baking
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
- duplicate tool names are automatically disambiguated

Only markdown files are collected as skill docs by the built-in loader.

How to use skills in EAA
------------------------

1. Point the task manager at one or more skill directories with ``skill_dirs``.
2. Build the task manager normally.
3. Use the interactive chat commands to inspect or launch skills.

Example:

.. code-block:: python

   from eaa.core.task_manager.base import BaseTaskManager

   task_manager = BaseTaskManager(
       llm_config=llm_config,
       tools=[acquisition_tool],
       skill_dirs=["./src/eaa/skills", "~/.eaa_skills"],
   )

Once loaded, the base chat loop supports:

- ``/skill`` to display the discovered skills
- ``/subtask <task description>`` to let the agent choose a skill, fetch its
  docs, and run a skill-driven subtask flow

What happens during ``/subtask``
--------------------------------

When you launch a subtask, the task manager:

- records the available skill catalog in context
- exposes the current task-manager metadata to the model
- runs the feedback-loop graph with instructions to select the right skill
- lets the model call the generated skill tool to fetch the markdown docs
- expands those docs back into conversation messages for the subtask

This means skills are currently best understood as documented agent playbooks
that can be injected on demand.

Installing bundled skills elsewhere
-----------------------------------

If you want to copy the packaged skills to a user-controlled directory, use the
CLI helper:

.. code-block:: bash

   python -m eaa.cli install-skills --destination ~/.eaa_skills

You can then add that directory to ``skill_dirs``.
