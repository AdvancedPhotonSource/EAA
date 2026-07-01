# Skills

## What skills are

In EAA, a skill is a reusable, markdown-first task package that the agent can
discover and use at runtime. Skills are not Python subclasses. Instead, they
are directories of documentation that the agent can inspect through normal
filesystem tools.

At load time, EAA:

- scans the configured skill directories for `SKILL.md`
- extracts metadata such as the skill name, description, and `SKILL.md` path
- injects that metadata into the system prompt
- injects only the selected `SKILL.md` into context when the user enters
  `/skill <name>`

This lets the agent pull in focused instructions for a workflow without baking
every workflow into one prompt or one task-manager class.

## Skills folder

The repository ships bundled skills under package-specific `skills/`
directories. Each skill lives in its own directory and must contain a
`SKILL.md` file.

A typical structure is:

```text
my-skill/
  SKILL.md
  references/
    api_reference.md
    figure.png
```

The current loader behavior is:

- if a configured directory itself contains `SKILL.md`, it is treated as one
  skill
- otherwise, EAA recursively searches below that directory for `SKILL.md`
- only `SKILL.md` is injected automatically when a user selects a skill
- supporting files remain available for explicit inspection through tools such
  as `ls` and `read_file`

## How to use skills in EAA

1. Point the task manager at one or more skill directories with `skill_dirs`.
2. Build the task manager normally.
3. Use the interactive chat commands to list or select skills.

Example:

```python
from eaa_core.task_manager.base import BaseTaskManager

task_manager = BaseTaskManager(
    llm_config=llm_config,
    tools=[acquisition_tool],
    skill_dirs=[
        "./packages/eaa-core/src/eaa_core/skills",
        "./packages/eaa-imaging/src/eaa_imaging/skills",
        "~/.eaa_skills",
    ],
)
```

Once loaded, the base chat loop supports:

- `/skill` to display the discovered skills
- `/skill <name>` to inject the selected skill's `SKILL.md`

Skills are currently best understood as documented agent playbooks that can be
loaded on demand through explicit user selection.

## Adding user skills

Create a user-controlled directory such as `~/.eaa_skills`, put one or more
skill directories below it, and add that path to `skill_dirs`:

```text
~/.eaa_skills/
  my-skill/
    SKILL.md
```

EAA will discover those skills the same way it discovers bundled skills.
