from eaa_core.task_manager.base import BaseTaskManager
from eaa_core.task_manager.nodes import NodeFactory
from eaa_core.task_manager.skills import SkillMetadata, discover_skills, resolve_skill
from eaa_core.task_manager.state import ChatGraphState, FeedbackLoopState
from eaa_imaging.task_manager.imaging.base import ImagingBaseTaskManager


class QueuedInputTaskManager(BaseTaskManager):
    """Task manager test double with queued user inputs."""

    def __init__(self, user_inputs: list[str], **kwargs):
        self.user_inputs = list(user_inputs)
        super().__init__(**kwargs)

    def get_user_input(self, *args, **kwargs) -> str:
        """Return the next queued user input."""
        return self.user_inputs.pop(0)

    def build_model(self, *args, **kwargs):
        """Skip model construction in command-ingestion tests."""

    def build_memory_store(self) -> None:
        """Skip memory setup in command-ingestion tests."""

    def build_chat_graph(self, checkpointer=None):
        """Skip graph construction in command-ingestion tests."""
        return None


def test_discover_skills_returns_skill_md_paths(tmp_path):
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\n"
        "name: demo\n"
        "description: Demo skill.\n"
        "---\n"
        "\n"
        "# Demo\n"
    )

    skills = discover_skills([str(tmp_path / "skills")])

    assert len(skills) == 1
    assert skills[0].name == "demo"
    assert skills[0].description == "Demo skill."
    assert skills[0].path == str(skill_file.resolve())


def test_resolve_skill_matches_requested_name_not_first_catalog_entry():
    skills = [
        SkillMetadata(
            name="roi-search-workflow",
            description="",
            path="/tmp/skills/roi-search-workflow/SKILL.md",
        ),
        SkillMetadata(
            name="scanning-microscope-focusing-with-landmark-feature-line-scan",
            description="",
            path=(
                "/tmp/skills/scanning-microscope-focusing-with-landmark-feature-line-scan/"
                "SKILL.md"
            ),
        ),
    ]

    skill = resolve_skill(
        skills,
        "scanning-microscope-focusing-with-landmark-feature-line-scan",
    )

    assert skill is not None
    assert skill.name == "scanning-microscope-focusing-with-landmark-feature-line-scan"


def test_base_task_manager_uses_builtin_core_skills_by_default():
    task_manager = BaseTaskManager(
        skill_dirs=None,
        use_coding_tools=False,
        build=False,
    )

    skill_names = {skill.name for skill in task_manager.skill_catalog}

    assert "monitor-status" in skill_names


def test_base_task_manager_allows_explicit_empty_skill_dirs():
    task_manager = BaseTaskManager(
        skill_dirs=[],
        use_coding_tools=False,
        build=False,
    )

    assert task_manager.skill_dirs == []
    assert task_manager.skill_catalog == []


def test_imaging_base_task_manager_uses_builtin_core_and_imaging_skills_by_default():
    task_manager = ImagingBaseTaskManager(
        skill_dirs=None,
        use_coding_tools=False,
        build=False,
    )

    skill_names = {skill.name for skill in task_manager.skill_catalog}

    assert "monitor-status" in skill_names
    assert "roi-search-workflow" in skill_names
    assert (
        "scanning-microscope-focusing-with-landmark-feature-line-scan" in skill_names
    )


def test_skill_command_injects_only_skill_md(tmp_path):
    skill_dir = tmp_path / "skills" / "demo"
    reference_dir = skill_dir / "references"
    reference_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\n"
        "name: demo\n"
        "description: Demo skill.\n"
        "---\n"
        "\n"
        "Use the demo workflow.\n"
    )
    (reference_dir / "extra.md").write_text("do not inject")
    task_manager = BaseTaskManager(
        skill_dirs=[str(tmp_path / "skills")],
        use_coding_tools=False,
        build=False,
    )

    messages = task_manager.expand_skill_command_in_text("/skill demo run it")

    assert len(messages) == 2
    assert "<name>demo</name>" in messages[0]["content"]
    assert str(skill_file.resolve()) in messages[0]["content"]
    assert "Use the demo workflow." in messages[0]["content"]
    assert "do not inject" not in messages[0]["content"]
    assert messages[1]["content"] == "run it"


def test_skill_selection_can_appear_inside_user_text(tmp_path):
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: demo\n"
        "description: Demo skill.\n"
        "---\n"
        "\n"
        "Use the demo workflow.\n"
    )
    task_manager = BaseTaskManager(
        skill_dirs=[str(tmp_path / "skills")],
        use_coding_tools=False,
        build=False,
    )

    messages = task_manager.expand_skill_command_in_text("please use /skill demo for this")

    assert len(messages) == 2
    assert "<name>demo</name>" in messages[0]["content"]
    assert messages[1]["content"] == "please use for this"


def test_chat_input_preserves_trailing_skill_command_text(tmp_path):
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: demo\n"
        "description: Demo skill.\n"
        "---\n"
        "\n"
        "Use the demo workflow.\n"
    )
    task_manager = QueuedInputTaskManager(
        ["/skill demo what does this skill talk about?"],
        skill_dirs=[str(tmp_path / "skills")],
        use_coding_tools=False,
        checkpoint_db_path=None,
        build=True,
    )
    state = ChatGraphState(await_user_input=True)

    NodeFactory(task_manager).await_or_ingest_user_input(state)

    assert len(state.messages) == 2
    assert "<name>demo</name>" in state.messages[0]["content"]
    assert state.messages[1]["content"] == "what does this skill talk about?"


def test_feedback_human_gate_preserves_trailing_skill_command_text(tmp_path):
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: demo\n"
        "description: Demo skill.\n"
        "---\n"
        "\n"
        "Use the demo workflow.\n"
    )
    task_manager = QueuedInputTaskManager(
        ["/skill demo what does this skill talk about?"],
        skill_dirs=[str(tmp_path / "skills")],
        use_coding_tools=False,
        checkpoint_db_path=None,
        build=True,
    )
    captured_context = []

    def fake_invoke_model_raw(*args, **kwargs):
        captured_context.extend(kwargs["context"])
        return {"role": "assistant", "content": "summary"}, kwargs.get("message")

    task_manager.invoke_model_raw = fake_invoke_model_raw
    state = FeedbackLoopState(await_user_input=True)

    NodeFactory(task_manager).handle_human_gate(state)

    assert len(captured_context) == 2
    assert "<name>demo</name>" in captured_context[0]["content"]
    assert captured_context[1]["content"] == "what does this skill talk about?"
