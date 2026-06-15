import json
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from eaa_core.task_manager.tool_executor import SerialToolExecutor
from eaa_core.tool.base import generate_openai_tool_schema
from eaa_core.tool.workspace import FileSystemTool, ImageRenderingTool, UvTool
import eaa_core.tool.workspace as workspace_tools


def build_tool_call(name, arguments):
    return {
        "id": f"call_{name}",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def parse_tool_content(result):
    return json.loads(result.message["content"])


def test_filesystem_tool_approval_rules_for_workspace_paths(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "inside.txt").write_text("inside")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    approvals = []

    def approval_handler(tool_name, arguments):
        approvals.append((tool_name, arguments))
        return False

    executor = SerialToolExecutor(approval_handler=approval_handler)
    executor.register_tools(FileSystemTool(workspace_path=str(workspace)))

    inside_result = executor.execute_tool_call(build_tool_call("read_file", {"file_path": "inside.txt"}))
    assert "inside" in parse_tool_content(inside_result)["result"]
    assert approvals == []

    outside_result = executor.execute_tool_call(
        build_tool_call("read_file", {"file_path": str(outside)})
    )
    assert parse_tool_content(outside_result)["error"] == "Tool execution was denied by the user."
    assert approvals == [("read_file", {"file_path": str(outside)})]


def test_filesystem_tool_allows_reading_whitelisted_skill_dirs(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skill_dir = tmp_path / "skills" / "example"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text("skill instructions")
    approvals = []

    executor = SerialToolExecutor(
        approval_handler=lambda tool_name, arguments: approvals.append((tool_name, arguments)) or False
    )
    executor.register_tools(
        FileSystemTool(
            workspace_path=str(workspace),
            read_whitelist_paths=[str(tmp_path / "skills")],
        )
    )

    read_result = executor.execute_tool_call(
        build_tool_call("read_file", {"file_path": str(skill_file)})
    )

    assert "skill instructions" in parse_tool_content(read_result)["result"]
    assert approvals == []


def test_approval_rule_errors_return_tool_error(tmp_path):
    executor = SerialToolExecutor(approval_handler=lambda tool_name, arguments: True)
    executor.register_tools(FileSystemTool(workspace_path=str(tmp_path)))

    result = executor.execute_tool_call(build_tool_call("read_file", {"file_path": ""}))

    assert parse_tool_content(result)["error"] == "Path must not be empty."


def test_filesystem_write_always_requires_approval(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source.txt").write_text("source")
    approvals = []

    def approval_handler(tool_name, arguments):
        approvals.append(tool_name)
        return False

    executor = SerialToolExecutor(approval_handler=approval_handler)
    executor.register_tools(FileSystemTool(workspace_path=str(workspace)))

    result = executor.execute_tool_call(
        build_tool_call("write_file", {"file_path": "created.txt", "content": "created"})
    )
    assert parse_tool_content(result)["error"] == "Tool execution was denied by the user."

    assert approvals == ["write_file"]
    assert not (workspace / "created.txt").exists()
    assert (workspace / "source.txt").exists()


def test_filesystem_tool_surface_excludes_shell_redundant_tools(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = FileSystemTool(workspace_path=str(workspace))
    visible_names = {spec.name for spec in tool.exposed_tools if spec.model_visible}

    assert visible_names == {
        "read_file",
        "write_file",
        "edit_file",
        "replace_file_lines",
    }


def test_image_rendering_tool_writes_rendered_png(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "image.png"
    Image.fromarray(np.arange(100, dtype=np.uint8).reshape(10, 10)).save(source)

    tool = ImageRenderingTool(workspace_path=str(workspace))
    result = tool.render_image_for_agent("image.png", add_axis_ticks=False)

    assert Path(result["img_path"]).exists()
    assert result["source_path"] == "image.png"


def test_uv_tool_anchors_project_in_working_directory(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    (parent / "pyproject.toml").write_text("[project]\nname = \"parent\"\nversion = \"0.1.0\"\n")
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/active-venv")
    monkeypatch.setenv("PYTHONHOME", "/tmp/python-home")

    tool = UvTool(working_directory=str(child))
    pyproject_path = tool.ensure_uv_project()
    env = tool.build_uv_env()

    assert pyproject_path == child / "pyproject.toml"
    assert pyproject_path.exists()
    assert "VIRTUAL_ENV" not in env
    assert "PYTHONHOME" not in env
    assert os.environ["VIRTUAL_ENV"] == "/tmp/active-venv"


def test_workspace_tool_schemas_include_annotated_descriptions(tmp_path):
    fs_tool = FileSystemTool(workspace_path=str(tmp_path))
    uv_tool = UvTool(working_directory=str(tmp_path))

    read_schema = generate_openai_tool_schema("read_file", fs_tool.read_file)
    uv_schema = generate_openai_tool_schema("uv", uv_tool.uv)

    read_properties = read_schema["function"]["parameters"]["properties"]
    uv_properties = uv_schema["function"]["parameters"]["properties"]
    assert read_properties["file_path"]["description"].startswith("Text file path to read.")
    assert read_properties["limit"]["type"] == "integer"
    assert uv_properties["arguments"]["type"] == "array"
    assert uv_properties["arguments"]["items"]["type"] == "string"


def test_uv_tool_casts_string_timeout_to_float(tmp_path, monkeypatch):
    seen_timeouts = []

    def fake_run_command(args, *, cwd, timeout_seconds, env):
        seen_timeouts.append(timeout_seconds)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(workspace_tools, "run_command", fake_run_command)
    tool = UvTool(working_directory=str(tmp_path))

    result = tool.uv("run", arguments=["python", "-V"], timeout_seconds="7.5")

    assert result["exit_code"] == 0
    assert seen_timeouts == [7.5]
    assert "img_path" not in result


def test_empty_image_paths_do_not_trigger_followup_images():
    assert SerialToolExecutor.extract_image_paths_from_tool_response('{"img_path": ""}') == []
    assert SerialToolExecutor.extract_image_paths_from_tool_response(
        '{"img_path": ["", "   ", "/tmp/image.png"]}'
    ) == ["/tmp/image.png"]
