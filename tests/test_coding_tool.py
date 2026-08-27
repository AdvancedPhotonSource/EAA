import logging
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

from eaa_core.tool.coding import BashCodingTool, SimplePythonEvalTool, PythonCodingTool
from eaa_core.tool.base import generate_openai_tool_schema

import test_utils as tutils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestCodingTool(tutils.BaseTester):
    def test_literal_eval_tool_evaluates_literal(self):
        tool = SimplePythonEvalTool()

        result = tool.evaluate("{'values': [1, 2, 3], 'enabled': True}")

        assert result == {"values": [1, 2, 3], "enabled": True}

    def test_literal_eval_tool_evaluates_arithmetic(self):
        tool = SimplePythonEvalTool()

        result = tool.evaluate(
            "(273.77 + ((290.77 - 291.835) - 3.5), "
            "235.58 + ((270.58 - 271.52) - 3.5))"
        )

        np.testing.assert_allclose(result, (269.205, 231.14))

    def test_literal_eval_tool_evaluates_sum(self):
        tool = SimplePythonEvalTool()

        result = tool.evaluate("sum([1, 2 * 3, -4])")

        assert result == 3

    def test_literal_eval_tool_rejects_other_calls(self):
        tool = SimplePythonEvalTool()

        with np.testing.assert_raises(ValueError):
            tool.evaluate("__import__('os').system('echo unsafe')")

    def test_literal_eval_tool_rejects_oversized_power(self):
        tool = SimplePythonEvalTool()

        with np.testing.assert_raises(ValueError):
            tool.evaluate("2 ** 10000")

    def test_literal_eval_tool_does_not_require_approval(self):
        tool = SimplePythonEvalTool()

        assert tool.require_approval is False
        assert tool.exposed_tools[0].require_approval is False

    def test_execute_code_calculates_mean(self):
        tool = PythonCodingTool()
        code = textwrap.dedent("""\
            import numpy as np
            print(np.mean(np.arange(10)))
        """)

        result = tool.execute_code(code)
        if self.debug:
            print(result)

        assert result["returncode"] == 0
        assert result["timeout"] is False
        assert result["stderr"] == ""
        assert result["stdout"].strip() == str(np.mean(np.arange(10)))

    def test_python_coding_tool_requires_approval_flag(self):
        tool = PythonCodingTool()
        assert tool.require_approval is True

    def test_coding_tool_names_use_execute_action(self):
        assert PythonCodingTool().exposed_tools[0].name == "python_coding_tool.execute"
        assert BashCodingTool().exposed_tools[0].name == "bash_coding_tool.execute"

    def test_bash_schema_exposes_optional_ui_tab_flag(self):
        exposed = BashCodingTool().exposed_tools[0]
        schema = exposed.schema or generate_openai_tool_schema(
            tool_name=exposed.name,
            func=exposed.function,
        )
        parameters = schema["function"]["parameters"]

        assert "create_ui_tab" in parameters["properties"]
        assert "read-only Bash tab" in parameters["properties"]["create_ui_tab"]["description"]
        assert "create_ui_tab" not in parameters["required"]

    def test_bash_defaults_to_no_terminal_and_preserves_result(self, tmp_path):
        tool = BashCodingTool(working_directory=str(tmp_path))
        starts = []
        tool.set_terminal_handlers(
            start_handler=lambda command: starts.append(command) or "bash-1",
            output_handler=lambda *_args: None,
            finish_handler=lambda *_args: None,
        )

        result = tool.execute_code("printf output; printf error >&2")

        assert starts == []
        assert result == {
            "stdout": "output",
            "stderr": "error",
            "returncode": 0,
            "timeout": False,
        }

    def test_bash_terminal_streams_input_unicode_and_both_streams(self, tmp_path):
        tool = BashCodingTool(working_directory=str(tmp_path))
        events = []
        tool.set_terminal_handlers(
            start_handler=lambda command: events.append(("start", command)) or "bash-1",
            output_handler=lambda terminal_id, chunks: events.append(
                ("output", terminal_id, chunks)
            ),
            finish_handler=lambda terminal_id, result: events.append(
                ("finish", terminal_id, dict(result))
            ),
        )

        result = tool.execute_code(
            "IFS= read -r value; printf '\\342'; sleep 0.12; "
            "printf '\\202\\254:%s' \"$value\"; sleep 0.12; printf err >&2",
            input_text="hello\n",
            create_ui_tab=True,
        )

        chunks = [chunk for event in events if event[0] == "output" for chunk in event[2]]
        assert events[0][0] == "start"
        assert events[-1][0] == "finish"
        assert result == {
            "stdout": "€:hello",
            "stderr": "err",
            "returncode": 0,
            "timeout": False,
        }
        assert "".join(chunk["text"] for chunk in chunks if chunk["stream"] == "stdout") == "€:hello"
        assert "".join(chunk["text"] for chunk in chunks if chunk["stream"] == "stderr") == "err"

    def test_bash_terminal_timeout_keeps_partial_output(self, tmp_path):
        tool = BashCodingTool(working_directory=str(tmp_path))
        finished = []
        tool.set_terminal_handlers(
            start_handler=lambda _command: "bash-1",
            output_handler=lambda *_args: None,
            finish_handler=lambda _terminal_id, result: finished.append(dict(result)),
        )

        started = time.monotonic()
        result = tool.execute_code(
            "printf before; sleep 5; printf after",
            timeout=0.2,
            create_ui_tab=True,
        )

        assert time.monotonic() - started < 2
        assert result["stdout"] == "before"
        assert result["stderr"] == ""
        assert result["returncode"] is None
        assert result["timeout"] is True
        assert "timed out" in result["error"]
        assert finished == [result]

    def test_bash_terminal_reports_launch_failure(self, monkeypatch, tmp_path):
        tool = BashCodingTool(working_directory=str(tmp_path))
        events = []
        tool.set_terminal_handlers(
            start_handler=lambda _command: events.append("start") or "bash-1",
            output_handler=lambda *_args: None,
            finish_handler=lambda _terminal_id, result: events.append(dict(result)),
        )
        monkeypatch.setattr(
            "eaa_core.tool.coding.subprocess.Popen",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("launch failed")),
        )

        result = tool.execute_code("true", create_ui_tab=True)

        assert events[0] == "start"
        assert result == {
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "timeout": False,
            "error": "launch failed",
        }
        assert events[-1] == result

    def test_bash_terminal_builds_bubblewrap_process(self, monkeypatch, tmp_path):
        tool = BashCodingTool(sandbox_type="bubblewrap")
        monkeypatch.setattr(
            "eaa_core.tool.coding.shutil.which",
            lambda name, path=None: "/usr/bin/bwrap" if name == "bwrap" else None,
        )
        monkeypatch.setattr(tool, "_bubblewrap_system_bind_paths", lambda: ["/usr"])
        monkeypatch.setattr(
            tool,
            "_bubblewrap_visible_bind_paths",
            lambda _workdir: [str(tmp_path)],
        )

        command, kwargs, cleanup = tool._stream_process_spec(
            "printf ok",
            env={"PATH": "/usr/bin"},
            cwd=str(tmp_path),
            exec_cwd=str(tmp_path),
            has_input=False,
        )

        assert command == [
            "/usr/bin/bwrap",
            "--die-with-parent",
            "--unshare-all",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--bind",
            "/tmp",
            "/tmp",
            "--ro-bind",
            "/usr",
            "/usr",
            "--bind",
            str(tmp_path),
            str(tmp_path),
            "--chdir",
            str(tmp_path),
            "/bin/bash",
            "-c",
            "printf ok",
        ]
        assert kwargs["env"] == {"PATH": "/usr/bin"}
        assert kwargs["start_new_session"] is True
        assert cleanup == []

    def test_bash_terminal_builds_container_process(self, monkeypatch, tmp_path):
        tool = BashCodingTool(sandbox_type="container", container_image="bash:test")
        monkeypatch.setattr(tool, "_select_container_runtime", lambda: "podman")

        command, kwargs, cleanup = tool._stream_process_spec(
            "printf ok",
            env={"VALUE": "one"},
            cwd=str(tmp_path),
            exec_cwd=str(tmp_path),
            has_input=True,
        )

        try:
            assert command == [
                "podman",
                "run",
                "--rm",
                "-i",
                "--workdir",
                str(tmp_path),
                "--tmpfs",
                str(tmp_path),
                "--env-file",
                cleanup[0],
                "bash:test",
                "bash",
                "-c",
                "printf ok",
            ]
            assert kwargs["stdin"] is subprocess.PIPE
            assert kwargs["start_new_session"] is True
            assert Path(cleanup[0]).read_text() == "VALUE=one\n"
        finally:
            Path(cleanup[0]).unlink()

    def test_execution_env_supplies_path_when_parent_path_is_missing(self, monkeypatch):
        monkeypatch.delenv("PATH", raising=False)

        env = PythonCodingTool()._build_execution_env()
        path_entries = env["PATH"].split(os.pathsep)

        assert path_entries[0] == str(Path(sys.executable).absolute().parent)
        assert "/usr/bin" in path_entries
        assert "/bin" in path_entries

    def test_execution_env_preserves_symlinked_venv_interpreter_path(
        self,
        monkeypatch,
        tmp_path,
    ):
        venv_bin = tmp_path / ".venv" / "bin"
        base_bin = tmp_path / "base" / "bin"
        venv_bin.mkdir(parents=True)
        base_bin.mkdir(parents=True)
        base_python = base_bin / "python3"
        base_python.write_text("")
        venv_python = venv_bin / "python3"
        venv_python.symlink_to(base_python)
        monkeypatch.setattr("eaa_core.tool.coding.sys.executable", str(venv_python))
        monkeypatch.setenv("PATH", str(base_bin))

        env = PythonCodingTool()._build_execution_env()
        path_entries = env["PATH"].split(os.pathsep)

        assert path_entries[0] == str(venv_bin)
        assert path_entries[1] == str(base_bin)

    def test_bubblewrap_visible_paths_include_editable_pth_entries(
        self,
        monkeypatch,
        tmp_path,
    ):
        site_dir = tmp_path / ".venv" / "lib" / "python3.11" / "site-packages"
        source_dir = tmp_path / "src" / "pkg"
        site_dir.mkdir(parents=True)
        source_dir.mkdir(parents=True)
        (site_dir / "__editable__.example.pth").write_text(f"{source_dir}\n")
        monkeypatch.setattr(
            "eaa_core.tool.coding.site.getsitepackages",
            lambda: [str(site_dir)],
        )
        monkeypatch.setattr(
            "eaa_core.tool.coding.site.getusersitepackages",
            lambda: str(tmp_path / "missing-user-site"),
        )
        monkeypatch.setattr("eaa_core.tool.coding.sys.path", [])

        paths = PythonCodingTool()._bubblewrap_visible_bind_paths(str(tmp_path))

        assert str(source_dir) in paths

    def test_bubblewrap_visible_paths_preserve_literal_and_real_paths(
        self,
        tmp_path,
    ):
        real_dir = tmp_path / "real" / "src"
        real_dir.mkdir(parents=True)
        linked_dir = tmp_path / "linked-src"
        linked_dir.symlink_to(real_dir, target_is_directory=True)

        paths = PythonCodingTool()._existing_bind_path_variants(str(linked_dir))

        assert str(linked_dir) in paths
        assert str(real_dir) in paths

    def test_bubblewrap_uses_normalized_path_and_absolute_runtime(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.delenv("PATH", raising=False)
        captured = {}

        def fake_which(name, path=None):
            captured["which_name"] = name
            captured["which_path"] = path
            if name == "bwrap" and path and "/usr/bin" in path.split(os.pathsep):
                return "/usr/bin/bwrap"
            return None

        def fake_run(command, **kwargs):
            captured["command"] = command
            captured["env"] = kwargs["env"]
            return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

        monkeypatch.setattr("eaa_core.tool.coding.shutil.which", fake_which)
        monkeypatch.setattr("eaa_core.tool.coding.subprocess.run", fake_run)

        result = BashCodingTool(sandbox_type="bubblewrap").execute_code(
            "python -c 'print(1)'",
            cwd=str(tmp_path),
        )

        assert result["returncode"] == 0
        assert captured["which_name"] == "bwrap"
        assert "/usr/bin" in captured["which_path"].split(os.pathsep)
        assert captured["command"][0] == "/usr/bin/bwrap"
        assert ["--bind", "/tmp", "/tmp"] == captured["command"][7:10]
        assert ["--tmpfs", "/tmp"] not in [
            captured["command"][index : index + 2]
            for index in range(len(captured["command"]) - 1)
        ]
        assert captured["env"]["PATH"] == captured["which_path"]

    def test_python_bubblewrap_executes_resolved_interpreter(
        self,
        monkeypatch,
        tmp_path,
    ):
        venv_bin = tmp_path / ".venv" / "bin"
        runtime_bin = tmp_path / "runtime" / "bin"
        site_dir = tmp_path / ".venv" / "lib" / "python3.14" / "site-packages"
        source_dir = tmp_path / "src" / "pkg"
        venv_bin.mkdir(parents=True)
        runtime_bin.mkdir(parents=True)
        site_dir.mkdir(parents=True)
        source_dir.mkdir(parents=True)
        runtime_python = runtime_bin / "python3.14"
        runtime_python.write_text("")
        (site_dir / "__editable__.example.pth").write_text(f"{source_dir}\n")
        venv_python = venv_bin / "python"
        venv_python.symlink_to(runtime_python)
        monkeypatch.setattr("eaa_core.tool.coding.sys.executable", str(venv_python))
        monkeypatch.setattr(
            "eaa_core.tool.coding.site.getsitepackages",
            lambda: [str(site_dir)],
        )
        monkeypatch.setattr(
            "eaa_core.tool.coding.site.getusersitepackages",
            lambda: str(tmp_path / "missing-user-site"),
        )
        monkeypatch.setattr("eaa_core.tool.coding.sys.path", [str(site_dir)])
        captured = {}

        def fake_which(name, path=None):
            if name == "bwrap":
                return "/usr/bin/bwrap"
            return None

        def fake_run(command, **kwargs):
            captured["command"] = command
            captured["env"] = kwargs["env"]
            return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

        monkeypatch.setattr("eaa_core.tool.coding.shutil.which", fake_which)
        monkeypatch.setattr("eaa_core.tool.coding.subprocess.run", fake_run)

        result = PythonCodingTool(sandbox_type="bubblewrap").execute_code(
            "print('ok')",
            cwd=str(tmp_path),
        )

        assert result["returncode"] == 0
        chdir_index = captured["command"].index("--chdir")
        python_command = captured["command"][chdir_index + 2]
        assert python_command == str(runtime_python)
        pythonpath = captured["env"]["PYTHONPATH"].split(os.pathsep)
        assert str(site_dir) in pythonpath
        assert str(source_dir) in pythonpath


if __name__ == "__main__":
    tester = TestCodingTool()
    tester.setup_method(name="", generate_data=False, generate_gold=False, debug=True)
    tester.test_execute_code_calculates_mean()
