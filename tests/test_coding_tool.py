import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from eaa_core.tool.coding import BashCodingTool, SimplePythonEvalTool, PythonCodingTool

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


if __name__ == "__main__":
    tester = TestCodingTool()
    tester.setup_method(name="", generate_data=False, generate_gold=False, debug=True)
    tester.test_execute_code_calculates_mean()
