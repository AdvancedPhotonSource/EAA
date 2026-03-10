"""Code execution tools for EAA agents."""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Annotated, Any, Dict, Optional

from eaa.core.tooling.base import BaseTool, ToolReturnType, check, tool


class CodingTool(BaseTool):
    """Shared behavior for code execution tools."""

    @check
    def __init__(
        self,
        *,
        default_timeout: Optional[float] = None,
        working_directory: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        run_in_sandbox: bool = False,
        container_image: Optional[str] = None,
        require_approval: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the coding tool base."""
        self._default_timeout = default_timeout
        self._working_directory = working_directory or os.getcwd()
        self._environment = environment or {}
        self._run_in_sandbox = run_in_sandbox
        self._container_image = container_image
        super().__init__(require_approval=require_approval, **kwargs)

    def _execute_in_container(
        self,
        command: list[str],
        *,
        env: Dict[str, str],
        timeout: Optional[float],
        input_text: Optional[str],
        workdir: Optional[str],
    ) -> subprocess.CompletedProcess[str]:
        runtime = self._select_container_runtime()
        if runtime is None:
            raise RuntimeError("No container runtime found (expected podman or docker).")
        env_file_path = self._write_env_file(env)
        container_workdir = workdir or "/workspace"
        container_cmd = [
            runtime,
            "run",
            "--rm",
            "-i",
            "--workdir",
            container_workdir,
            "--tmpfs",
            container_workdir,
            "--env-file",
            env_file_path,
            self._container_image,
            *command,
        ]
        try:
            return subprocess.run(
                container_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                input=input_text,
            )
        finally:
            os.unlink(env_file_path)

    @staticmethod
    def _select_container_runtime() -> Optional[str]:
        for runtime in ("podman", "docker"):
            if shutil.which(runtime):
                return runtime
        return None

    @staticmethod
    def _write_env_file(env: Dict[str, str]) -> str:
        env_file = tempfile.NamedTemporaryFile("w", delete=False)
        for key, value in env.items():
            if "\n" in value or "\r" in value:
                continue
            env_file.write(f"{key}={value}\n")
        env_file.flush()
        env_file.close()
        return env_file.name

    @staticmethod
    def _prepare_source_code(code: str) -> str:
        """Normalize model-generated source code before execution."""
        normalized = code.replace("\r\n", "\n").replace("\r", "\n")
        stripped = normalized.strip()
        if (
            "\n" not in normalized
            and "\\n" in normalized
            and len(stripped) >= 2
            and stripped[0] == stripped[-1]
            and stripped[0] in {"'", '"'}
        ):
            try:
                decoded = ast.literal_eval(stripped)
                if isinstance(decoded, str):
                    if "\n" not in decoded and "\\n" in decoded:
                        decoded = decoded.replace("\\n", "\n")
                    normalized = decoded
            except (SyntaxError, ValueError):
                pass
        quote_map = str.maketrans(
            {
                "“": '"',
                "”": '"',
                "„": '"',
                "‟": '"',
                "‘": "'",
                "’": "'",
                "‚": "'",
                "‛": "'",
            }
        )
        return normalized.translate(quote_map)


class PythonCodingTool(CodingTool):
    """Expose a tool that executes Python code in an isolated subprocess."""

    name: str = "python_coding"

    @check
    def __init__(
        self,
        *,
        default_timeout: Optional[float] = None,
        working_directory: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        run_in_sandbox: bool = False,
        container_image: Optional[str] = None,
        require_approval: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Python coding tool."""
        super().__init__(
            default_timeout=default_timeout,
            working_directory=working_directory,
            environment=environment,
            run_in_sandbox=run_in_sandbox,
            container_image=container_image or self._default_python_image(),
            require_approval=require_approval,
            **kwargs,
        )

    @tool(name="execute_python_code", return_type=ToolReturnType.DICT)
    def execute_code(
        self,
        code: Annotated[str, "Python source code to execute in a subprocess."],
        *,
        timeout: Annotated[
            Optional[float],
            "Maximum seconds to wait for execution. Overrides the tool-level default; omit to use the default.",
        ] = None,
        cwd: Annotated[
            Optional[str],
            "Working directory for the subprocess. Overrides the tool-level default; omit to use the default.",
        ] = None,
        input_text: Annotated[
            Optional[str],
            "Text piped into the subprocess's stdin.",
        ] = None,
        last_print_is_image_path: Annotated[
            bool,
            "When True, the last non-empty stdout line is treated as a PNG image path.",
        ] = False,
    ) -> Dict[str, Any]:
        """Execute Python source code in an isolated subprocess."""
        if not isinstance(code, str):
            raise TypeError("code must be a string containing Python source")
        prepared_code = self._prepare_source_code(code)
        exec_timeout = timeout if timeout is not None else self._default_timeout
        exec_cwd = cwd or self._working_directory
        env = os.environ.copy()
        env.update(self._environment)
        tmp_file = None
        try:
            if self._run_in_sandbox:
                result = self._execute_in_container(
                    ["python", "-c", prepared_code],
                    env=env,
                    timeout=exec_timeout,
                    input_text=input_text,
                    workdir=cwd,
                )
            else:
                tmp_file = tempfile.NamedTemporaryFile(
                    "w",
                    suffix=".py",
                    delete=False,
                    dir=exec_cwd,
                )
                tmp_file.write(prepared_code)
                tmp_file.flush()
                tmp_file.close()
                result = subprocess.run(
                    [sys.executable, tmp_file.name],
                    capture_output=True,
                    text=True,
                    cwd=exec_cwd,
                    env=env,
                    timeout=exec_timeout,
                    input=input_text,
                )
            result_dict: Dict[str, Any] = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "timeout": False,
            }
            if last_print_is_image_path:
                non_empty_lines = [line for line in result.stdout.splitlines() if line.strip()]
                if non_empty_lines:
                    candidate = non_empty_lines[-1].strip()
                    if os.path.isfile(candidate) and os.path.splitext(candidate)[1].lower() == ".png":
                        result_dict["image_path"] = candidate
            return result_dict
        except subprocess.TimeoutExpired as exc:
            return {
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "returncode": None,
                "timeout": True,
                "error": f"Execution timed out after {exec_timeout} seconds",
            }
        except Exception as exc:  # pragma: no cover
            return {
                "stdout": "",
                "stderr": "",
                "returncode": None,
                "timeout": False,
                "error": str(exc),
            }
        finally:
            if not self._run_in_sandbox and tmp_file is not None:
                try:
                    os.unlink(tmp_file.name)
                except OSError:
                    pass

    @staticmethod
    def _default_python_image() -> str:
        """Return the default Python container image name."""
        return f"python:{sys.version_info.major}.{sys.version_info.minor}"


class BashCodingTool(CodingTool):
    """Expose a tool that executes Bash code in an isolated subprocess."""

    name: str = "bash_coding"

    @check
    def __init__(
        self,
        *,
        default_timeout: Optional[float] = None,
        working_directory: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        shell_path: str = "/bin/bash",
        run_in_sandbox: bool = False,
        container_image: Optional[str] = None,
        require_approval: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Bash coding tool."""
        self._shell_path = shell_path
        super().__init__(
            default_timeout=default_timeout,
            working_directory=working_directory,
            environment=environment,
            run_in_sandbox=run_in_sandbox,
            container_image=container_image or "bash:latest",
            require_approval=require_approval,
            **kwargs,
        )

    @tool(name="execute_bash_code", return_type=ToolReturnType.DICT)
    def execute_code(
        self,
        code: str,
        *,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        input_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute Bash code in a subprocess and capture the result."""
        if not isinstance(code, str):
            raise TypeError("code must be a string containing Bash source")
        exec_timeout = timeout if timeout is not None else self._default_timeout
        exec_cwd = cwd or self._working_directory
        env = os.environ.copy()
        env.update(self._environment)
        tmp_file = None
        try:
            if self._run_in_sandbox:
                result = self._execute_in_container(
                    ["bash", "-c", code],
                    env=env,
                    timeout=exec_timeout,
                    input_text=input_text,
                    workdir=cwd,
                )
            else:
                tmp_file = tempfile.NamedTemporaryFile(
                    "w",
                    suffix=".sh",
                    delete=False,
                    dir=exec_cwd,
                )
                tmp_file.write(code)
                tmp_file.flush()
                tmp_file.close()
                result = subprocess.run(
                    [self._shell_path, tmp_file.name],
                    capture_output=True,
                    text=True,
                    cwd=exec_cwd,
                    env=env,
                    timeout=exec_timeout,
                    input=input_text,
                )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "timeout": False,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "returncode": None,
                "timeout": True,
                "error": f"Execution timed out after {exec_timeout} seconds",
            }
        except Exception as exc:  # pragma: no cover
            return {
                "stdout": "",
                "stderr": "",
                "returncode": None,
                "timeout": False,
                "error": str(exc),
            }
        finally:
            if not self._run_in_sandbox and tmp_file is not None:
                try:
                    os.unlink(tmp_file.name)
                except OSError:
                    pass
