"""Code execution tools for EAA agents."""

from __future__ import annotations

import ast
import operator
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Annotated, Any, Dict, Literal, Optional, Sequence

from eaa_core.tool.base import BaseTool, check, tool

SandboxType = Literal["bubblewrap", "container"] | None


class SimplePythonEvalTool(BaseTool):
    """Safely evaluate Python literals and basic arithmetic."""

    name: str = "literal_eval"
    _max_code_length = 2_000
    _max_ast_nodes = 200
    _max_integer_bits = 4_096
    _max_exponent = 1_000
    _binary_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary_operators = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    @tool(name="simple_python_eval_tool.evaluate_python_expression", require_approval=False)
    def evaluate(
        self,
        code: Annotated[str, "Python literal or basic arithmetic expression."],
    ) -> Any:
        """Evaluate literals, basic arithmetic, and calls to ``sum`` safely.
        This tool does not need approval.

        More complicated code should be executed with ``PythonCodingTool`` or
        by writing a script and running it with ``UvTool``.
        """
        if not isinstance(code, str):
            raise TypeError("code must be a string containing a Python literal")
        if len(code) > self._max_code_length:
            raise ValueError("expression is too long")

        expression = ast.parse(code, mode="eval")
        if sum(1 for _ in ast.walk(expression)) > self._max_ast_nodes:
            raise ValueError("expression is too complex")
        return self._evaluate_node(expression.body)

    def _evaluate_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            self._check_result_size(node.value)
            return node.value
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values = [self._evaluate_node(item) for item in node.elts]
            return {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](values)
        if isinstance(node, ast.Dict):
            return {
                self._evaluate_node(key): self._evaluate_node(value)
                for key, value in zip(node.keys, node.values)
            }
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._unary_operators:
            operand = self._require_number(self._evaluate_node(node.operand))
            result = self._unary_operators[type(node.op)](operand)
            self._check_result_size(result)
            return result
        if isinstance(node, ast.BinOp) and type(node.op) in self._binary_operators:
            left = self._require_number(self._evaluate_node(node.left))
            right = self._require_number(self._evaluate_node(node.right))
            if isinstance(node.op, ast.Pow):
                if isinstance(right, complex) or abs(right) > self._max_exponent:
                    raise ValueError("exponent is too large")
            result = self._binary_operators[type(node.op)](left, right)
            self._check_result_size(result)
            return result
        if isinstance(node, ast.Call):
            return self._evaluate_sum(node)
        raise ValueError(f"unsupported expression: {type(node).__name__}")

    def _evaluate_sum(self, node: ast.Call) -> Any:
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id != "sum"
            or node.keywords
            or len(node.args) not in (1, 2)
        ):
            raise ValueError("only sum(iterable[, start]) is allowed")

        values = self._evaluate_node(node.args[0])
        if not isinstance(values, (list, tuple, set)):
            raise ValueError("sum() requires a literal list, tuple, or set")
        numbers = [self._require_number(value) for value in values]
        start = (
            self._require_number(self._evaluate_node(node.args[1]))
            if len(node.args) == 2
            else 0
        )
        result = sum(numbers, start)
        self._check_result_size(result)
        return result

    @staticmethod
    def _require_number(value: Any) -> int | float | complex:
        if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
            raise ValueError("arithmetic operands must be numbers")
        return value

    def _check_result_size(self, value: Any) -> None:
        if isinstance(value, int) and value.bit_length() > self._max_integer_bits:
            raise ValueError("integer result is too large")


class CodingTool(BaseTool):
    """Shared behavior for code execution tools."""

    @check
    def __init__(
        self,
        *,
        default_timeout: Optional[float] = None,
        working_directory: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        sandbox_type: SandboxType = None,
        bubblewrap_visible_dirs: Optional[Sequence[str]] = None,
        container_image: Optional[str] = None,
        require_approval: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the coding tool base."""
        self._default_timeout = default_timeout
        self._working_directory = working_directory or os.getcwd()
        self._environment = environment or {}
        self.sandbox_type = self._normalize_sandbox_type(sandbox_type)
        self.bubblewrap_visible_dirs = (
            list(bubblewrap_visible_dirs) if bubblewrap_visible_dirs is not None else None
        )
        self._container_image = container_image
        super().__init__(require_approval=require_approval, **kwargs)

    def set_sandbox_type(
        self,
        sandbox_type: SandboxType,
        *,
        visible_dirs: Optional[Sequence[str]] = None,
    ) -> None:
        """Configure the sandbox mode used for code execution.

        Parameters
        ----------
        sandbox_type : {"bubblewrap", "container"} or None
            Sandbox implementation to use. ``None`` executes directly on the
            host with the configured working directory and environment.
        visible_dirs : sequence of str, optional
            Additional paths to bind into bubblewrap at the same absolute paths.
            The current working directory is always included.
        """
        self.sandbox_type = self._normalize_sandbox_type(sandbox_type)
        self.bubblewrap_visible_dirs = list(visible_dirs) if visible_dirs is not None else None

    @staticmethod
    def _normalize_sandbox_type(sandbox_type: SandboxType) -> SandboxType:
        if sandbox_type not in (None, "bubblewrap", "container"):
            raise ValueError(
                "sandbox_type must be one of None, 'bubblewrap', or 'container'."
            )
        return sandbox_type

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

    def _execute_in_bubblewrap(
        self,
        command: list[str],
        *,
        env: Dict[str, str],
        timeout: Optional[float],
        input_text: Optional[str],
        workdir: str,
    ) -> subprocess.CompletedProcess[str]:
        if shutil.which("bwrap") is None:
            raise RuntimeError("No bubblewrap runtime found (expected `bwrap`).")
        bubblewrap_cmd = [
            "bwrap",
            "--die-with-parent",
            "--unshare-all",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
        ]
        for path in self._bubblewrap_system_bind_paths():
            bubblewrap_cmd.extend(["--ro-bind", path, path])
        for path in self._bubblewrap_visible_bind_paths(workdir):
            bubblewrap_cmd.extend(["--bind", path, path])
        bubblewrap_cmd.extend(["--chdir", workdir, *command])
        return subprocess.run(
            bubblewrap_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            input=input_text,
            env=env,
        )

    @staticmethod
    def _bubblewrap_system_bind_paths() -> list[str]:
        paths = ["/usr", "/bin", "/lib", "/lib64", "/etc"]
        paths.extend([sys.base_prefix, sys.exec_prefix])
        resolved: list[str] = []
        for path in paths:
            abs_path = os.path.abspath(os.path.expanduser(path))
            if not os.path.exists(abs_path) or abs_path in resolved:
                continue
            resolved.append(abs_path)
        return resolved

    def _bubblewrap_visible_bind_paths(self, workdir: str) -> list[str]:
        paths = list(self.bubblewrap_visible_dirs or [])
        paths.append(os.getcwd())
        paths.append(workdir)
        resolved: list[str] = []
        for path in paths:
            real_path = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
            if not os.path.exists(real_path) or real_path in resolved:
                continue
            resolved.append(real_path)
        return resolved

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
        sandbox_type: SandboxType = None,
        bubblewrap_visible_dirs: Optional[Sequence[str]] = None,
        container_image: Optional[str] = None,
        require_approval: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Python coding tool."""
        super().__init__(
            default_timeout=default_timeout,
            working_directory=working_directory,
            environment=environment,
            sandbox_type=sandbox_type,
            bubblewrap_visible_dirs=bubblewrap_visible_dirs,
            container_image=container_image or self._default_python_image(),
            require_approval=require_approval,
            **kwargs,
        )

    @tool(name="python_coding_tool.execute")
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
        exec_timeout = float(timeout) if timeout is not None else self._default_timeout
        exec_cwd = cwd or self._working_directory
        env = os.environ.copy()
        env.update(self._environment)
        tmp_file = None
        try:
            if self.sandbox_type == "container":
                result = self._execute_in_container(
                    ["python", "-c", prepared_code],
                    env=env,
                    timeout=exec_timeout,
                    input_text=input_text,
                    workdir=cwd,
                )
            elif self.sandbox_type == "bubblewrap":
                result = self._execute_in_bubblewrap(
                    [sys.executable, "-c", prepared_code],
                    env=env,
                    timeout=exec_timeout,
                    input_text=input_text,
                    workdir=exec_cwd,
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
                        result_dict["img_path"] = candidate
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
            if self.sandbox_type is None and tmp_file is not None:
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
        sandbox_type: SandboxType = None,
        bubblewrap_visible_dirs: Optional[Sequence[str]] = None,
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
            sandbox_type=sandbox_type,
            bubblewrap_visible_dirs=bubblewrap_visible_dirs,
            container_image=container_image or "bash:latest",
            require_approval=require_approval,
            **kwargs,
        )

    @tool(name="bash_coding_tool.execute")
    def execute_code(
        self,
        code: Annotated[str, "Bash source code to execute in a subprocess."],
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
    ) -> Dict[str, Any]:
        """Execute Bash code in a subprocess and capture the result."""
        if not isinstance(code, str):
            raise TypeError("code must be a string containing Bash source")
        exec_timeout = float(timeout) if timeout is not None else self._default_timeout
        exec_cwd = cwd or self._working_directory
        env = os.environ.copy()
        env.update(self._environment)
        tmp_file = None
        try:
            if self.sandbox_type == "container":
                result = self._execute_in_container(
                    ["bash", "-c", code],
                    env=env,
                    timeout=exec_timeout,
                    input_text=input_text,
                    workdir=cwd,
                )
            elif self.sandbox_type == "bubblewrap":
                result = self._execute_in_bubblewrap(
                    [self._shell_path, "-c", code],
                    env=env,
                    timeout=exec_timeout,
                    input_text=input_text,
                    workdir=exec_cwd,
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
            if self.sandbox_type is None and tmp_file is not None:
                try:
                    os.unlink(tmp_file.name)
                except OSError:
                    pass
