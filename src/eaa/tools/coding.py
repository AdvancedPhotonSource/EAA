"""Python code execution tool for LLM agents."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional

from eaa.tools.base import BaseTool, ToolReturnType, check


class PythonCodingTool(BaseTool):
    """Expose a tool that executes Python code in an isolated subprocess."""

    name: str = "python_coding"

    @check
    def __init__(
        self,
        *,
        default_timeout: Optional[float] = None,
        working_directory: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None,
        require_approval: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the coding tool.

        Parameters
        ----------
        default_timeout : float, optional
            Timeout (in seconds) applied when executing code unless a call
            overrides it.
        working_directory : str, optional
            Working directory for executed code. Defaults to the current
            working directory when the tool is instantiated.
        environment : Dict[str, str], optional
            Environment variables to overlay on top of the current process
            environment when running code.
        """
        self._default_timeout = default_timeout
        self._working_directory = working_directory or os.getcwd()
        self._environment = environment or {}

        super().__init__(require_approval=require_approval, **kwargs)

        self.exposed_tools = [
            {
                "name": "execute_code",
                "function": self.execute_code,
                "return_type": ToolReturnType.DICT,
            }
        ]

    def execute_code(
        self,
        code: str,
        *,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        input_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute Python code in a subprocess and capture the result.

        Parameters
        ----------
        code : str
            Python source code to execute.
        timeout : float, optional
            Timeout (in seconds) for this execution. Falls back to the tool
            default when omitted.
        cwd : str, optional
            Working directory for this execution. Defaults to the tool's
            configured working directory.
        input_text : str, optional
            Text supplied to the subprocess via standard input.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing stdout, stderr, returncode, and a
            timeout flag. If execution fails before the subprocess starts,
            an error message is included.
        """
        if not isinstance(code, str):
            raise TypeError("code must be a string containing Python source")

        exec_timeout = timeout if timeout is not None else self._default_timeout
        exec_cwd = cwd or self._working_directory
        env = os.environ.copy()
        env.update(self._environment)

        tmp_file = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        try:
            tmp_file.write(code)
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
        except Exception as exc:  # pragma: no cover - best effort reporting
            return {
                "stdout": "",
                "stderr": "",
                "returncode": None,
                "timeout": False,
                "error": str(exc),
            }
        finally:
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass
