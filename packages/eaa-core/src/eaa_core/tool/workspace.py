"""Workspace file, image rendering, and uv execution tools."""

from __future__ import annotations

import mimetypes
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from fnmatch import fnmatch
from pathlib import Path
from typing import Annotated, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from eaa_core.tool.base import BaseTool, check, tool


TEXT_FILE_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".tsv",
    ".ini",
    ".cfg",
    ".sh",
    ".rst",
}


def truncate_text(text: str, max_chars: int = 12000) -> str:
    """Return text shortened to a bounded size for tool responses."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 120] + "\n...[truncated]..."


def format_lines_with_numbers(lines: list[str], *, start_line: int) -> str:
    """Format text lines with stable line-number prefixes."""
    width = max(4, len(str(start_line + len(lines))))
    return "\n".join(f"{line_no:>{width}} | {line}" for line_no, line in enumerate(lines, start=start_line))


def is_probably_text_file(path: Path) -> bool:
    """Return whether a path appears to contain text data."""
    mime_type, _encoding = mimetypes.guess_type(path.name)
    return mime_type is None or mime_type.startswith("text/") or path.suffix.lower() in TEXT_FILE_SUFFIXES


def run_command(
    args: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and terminate its process group on timeout."""
    timeout_seconds = float(timeout_seconds)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    popen_kwargs: dict[str, Any] = {
        "args": args,
        "cwd": cwd,
        "env": env,
        "text": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(**popen_kwargs)
    process_group_id: int | None = None
    if os.name != "nt":
        try:
            process_group_id = os.getpgid(process.pid)
        except ProcessLookupError:
            process_group_id = None

    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        terminate_process(process, process_group_id=process_group_id)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(args, timeout_seconds, output=stdout, stderr=stderr) from exc

    terminate_remaining_descendants(process_group_id)
    return subprocess.CompletedProcess(
        args=args,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def terminate_process(process: subprocess.Popen[str], *, process_group_id: int | None) -> None:
    """Terminate a process and its group when available."""
    if os.name == "nt":
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        return

    if process_group_id is not None:
        signal_process_group(process_group_id, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            signal_process_group(process_group_id, signal.SIGKILL)
            process.wait(timeout=2)
        return

    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def terminate_remaining_descendants(process_group_id: int | None) -> None:
    """Clean up leftover descendants in a process group."""
    if os.name == "nt" or process_group_id is None:
        return
    signal_process_group(process_group_id, signal.SIGTERM)
    time.sleep(0.1)
    signal_process_group(process_group_id, signal.SIGKILL)


def signal_process_group(process_group_id: int, sig: int) -> None:
    """Send a signal to a process group, ignoring already-exited groups."""
    try:
        os.killpg(process_group_id, sig)
    except ProcessLookupError:
        return


class WorkspacePathMixin:
    """Path helpers for tools rooted in a workspace directory."""

    workspace_path: Path

    def resolve_path(self, path: str) -> Path:
        """Resolve a user path as absolute or workspace-relative."""
        normalized = path.strip()
        if not normalized:
            raise ValueError("Path must not be empty.")
        candidate = Path(normalized)
        if not candidate.is_absolute():
            candidate = self.workspace_path / candidate
        return candidate.expanduser().resolve()

    def relative_or_absolute_path(self, path: Path) -> str:
        """Return a workspace-relative path when possible, else an absolute path."""
        try:
            return str(path.relative_to(self.workspace_path)).replace("\\", "/")
        except ValueError:
            return str(path)

    def is_in_workspace(self, path: Path) -> bool:
        """Return whether a path is inside the configured workspace."""
        try:
            path.resolve().relative_to(self.workspace_path)
        except ValueError:
            return False
        return True

    def any_path_outside_workspace(self, arguments: dict[str, Any], names: tuple[str, ...]) -> bool:
        """Return whether any named path argument resolves outside the workspace."""
        for name in names:
            value = arguments.get(name)
            if isinstance(value, str) and not self.is_in_workspace(self.resolve_path(value)):
                return True
        return False


class FileSystemTool(WorkspacePathMixin, BaseTool):
    """Expose workspace-aware filesystem tools."""

    name: str = "file_system"

    @check
    def __init__(
        self,
        *,
        workspace_path: Optional[str] = None,
        require_approval: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the filesystem tool.

        Parameters
        ----------
        workspace_path : str, optional
            Root path used to resolve relative paths. Defaults to the current
            working directory.
        require_approval : bool, default=False
            Fallback approval setting for tools without stricter method-level
            approval rules.
        **kwargs
            Additional arguments forwarded to :class:`BaseTool`.
        """
        self.workspace_path = Path(workspace_path or os.getcwd()).expanduser().resolve()
        super().__init__(require_approval=require_approval, **kwargs)

    def requires_approval_for_path(self, arguments: dict[str, Any]) -> bool:
        """Return whether a single target path sits outside the workspace."""
        return self.any_path_outside_workspace(arguments, ("path",))

    def requires_approval_for_directory_path(self, arguments: dict[str, Any]) -> bool:
        """Return whether a directory path sits outside the workspace."""
        return self.any_path_outside_workspace(arguments, ("directory_path",))

    def requires_approval_for_file_path(self, arguments: dict[str, Any]) -> bool:
        """Return whether a file path sits outside the workspace."""
        return self.any_path_outside_workspace(arguments, ("file_path",))

    def requires_approval_for_copy(self, arguments: dict[str, Any]) -> bool:
        """Return whether copy source or destination sits outside the workspace."""
        return self.any_path_outside_workspace(arguments, ("source_path", "destination_path"))

    @tool(name="ls", require_approval="requires_approval_for_path")
    def ls(
        self,
        path: Annotated[str, "Directory path to list. Relative paths are resolved from the workspace."] = ".",
    ) -> dict[str, Any]:
        """List direct children in a directory."""
        target = self.workspace_path if path == "." else self.resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"Directory not found: {target}")
        if not target.is_dir():
            raise NotADirectoryError(f"Expected a directory path, got: {target}")
        entries = []
        for child in sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            stat_result = child.stat()
            entries.append(
                {
                    "path": self.relative_or_absolute_path(child) + ("/" if child.is_dir() else ""),
                    "is_dir": child.is_dir(),
                    "size": 0 if child.is_dir() else stat_result.st_size,
                    "modified_at": stat_result.st_mtime,
                }
            )
        return {
            "ok": True,
            "path": "." if target == self.workspace_path else self.relative_or_absolute_path(target),
            "entries": entries,
        }

    @tool(name="mkdir", require_approval="requires_approval_for_directory_path")
    def mkdir(
        self,
        directory_path: Annotated[str, "Directory path to create. Relative paths are resolved from the workspace."],
        parents: Annotated[bool, "Whether to create parent directories as needed."] = True,
        exist_ok: Annotated[bool, "Whether an existing directory should be treated as success."] = True,
    ) -> dict[str, Any]:
        """Create a directory."""
        target = self.resolve_path(directory_path)
        target.mkdir(parents=parents, exist_ok=exist_ok)
        return {"ok": True, "directory_path": self.relative_or_absolute_path(target)}

    @tool(name="read_file", require_approval="requires_approval_for_file_path")
    def read_file(
        self,
        file_path: Annotated[str, "Text file path to read. Relative paths are resolved from the workspace."],
        offset: Annotated[int, "Zero-based line offset to start reading from."] = 0,
        limit: Annotated[int, "Maximum number of lines to read."] = 200,
    ) -> str:
        """Read a text file with line numbers."""
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit <= 0:
            raise ValueError("limit must be positive")
        target = self.resolve_path(file_path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File not found: {target}")
        if not is_probably_text_file(target):
            raise ValueError("File does not look like text. Use render_image_for_agent for images.")
        try:
            content = target.read_text()
        except UnicodeDecodeError as exc:
            raise ValueError(f"Unable to decode file as text: {target}") from exc
        if not content:
            return "System reminder: File exists but has empty contents"
        lines = content.splitlines()
        if offset >= len(lines):
            raise ValueError(f"Line offset {offset} exceeds file length ({len(lines)} lines).")
        return format_lines_with_numbers(lines[offset : offset + limit], start_line=offset + 1)

    @tool(name="write_file", require_approval=True)
    def write_file(
        self,
        file_path: Annotated[str, "File path to create or overwrite. Relative paths are resolved from the workspace."],
        content: Annotated[str, "Complete text content to write."],
    ) -> dict[str, Any]:
        """Create or overwrite a text file."""
        target = self.resolve_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return {
            "ok": True,
            "file_path": self.relative_or_absolute_path(target),
            "bytes_written": len(content.encode("utf-8")),
        }

    @tool(name="edit_file", require_approval=True)
    def edit_file(
        self,
        file_path: Annotated[str, "Text file path to edit. Relative paths are resolved from the workspace."],
        old_string: Annotated[str, "Exact existing text to replace."],
        new_string: Annotated[str, "Replacement text."],
        replace_all: Annotated[bool, "Whether to replace every occurrence rather than requiring exactly one."] = False,
    ) -> dict[str, Any]:
        """Replace exact text in a file."""
        target = self.resolve_path(file_path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File not found: {target}")
        content = target.read_text()
        occurrences = content.count(old_string)
        if occurrences == 0:
            raise ValueError("old_string was not found in the file.")
        if not replace_all and occurrences != 1:
            raise ValueError(
                f"old_string matched {occurrences} times. Pass replace_all=True to replace all occurrences."
            )
        updated = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
        target.write_text(updated)
        return {
            "ok": True,
            "file_path": self.relative_or_absolute_path(target),
            "occurrences": occurrences if replace_all else 1,
        }

    @tool(name="replace_file_lines", require_approval=True)
    def replace_file_lines(
        self,
        file_path: Annotated[str, "Text file path to edit. Relative paths are resolved from the workspace."],
        new_text: Annotated[str, "Replacement text for the selected line range. Use an empty string to delete the range."],
        start_line: Annotated[int, "1-based first line number to replace."],
        end_line: Annotated[
            Optional[int],
            "Optional 1-based last line number to replace, inclusive. Omit to insert at start_line.",
        ] = None,
    ) -> dict[str, Any]:
        """Replace or insert text in a 1-based line range."""
        if start_line <= 0:
            raise ValueError("start_line must be a positive 1-based line number.")
        if end_line is not None and end_line < start_line:
            raise ValueError("end_line must be greater than or equal to start_line.")
        target = self.resolve_path(file_path)
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"File not found: {target}")
        content = target.read_text()
        lines = content.splitlines(keepends=True)
        replacement_lines = new_text.splitlines(keepends=True)
        if not lines or start_line > len(lines):
            separator = "\n" if content and not content.endswith("\n") and new_text else ""
            target.write_text(content + separator + new_text)
            return {
                "ok": True,
                "file_path": self.relative_or_absolute_path(target),
                "start_line": start_line,
                "end_line": end_line,
                "appended": True,
            }
        if end_line is None:
            insertion_index = start_line - 1
            updated_lines = lines[:insertion_index] + replacement_lines + lines[insertion_index:]
        else:
            if end_line > len(lines):
                raise ValueError(
                    f"Requested line range {start_line}-{end_line} exceeds file length ({len(lines)} lines)."
                )
            updated_lines = lines[: start_line - 1] + replacement_lines + lines[end_line:]
        target.write_text("".join(updated_lines))
        return {
            "ok": True,
            "file_path": self.relative_or_absolute_path(target),
            "start_line": start_line,
            "end_line": end_line,
            "appended": False,
        }

    @tool(name="glob", require_approval="requires_approval_for_path")
    def glob(
        self,
        pattern: Annotated[str, "Glob pattern such as `**/*.py` or `data/*.png`."],
        path: Annotated[str, "Base directory to search from. Relative paths are resolved from the workspace."] = ".",
    ) -> dict[str, Any]:
        """Find files below a directory that match a glob pattern."""
        normalized_pattern = pattern.lstrip("/")
        if ".." in Path(normalized_pattern).parts:
            raise ValueError("Path traversal not allowed in glob pattern.")
        base_path = self.workspace_path if path == "." else self.resolve_path(path)
        if not base_path.exists() or not base_path.is_dir():
            raise NotADirectoryError(f"Search base is not a directory: {base_path}")
        matches = [
            self.relative_or_absolute_path(match)
            for match in sorted(base_path.rglob(normalized_pattern))
            if match.is_file()
        ]
        return {
            "ok": True,
            "pattern": pattern,
            "path": "." if base_path == self.workspace_path else self.relative_or_absolute_path(base_path),
            "matches": matches,
        }

    @tool(name="grep", require_approval="requires_approval_for_path")
    def grep(
        self,
        pattern: Annotated[str, "Literal text to search for."],
        path: Annotated[str, "File or directory path to search in. Relative paths are resolved from the workspace."] = ".",
        glob: Annotated[Optional[str], "Optional glob filter such as `**/*.py`."] = None,
        output_mode: Annotated[str, "Output mode: `files_with_matches`, `content`, or `count`."] = "files_with_matches",
    ) -> dict[str, Any]:
        """Search for literal text in files."""
        if not pattern:
            raise ValueError("pattern must not be empty")
        if output_mode not in {"files_with_matches", "content", "count"}:
            raise ValueError("output_mode must be one of: files_with_matches, content, count")
        target = self.workspace_path if path == "." else self.resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"Search path not found: {target}")
        search_paths = [target] if target.is_file() else [candidate for candidate in target.rglob("*") if candidate.is_file()]
        matches: dict[str, list[dict[str, Any]]] = {}
        for candidate in sorted(search_paths):
            relative_candidate = self.relative_or_absolute_path(candidate)
            if glob is not None and not fnmatch(relative_candidate, glob):
                continue
            if not is_probably_text_file(candidate):
                continue
            try:
                lines = candidate.read_text().splitlines()
            except UnicodeDecodeError:
                continue
            hit_rows = [
                {"line": line_number, "text": line}
                for line_number, line in enumerate(lines, start=1)
                if pattern in line
            ]
            if hit_rows:
                matches[relative_candidate] = hit_rows
        if output_mode == "files_with_matches":
            results: Any = sorted(matches)
        elif output_mode == "count":
            results = {file_path: len(rows) for file_path, rows in matches.items()}
        else:
            results = matches
        return {"ok": True, "pattern": pattern, "output_mode": output_mode, "results": results}

    @tool(name="copy_file", require_approval="requires_approval_for_copy")
    def copy_file(
        self,
        source_path: Annotated[str, "Source file path to copy. Relative paths are resolved from the workspace."],
        destination_path: Annotated[str, "Destination file path. Relative paths are resolved from the workspace."],
        overwrite: Annotated[bool, "Whether to replace an existing destination file."] = False,
    ) -> dict[str, Any]:
        """Copy a file."""
        source = self.resolve_path(source_path)
        destination = self.resolve_path(destination_path)
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")
        if destination.exists():
            if not overwrite:
                raise FileExistsError(f"Destination already exists: {destination}")
            if destination.is_dir():
                raise IsADirectoryError(f"Destination is a directory: {destination}")
            destination.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return {
            "ok": True,
            "source_path": self.relative_or_absolute_path(source),
            "destination_path": self.relative_or_absolute_path(destination),
        }

    @tool(name="move_file", require_approval=True)
    def move_file(
        self,
        source_path: Annotated[str, "Source file path to move or rename. Relative paths are resolved from the workspace."],
        destination_path: Annotated[str, "Destination file path. Relative paths are resolved from the workspace."],
        overwrite: Annotated[bool, "Whether to replace an existing destination file."] = False,
    ) -> dict[str, Any]:
        """Move or rename a file."""
        source = self.resolve_path(source_path)
        destination = self.resolve_path(destination_path)
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")
        if destination.exists():
            if not overwrite:
                raise FileExistsError(f"Destination already exists: {destination}")
            if destination.is_dir():
                raise IsADirectoryError(f"Destination is a directory: {destination}")
            destination.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        return {
            "ok": True,
            "source_path": self.relative_or_absolute_path(source),
            "destination_path": self.relative_or_absolute_path(destination),
        }

    @tool(name="delete_file", require_approval=True)
    def delete_file(
        self,
        file_path: Annotated[str, "File path to delete. Relative paths are resolved from the workspace."],
        missing_ok: Annotated[bool, "Whether a missing file should be treated as a successful no-op."] = False,
    ) -> dict[str, Any]:
        """Delete a file."""
        target = self.resolve_path(file_path)
        if not target.exists():
            if missing_ok:
                return {"ok": True, "deleted": False, "file_path": file_path}
            raise FileNotFoundError(f"File not found: {target}")
        if not target.is_file():
            raise IsADirectoryError(f"Expected a file path, got: {target}")
        target.unlink()
        return {"ok": True, "deleted": True, "file_path": self.relative_or_absolute_path(target)}


class ImageRenderingTool(WorkspacePathMixin, BaseTool):
    """Render image files to temporary PNGs for model inspection."""

    name: str = "image_rendering"

    @check
    def __init__(
        self,
        *,
        workspace_path: Optional[str] = None,
        render_directory: Optional[str] = None,
        require_approval: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the image rendering tool."""
        self.workspace_path = Path(workspace_path or os.getcwd()).expanduser().resolve()
        default_render_directory = self.workspace_path / ".tmp" / "rendered_images"
        self.render_directory = Path(render_directory).expanduser().resolve() if render_directory else default_render_directory
        super().__init__(require_approval=require_approval, **kwargs)

    def requires_approval_for_image_path(self, arguments: dict[str, Any]) -> bool:
        """Return whether the source image path sits outside the workspace."""
        return self.any_path_outside_workspace(arguments, ("image_path",))

    @staticmethod
    def validated_percentile_range(
        min_percentile: Optional[float],
        max_percentile: Optional[float],
    ) -> tuple[Optional[float], Optional[float]]:
        """Validate optional percentile display bounds."""
        if min_percentile is None and max_percentile is None:
            return None, None
        if min_percentile is None or max_percentile is None:
            raise ValueError("Both min_percentile and max_percentile must be provided together.")
        min_value = float(min_percentile)
        max_value = float(max_percentile)
        if not 0.0 <= min_value <= 100.0:
            raise ValueError("min_percentile must be between 0 and 100.")
        if not 0.0 <= max_value <= 100.0:
            raise ValueError("max_percentile must be between 0 and 100.")
        if min_value >= max_value:
            raise ValueError("min_percentile must be less than max_percentile.")
        return min_value, max_value

    @staticmethod
    def validated_display_value_range(
        vmin: Optional[float],
        vmax: Optional[float],
    ) -> tuple[Optional[float], Optional[float]]:
        """Validate optional explicit display bounds."""
        if vmin is None and vmax is None:
            return None, None
        if vmin is None or vmax is None:
            raise ValueError("Both vmin and vmax must be provided together.")
        min_value = float(vmin)
        max_value = float(vmax)
        if min_value >= max_value:
            raise ValueError("vmin must be less than vmax.")
        return min_value, max_value

    @classmethod
    def validate_image_display_controls(
        cls,
        source: Path,
        min_percentile: Optional[float],
        max_percentile: Optional[float],
        vmin: Optional[float],
        vmax: Optional[float],
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Validate mutually exclusive image rendering display controls."""
        percentile_min, percentile_max = cls.validated_percentile_range(min_percentile, max_percentile)
        value_min, value_max = cls.validated_display_value_range(vmin, vmax)
        if percentile_min is not None and value_min is not None:
            raise ValueError("Percentile bounds and explicit vmin/vmax are mutually exclusive.")
        if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"} and (
            percentile_min is not None or value_min is not None
        ):
            raise ValueError("PNG, JPEG, and PDF sources do not accept display scaling controls.")
        return percentile_min, percentile_max, value_min, value_max

    @staticmethod
    def validate_log_scale(source: Path, log_scale: bool) -> None:
        """Validate that log-scale rendering is only requested for TIFF sources."""
        if log_scale and source.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError("log_scale is only supported for TIFF sources.")

    @staticmethod
    def apply_log_scale_for_render(
        array: np.ndarray,
        render_kwargs: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply log1p scaling to an image array and display bounds."""
        finite_values = array[np.isfinite(array)]
        if finite_values.size == 0:
            raise ValueError("Image does not contain any finite values for log-scale rendering.")
        transformed_array = np.log1p(np.clip(array, 0, None))
        transformed_kwargs = dict(render_kwargs)
        if "vmin" in transformed_kwargs:
            transformed_kwargs["vmin"] = float(np.log1p(max(transformed_kwargs["vmin"], 0.0)))
        if "vmax" in transformed_kwargs:
            transformed_kwargs["vmax"] = float(np.log1p(max(transformed_kwargs["vmax"], 0.0)))
        return transformed_array, transformed_kwargs

    @tool(name="render_image_for_agent", require_approval="requires_approval_for_image_path")
    def render_image_for_agent(
        self,
        image_path: Annotated[str, "Image path to visualize. Relative paths are resolved from the workspace."],
        add_axis_ticks: Annotated[bool, "Whether to show axis ticks on the rendered image."] = True,
        add_colorbar: Annotated[bool, "Whether to add a colorbar for 2D images."] = False,
        min_percentile: Annotated[Optional[float], "Optional TIFF-only lower percentile for display scaling."] = None,
        max_percentile: Annotated[Optional[float], "Optional TIFF-only upper percentile for display scaling."] = None,
        vmin: Annotated[Optional[float], "Optional TIFF-only lower display bound."] = None,
        vmax: Annotated[Optional[float], "Optional TIFF-only upper display bound."] = None,
        log_scale: Annotated[bool, "Whether to apply TIFF-only log1p rendering."] = False,
    ) -> dict[str, Any]:
        """Render a TIFF, PNG, JPEG, or first-page PDF to a temporary PNG."""
        source = self.resolve_path(image_path)
        if not source.exists():
            raise FileNotFoundError(f"Image not found: {source}")
        display_min, display_max, display_vmin, display_vmax = self.validate_image_display_controls(
            source,
            min_percentile,
            max_percentile,
            vmin,
            vmax,
        )
        self.validate_log_scale(source, log_scale)
        self.render_directory.mkdir(parents=True, exist_ok=True)
        target = self.render_directory / f"{uuid.uuid4().hex}.png"
        if source.suffix.lower() == ".pdf":
            try:
                import pypdfium2 as pdfium
            except ImportError as exc:
                raise RuntimeError("PDF rendering requires the optional `pypdfium2` package.") from exc
            pdf = pdfium.PdfDocument(str(source))
            try:
                if len(pdf) == 0:
                    raise ValueError("PDF does not contain any pages.")
                array = np.asarray(pdf[0].render(scale=2.0).to_pil())
            finally:
                pdf.close()
        else:
            with Image.open(source) as img:
                array = np.asarray(img)
        render_kwargs: dict[str, Any] = {}
        if display_min is not None and display_max is not None:
            finite_values = array[np.isfinite(array)]
            if finite_values.size == 0:
                raise ValueError("Image does not contain any finite values for percentile scaling.")
            percentile_vmin = float(np.percentile(finite_values, display_min))
            percentile_vmax = float(np.percentile(finite_values, display_max))
            if np.isclose(percentile_vmin, percentile_vmax):
                percentile_vmax = percentile_vmin + 1e-12
            render_kwargs["vmin"] = percentile_vmin
            render_kwargs["vmax"] = percentile_vmax
        elif display_vmin is not None and display_vmax is not None:
            render_kwargs["vmin"] = display_vmin
            render_kwargs["vmax"] = display_vmax
        if log_scale:
            array, render_kwargs = self.apply_log_scale_for_render(array, render_kwargs)
        fig, ax = plt.subplots(figsize=(8, 8))
        if add_colorbar and array.ndim != 2:
            raise ValueError("add_colorbar is only supported for 2D images.")
        if array.ndim == 2:
            mappable = ax.imshow(array, cmap="gray", **render_kwargs)
        else:
            mappable = ax.imshow(array, **render_kwargs)
        ax.set_title(source.name)
        if add_colorbar:
            fig.colorbar(mappable, ax=ax)
        if not add_axis_ticks:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(target, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return {
            "img_path": str(target),
            "source_path": self.relative_or_absolute_path(source),
            "add_axis_ticks": add_axis_ticks,
            "add_colorbar": add_colorbar,
            "min_percentile": display_min,
            "max_percentile": display_max,
            "vmin": display_vmin,
            "vmax": display_vmax,
            "log_scale": log_scale,
        }


class UvTool(WorkspacePathMixin, BaseTool):
    """Expose uv project and environment management commands."""

    name: str = "uv"

    @check
    def __init__(
        self,
        *,
        working_directory: Optional[str] = None,
        require_approval: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the uv tool.

        Parameters
        ----------
        working_directory : str, optional
            Directory where uv commands run. Defaults to the current working
            directory.
        require_approval : bool, default=True
            Whether uv execution requires approval. This defaults to true and
            should normally stay enabled because uv can install packages and
            execute project scripts.
        **kwargs
            Additional arguments forwarded to :class:`BaseTool`.
        """
        self.workspace_path = Path(working_directory or os.getcwd()).expanduser().resolve()
        super().__init__(require_approval=require_approval, **kwargs)

    def ensure_uv_project(self) -> Path:
        """Ensure uv anchors itself in the working directory, not a parent project."""
        pyproject_path = self.workspace_path / "pyproject.toml"
        if pyproject_path.exists():
            return pyproject_path
        project_name = re.sub(r"[^a-z0-9]+", "-", self.workspace_path.name.lower()).strip("-") or "eaa-workspace"
        pyproject_path.write_text(
            "\n".join(
                [
                    "[project]",
                    f'name = "{project_name}"',
                    'version = "0.1.0"',
                    'requires-python = ">=3.11"',
                    "dependencies = []",
                    "",
                ]
            )
        )
        return pyproject_path

    @staticmethod
    def build_uv_env() -> dict[str, str]:
        """Return an environment suitable for uv-managed subprocesses."""
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        env.pop("PYTHONHOME", None)
        return env

    @tool(name="uv", require_approval=True)
    def uv(
        self,
        subcommand: Annotated[str, "The uv subcommand to run, such as `run`, `add`, `remove`, `venv`, `pip`, or `sync`."],
        arguments: Annotated[Optional[list[str]], "Arguments passed after the uv subcommand."] = None,
        timeout_seconds: Annotated[float, "Maximum execution time in seconds before the command is aborted."] = 600,
        last_stdout_line_is_image_path: Annotated[
            bool,
            "If true, interpret the last stdout line as an image path and expose it as `img_path`.",
        ] = False,
    ) -> dict[str, Any]:
        """Run a bounded uv command from the working directory."""
        normalized_subcommand = subcommand.strip()
        if not normalized_subcommand:
            raise ValueError("subcommand must not be empty")
        args = arguments or []
        command_timeout = float(timeout_seconds)
        self.ensure_uv_project()
        completed = run_command(
            ["uv", normalized_subcommand, *args],
            cwd=self.workspace_path,
            timeout_seconds=command_timeout,
            env=self.build_uv_env(),
        )
        stdout = completed.stdout or ""
        img_path = None
        if last_stdout_line_is_image_path and stdout.strip():
            candidate = stdout.strip().splitlines()[-1].strip()
            if candidate:
                path = self.resolve_path(candidate)
                if path.exists():
                    img_path = str(path)
        result = {
            "stdout": truncate_text(stdout),
            "stderr": truncate_text(completed.stderr or ""),
            "exit_code": completed.returncode,
            "subcommand": normalized_subcommand,
            "arguments": args,
        }
        if img_path:
            result["img_path"] = img_path
        return result
