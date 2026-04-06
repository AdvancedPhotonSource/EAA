from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Sequence, Tuple

from eaa_core.message_proc import generate_openai_message

logger = logging.getLogger(__name__)
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*]\(([^)]+)\)")
MESSAGE_BREAK_PATTERN = re.compile(r"(?im)^\s*<message_break>\s*$")


@dataclass(frozen=True)
class SkillMetadata:
    """Metadata for a discovered skill."""

    name: str
    description: str
    tool_name: str
    path: str


def load_skills(skill_dirs: Sequence[str]) -> List[SkillMetadata]:
    """Discover skills under the configured directories."""
    skills: List[SkillMetadata] = []
    seen_tool_names: set[str] = set()
    for base_dir in skill_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            logger.warning("Skill directory not found: %s", base_dir)
            continue
        if (base_path / "SKILL.md").exists():
            skill_paths = [base_path]
        else:
            skill_paths = sorted({path.parent for path in base_path.rglob("SKILL.md")})
        for skill_path in skill_paths:
            metadata = parse_skill_metadata(skill_path)
            if metadata is None:
                continue
            tool_name = ensure_unique_tool_name(metadata.tool_name, seen_tool_names)
            if tool_name != metadata.tool_name:
                metadata = SkillMetadata(
                    name=metadata.name,
                    description=metadata.description,
                    tool_name=tool_name,
                    path=metadata.path,
                )
            skills.append(metadata)
    return skills


def parse_skill_metadata(skill_dir: Path) -> SkillMetadata | None:
    """Parse a skill directory into metadata."""
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        return None
    text = skill_file.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    frontmatter, body_lines = parse_frontmatter(lines)
    name = frontmatter.get("name")
    description = frontmatter.get("description")
    name_index = 0
    if not name:
        name, name_index = extract_name(body_lines)
    if not name:
        name = skill_dir.name
        name_index = 0
    if not description:
        description = extract_description(body_lines, name_index)
    if not description:
        description = "No description provided."
    return SkillMetadata(
        name=name,
        description=description,
        tool_name=make_tool_name(name),
        path=str(skill_dir),
    )


def parse_frontmatter(lines: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """Split markdown frontmatter from the body."""
    if not lines or lines[0].strip() != "---":
        return {}, lines
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return read_frontmatter(lines[1:index]), lines[index + 1 :]
    return {}, lines


def read_frontmatter(lines: List[str]) -> Dict[str, str]:
    """Read a simple YAML-like frontmatter block."""
    frontmatter: Dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        frontmatter[key.strip()] = value.strip().strip("\"").strip("'")
    return frontmatter


def extract_name(lines: List[str]) -> Tuple[str | None, int]:
    """Extract the skill name from markdown lines."""
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            name = stripped.lstrip("#").strip()
            if name:
                return name, index
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("name:"):
            name = stripped.split(":", 1)[1].strip()
            if name:
                return name, index
    return None, 0


def extract_description(lines: List[str], name_index: int) -> str:
    """Extract a short description from markdown lines."""
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("description:"):
            return stripped.split(":", 1)[1].strip()
    return extract_first_paragraph(lines, start_index=name_index + 1)


def extract_first_paragraph(lines: List[str], start_index: int) -> str:
    """Extract the first non-empty paragraph from markdown lines."""
    paragraph: List[str] = []
    for line in lines[start_index:]:
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        paragraph.append(stripped)
    return " ".join(paragraph)


def make_tool_name(name: str) -> str:
    """Convert a skill name to a tool-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return f"skill-{slug or 'skill'}"


def ensure_unique_tool_name(tool_name: str, seen: set[str]) -> str:
    """Ensure that discovered skill tool names are unique."""
    if tool_name not in seen:
        seen.add(tool_name)
        return tool_name
    index = 2
    while True:
        candidate = f"{tool_name}-{index}"
        if candidate not in seen:
            seen.add(candidate)
            return candidate
        index += 1


def resolve_skill_metadata(
    skill_catalog: Sequence[SkillMetadata],
    skill_name: str,
) -> SkillMetadata | None:
    """Resolve a skill by name, tool name, or filesystem path."""
    normalized = skill_name.strip()
    if len(normalized) == 0:
        return None
    for skill in skill_catalog:
        if normalized in {skill.name, skill.tool_name, skill.path}:
            return skill
    normalized_lower = normalized.lower()
    for skill in skill_catalog:
        if normalized_lower in {
            skill.name.lower(),
            skill.tool_name.lower(),
            skill.path.lower(),
        }:
            return skill
    return None


def collect_skill_docs(
    skill_dir: Path,
    *,
    max_doc_bytes: int = 200_000,
) -> Tuple[Dict[str, str], List[str], Dict[str, List[str]]]:
    """Collect markdown docs for a skill."""
    files: Dict[str, str] = {}
    skipped: List[str] = []
    images_by_file: Dict[str, List[str]] = {}
    doc_paths: list[Path] = []
    for path in skill_dir.rglob("*"):
        if path.is_dir() or path.name.startswith(".") or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() == ".md":
            doc_paths.append(path)

    def path_sort_key(path: Path) -> tuple[int, str]:
        relative_path = str(path.relative_to(skill_dir))
        return (0 if relative_path.lower() == "skill.md" else 1, relative_path.lower())

    for path in sorted(doc_paths, key=path_sort_key):
        relative_path = str(path.relative_to(skill_dir))
        try:
            if path.stat().st_size > max_doc_bytes:
                skipped.append(relative_path)
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            skipped.append(relative_path)
            continue
        files[relative_path] = content
        images_by_file[relative_path] = extract_markdown_image_paths(content, markdown_path=path)
    return files, skipped, images_by_file


def extract_markdown_image_paths(
    markdown_text: str,
    markdown_path: Path | None = None,
) -> List[str]:
    """Extract image paths referenced by markdown image tags."""
    image_paths: List[str] = []
    for match in MARKDOWN_IMAGE_PATTERN.finditer(markdown_text):
        image_path = match.group(1).strip()
        if markdown_path is not None and not re.match(r"^[a-zA-Z]+://", image_path):
            image_path = str((markdown_path.parent / image_path).resolve())
        image_paths.append(image_path)
    return image_paths


def split_markdown_into_message_sections(
    markdown_text: str,
    markdown_path: Path | None = None,
) -> List[Dict[str, Any]]:
    """Split markdown content into prompt sections."""
    parts = MESSAGE_BREAK_PATTERN.split(markdown_text)
    sections: List[Dict[str, Any]] = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        sections.append(
            {
                "text": stripped,
                "image_paths": extract_markdown_image_paths(stripped, markdown_path=markdown_path),
            }
        )
    return sections


def build_skill_messages(
    files: Dict[str, str],
    skill_root: Path | None = None,
) -> List[Dict[str, Any]]:
    """Build OpenAI message payloads from collected skill markdown files."""
    messages: List[Dict[str, Any]] = []
    for relative_path, file_content in files.items():
        if not isinstance(relative_path, str) or not isinstance(file_content, str):
            continue
        markdown_path = skill_root / relative_path if skill_root is not None else None
        for section in split_markdown_into_message_sections(
            file_content,
            markdown_path=markdown_path,
        ):
            if len(section["image_paths"]) == 0:
                messages.append(generate_openai_message(content=section["text"], role="user"))
                continue
            try:
                messages.append(
                    generate_openai_message(
                        content=section["text"],
                        role="user",
                        image_path=section["image_paths"][0],
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load skill image '%s': %s",
                    section["image_paths"][0],
                    exc,
                )
                messages.append(generate_openai_message(content=section["text"], role="user"))
            for image_path in section["image_paths"][1:]:
                try:
                    messages.append(generate_openai_message(content="", role="user", image_path=image_path))
                except Exception as exc:
                    logger.warning("Failed to load skill image '%s': %s", image_path, exc)
    return messages
