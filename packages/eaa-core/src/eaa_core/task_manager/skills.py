from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Sequence

from eaa_core.message_proc import generate_openai_message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillMetadata:
    """Metadata for a discovered skill."""

    name: str
    description: str
    path: str


def discover_skills(skill_dirs: Sequence[str]) -> list[SkillMetadata]:
    """Discover skill metadata under configured skill directories.

    Parameters
    ----------
    skill_dirs : Sequence[str]
        Directories to scan. Each entry can be either a skill directory
        containing ``SKILL.md`` or a parent directory containing skill
        subdirectories.

    Returns
    -------
    list[SkillMetadata]
        Discovered skill metadata with paths pointing to ``SKILL.md`` files.
    """
    skills: list[SkillMetadata] = []
    for base_dir in skill_dirs:
        base_path = Path(base_dir).expanduser()
        if not base_path.exists():
            logger.warning("Skill directory not found: %s", base_dir)
            continue
        if (base_path / "SKILL.md").exists():
            skill_files = [base_path / "SKILL.md"]
        else:
            skill_files = sorted(base_path.rglob("SKILL.md"))
        for skill_file in skill_files:
            metadata = parse_skill_metadata(skill_file)
            if metadata is not None:
                skills.append(metadata)
    return skills


def parse_skill_metadata(skill_file: Path) -> SkillMetadata | None:
    """Parse one ``SKILL.md`` file into metadata.

    Parameters
    ----------
    skill_file : Path
        Skill markdown file to parse.

    Returns
    -------
    SkillMetadata or None
        Parsed metadata, or ``None`` when the file does not exist.
    """
    if not skill_file.exists() or not skill_file.is_file():
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
        name = skill_file.parent.name
    if not description:
        description = extract_description(body_lines, name_index)
    if not description:
        description = "No description provided."
    return SkillMetadata(
        name=name,
        description=description,
        path=str(skill_file.resolve()),
    )


def parse_frontmatter(lines: list[str]) -> tuple[dict[str, str], list[str]]:
    """Split markdown frontmatter from the body."""
    if not lines or lines[0].strip() != "---":
        return {}, lines
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return read_frontmatter(lines[1:index]), lines[index + 1 :]
    return {}, lines


def read_frontmatter(lines: list[str]) -> dict[str, str]:
    """Read a simple YAML-like frontmatter block."""
    frontmatter: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        frontmatter[key.strip()] = value.strip().strip("\"").strip("'")
    return frontmatter


def extract_name(lines: list[str]) -> tuple[str | None, int]:
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


def extract_description(lines: list[str], name_index: int) -> str:
    """Extract a short description from markdown lines."""
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("description:"):
            return stripped.split(":", 1)[1].strip()
    return extract_first_paragraph(lines, start_index=name_index + 1)


def extract_first_paragraph(lines: list[str], start_index: int) -> str:
    """Extract the first non-empty paragraph from markdown lines."""
    paragraph: list[str] = []
    for line in lines[start_index:]:
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        paragraph.append(stripped)
    return " ".join(paragraph)


def resolve_skill(
    skill_catalog: Sequence[SkillMetadata],
    skill_name: str,
) -> SkillMetadata | None:
    """Resolve a skill by name, skill file path, or skill directory path."""
    normalized = skill_name.strip()
    if not normalized:
        return None
    normalized_path = str(Path(normalized).expanduser())
    for skill in skill_catalog:
        skill_dir = str(Path(skill.path).parent)
        if normalized in {skill.name, skill.path, skill_dir}:
            return skill
        if normalized_path in {skill.path, skill_dir}:
            return skill
    normalized_lower = normalized.lower()
    normalized_path_lower = normalized_path.lower()
    for skill in skill_catalog:
        skill_dir = str(Path(skill.path).parent)
        if normalized_lower in {
            skill.name.lower(),
            skill.path.lower(),
            skill_dir.lower(),
        }:
            return skill
        if normalized_path_lower in {
            skill.path.lower(),
            skill_dir.lower(),
        }:
            return skill
    return None


def build_skill_context_message(skill: SkillMetadata) -> dict[str, object]:
    """Build the context message for one explicitly selected skill.

    Parameters
    ----------
    skill : SkillMetadata
        Skill to load.

    Returns
    -------
    dict[str, object]
        OpenAI-compatible user message containing only ``SKILL.md``.
    """
    skill_file = Path(skill.path)
    text = skill_file.read_text(encoding="utf-8", errors="replace")
    content = "\n".join(
        [
            "<skill>",
            f"<name>{skill.name}</name>",
            f"<path>{skill.path}</path>",
            text.rstrip(),
            "</skill>",
        ]
    )
    return generate_openai_message(content=content, role="user")


def skill_catalog_to_dicts(skill_catalog: Sequence[SkillMetadata]) -> list[dict[str, str]]:
    """Serialize skill metadata for prompts, APIs, or WebUI storage."""
    return [
        {
            "name": skill.name,
            "description": skill.description,
            "path": skill.path,
        }
        for skill in skill_catalog
    ]
