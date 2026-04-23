from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from eaa_core.skill import (
    SkillMetadata,
    build_skill_messages,
    collect_skill_docs,
    load_skills,
    resolve_skill_metadata,
)
from eaa_core.tool.base import BaseTool, tool


class SkillLibraryTool(BaseTool):
    """Expose the configured skill catalog and skill-loading helpers."""

    def __init__(
        self,
        skill_dirs: Sequence[str] = (),
        *,
        max_doc_bytes: int = 200_000,
        require_approval: bool = False,
        **kwargs: Any,
    ) -> None:
        self.skill_dirs = list(skill_dirs)
        self.skill_catalog: List[SkillMetadata] = []
        self.max_doc_bytes = max_doc_bytes
        self.refresh_skill_catalog()
        super().__init__(require_approval=require_approval, **kwargs)

    def refresh_skill_catalog(self) -> List[SkillMetadata]:
        """Reload the skill catalog from the configured directories."""
        self.skill_catalog = load_skills(self.skill_dirs)
        return list(self.skill_catalog)

    @tool(name="get_skill_catalog")
    def get_skill_catalog(self) -> List[Dict[str, str]]:
        """Return the available skill catalog."""
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "path": skill.path,
            }
            for skill in self.skill_catalog
        ]

    @tool(name="load_skill")
    def load_skill(self, skill_name: str) -> Dict[str, Any]:
        """Load a skill by name or path and return built messages."""
        metadata = resolve_skill_metadata(self.skill_catalog, skill_name)
        if metadata is None:
            raise ValueError(f"Unknown skill requested: {skill_name}")
        files, skipped, images_by_file = collect_skill_docs(
            Path(metadata.path),
            max_doc_bytes=self.max_doc_bytes,
        )
        messages = build_skill_messages(
            files=files,
            skill_root=Path(metadata.path),
        )
        return {
            "name": metadata.name,
            "description": metadata.description,
            "path": metadata.path,
            "files": files,
            "messages": messages,
            "images_by_file": images_by_file,
            "skipped_files": skipped,
        }
