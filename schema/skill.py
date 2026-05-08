"""
Skill data models — aligned with the Agent Skills open standard (agentskills.io).

Simplified for research: keeps name, description, body, references, scripts,
and source_trajectories. Drops license, allowed-tools, compatibility, and
per-skill metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, model_validator


class SkillFile(BaseModel):
    """An inline file attached to a skill.

    Simulates the ``references/`` or ``scripts/`` directory in the Agent Skills
    standard.  File type is inferred from the extension (e.g. ``.py`` = Python,
    ``.sh`` = bash, ``.md`` = Markdown).
    """

    filename: str  # e.g. "REFERENCE.md", "extract.py", "run.sh"
    content: str   # file content


class Skill(BaseModel):
    """A single extracted skill, following the Agent Skills open standard.

    The ``name`` field (slug format) serves as the unique identifier — there is
    no separate ``id`` field.  Per-skill ``metadata`` is also removed; extraction
    internals (update_count, created_at, etc.) are tracked only in the
    SkillStore's operation history.
    """

    # --- SKILL.md frontmatter ---
    name: str         # slug: lowercase letters, digits, hyphens; max 64 chars
    description: str  # 1–1024 chars; what this skill does and when to use it

    # --- SKILL.md body ---
    body: str = ""    # Markdown instructions (procedures, strategies, etc.)

    # --- Attached resources (inline, simulating directories) ---
    references: list[SkillFile] = Field(default_factory=list)  # references/
    scripts: list[SkillFile] = Field(default_factory=list)     # scripts/

    # --- Provenance (research-specific, not in the standard) ---
    source_trajectories: list[str] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Backward compatibility with legacy skill_set.json files
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # content → body
            if "content" in data and "body" not in data:
                data["body"] = data.pop("content")
            # Drop old id field (name is now the identifier)
            data.pop("id", None)
            # Drop old per-skill metadata field
            data.pop("metadata", None)
        return data


class SkillSet(BaseModel):
    """The output of a skill extraction run — a collection of skills."""

    skills: list[Skill] = Field(default_factory=list)
    extractor_model: str = ""           # Model used for extraction
    extraction_method: str = ""         # Name of the extraction method
    extraction_config: dict = Field(default_factory=dict)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict = Field(default_factory=dict)  # SkillSet-level metadata
