"""
Proposal data models — structured representations for skill extraction proposals.

Used by the parallel (map-reduce) extraction method. Each ProposalSet contains
a list of SkillProposals (add/update/delete operations) that can be merged
hierarchically before being applied to the SkillStore.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SkillProposal(BaseModel):
    """A single proposed skill store operation.

    - **add**: Propose a new skill. Requires ``name``, ``description``,
      ``body``.
    - **update**: Modify an existing skill. Requires ``skill_name``, plus
      any fields to change (``name``, ``description``, ``body``).
    - **delete**: Remove a skill. Requires ``skill_name``.
    """

    action: Literal["add", "update", "delete"]
    skill_name: str = ""  # Required for update/delete (slug identifier)
    name: str = ""        # New name for add; optional new name for update
    description: str = ""
    body: str = ""
    references: list[dict] = Field(default_factory=list)  # [{filename, content}]
    scripts: list[dict] = Field(default_factory=list)      # [{filename, content}]
    reasoning: str = ""  # Explanation of why this operation is proposed
    source_trajectory_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy(cls, data):
        """Backward compat: content→body, skill_id→skill_name."""
        if isinstance(data, dict):
            if "content" in data and "body" not in data:
                data["body"] = data.pop("content")
            if "skill_id" in data and "skill_name" not in data:
                data["skill_name"] = data.pop("skill_id")
        return data


class ProposalSet(BaseModel):
    """A collection of skill proposals — the output of a map or reduce step.

    The same data structure is used for:
    - Map output (proposals from a single trajectory)
    - Intermediate reduce output (merged proposals from a group)
    - Final reduce input (merged proposals ready for tool-call application)
    """

    proposals: list[SkillProposal] = Field(default_factory=list)
    source_trajectory_ids: list[str] = Field(default_factory=list)
    summary: str = ""
