"""
Read-only skill provider — loads a SkillSet from JSON and exposes
list_skills / view_skill / read_skill_file as OpenAI function-calling tools.

Progressive disclosure (aligned with Agent Skills standard):
  1. list_skills  → name + description (~100 tokens per skill)
  2. view_skill   → full body + file name listing
  3. read_skill_file → specific attached file content

Designed to be composable: the tool_schemas can be merged into any
tool list, and dispatch() can be called from any agent loop.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from skilllens.schema.skill import Skill, SkillSet

logger = logging.getLogger(__name__)


def _json_response(**kwargs) -> str:
    """Return a JSON-encoded response string."""
    return json.dumps(kwargs, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI Chat Completions format)
# ---------------------------------------------------------------------------

SKILL_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": (
                "List all skills in the skill library. "
                "Returns name and a short description for each skill. "
                "Does NOT include full body or attached files — use "
                "view_skill for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_skill",
            "description": (
                "View a specific skill's full body (Markdown instructions). "
                "Also lists attached reference and script file names, but "
                "does NOT include their contents — use read_skill_file for that."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The slug name of the skill to view.",
                    },
                },
                "required": ["skill_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill_file",
            "description": (
                "Read the full content of an attached file (reference or "
                "script) belonging to a skill. Use view_skill first to see "
                "the list of available file names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The slug name of the skill.",
                    },
                    "filename": {
                        "type": "string",
                        "description": (
                            "The filename to read "
                            "(e.g. 'REFERENCE.md', 'extract.py')."
                        ),
                    },
                },
                "required": ["skill_name", "filename"],
            },
        },
    },
]

SKILL_TOOL_NAMES = {"list_skills", "view_skill", "read_skill_file"}


# ---------------------------------------------------------------------------
# ReadOnlySkillProvider
# ---------------------------------------------------------------------------

class ReadOnlySkillProvider:
    """In-memory read-only skill store loaded from a SkillSet.

    Exposes ``list_skills``, ``view_skill``, and ``read_skill_file`` as
    tools compatible with OpenAI function-calling format. No mutation.
    """

    def __init__(self, skill_set: SkillSet):
        self._skill_set = skill_set
        # Index by name for O(1) lookup
        self._skills: dict[str, dict] = {}
        for skill in skill_set.skills:
            self._skills[skill.name] = {
                "name": skill.name,
                "description": skill.description,
                "body": skill.body,
                "references": [
                    {"filename": r.filename, "content": r.content}
                    for r in skill.references
                ],
                "scripts": [
                    {"filename": s.filename, "content": s.content}
                    for s in skill.scripts
                ],
            }
        logger.info(
            "ReadOnlySkillProvider loaded %d skills (extractor=%s, method=%s)",
            len(self._skills),
            skill_set.extractor_model,
            skill_set.extraction_method,
        )

        # Track tool call history for analysis
        self.call_history: list[dict] = []

    @classmethod
    def from_file(cls, path: str | Path) -> "ReadOnlySkillProvider":
        """Load a SkillSet from a skill_set.json file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Skill set file not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        skill_set = SkillSet(**data)
        return cls(skill_set)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tool_schemas(self) -> list[dict]:
        """Return tool schemas in Chat Completions format."""
        return SKILL_TOOL_SCHEMAS

    @property
    def skill_count(self) -> int:
        return len(self._skills)

    @property
    def is_single_skill(self) -> bool:
        """True when the provider contains exactly one skill."""
        return len(self._skills) == 1

    def get_single_skill(self) -> dict:
        """Return the single skill dict.  Raises ValueError if count != 1."""
        if not self.is_single_skill:
            raise ValueError(
                f"get_single_skill() requires exactly 1 skill, got {len(self._skills)}"
            )
        return next(iter(self._skills.values()))

    # ------------------------------------------------------------------
    # Tool methods
    # ------------------------------------------------------------------

    def list_skills(self) -> str:
        """List all skills (name, description only — no body or files)."""
        if not self._skills:
            result = _json_response(
                skills=[],
                message="Skill library is empty.",
                count=0,
            )
        else:
            summaries = [
                {
                    "name": s["name"],
                    "description": s["description"],
                }
                for s in self._skills.values()
            ]
            result = _json_response(
                skills=summaries,
                count=len(summaries),
            )
        self._log("list_skills", "")
        return result

    def view_skill(self, skill_name: str) -> str:
        """View skill body + list of attached file names."""
        skill = self._skills.get(skill_name)
        if skill is None:
            self._log("view_skill", skill_name, status="not_found")
            return _json_response(
                status="error",
                reason=f"Skill '{skill_name}' not found. "
                       f"Use list_skills to see available skill names.",
            )

        # Progressive disclosure: body in full, files by name only
        view = {
            "name": skill["name"],
            "description": skill["description"],
            "body": skill["body"],
            "references": [r["filename"] for r in skill.get("references", [])],
            "scripts": [s["filename"] for s in skill.get("scripts", [])],
        }
        self._log("view_skill", skill_name, status="ok")
        return _json_response(status="ok", skill=view)

    def read_skill_file(self, skill_name: str, filename: str) -> str:
        """Read the full content of an attached file."""
        skill = self._skills.get(skill_name)
        if skill is None:
            self._log("read_skill_file", skill_name, status="skill_not_found",
                       filename=filename)
            return _json_response(
                status="error",
                reason=f"Skill '{skill_name}' not found.",
            )

        # Search in references and scripts
        for f in skill.get("references", []):
            if f["filename"] == filename:
                self._log("read_skill_file", skill_name, status="ok",
                           filename=filename, source="references")
                return _json_response(
                    status="ok",
                    filename=filename,
                    content=f["content"],
                )
        for f in skill.get("scripts", []):
            if f["filename"] == filename:
                self._log("read_skill_file", skill_name, status="ok",
                           filename=filename, source="scripts")
                return _json_response(
                    status="ok",
                    filename=filename,
                    content=f["content"],
                )

        available = (
            [r["filename"] for r in skill.get("references", [])]
            + [s["filename"] for s in skill.get("scripts", [])]
        )
        self._log("read_skill_file", skill_name, status="file_not_found",
                   filename=filename)
        return _json_response(
            status="error",
            reason=f"File '{filename}' not found in skill '{skill_name}'.",
            available_files=available,
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def handles(self, function_name: str) -> bool:
        """Check if this provider handles the given tool name."""
        return function_name in SKILL_TOOL_NAMES

    def dispatch(self, function_name: str, arguments: dict) -> str:
        """Route a tool call to the appropriate method."""
        if function_name == "list_skills":
            return self.list_skills()
        elif function_name == "view_skill":
            return self.view_skill(**arguments)
        elif function_name == "read_skill_file":
            return self.read_skill_file(**arguments)
        else:
            return _json_response(
                status="error",
                reason=f"Unknown skill tool: '{function_name}'.",
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log(self, action: str, skill_name: str, **extra) -> None:
        """Record a tool call to history."""
        entry = {
            "action": action,
            "skill_name": skill_name,
            **extra,
        }
        self.call_history.append(entry)
        logger.debug("SkillProvider %s: %s", action, skill_name or "(n/a)")
