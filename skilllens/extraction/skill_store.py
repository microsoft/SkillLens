"""
SkillStore — in-memory skill manager exposed as OpenAI function-calling tools.

Maintains a dict of skills keyed by slug name, with programmatic constraints
(max count, max chars).  Provides 7 tools: list_skills, view_skill,
read_skill_file, add_skill, update_skill, delete_skill, finish_extraction.

Aligned with the Agent Skills open standard (agentskills.io):
- Skill name (slug format) is the unique identifier (no separate id)
- Skills have body (Markdown instructions), references, and scripts
- Progressive disclosure: view_skill returns body + file names,
  read_skill_file returns specific file content
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime, timezone

from skilllens.schema.skill import Skill, SkillFile, SkillSet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str, max_len: int = 64) -> str:
    """Convert text to a slug matching Agent Skills name constraints.

    Rules: lowercase, alphanumeric + hyphens, no leading/trailing/double hyphens.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)  # no consecutive hyphens
    text = text.strip("-")
    if len(text) > max_len:
        text = text[:max_len].rstrip("-")
    return text or "skill"


def _json_response(**kwargs) -> str:
    """Return a JSON-encoded response string."""
    return json.dumps(kwargs, ensure_ascii=False)


def _skill_char_count(skill: dict) -> int:
    """Total chars for a skill: description + body + all file contents."""
    total = len(skill.get("description", "")) + len(skill.get("body", ""))
    for ref in skill.get("references", []):
        total += len(ref.get("content", ""))
    for scr in skill.get("scripts", []):
        total += len(scr.get("content", ""))
    return total


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": (
                "List all skills in the skill store. "
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
                "Read the full content of an attached file (reference or script) "
                "belonging to a skill. Use view_skill first to see the list of "
                "available file names."
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
                        "description": "The filename to read (e.g. 'REFERENCE.md', 'extract.py').",
                    },
                },
                "required": ["skill_name", "filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_skill",
            "description": "Add a new skill to the store.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Skill name in slug format: lowercase letters, digits, "
                            "and hyphens only, max 64 characters "
                            "(e.g. 'debug-failing-test', 'normalize-data-types')."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "One or two sentences ONLY: what class of problems "
                            "this skill addresses and when to apply it. "
                            "Keep it short and scannable — no procedural detail."
                        ),
                    },
                    "body": {
                        "type": "string",
                        "description": (
                            "The skill's main instructions in Markdown format. "
                            "Should capture reusable procedural knowledge that "
                            "another agent can follow independently. May include: "
                            "step-by-step procedures, strategies and heuristics, "
                            "common pitfalls and workarounds, applicable scenarios, "
                            "key commands or code patterns, and verification methods."
                        ),
                    },
                    "references": {
                        "type": "array",
                        "description": (
                            "Optional attached reference documents. Each item is "
                            "an object with 'filename' and 'content' fields."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {
                                    "type": "string",
                                    "description": "File name (e.g. 'REFERENCE.md').",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "File content.",
                                },
                            },
                            "required": ["filename", "content"],
                        },
                    },
                    "scripts": {
                        "type": "array",
                        "description": (
                            "Optional attached executable scripts. Each item is "
                            "an object with 'filename' and 'content' fields."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {
                                    "type": "string",
                                    "description": "File name with extension (e.g. 'extract.py', 'run.sh').",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "File content.",
                                },
                            },
                            "required": ["filename", "content"],
                        },
                    },
                },
                "required": ["name", "description", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_skill",
            "description": (
                "Update an existing skill. Only provide the fields you want "
                "to change — omitted fields are left unchanged. "
                "Note: references and scripts are replaced entirely if provided."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The slug name of the skill to update.",
                    },
                    "name": {
                        "type": "string",
                        "description": "New skill name (slug format).",
                    },
                    "description": {
                        "type": "string",
                        "description": "New description (1-2 sentences only).",
                    },
                    "body": {
                        "type": "string",
                        "description": (
                            "New Markdown body. Should capture reusable "
                            "procedural knowledge that another agent can follow "
                            "independently."
                        ),
                    },
                    "references": {
                        "type": "array",
                        "description": (
                            "New list of reference files (replaces all existing). "
                            "Each item: {filename, content}."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["filename", "content"],
                        },
                    },
                    "scripts": {
                        "type": "array",
                        "description": (
                            "New list of script files (replaces all existing). "
                            "Each item: {filename, content}."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["filename", "content"],
                        },
                    },
                },
                "required": ["skill_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_skill",
            "description": "Remove a skill from the store.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The slug name of the skill to delete.",
                    },
                },
                "required": ["skill_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_extraction",
            "description": (
                "Call this when you are done processing the current trajectory. "
                "Provide a brief summary of what changes you made to the skill store."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": (
                            "Brief summary of skill operations performed "
                            "for this trajectory."
                        ),
                    },
                },
                "required": ["summary"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# SkillStore
# ---------------------------------------------------------------------------

class SkillStore:
    """In-memory skill store with programmatic constraints.

    Parameters
    ----------
    max_skills : int
        Maximum number of skills allowed. ``add_skill`` returns an error
        when this limit is reached.
    max_skill_chars : int
        Maximum total characters for a single skill
        (description + body + references + scripts).  0 means no per-skill limit.
    max_total_chars : int
        Maximum total characters across all skills.
        0 means no total limit.
    """

    def __init__(
        self,
        *,
        max_skills: int = 10,
        max_skill_chars: int = 2000,
        max_total_chars: int = 0,
    ):
        self._skills: dict[str, dict] = {}  # name → skill dict
        self._name_counts: dict[str, int] = {}  # slug → occurrence count

        # Constraints
        self.max_skills = max_skills
        self.max_skill_chars = max_skill_chars
        self.max_total_chars = max_total_chars

        # Operation history for logging/analysis
        self.history: list[dict] = []

        # Summaries collected from finish_extraction calls
        self.summaries: list[dict] = []

        # Current trajectory context (set via set_trajectory_context)
        self._traj_index: int = -1
        self._traj_id: str = ""
        self._traj_op_counter: int = 0  # resets each trajectory

    # ------------------------------------------------------------------
    # Retry support
    # ------------------------------------------------------------------

    def reset_for_retry(self) -> None:
        """Clear skills and summaries so final reduce can be retried.

        Keeps constraints, history (for audit), and trajectory context.
        """
        self._skills.clear()
        self._name_counts.clear()
        self.summaries.clear()
        self._traj_op_counter = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tool_schemas(self) -> list[dict]:
        """Return OpenAI function-calling tool definitions."""
        return TOOL_SCHEMAS

    @property
    def skill_count(self) -> int:
        return len(self._skills)

    @property
    def total_chars(self) -> int:
        """Total chars across all skills."""
        return sum(_skill_char_count(s) for s in self._skills.values())

    # ------------------------------------------------------------------
    # Trajectory context
    # ------------------------------------------------------------------

    def set_trajectory_context(self, index: int, trajectory_id: str) -> None:
        """Set the current trajectory context for history logging."""
        self._traj_index = index
        self._traj_id = trajectory_id
        self._traj_op_counter = 0

    # ------------------------------------------------------------------
    # Tool methods
    # ------------------------------------------------------------------

    def list_skills(self) -> str:
        """List all skills (name, description only)."""
        if not self._skills:
            result = _json_response(
                skills=[],
                message="Skill store is empty.",
                count=0,
                max_skills=self.max_skills,
            )
            self._log("list_skills", "", status="ok", result_count=0)
            return result

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
            max_skills=self.max_skills,
        )
        self._log("list_skills", "", status="ok", result_count=len(summaries))
        return result

    def view_skill(self, skill_name: str) -> str:
        """View skill body + list of attached file names."""
        skill = self._skills.get(skill_name)
        if skill is None:
            self._log("view_skill", skill_name, status="error", reason="not_found")
            return _json_response(
                status="error",
                reason=f"Skill '{skill_name}' not found.",
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
        """Read the full content of an attached file (reference or script)."""
        skill = self._skills.get(skill_name)
        if skill is None:
            self._log("read_skill_file", skill_name, status="error",
                       reason="skill_not_found", filename=filename)
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

        self._log("read_skill_file", skill_name, status="error",
                   reason="file_not_found", filename=filename)
        available = (
            [r["filename"] for r in skill.get("references", [])]
            + [s["filename"] for s in skill.get("scripts", [])]
        )
        return _json_response(
            status="error",
            reason=f"File '{filename}' not found in skill '{skill_name}'.",
            available_files=available,
        )

    def add_skill(
        self,
        name: str,
        description: str,
        body: str,
        references: list[dict] | None = None,
        scripts: list[dict] | None = None,
    ) -> str:
        """Add a new skill."""
        references = references or []
        scripts = scripts or []

        # Slugify the name
        slug = _slugify(name)
        if not slug:
            self._log("add_skill", "", status="error", reason="invalid_name", name=name)
            return _json_response(
                status="error",
                reason=f"Invalid skill name: '{name}'. Use lowercase letters, digits, and hyphens.",
            )

        # Handle name collision: append counter
        if slug in self._skills:
            count = self._name_counts.get(slug, 1) + 1
            self._name_counts[slug] = count
            slug = f"{slug}-{count}"

        # Check count limit
        if len(self._skills) >= self.max_skills:
            self._log("add_skill", "", status="error", reason="skill_limit_reached", name=slug)
            return _json_response(
                status="error",
                reason=(
                    f"Skill limit reached ({len(self._skills)}/{self.max_skills}). "
                    f"Delete or merge existing skills first."
                ),
            )

        # Build skill dict for char counting
        skill = {
            "name": slug,
            "description": description,
            "body": body,
            "references": references,
            "scripts": scripts,
            "source_trajectories": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "update_count": 0,
        }

        # Check per-skill char limit
        char_count = _skill_char_count(skill)
        if self.max_skill_chars > 0 and char_count > self.max_skill_chars:
            self._log("add_skill", "", status="error", reason="skill_too_long",
                       name=slug, char_count=char_count, limit=self.max_skill_chars)
            return _json_response(
                status="error",
                reason=(
                    f"Skill content too long: {char_count} chars "
                    f"(limit: {self.max_skill_chars}). "
                    f"You MUST shorten the body to fit within {self.max_skill_chars} characters. "
                    f"Current excess: {char_count - self.max_skill_chars} chars."
                ),
            )

        # Check total char budget
        if self.max_total_chars > 0:
            new_total = self.total_chars + char_count
            if new_total > self.max_total_chars:
                self._log("add_skill", "", status="error", reason="total_budget_exceeded",
                           name=slug, char_count=char_count, total_would_be=new_total)
                return _json_response(
                    status="error",
                    reason=(
                        f"Total character budget exceeded: "
                        f"current={self.total_chars}, new_skill={char_count}, "
                        f"would_be={new_total}, limit={self.max_total_chars}. "
                        f"Delete or shorten existing skills first."
                    ),
                )

        self._skills[slug] = skill

        self._log("add", slug, name=slug)
        return _json_response(
            status="ok",
            name=slug,
            message=f"Skill '{slug}' added ({len(self._skills)}/{self.max_skills}).",
        )

    def update_skill(
        self,
        skill_name: str,
        name: str | None = None,
        description: str | None = None,
        body: str | None = None,
        references: list[dict] | None = None,
        scripts: list[dict] | None = None,
    ) -> str:
        """Update an existing skill (only provided fields are changed)."""
        skill = self._skills.get(skill_name)
        if skill is None:
            self._log("update_skill", skill_name, status="error", reason="not_found")
            return _json_response(
                status="error",
                reason=f"Skill '{skill_name}' not found.",
            )

        # Compute new values
        new_desc = description if description is not None else skill["description"]
        new_body = body if body is not None else skill["body"]
        new_refs = references if references is not None else skill["references"]
        new_scripts = scripts if scripts is not None else skill["scripts"]

        # Build a temporary dict for char counting
        temp = {
            "description": new_desc,
            "body": new_body,
            "references": new_refs,
            "scripts": new_scripts,
        }
        new_char_count = _skill_char_count(temp)

        if self.max_skill_chars > 0 and new_char_count > self.max_skill_chars:
            self._log("update_skill", skill_name, status="error", reason="skill_too_long",
                       char_count=new_char_count, limit=self.max_skill_chars)
            return _json_response(
                status="error",
                reason=(
                    f"Updated skill content too long: {new_char_count} chars "
                    f"(limit: {self.max_skill_chars}). "
                    f"Please shorten the body or attached files."
                ),
            )

        # Check total char budget
        if self.max_total_chars > 0:
            old_char_count = _skill_char_count(skill)
            delta = new_char_count - old_char_count
            if self.total_chars + delta > self.max_total_chars:
                self._log("update_skill", skill_name, status="error",
                           reason="total_budget_exceeded",
                           delta=delta, total_would_be=self.total_chars + delta)
                return _json_response(
                    status="error",
                    reason=(
                        f"Total character budget would be exceeded after update. "
                        f"Current total: {self.total_chars}, delta: +{delta}, "
                        f"limit: {self.max_total_chars}."
                    ),
                )

        # Handle rename
        new_name = skill_name
        if name is not None and name != skill_name:
            new_slug = _slugify(name)
            if new_slug in self._skills and new_slug != skill_name:
                self._log("update_skill", skill_name, status="error",
                           reason="name_collision", new_name=new_slug)
                return _json_response(
                    status="error",
                    reason=f"Skill name '{new_slug}' already exists.",
                )
            # Remove old key, insert new key
            del self._skills[skill_name]
            new_name = new_slug
            skill["name"] = new_name

        # Apply updates
        skill["description"] = new_desc
        skill["body"] = new_body
        skill["references"] = new_refs
        skill["scripts"] = new_scripts
        skill["update_count"] = skill.get("update_count", 0) + 1
        skill["last_updated"] = datetime.now(timezone.utc).isoformat()

        self._skills[new_name] = skill

        self._log("update", new_name, old_name=skill_name if new_name != skill_name else "")
        return _json_response(
            status="ok",
            message=f"Skill '{new_name}' updated.",
        )

    def delete_skill(self, skill_name: str) -> str:
        """Delete a skill."""
        if skill_name not in self._skills:
            self._log("delete_skill", skill_name, status="error", reason="not_found")
            return _json_response(
                status="error",
                reason=f"Skill '{skill_name}' not found.",
            )

        del self._skills[skill_name]

        self._log("delete", skill_name)
        return _json_response(
            status="ok",
            message=f"Skill '{skill_name}' deleted. "
                    f"Remaining: {len(self._skills)}/{self.max_skills}.",
        )

    def finish_extraction(self, summary: str) -> str:
        """Signal that processing of the current trajectory is complete."""
        self.summaries.append({
            "summary": summary,
            "skill_count": len(self._skills),
            "total_chars": self.total_chars,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._log("finish", "", summary=summary)
        return _json_response(status="done")

    # ------------------------------------------------------------------
    # Dispatch (unified tool call handler)
    # ------------------------------------------------------------------

    def dispatch(self, function_name: str, arguments: dict) -> str:
        """Route a tool call to the appropriate method."""
        try:
            if function_name == "list_skills":
                return self.list_skills()
            elif function_name == "view_skill":
                return self.view_skill(**arguments)
            elif function_name == "read_skill_file":
                return self.read_skill_file(**arguments)
            elif function_name == "add_skill":
                return self.add_skill(**arguments)
            elif function_name == "update_skill":
                return self.update_skill(**arguments)
            elif function_name == "delete_skill":
                return self.delete_skill(**arguments)
            elif function_name == "finish_extraction":
                return self.finish_extraction(**arguments)
            else:
                self._log("unknown_tool", "", status="error", reason="unknown_tool",
                           function_name=function_name)
                return _json_response(
                    status="error",
                    reason=f"Unknown tool: '{function_name}'.",
                )
        except TypeError as e:
            self._log(function_name, "", status="error",
                       reason=f"Invalid arguments: {e}")
            return _json_response(
                status="error",
                reason=f"Invalid arguments for '{function_name}': {e}",
            )

    # ------------------------------------------------------------------
    # Export & snapshot
    # ------------------------------------------------------------------

    def to_skill_set(
        self,
        extractor_model: str = "",
        extraction_method: str = "sequential",
        extraction_config: dict | None = None,
    ) -> SkillSet:
        """Export the current store as a ``SkillSet``."""
        skills = [
            Skill(
                name=s["name"],
                description=s["description"],
                body=s.get("body", ""),
                references=[
                    SkillFile(filename=r["filename"], content=r["content"])
                    for r in s.get("references", [])
                ],
                scripts=[
                    SkillFile(filename=sc["filename"], content=sc["content"])
                    for sc in s.get("scripts", [])
                ],
                source_trajectories=s.get("source_trajectories", []),
            )
            for s in self._skills.values()
        ]

        return SkillSet(
            skills=skills,
            extractor_model=extractor_model,
            extraction_method=extraction_method,
            extraction_config=extraction_config or {},
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={
                "constraints": {
                    "max_skills": self.max_skills,
                    "max_skill_chars": self.max_skill_chars,
                    "max_total_chars": self.max_total_chars,
                },
                "final_skill_count": len(self._skills),
                "final_total_chars": self.total_chars,
                "total_operations": len(self.history),
            },
        )

    def snapshot(self) -> dict:
        """Return a serializable snapshot of the current state."""
        return {
            "skills": list(self._skills.values()),
            "skill_count": len(self._skills),
            "total_chars": self.total_chars,
            "constraints": {
                "max_skills": self.max_skills,
                "max_skill_chars": self.max_skill_chars,
                "max_total_chars": self.max_total_chars,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Internal logging
    # ------------------------------------------------------------------

    def _log(self, action: str, skill_name: str, **extra) -> None:
        """Record an operation to history."""
        self._traj_op_counter += 1
        entry = {
            "trajectory_index": self._traj_index,
            "trajectory_id": self._traj_id,
            "tag": f"traj{self._traj_index}_op{self._traj_op_counter}",
            "action": action,
            "skill_name": skill_name,
            "skill_count": len(self._skills),
            "total_chars": self.total_chars,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        self.history.append(entry)
        logger.debug("SkillStore %s: %s", action, skill_name or "(n/a)")
