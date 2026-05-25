"""
Prompt templates for the parallel extraction method — Map phase.

Each map subagent analyses a single trajectory and produces a structured
ProposalSet (JSON) of add/update/delete operations, given a read-only
snapshot of the current skill store.
"""

from __future__ import annotations

import json

from skilllens.schema.trajectory import Step, Trajectory


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_MAP_SYSTEM_PROMPT_TEMPLATE = """\
You are a **Skill Extraction Sub-Agent** working in parallel with other \
sub-agents. Your task is to analyse a single agent trajectory and propose \
changes to a shared skill store.

## Your Role

You are ONE of several sub-agents processing different trajectories \
simultaneously. Your proposals will be **merged with those of other \
sub-agents** before being applied to the skill store. Therefore:

- **Focus on genuinely novel, reusable knowledge** from THIS trajectory.
- Do NOT try to build a complete skill set on your own.
- Assume other sub-agents handle other trajectories.

## Skill Quality Requirements

Each skill must satisfy ALL of the following criteria:

1. **High-level and domain-general**: A skill should capture a broad \
problem-solving strategy, design principle, or methodological pattern that \
applies to an entire CLASS of tasks within a domain — not a solution to \
one particular problem instance. Abstract away from the specific task, \
dataset, or environment in this trajectory to the underlying general pattern.

2. **Maximally broad coverage**: Prefer skills that cover WIDER categories \
of situations. When you identify a useful pattern, ask whether it can be \
stated more generally to cover more scenarios. A single well-generalized \
skill is worth more than several narrow ones.

3. **Information-dense and actionable**: Despite being general, each skill \
must contain CONCRETE procedural guidance — step-by-step strategies, \
decision criteria, diagnostic heuristics, and verification methods that \
an agent can directly follow. Avoid platitudes and surface-level advice. \
Every sentence should carry information that would materially help an agent \
solve a task.

4. **Self-contained**: Another agent must be able to follow the skill \
independently on a different task without needing context from the \
original trajectory.

**Litmus test**: "Would this skill meaningfully help an agent facing a \
DIFFERENT task in the SAME broad domain?" If no — it is too specific. \
If it reads like obvious advice anyone would already know — it lacks \
information density.

## Skill Structure (Agent Skills Standard)

Each skill follows the Agent Skills open standard and has these fields:
- **`name`**: Slug format (lowercase letters, digits, hyphens; max 64 chars). \
Names a general capability or methodology, not a task description.
- **`description`**: **One or two sentences only.** State WHAT class of \
problems this skill addresses and WHEN to apply it. Must be short enough \
to scan at a glance — no procedural detail here.
- **`body`**: A **Markdown knowledge document** with concrete procedural \
guidance. This is where ALL the detail goes — step-by-step strategies, \
decision criteria, diagnostic heuristics, verification methods. Should be \
dense with useful information while remaining broadly applicable.
- **`references`** (optional): Attached reference documents. \
Each is a {{filename, content}} object.
- **`scripts`** (optional): Attached executable scripts. \
Each is a {{filename, content}} object.

## Available Actions

| Action | When to use | Required fields |
|--------|-------------|-----------------|
| `add` | New reusable knowledge not covered by existing skills | `name`, `description`, `body` |
| `update` | Trajectory reveals improvements or additions to an existing skill | `skill_name`, plus fields to change |
| `delete` | An existing skill is superseded, incorrect, or too narrow | `skill_name` |

## Guidelines

- **Generalize aggressively**: Strip away all task-specific details \
(specific names, identifiers, error messages, tool-specific APIs) and \
replace them with general categories and patterns. The skill should read \
as if it was written by someone who has seen hundreds of similar tasks, \
not someone reporting on one trajectory.
- **Maximize coverage per skill**: If two patterns from this trajectory \
could be unified into a broader principle, do so.
- **Maintain information density**: General does NOT mean vague. Every \
step, heuristic, and decision criterion in the content should provide \
actionable guidance that goes beyond common sense.
- For `update`, only include the fields you want to change.
- For `add`, verify it is not a near-duplicate of an existing skill.
- Include a brief `reasoning` for each proposal.
- It is OK to propose nothing if the trajectory has no generalizable insights.

## Constraints

The skill store has hard limits:
{constraints_block}
Keep these in mind when proposing `add` operations — do not exceed per-skill \
char limits (description + body + attached file contents combined).

## Output Format

Output a **single JSON object** (no extra text before or after):

```json
{{
  "proposals": [
    {{
      "action": "add",
      "name": "skill-name-in-slug-format",
      "description": "One paragraph description.",
      "body": "Markdown procedural content...",
      "reasoning": "Why this skill should be added."
    }},
    {{
      "action": "update",
      "skill_name": "existing-skill-name",
      "body": "Updated markdown content...",
      "reasoning": "Why this update improves the skill."
    }},
    {{
      "action": "delete",
      "skill_name": "obsolete-skill-name",
      "reasoning": "Why this skill should be removed."
    }}
  ],
  "summary": "Brief summary of what you found in this trajectory."
}}
```

If the trajectory contains no useful new knowledge, output:
```json
{{"proposals": [], "summary": "No new insights found."}}
```\
"""


def build_map_system_prompt(
    *,
    max_skills: int = 10,
    max_skill_chars: int = 2000,
    max_total_chars: int = 0,
) -> str:
    """Build the system prompt for a map sub-agent."""
    lines = []
    lines.append(f"- **Maximum number of skills**: {max_skills}")
    if max_skill_chars > 0:
        lines.append(
            f"- **Maximum characters per skill** (description + body + files): "
            f"{max_skill_chars:,}"
        )
    if max_total_chars > 0:
        lines.append(
            f"- **Total character budget** across all skills: "
            f"{max_total_chars:,}"
        )
    else:
        lines.append("- **Total character budget**: unlimited")

    constraints_block = "\n".join(lines)
    return _MAP_SYSTEM_PROMPT_TEMPLATE.format(constraints_block=constraints_block)


# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------

def format_store_snapshot(snapshot: dict) -> str:
    """Format a SkillStore snapshot as a readable markdown section.

    Parameters
    ----------
    snapshot : dict
        Output of ``SkillStore.snapshot()`` — has ``skills``, ``skill_count``,
        ``constraints`` keys.
    """
    skills = snapshot.get("skills", [])
    max_skills = snapshot.get("constraints", {}).get("max_skills", "?")

    if not skills:
        return (
            f"## Current Skill Store (0/{max_skills} skills)\n\n"
            "The skill store is empty."
        )

    lines = [f"## Current Skill Store ({len(skills)}/{max_skills} skills)\n"]
    for i, s in enumerate(skills, 1):
        name = s.get("name", "?")
        desc = s.get("description", "")
        # Truncate long descriptions for prompt efficiency
        if len(desc) > 200:
            desc = desc[:200] + "..."
        lines.append(f"{i}. **[{name}]** {desc}")

    return "\n".join(lines)


def build_map_user_message(
    traj: Trajectory,
    store_snapshot: dict,
    *,
    include_feedback: bool = True,
) -> str:
    """Build the user message for a map sub-agent.

    Parameters
    ----------
    traj : Trajectory
        The trajectory to analyse.
    store_snapshot : dict
        Output of ``SkillStore.snapshot()``.
    include_feedback : bool
        Whether to include outcome/reward information.
    """
    parts: list[str] = []

    # Part 1: Store snapshot
    parts.append(format_store_snapshot(store_snapshot))
    parts.append("")
    parts.append("---")
    parts.append("")

    # Part 2: Trajectory
    parts.append("## Trajectory")
    parts.append("")

    if include_feedback and traj.outcome:
        parts.append(f"**Outcome**: {traj.outcome} (reward={traj.reward})")
        parts.append("")

    parts.append("---")
    parts.append("")
    parts.append("### Agent Steps")
    parts.append("")

    for step_idx, step in enumerate(traj.steps, 1):
        parts.append(_render_step(step, step_idx))
        parts.append("")

    # Final answer
    if traj.final_answer:
        parts.append("### Final Answer")
        parts.append("")
        fa = traj.final_answer
        if len(fa) > 3000:
            fa = fa[:3000] + f"\n\n...[truncated, {len(traj.final_answer)} chars total]"
        parts.append(fa)

    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append(
        "**[END OF TRAJECTORY]** Now analyse this trajectory and output your "
        "proposals as a JSON object. Remember: extract **high-level, "
        "domain-general strategies and methodologies** with broad coverage "
        "across a class of tasks. Generalize aggressively — strip away all "
        "task-specific details — while keeping the content information-dense "
        "and actionable."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Step renderer (shared with sequential prompts)
# ---------------------------------------------------------------------------

def _render_step(step: Step, index: int) -> str:
    """Render a single step into readable text."""
    parts: list[str] = []

    parts.append(f"**Step {index}** [{step.role}]")

    if step.content:
        parts.append(step.content)

    if step.tool_calls:
        for tc in step.tool_calls:
            fn = (
                tc.get("function_name")
                or tc.get("name")
                or tc.get("function", {}).get("name", "?")
            )
            args_raw = (
                tc.get("arguments")
                or tc.get("function", {}).get("arguments", "")
            )
            if isinstance(args_raw, dict):
                args_str = json.dumps(args_raw, ensure_ascii=False)
            else:
                args_str = str(args_raw)
            parts.append(f"  -> Tool call: `{fn}` -- {args_str}")

    if step.observation:
        parts.append(f"  <- Observation: {step.observation}")

    return "\n".join(parts)
