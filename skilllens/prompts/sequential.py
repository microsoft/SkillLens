"""
Prompt templates for the naive tool-use extraction method.

The LLM acts as a ReAct agent, using tools (list_skills, view_skill,
read_skill_file, add_skill, update_skill, delete_skill, finish_extraction)
to manage an in-memory skill store.
"""

from __future__ import annotations

import json

from skilllens.schema.trajectory import Step, Trajectory


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are a **Skill Extraction Engine**. You maintain a skill store containing \
reusable procedural knowledge extracted from agent trajectories.

You will see agent trajectories one at a time. For each trajectory, use the \
available tools to inspect and update your skill store.

## Available Tools

| Tool | Purpose |
|------|---------|
| `list_skills` | See all skills (name, description). Does NOT show body or files. |
| `view_skill` | Read a skill's full body (Markdown instructions) and see attached file names. |
| `read_skill_file` | Read the content of a specific attached file (reference or script). |
| `add_skill` | Add a new skill to the store. |
| `update_skill` | Modify an existing skill. Only provide fields to change. |
| `delete_skill` | Remove a skill from the store. |
| `finish_extraction` | **Call this when done** with the current trajectory. |

## Workflow

1. Call `list_skills` to see what is already in the store.
2. Read the trajectory carefully — identify reusable procedural knowledge.
3. Decide what skills to add, update, merge, or remove.
4. Use the tools to make your changes.
5. When done, call `finish_extraction` with a brief summary of what you did.

## Skill Structure (Agent Skills Standard)

Each skill follows the Agent Skills open standard and has these fields:
- **`name`**: Slug format (lowercase letters, digits, hyphens; max 64 chars). \
Names a general capability or methodology, not a specific task.
- **`description`**: **One or two sentences only.** State WHAT class of \
problems this skill addresses and WHEN to apply it. Must be short enough \
to scan at a glance — no procedural detail here.
- **`body`**: A **Markdown knowledge document** where ALL procedural \
detail goes — step-by-step strategies, decision criteria, diagnostic \
heuristics, verification methods. Should be dense with useful information \
while remaining broadly applicable.
- **`references`** (optional): Attached reference documents \
(e.g. detailed guides, domain references). Each is a {{filename, content}} object.
- **`scripts`** (optional): Attached executable scripts \
(e.g. diagnostic scripts, helper utilities). Each is a {{filename, content}} object.

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

## Guidelines

- **Generalize aggressively**: Strip away all task-specific details \
(specific names, identifiers, error messages, tool-specific APIs) and \
replace them with general categories and patterns. The skill should read \
as if it was written by someone who has seen hundreds of similar tasks, \
not someone reporting on one trajectory.
- **Maximize coverage per skill**: If two patterns from this trajectory \
could be unified into a broader principle, do so.
- **Maintain information density**: General does NOT mean vague. Every \
step, heuristic, and decision criterion in the body should provide \
actionable guidance that goes beyond common sense.
- **Prefer updating/refining** existing skills over adding near-duplicates. \
If a trajectory demonstrates a better approach to something an existing \
skill already covers, update that skill.
- **When the store is full**, decide whether the new knowledge is valuable \
enough to replace an existing skill. You may delete a less useful skill \
to make room.
- **Use references and scripts** when the knowledge benefits from \
supplementary documentation or reusable code snippets.

## Constraints

The skill store has hard limits enforced programmatically:
{constraints_block}
If you hit a limit, the tool will return an error message. Adjust your
approach accordingly (e.g., shorten the body, delete a skill first, etc.).

**Character budget per skill** counts: description + body + all reference \
file contents + all script file contents.

**IMPORTANT**: You MUST call `finish_extraction` when you are done processing \
the current trajectory. Do NOT simply stop responding.\
"""


def build_system_prompt(
    *,
    max_skills: int = 10,
    max_skill_chars: int = 2000,
    max_total_chars: int = 0,
) -> str:
    """Build the system prompt with concrete constraint values."""
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
    return _SYSTEM_PROMPT_TEMPLATE.format(constraints_block=constraints_block)


# Keep a default for backward compatibility / simple tests
SYSTEM_PROMPT = build_system_prompt()


# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------

def build_user_message(
    traj: Trajectory,
    index: int,
    total: int,
    *,
    include_feedback: bool = True,
) -> str:
    """Build the user-turn message for one extraction round."""
    parts: list[str] = []

    # Header
    parts.append(f"## Trajectory")
    parts.append("")

    # Feedback (optional)
    if include_feedback and traj.outcome:
        parts.append("")
        parts.append(f"**Outcome**: {traj.outcome} (reward={traj.reward})")

    parts.append("")
    parts.append("---")
    parts.append("")

    # Steps (full content, no truncation)
    parts.append("### Agent Steps")
    parts.append("")

    for step_idx, step in enumerate(traj.steps, 1):
        parts.append(_render_step(step, step_idx))
        parts.append("")

    # Final answer
    if traj.final_answer:
        parts.append("### Final Answer")
        parts.append("")
        # Truncate very long final answers (e.g. full patches)
        fa = traj.final_answer
        if len(fa) > 3000:
            fa = fa[:3000] + f"\n\n…[truncated, {len(traj.final_answer)} chars total]"
        parts.append(fa)

    # Role reminder
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append(
        "**[END OF TRAJECTORY]** Now analyze this trajectory as the Skill "
        "Extraction Engine. Extract **high-level, domain-general strategies** "
        "with broad coverage — generalize aggressively while keeping content "
        "information-dense and actionable. Start by calling `list_skills`, "
        "then add/update/delete skills as needed. When done, call "
        "`finish_extraction`."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Step renderer
# ---------------------------------------------------------------------------

def _render_step(step: Step, index: int) -> str:
    """Render a single step into readable text."""
    parts: list[str] = []

    # Header
    parts.append(f"**Step {index}** [{step.role}]")

    # Content
    if step.content:
        parts.append(step.content)

    # Tool calls
    if step.tool_calls:
        for tc in step.tool_calls:
            fn = (
                tc.get("function_name")
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
            parts.append(f"  → Tool call: `{fn}` — {args_str}")

    # Observation
    if step.observation:
        parts.append(f"  ← Observation: {step.observation}")

    return "\n".join(parts)
