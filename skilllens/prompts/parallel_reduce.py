"""
Prompt templates for the parallel extraction method — Reduce phase.

Two prompt sets:
1. **Intermediate reduce**: merges multiple ModeSets into one (JSON output).
2. **Final reduce**: synthesises merged modes into skills via tool-calling.
"""

from __future__ import annotations

from skilllens.prompts.parallel_map import format_store_snapshot
from skilllens.schema.modes import Mode, ModeSet


# =========================================================================
# 1. Intermediate Reduce — JSON output
# =========================================================================

_INTERMEDIATE_REDUCE_SYSTEM_TEMPLATE = """\
You are a **Mode Merger**. You receive mode sets from multiple parallel \
sub-agents, each of which analysed a different agent trajectory and extracted \
success modes (from successful trajectories) and/or failure modes (from \
failed trajectories).

Your task is to **merge, deduplicate, and consolidate** these mode sets \
into a single unified mode set.

## Merging Guidelines

1. **Deduplicate**: If multiple modes describe the same or overlapping \
pattern, combine them into ONE stronger mode with the best description.
2. **Generalise**: When merging similar modes, raise the abstraction level \
to cover more scenarios. A single well-generalised mode is worth more than \
several narrow ones.
3. **Preserve type**: Keep success modes and failure modes separate. Do NOT \
convert between types.
4. **Preserve quality**: Keep modes with concrete, actionable descriptions. \
Drop vague or low-value modes.
5. **Merge evidence**: When combining modes, merge their evidence and \
source_trajectory_ids.
6. **Prioritise**: If there are too many modes, keep the most important and \
broadly applicable ones. Aim for at most {max_modes_per_type} modes per type \
(success/failure).

## Mode Quality Requirements

When merging, **actively raise the generality level**. Each merged mode must:

1. **Be transferable**: Apply to a broad class of tasks, not just the \
original ones.
2. **Be information-dense**: Concrete enough that someone can act on it.
3. **Be non-obvious**: Go beyond common sense.
4. **Be free of task-specific details**: No specific names, identifiers, \
error messages.

## Output Format

Output a **single JSON object** (no extra text before or after):

```json
{{
  "success_modes": [
    {{
      "type": "success",
      "pattern": "pattern-name",
      "description": "Merged description covering multiple trajectories."
    }}
  ],
  "failure_modes": [
    {{
      "type": "failure",
      "pattern": "pattern-name",
      "description": "Merged description covering multiple trajectories."
    }}
  ],
  "summary": "Brief summary of merge decisions."
}}
```\
"""


def build_intermediate_reduce_system_prompt(
    *,
    max_modes_per_type: int = 10,
) -> str:
    """Build the system prompt for an intermediate reduce step."""
    return _INTERMEDIATE_REDUCE_SYSTEM_TEMPLATE.format(
        max_modes_per_type=max_modes_per_type,
    )


def _format_mode_set(ms: ModeSet, label: str) -> str:
    """Format a ModeSet as readable markdown for prompt injection."""
    lines = [f"### {label}"]
    lines.append(
        f"Source trajectories: "
        f"{', '.join(ms.source_trajectory_ids) or 'unknown'}"
    )
    if ms.summary:
        lines.append(f"Summary: {ms.summary}")
    lines.append("")

    if ms.success_modes:
        lines.append(f"**Success Modes** ({len(ms.success_modes)}):")
        for i, m in enumerate(ms.success_modes, 1):
            lines.append(f"  {i}. **{m.pattern}**: {m.description}")
        lines.append("")

    if ms.failure_modes:
        lines.append(f"**Failure Modes** ({len(ms.failure_modes)}):")
        for i, m in enumerate(ms.failure_modes, 1):
            lines.append(f"  {i}. **{m.pattern}**: {m.description}")
        lines.append("")

    if not ms.success_modes and not ms.failure_modes:
        lines.append("_(no modes)_")

    return "\n".join(lines)


def build_intermediate_reduce_user_message(
    mode_sets: list[ModeSet],
    store_snapshot: dict,
) -> str:
    """Build the user message for an intermediate reduce step."""
    parts: list[str] = []

    # Store context
    parts.append(format_store_snapshot(store_snapshot))
    parts.append("")
    parts.append("---")
    parts.append("")

    # Mode sets
    parts.append(f"## Mode Sets to Merge ({len(mode_sets)} sets)")
    parts.append("")

    total_success = sum(len(ms.success_modes) for ms in mode_sets)
    total_failure = sum(len(ms.failure_modes) for ms in mode_sets)
    parts.append(
        f"Total: {total_success} success modes + {total_failure} failure modes"
    )
    parts.append("")

    for idx, ms in enumerate(mode_sets, 1):
        parts.append(_format_mode_set(ms, f"Set {idx}"))
        parts.append("")

    parts.append("---")
    parts.append("")
    parts.append(
        "**Now merge these mode sets into a single unified mode set.** "
        "Deduplicate, consolidate, and generalise. "
        "Keep success and failure modes separate. "
        "Output a JSON object."
    )

    return "\n".join(parts)


# =========================================================================
# 2. Final Reduce — tool-calling
# =========================================================================

_FINAL_REDUCE_SYSTEM_TEMPLATE = """\
You are a **Skill Synthesis Engine**. You receive a set of consolidated \
**success modes** and **failure modes** extracted from many agent \
trajectories, and must synthesise them into skills using the available tools.

## Context

The modes you receive represent distilled patterns:
- **Success modes**: Strategies, methodologies, and decision patterns that \
led to successful task completion across multiple trajectories.
- **Failure modes**: Error patterns, anti-patterns, and pitfalls that \
caused task failures across multiple trajectories.

Your job is to combine these into **cohesive, actionable skills** that \
an agent can use to improve its performance on similar tasks.

## Available Tools

| Tool | Purpose |
|------|---------|
| `list_skills` | See all skills (name, description). Does NOT show body or files. |
| `view_skill` | Read a skill's full body and see attached file names. |
| `read_skill_file` | Read the content of a specific attached file. |
| `add_skill` | Add a new skill to the store. |
| `update_skill` | Modify an existing skill. Only provide fields to change. |
| `delete_skill` | Remove a skill from the store. |
| `finish_extraction` | **Call this when done** processing the modes. |

## Synthesis Strategy

When creating skills from modes:

1. **Integrate success and failure modes**: A good skill should include \
both what TO DO (from success modes) and what to AVOID (from failure modes). \
This creates complete, actionable guidance.

2. **Organise thematically**: Group related success and failure modes into \
coherent skills around themes (e.g. "data validation", "error handling", \
"formula construction").

3. **Structure the body clearly**: Use the skill body to provide:
   - **Recommended approaches** (derived from success modes)
   - **Common pitfalls to avoid** (derived from failure modes)
   - **Decision criteria** for when to apply the skill
   - **Verification methods** to check correctness

4. **Maintain information density**: Every sentence should carry actionable \
information. No platitudes.

5. **Keep descriptions short**: The `description` field must be 1-2 \
sentences only. All detail goes in `body`.

## Skill Structure (Agent Skills Standard)

- **`name`**: Slug format (lowercase + hyphens, max 64 chars).
- **`description`**: 1-2 sentences: what class of problems and when to apply.
- **`body`**: Full Markdown procedural content with strategies, pitfalls, \
decision criteria, and verification methods.
- **`references`** (optional): Attached reference documents.
- **`scripts`** (optional): Attached executable scripts.

## Constraints

{constraints_block}
If you hit a character limit error, you MUST shorten the body significantly (remove examples, merge bullet points, cut less critical details) and retry `add_skill` until it succeeds.

## CRITICAL RULES

1. **You MUST submit skills via the `add_skill` tool.** Writing skill content as plain text in your response does NOT count. Only tool calls are recorded.
2. **After adding skills, you MUST call `finish_extraction`.** Do NOT simply stop responding.
3. **Never give up.** If `add_skill` returns an error, fix the issue and call `add_skill` again. Keep retrying until successful.\
"""


def build_final_reduce_system_prompt(
    *,
    max_skills: int = 10,
    max_skill_chars: int = 2000,
    max_total_chars: int = 0,
    meta_skill_guidance: str = "",
) -> str:
    """Build the system prompt for the final reduce (tool-calling) step."""
    lines = []
    lines.append(f"- **Maximum number of skills**: {max_skills}")
    if max_skill_chars > 0:
        lines.append(
            f"- **Maximum characters per skill** (description + body + files): "
            f"**{max_skill_chars:,} characters**. This is strictly enforced — "
            f"the tool will reject any skill exceeding this limit."
        )
    if max_total_chars > 0:
        lines.append(
            f"- **Total character budget** across all skills: "
            f"{max_total_chars:,}"
        )
    else:
        lines.append("- **Total character budget**: unlimited")

    constraints_block = "\n".join(lines)
    result = _FINAL_REDUCE_SYSTEM_TEMPLATE.format(
        constraints_block=constraints_block
    )

    if meta_skill_guidance:
        result += "\n\n## Extraction Quality Guidance\n\n" + meta_skill_guidance

    return result


def build_final_reduce_user_message(
    mode_sets: list[ModeSet],
) -> str:
    """Build the user message for the final reduce step.

    Parameters
    ----------
    mode_sets : list[ModeSet]
        The final merged mode sets (usually 1-G sets) to synthesise.
    """
    parts: list[str] = []
    parts.append("## Consolidated Modes for Skill Synthesis")
    parts.append("")

    # Flatten all modes
    all_success: list[Mode] = []
    all_failure: list[Mode] = []
    for ms in mode_sets:
        all_success.extend(ms.success_modes)
        all_failure.extend(ms.failure_modes)

    if not all_success and not all_failure:
        parts.append(
            "No modes to synthesise. Call `finish_extraction` immediately."
        )
        return "\n".join(parts)

    parts.append(
        f"You have **{len(all_success)} success modes** and "
        f"**{len(all_failure)} failure modes** to synthesise into skills."
    )
    parts.append("")

    if all_success:
        parts.append("### Success Modes")
        parts.append("")
        for i, m in enumerate(all_success, 1):
            parts.append(f"**{i}. {m.pattern}**")
            parts.append(f"  {m.description}")
            if m.evidence:
                parts.append(f"  _Evidence_: {m.evidence}")
            parts.append("")

    if all_failure:
        parts.append("### Failure Modes")
        parts.append("")
        for i, m in enumerate(all_failure, 1):
            parts.append(f"**{i}. {m.pattern}**")
            parts.append(f"  {m.description}")
            if m.evidence:
                parts.append(f"  _Evidence_: {m.evidence}")
            parts.append("")

    parts.append("---")
    parts.append("")
    parts.append(
        "**Start by calling `list_skills` to see the current store, then "
        "synthesise the modes above into cohesive skills. Integrate success "
        "modes (what to do) and failure modes (what to avoid) into complete, "
        "actionable skills. When done, call `finish_extraction`.**"
    )

    return "\n".join(parts)
