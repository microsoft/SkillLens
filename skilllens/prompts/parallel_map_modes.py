"""
Prompt templates for the parallel extraction method — Map phase.

Each map subagent analyses a single trajectory and produces a ModeSet (JSON)
containing success_modes or failure_modes depending on the trajectory outcome,
given a read-only snapshot of the current skill store.
"""

from __future__ import annotations

import json
import re

from skilllens.prompts.parallel_map import (
    _render_step,
    format_store_snapshot,
)
from skilllens.schema.trajectory import Trajectory


# ---------------------------------------------------------------------------
# Compact trajectory rendering for interactive-env trajectories (e.g. ALFWorld)
# ---------------------------------------------------------------------------

_ALFWORLD_HEADER_RE = re.compile(
    r"You are an expert agent operating in the ALFRED Embodied Environment"
)


def _is_interactive_env_trajectory(traj: Trajectory) -> bool:
    """Detect ALFWorld-style trajectories with redundant per-step prompts."""
    if len(traj.steps) < 4:
        return False
    # Check first user message for ALFWorld header
    for s in traj.steps[:2]:
        if s.role == "user" and s.content and _ALFWORLD_HEADER_RE.search(s.content):
            return True
    return False


def _render_compact_trajectory(traj: Trajectory) -> str:
    """Render an interactive-env trajectory in compact format.

    Instead of repeating the full prompt each step, output:
      - Task description (once)
      - Environment objects (once)
      - Sequence of (think, action, observation) tuples
    """
    parts: list[str] = []
    steps = traj.steps

    # Extract task description and initial observation from first user message
    first_user = None
    for s in steps:
        if s.role == "user" and s.content:
            first_user = s.content
            break

    if first_user:
        # Extract task
        task_match = re.search(r"Your task is to:\s*(.+?)(?:\n|$)", first_user)
        task_desc = task_match.group(1).strip() if task_match else "unknown"
        parts.append(f"**Task**: {task_desc}")
        parts.append("")

        # Extract initial observation (environment description)
        obs_match = re.search(
            r"(?:Your current observation is:|observation is:)\s*(.+?)(?:\nYour task|\nYour admissible)",
            first_user, re.DOTALL,
        )
        if obs_match:
            parts.append(f"**Initial observation**: {obs_match.group(1).strip()}")
            parts.append("")

    parts.append("### Steps")
    parts.append("")

    # Walk through user/agent pairs and extract (think, action, observation)
    step_num = 0
    i = 0
    while i < len(steps):
        s = steps[i]

        if s.role == "agent" and s.content:
            step_num += 1
            content = s.content

            # Extract think block
            think = ""
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if think_match:
                think = think_match.group(1).strip()

            # Extract action
            action = ""
            action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
            if action_match:
                action = action_match.group(1).strip()

            # Get observation from tool_call observation or next user message
            obs = ""
            if s.observation:
                obs = s.observation
            elif s.tool_calls:
                for tc in s.tool_calls:
                    if isinstance(tc, dict) and tc.get("observation"):
                        obs = tc["observation"]
                        break

            # Render compact step
            parts.append(f"**Step {step_num}**")
            if think:
                parts.append(f"  [think] {think}")
            if action:
                parts.append(f"  [action] {action}")
            if obs:
                parts.append(f"  [obs] {obs}")
            parts.append("")

        i += 1

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_MODE_MAP_SYSTEM_TEMPLATE = """\
You are a **Mode Extraction Sub-Agent** working in parallel with other \
sub-agents. Your task is to analyse a single agent trajectory and extract \
**{mode_type}** — high-level, reusable, transferable patterns.

## Your Role

You are ONE of several sub-agents processing different trajectories \
simultaneously. Your extracted modes will be **merged with those of other \
sub-agents** before being synthesised into skills. Therefore:

- **Focus on genuinely novel, reusable patterns** from THIS trajectory.
- Do NOT try to be exhaustive — capture only the most important patterns.
- Assume other sub-agents handle other trajectories.

## What is a Mode?

A **mode** is a high-level behavioral pattern that is:
1. **Transferable**: Applies to a broad class of tasks, not just this one.
2. **Actionable**: Concrete enough that an agent can follow it.
3. **Non-obvious**: Goes beyond common sense — captures insight from \
experience.
4. **Self-contained**: Understandable without the original trajectory context.

{mode_type_guidance}

## Mode Quality Requirements

Each mode must satisfy ALL of the following:

1. **High-level and domain-general**: A mode should capture a broad \
problem-solving pattern, strategy, or anti-pattern that applies to an \
entire CLASS of tasks — not a description of what happened in one \
particular problem instance. Abstract away from specific details.

2. **Maximally broad coverage**: Prefer modes that cover WIDER categories \
of situations. When you identify a useful pattern, ask whether it can be \
stated more generally.

3. **Information-dense**: Despite being general, each mode must contain \
CONCRETE descriptions of the pattern — what it looks like, when it \
applies, why it matters. Avoid platitudes.

4. **Free of task-specific details**: No specific file names, variable \
names, error messages, or API calls. Use general categories instead.

## Constraints

- Extract at most **{max_modes}** modes from this trajectory.
- Each mode should be concise: pattern name + description (2-4 sentences).
{constraints_block}

## Output Format

Output a **single JSON object** (no extra text before or after):

```json
{{
  "{mode_list_key}": [
    {{
      "type": "{mode_type_value}",
      "pattern": "short-descriptive-name",
      "description": "2-4 sentences describing the pattern, when it applies, and why it matters."
    }}
  ],
  "summary": "Brief summary of what you observed in this trajectory."
}}
```

If the trajectory contains no useful patterns, output:
```json
{{"{mode_list_key}": [], "summary": "No notable patterns found."}}
```\
"""

_SUCCESS_GUIDANCE = """\
### Success Modes

You are extracting **success modes** from a trajectory where the agent \
**successfully completed** the task. Focus on:

- **Effective strategies**: What high-level approaches led to success? \
(e.g. "validate intermediate results before proceeding", "decompose \
complex operations into atomic steps")
- **Good decision patterns**: What decision-making patterns were effective? \
(e.g. "check data shape before applying transformations", "use defensive \
error handling around I/O operations")
- **Methodological insights**: What general methodologies or workflows \
proved effective? (e.g. "iterative refinement with validation checkpoints")

Ask: "What did this agent do RIGHT that other agents facing similar tasks \
should also do?"\
"""

_FAILURE_GUIDANCE = """\
### Failure Modes

You are extracting **failure modes** from a trajectory where the agent \
**failed** the task. Focus on:

- **Error patterns**: What categories of mistakes did the agent make? \
(e.g. "assumed data format without verification", "applied operations \
in wrong order")
- **Anti-patterns**: What approaches should be avoided? \
(e.g. "modifying data in-place without backup", "ignoring edge cases \
in data types")
- **Pitfalls and traps**: What non-obvious traps did the agent fall into? \
(e.g. "formula syntax differs between programmatic and UI interfaces", \
"silent type coercion causing incorrect results")

Ask: "What should an agent AVOID doing when facing similar tasks?"\
"""


def build_mode_map_system_prompt(
    *,
    max_modes_per_trajectory: int = 3,
    max_skills: int = 10,
    max_skill_chars: int = 2000,
    max_total_chars: int = 0,
    is_success: bool = True,
    meta_skill_guidance: str = "",
) -> str:
    """Build the system prompt for a mode map sub-agent.

    Parameters
    ----------
    is_success : bool
        If True, build prompt for success mode extraction (resolved traj).
        If False, build prompt for failure mode extraction (unresolved traj).
    """
    if is_success:
        mode_type = "success modes"
        mode_type_guidance = _SUCCESS_GUIDANCE
        mode_list_key = "success_modes"
        mode_type_value = "success"
    else:
        mode_type = "failure modes"
        mode_type_guidance = _FAILURE_GUIDANCE
        mode_list_key = "failure_modes"
        mode_type_value = "failure"

    # Constraints block
    lines = []
    lines.append(f"- The final skill store allows at most **{max_skills}** skills.")
    if max_skill_chars > 0:
        lines.append(
            f"- Each skill can have at most **{max_skill_chars:,}** characters."
        )
    if max_total_chars > 0:
        lines.append(
            f"- Total character budget across all skills: **{max_total_chars:,}**."
        )
    constraints_block = "\n".join(lines)

    result = _MODE_MAP_SYSTEM_TEMPLATE.format(
        mode_type=mode_type,
        mode_type_guidance=mode_type_guidance,
        max_modes=max_modes_per_trajectory,
        mode_list_key=mode_list_key,
        mode_type_value=mode_type_value,
        constraints_block=constraints_block,
    )

    if meta_skill_guidance:
        result += "\n\n## Extraction Quality Guidance\n\n" + meta_skill_guidance

    return result


# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------

def build_mode_map_user_message(
    traj: Trajectory,
    store_snapshot: dict,
    *,
    include_feedback: bool = True,
) -> str:
    """Build the user message for a mode map sub-agent.

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

    # Part 1: Store snapshot (for context on what skills exist)
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

    if _is_interactive_env_trajectory(traj):
        # Compact rendering: deduplicate headers/history for ALFWorld-style trajs
        parts.append(_render_compact_trajectory(traj))
    else:
        # Default: render each step verbatim
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

    outcome_label = traj.outcome or "unknown"
    if outcome_label == "resolved":
        parts.append(
            "**[END OF TRAJECTORY — SUCCESSFUL]** Now analyse this trajectory "
            "and extract **success modes** — high-level strategies, "
            "methodologies, and decision patterns that contributed to success. "
            "Generalise aggressively. Output a JSON object."
        )
    else:
        parts.append(
            "**[END OF TRAJECTORY — FAILED]** Now analyse this trajectory "
            "and extract **failure modes** — error patterns, anti-patterns, "
            "and pitfalls that led to failure. "
            "Generalise aggressively. Output a JSON object."
        )

    return "\n".join(parts)
