"""
ALFWorld prompt templates and skill-augmented prompt builder.

Prompt templates are copied verbatim from the original eval script
(alfworld/eval_alfworld_api_noray.py) to guarantee baseline equivalence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skilllens.inference.skill_provider import ReadOnlySkillProvider

# ---------------------------------------------------------------------------
# Prompt templates — EXACT copies from eval_alfworld_api_noray.py (lines 96-117)
# DO NOT modify these without also updating the original file.
# ---------------------------------------------------------------------------

ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

HISTORY_LENGTH = 2

# ---------------------------------------------------------------------------
# The "Now it's your turn" block — used to split prompt for skill injection
# ---------------------------------------------------------------------------

_ACTION_INSTRUCTION_BLOCK = """Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""


# ---------------------------------------------------------------------------
# build_prompt — EXACT copy from eval_alfworld_api_noray.py (lines 190-223)
# ---------------------------------------------------------------------------

def build_prompt(
    text_obs: str,
    admissible_commands: list[str],
    task_desc: str,
    history_records: list[dict],
    init: bool = False,
) -> str:
    """Build prompt for a single environment given its history records.

    This is a verbatim copy of the original build_prompt().
    """
    reformatted_actions = "\n ".join(
        f"'{s}'" for s in admissible_commands if s != "help"
    )
    step_count = len(history_records)

    if init or step_count == 0:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=text_obs,
            admissible_actions=reformatted_actions,
        )
    else:
        recent = history_records[-HISTORY_LENGTH:]
        valid_len = len(recent)
        start_idx = step_count - valid_len
        lines = []
        for j, rec in enumerate(recent):
            step_num = start_idx + j + 1
            lines.append(
                f"[Observation {step_num}: '{rec['text_obs']}', "
                f"Action {step_num}: '{rec['action']}']"
            )
        action_history = "\n".join(lines)

        return ALFWORLD_TEMPLATE.format(
            task_description=task_desc,
            step_count=step_count,
            history_length=valid_len,
            action_history=action_history,
            current_step=step_count + 1,
            current_observation=text_obs,
            admissible_actions=reformatted_actions,
        )


# ---------------------------------------------------------------------------
# build_prompt_with_skill — skill-augmented version
# ---------------------------------------------------------------------------

def build_prompt_with_skill(
    text_obs: str,
    admissible_commands: list[str],
    task_desc: str,
    history_records: list[dict],
    init: bool = False,
    skill_provider: ReadOnlySkillProvider | None = None,
) -> tuple[str, bool]:
    """Build prompt with optional skill injection.

    Returns
    -------
    (prompt, has_tool_protocol)
        prompt: the final prompt string
        has_tool_protocol: True if the multi-skill tool protocol is active
            (caller should handle ```skill``` blocks in model responses)
    """
    base_prompt = build_prompt(text_obs, admissible_commands, task_desc,
                               history_records, init=init)

    if skill_provider is None or skill_provider.skill_count == 0:
        return base_prompt, False

    # Import skill tools (benchmark-agnostic functions)
    from skilllens.inference.spreadsheetbench.skill_tools import (
        build_single_skill_prompt_section,
        build_skill_prompt_section,
    )

    if skill_provider.is_single_skill:
        # Single skill: inline content before "Now it's your turn" block
        skill_section = build_single_skill_prompt_section(
            skill_provider.get_single_skill()
        )
        # Split at the action instruction block and insert skill section
        if _ACTION_INSTRUCTION_BLOCK in base_prompt:
            before, after = base_prompt.split(_ACTION_INSTRUCTION_BLOCK, 1)
            prompt = (
                before.rstrip()
                + "\n\n"
                + skill_section
                + "\n\n"
                + _ACTION_INSTRUCTION_BLOCK
                + after
            )
        else:
            # Fallback: just append
            prompt = base_prompt.rstrip() + "\n\n" + skill_section
        return prompt, False
    else:
        # Multiple skills: insert tool protocol before "Now it's your turn"
        skill_section = build_skill_prompt_section(skill_provider.skill_count)
        if _ACTION_INSTRUCTION_BLOCK in base_prompt:
            before, after = base_prompt.split(_ACTION_INSTRUCTION_BLOCK, 1)
            prompt = (
                before.rstrip()
                + "\n\n"
                + skill_section
                + "\n\n"
                + _ACTION_INSTRUCTION_BLOCK
                + after
            )
        else:
            prompt = base_prompt.rstrip() + "\n\n" + skill_section
        return prompt, True
