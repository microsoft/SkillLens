"""
Text-mode skill tools for SpreadsheetBench.

SpreadsheetBench uses plain text conversation (not OpenAI function calling),
so we define a text-based protocol for skill tool invocation:

    ```skill
    list_skills
    ```

    ```skill
    view_skill <skill_name>
    ```

    ```skill
    read_skill_file <skill_name> <filename>
    ```

This module provides:
- extract_skill_call(): parse a ```skill``` block from LLM response
- build_skill_prompt_section(): generate the skill section to append to prompts
- build_single_skill_prompt_section(): inline a single skill into the prompt
"""

from __future__ import annotations

import re
from typing import Optional


# Regex to match ```skill ... ``` blocks (similar to ```python ... ```)
_SKILL_BLOCK_RE = re.compile(
    r"```skill\s*\n(.*?)```",
    re.DOTALL,
)


def extract_skill_call(response: str) -> Optional[tuple[str, dict]]:
    """Parse a ```skill``` block from the LLM response.

    Returns:
        (function_name, arguments_dict) if a skill block is found, else None.

    Examples:
        "```skill\nlist_skills\n```"  -> ("list_skills", {})
        "```skill\nview_skill s-01\n```"  -> ("view_skill", {"skill_name": "s-01"})
        "```skill\nread_skill_file s-01 REF.md\n```"
            -> ("read_skill_file", {"skill_name": "s-01", "filename": "REF.md"})
    """
    match = _SKILL_BLOCK_RE.search(response)
    if not match:
        return None

    content = match.group(1).strip()
    if not content:
        return None

    parts = content.split(None)  # split on whitespace
    function_name = parts[0]

    if function_name == "list_skills":
        return ("list_skills", {})
    elif function_name == "view_skill":
        if len(parts) < 2:
            return ("view_skill", {"skill_name": ""})
        return ("view_skill", {"skill_name": parts[1].strip()})
    elif function_name == "read_skill_file":
        if len(parts) < 3:
            # Partial args — return what we have
            skill_name = parts[1].strip() if len(parts) > 1 else ""
            return ("read_skill_file", {"skill_name": skill_name, "filename": ""})
        return ("read_skill_file", {
            "skill_name": parts[1].strip(),
            "filename": " ".join(parts[2:]).strip(),  # filename may contain spaces
        })
    else:
        return None


def has_skill_call(response: str) -> bool:
    """Check if the response contains a ```skill``` block."""
    return _SKILL_BLOCK_RE.search(response) is not None


def build_skill_prompt_section(skill_count: int) -> str:
    """Generate the skill library section to append to the prompt.

    This tells the model about available skill tools and how to use them.
    Used when there are **multiple** skills (skill_count > 1).
    """
    return f"""
## Skill Library

You have access to a **Skill Library** containing {skill_count} reusable procedural skills \
extracted from previous successful problem-solving experiences. These skills may help you \
solve the current task more effectively.

**Available skill tools:**
- `list_skills`: See all available skills (name, description). Use this first to find relevant skills.
- `view_skill <skill_name>`: Read the full body of a specific skill, including step-by-step \
procedures. Also shows names of attached reference and script files.
- `read_skill_file <skill_name> <filename>`: Read the content of a specific attached file \
(reference document or script).

**How to use skill tools:**
To call a skill tool, use a ```skill``` code block (NOT a ```python``` block):

```skill
list_skills
```

```skill
view_skill <skill_name>
```

```skill
read_skill_file <skill_name> <filename>
```

**Workflow:**
1. At the start of your work, call `list_skills` to see what skills are available.
2. If any skill seems relevant to the current task, call `view_skill` to read the full body.
3. If the skill has attached reference or script files, use `read_skill_file` to read them.
4. Use the skill's guidance as a reference — adapt it to the specific task at hand.
5. After consulting skills, proceed with ```python``` code blocks to solve the task.

**Important notes:**
- Skill tools are read-only and do not execute any code or modify files.
- Each response should contain EITHER a ```skill``` block OR a ```python``` block, not both.
- Skills are optional aids, not mandatory procedures. Use your own judgment.
""".strip()


def build_single_skill_prompt_section(skill: dict) -> str:
    """Generate a prompt section that inlines a single skill directly.

    When there is only one skill, we skip the list_skills / view_skill tool
    protocol and embed the skill content directly into the system prompt
    for simpler and more efficient consumption by the model.

    Args:
        skill: dict with keys ``name``, ``description``, ``body``,
               and optional ``references``, ``scripts``.
    """
    parts = [f"""
## Skill Reference

Below is a reusable procedural skill extracted from previous successful problem-solving \
experiences. It may help you solve the current task more effectively. Use it as a \
reference — adapt it to the specific task at hand.

### {skill['name']}

{skill['description']}

{skill['body']}"""]

    # Inline references if any
    references = skill.get("references", [])
    if references:
        parts.append("\n#### Reference Files\n")
        for ref in references:
            fn = ref.get("filename", ref) if isinstance(ref, dict) else ref
            content = ref.get("content", "") if isinstance(ref, dict) else ""
            if content:
                parts.append(f"**{fn}**:\n{content}\n")

    # Inline scripts if any
    scripts = skill.get("scripts", [])
    if scripts:
        parts.append("\n#### Script Files\n")
        for scr in scripts:
            fn = scr.get("filename", scr) if isinstance(scr, dict) else scr
            content = scr.get("content", "") if isinstance(scr, dict) else ""
            if content:
                parts.append(f"**{fn}**:\n```\n{content}\n```\n")

    parts.append(
        "\n**Note:** This skill is an optional aid, not a mandatory procedure. "
        "Use your own judgment."
    )

    return "\n".join(parts).strip()
