"""SpreadsheetBench skill-augmented inference.

Text-mode integration: uses ```skill``` code blocks (not function calling)
for skill tool invocation alongside ```python``` code blocks for execution.
"""

__all__ = [
    "extract_skill_call",
    "has_skill_call",
    "build_skill_prompt_section",
]

from skilllens.inference.spreadsheetbench.skill_tools import (
    extract_skill_call,
    has_skill_call,
    build_skill_prompt_section,
)
