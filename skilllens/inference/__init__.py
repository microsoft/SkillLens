"""
Skill-augmented inference module.

Provides read-only skill tools (list_skills, view_skill) that can be
injected into agent loops, allowing the target model to consult a skill
library during problem solving.

Benchmark-specific adapters live in sub-packages:
  - swebench/     — mini-swe-agent integration
  - spreadsheetbench/ — SpreadsheetBench text-mode integration
"""

from skilllens.inference.skill_provider import ReadOnlySkillProvider

__all__ = [
    "ReadOnlySkillProvider",
]
