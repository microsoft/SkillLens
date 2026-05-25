"""SWE-bench skill-augmented inference.

Requires litellm and minisweagent — only import in SWE-bench environments.
"""

__all__ = [
    "SkillAwareAgent",
    "SkillAwareLitellmModel",
    "build_skill_section",
    "build_single_skill_section",
]


def __getattr__(name):
    if name in __all__:
        from skilllens.inference.swebench.skill_agent import (
            SkillAwareAgent,
            SkillAwareLitellmModel,
            build_skill_section,
            build_single_skill_section,
        )
        _exports = {
            "SkillAwareAgent": SkillAwareAgent,
            "SkillAwareLitellmModel": SkillAwareLitellmModel,
            "build_skill_section": build_skill_section,
            "build_single_skill_section": build_single_skill_section,
        }
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
