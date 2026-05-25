"""
Skill-aware wrappers for mini-swe-agent's LitellmModel and DefaultAgent.

These subclasses add read-only skill tools (list_skills, view_skill) on top
of the standard bash tool.  When the LLM calls a skill tool, the result is
produced in-memory by the ReadOnlySkillProvider — it never touches the
Docker environment.

Usage:
    provider = ReadOnlySkillProvider.from_file("skill_set.json")
    model = SkillAwareLitellmModel(skill_provider=provider, model_name="azure/gpt-5-mini", ...)
    agent = SkillAwareAgent(model, env, skill_provider=provider, **agent_config)
    agent.run(task)

When skill_provider is None, both classes behave identically to their parents.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import litellm
from jinja2 import StrictUndefined, Template

from skilllens.benchmarks.swebench.minisweagent.agents.default import DefaultAgent
from skilllens.benchmarks.swebench.minisweagent.exceptions import FormatError
from skilllens.benchmarks.swebench.minisweagent.models.litellm_model import (
    BASH_TOOL,
    LitellmModel,
    LitellmModelConfig,
)

from skilllens.inference.skill_provider import (
    ReadOnlySkillProvider,
    SKILL_TOOL_NAMES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt section for skill library
# ---------------------------------------------------------------------------

SKILL_SECTION_TEMPLATE = """
## Skill Library

You have access to a **Skill Library** containing {skill_count} reusable procedural skills \
extracted from previous successful problem-solving experiences. These skills may help you \
solve the current task more effectively.

**Available skill tools:**
- `list_skills`: See all available skills (name, description). Use this first to find relevant skills.
- `view_skill(skill_name)`: Read the full body of a specific skill, including step-by-step \
procedures. Also shows names of attached reference and script files.
- `read_skill_file(skill_name, filename)`: Read the content of a specific attached file \
(reference document or script).

**How to use:**
1. At the start of your work, call `list_skills` to see what skills are available.
2. If any skill seems relevant to the current task, call `view_skill` to read the full body.
3. If the skill has attached files, use `read_skill_file` to read them as needed.
4. Use the skill's guidance as a reference — adapt it to the specific task at hand.
5. Skills are optional aids, not mandatory procedures. Use your own judgment.

**Note:** Skill tools are read-only and do not interact with the file system. \
You still need to use the `bash` tool for all file operations and command execution.
"""

SINGLE_SKILL_SECTION_TEMPLATE = """
## Skill Reference

Below is a reusable procedural skill extracted from previous successful problem-solving \
experiences. It may help you solve the current task more effectively. Use it as a \
reference — adapt it to the specific task at hand.

### {name}

{description}

{body}

**Note:** This skill is an optional aid, not a mandatory procedure. Use your own judgment.
"""


def build_skill_section(skill_count: int) -> str:
    """Return the skill library section to inject into the system prompt."""
    return SKILL_SECTION_TEMPLATE.format(skill_count=skill_count).strip()


def build_single_skill_section(skill: dict) -> str:
    """Return an inline skill section when there is only one skill."""
    return SINGLE_SKILL_SECTION_TEMPLATE.format(
        name=skill["name"],
        description=skill["description"],
        body=skill["body"],
    ).strip()


# ---------------------------------------------------------------------------
# SkillAwareLitellmModel
# ---------------------------------------------------------------------------

class SkillAwareLitellmModel(LitellmModel):
    """LitellmModel extended with read-only skill tools.

    Overrides:
    - ``_query()``:  adds skill tool schemas to the API call.
    - ``_parse_actions()``:  recognises skill tool calls alongside bash.
    """

    def __init__(
        self,
        skill_provider: ReadOnlySkillProvider | None = None,
        *,
        config_class: Callable = LitellmModelConfig,
        **kwargs,
    ):
        super().__init__(config_class=config_class, **kwargs)
        self.skill_provider = skill_provider

    # -- override: inject skill tool schemas into API call -----------------

    def _query(self, messages: list[dict[str, str]], **kwargs):
        tools = [BASH_TOOL]
        if self.skill_provider is not None:
            tools.extend(self.skill_provider.tool_schemas)
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=tools,
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    # -- override: parse both bash and skill tool calls --------------------

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls, recognising both bash and skill tools.

        Returns a list of action dicts.  Each has a ``"type"`` key:
        - ``"bash"``       — normal env action with ``command`` key.
        - ``"skill_tool"`` — already-resolved skill action with ``result`` key.
        """
        tool_calls = response.choices[0].message.tool_calls or []

        # Preserve the reasoning-content fallback from the existing patch
        if not tool_calls:
            tool_calls = self._extract_tool_calls_from_reasoning(response)

        if not tool_calls:
            raise FormatError(
                {
                    "role": "user",
                    "content": Template(
                        self.config.format_error_template,
                        undefined=StrictUndefined,
                    ).render(
                        error=(
                            "No tool calls found in the response. "
                            "Every response MUST include at least one tool call."
                        ),
                        actions=[],
                    ),
                    "extra": {"interrupt_type": "FormatError"},
                }
            )

        actions: list[dict] = []
        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                fn_args = {}

            if fn_name == "bash":
                # Standard bash action — same format as original
                if not isinstance(fn_args, dict) or "command" not in fn_args:
                    raise FormatError(
                        {
                            "role": "user",
                            "content": Template(
                                self.config.format_error_template,
                                undefined=StrictUndefined,
                            ).render(
                                error="Missing 'command' argument in bash tool call.",
                                actions=[],
                            ),
                            "extra": {"interrupt_type": "FormatError"},
                        }
                    )
                actions.append({
                    "type": "bash",
                    "command": fn_args["command"],
                    "tool_call_id": tool_call.id,
                })

            elif self.skill_provider is not None and self.skill_provider.handles(fn_name):
                # Skill tool — resolve immediately, carry result
                result = self.skill_provider.dispatch(fn_name, fn_args)
                actions.append({
                    "type": "skill_tool",
                    "function_name": fn_name,
                    "arguments": fn_args,
                    "tool_call_id": tool_call.id,
                    "result": result,
                })
                logger.info(
                    "Skill tool call: %s(%s) → %s",
                    fn_name,
                    json.dumps(fn_args, ensure_ascii=False)[:80],
                    result[:120],
                )

            else:
                raise FormatError(
                    {
                        "role": "user",
                        "content": Template(
                            self.config.format_error_template,
                            undefined=StrictUndefined,
                        ).render(
                            error=f"Unknown tool '{fn_name}'.",
                            actions=[],
                        ),
                        "extra": {"interrupt_type": "FormatError"},
                    }
                )

        return actions

    # -- serialization: include skill info ---------------------------------

    def serialize(self) -> dict:
        data = super().serialize()
        if self.skill_provider is not None:
            data["info"]["config"]["skill_provider"] = {
                "skill_count": self.skill_provider.skill_count,
                "call_history": self.skill_provider.call_history,
            }
        return data


# ---------------------------------------------------------------------------
# SkillAwareAgent
# ---------------------------------------------------------------------------

class SkillAwareAgent(DefaultAgent):
    """DefaultAgent with skill-tool interception in execute_actions().

    Skill tool calls (type="skill_tool") are resolved in-memory and never
    sent to the Docker environment.  Bash actions go through the standard
    env.execute() path.  The observation messages for both types are
    properly ordered and added to the conversation.
    """

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions: skill tools in-memory, bash via env.execute().

        Iterates through the actions list in original order, preserving the
        correspondence between tool_call and tool_result that the LLM API
        requires.
        """
        actions = message.get("extra", {}).get("actions", [])
        outputs: list[dict] = []

        for action in actions:
            if action.get("type") == "skill_tool":
                # Skill tool — already resolved, wrap as env-compatible output
                outputs.append({
                    "output": action["result"],
                    "returncode": 0,
                    "exception_info": "",
                })
            else:
                # Bash action — execute in Docker environment
                outputs.append(self.env.execute(action))

        return self.add_messages(
            *self.model.format_observation_messages(
                message, outputs, self.get_template_vars()
            )
        )
