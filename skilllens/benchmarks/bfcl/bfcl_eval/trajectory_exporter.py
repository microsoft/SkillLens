"""
Convert BFCL result dicts into the Trajectory schema used by skill_distill_bench.

Works with both single-turn and multi-turn results, FC and prompting modes.
The conversion relies on the ``inference_log`` that BFCL's BaseHandler already
produces — no monkey-patching needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize(obj: Any) -> str:
    """Best-effort JSON serialization, falling back to str()."""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def _build_tools_system_content(functions: list[dict]) -> str:
    """Build a system-prompt-like string from BFCL function definitions.

    FC models receive tool schemas via the API ``tools`` parameter (not in messages),
    so we reconstruct a readable representation for the trajectory.
    """
    lines = [f"You have access to {len(functions)} tools:\n"]
    for func in functions:
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        lines.append(f"- {name}: {desc}")
        if params.get("properties"):
            param_names = list(params["properties"].keys())
            required = params.get("required", [])
            param_parts = []
            for p in param_names:
                pinfo = params["properties"][p]
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                marker = " (required)" if p in required else ""
                param_parts.append(f"    {p}: {ptype}{marker} — {pdesc}")
            lines.append("  Parameters:")
            lines.extend(param_parts)
    return "\n".join(lines)


def _extract_tool_calls_from_result(model_responses) -> list[dict] | None:
    """
    FC model responses are lists like [{"func_name": "args_json"}, ...].
    Convert to a normalized tool_calls list.
    """
    if not isinstance(model_responses, list):
        return None
    tool_calls = []
    for item in model_responses:
        if isinstance(item, dict):
            for name, args in item.items():
                tool_calls.append({
                    "function": {"name": name, "arguments": args},
                })
    return tool_calls if tool_calls else None


def _is_multi_turn(result_dict: dict) -> bool:
    """Check if this is a multi-turn result by looking at the id or inference_log structure."""
    entry_id = result_dict.get("id", "")
    return any(
        prefix in entry_id
        for prefix in ("multi_turn", "web_search", "memory_")
    )


# ---------------------------------------------------------------------------
# Single-turn conversion
# ---------------------------------------------------------------------------

def _convert_single_turn(result_dict: dict, model_name: str, test_entry: dict | None) -> dict:
    """Convert a single-turn BFCL result into Trajectory format."""
    entry_id = result_dict["id"]
    category = entry_id.rsplit("_", 1)[0]
    result = result_dict.get("result", "")

    steps = []

    # 0. System step — tool/function definitions (FC models pass these via API tools param)
    if test_entry and test_entry.get("function"):
        steps.append({
            "role": "system",
            "content": _build_tools_system_content(test_entry["function"]),
            "metadata": {"type": "tool_definitions"},
        })

    # 1. User step — from test_entry if available, otherwise from result id
    if test_entry and "question" in test_entry:
        user_content = _serialize(test_entry["question"])
    else:
        user_content = f"[Task: {entry_id}]"

    steps.append({
        "role": "user",
        "content": user_content,
        "metadata": {
            "type": "task_input",
            "functions_available": len(test_entry.get("function", [])) if test_entry else 0,
        },
    })

    # 2. Agent step — the model's response
    if isinstance(result, str):
        agent_content = result
        tool_calls = None
    elif isinstance(result, list):
        # FC model: result is list of {func_name: args}
        tool_calls = _extract_tool_calls_from_result(result)
        agent_content = ""
    else:
        agent_content = _serialize(result)
        tool_calls = None

    steps.append({
        "role": "agent",
        "content": agent_content,
        "timestamp": None,
        "tool_calls": tool_calls,
        "observation": None,
        "metadata": {
            "input_token": result_dict.get("input_token_count", 0),
            "output_token": result_dict.get("output_token_count", 0),
            "latency": result_dict.get("latency", 0),
        },
    })

    # Final answer
    if tool_calls:
        final_answer = _serialize(tool_calls)
    else:
        final_answer = agent_content

    return _build_trajectory(
        entry_id=entry_id,
        category=category,
        model_name=model_name,
        steps=steps,
        final_answer=final_answer,
        result_dict=result_dict,
    )


# ---------------------------------------------------------------------------
# Multi-turn conversion
# ---------------------------------------------------------------------------

def _convert_multi_turn(result_dict: dict, model_name: str, test_entry: dict | None) -> dict:
    """Convert a multi-turn BFCL result into Trajectory format."""
    entry_id = result_dict["id"]
    category = entry_id.rsplit("_", 1)[0]
    inference_log = result_dict.get("inference_log", [])
    result = result_dict.get("result", [])

    steps = []

    # 0. System step — tool/function definitions
    if test_entry and test_entry.get("function"):
        steps.append({
            "role": "system",
            "content": _build_tools_system_content(test_entry["function"]),
            "metadata": {"type": "tool_definitions"},
        })

    # Walk through inference_log entries
    # Structure: list of items, each is either:
    #   - a list of state_info dicts
    #   - a dict with "begin_of_turn_query" + "step_0", "step_1", ...
    for log_entry in inference_log:
        # State info (list of dicts with role=state_info) — skip
        if isinstance(log_entry, list):
            continue

        if not isinstance(log_entry, dict):
            continue

        # Turn log
        turn_query = log_entry.get("begin_of_turn_query", [])

        # Add user message for this turn
        if turn_query:
            user_content = ""
            for msg in turn_query:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_content = msg.get("content", "")
                    break
            if not user_content:
                user_content = _serialize(turn_query)

            steps.append({
                "role": "user",
                "content": user_content,
                "metadata": {"type": "turn_query"},
            })

        # Walk through steps in this turn
        step_idx = 0
        while f"step_{step_idx}" in log_entry:
            step_entries = log_entry[f"step_{step_idx}"]
            if not isinstance(step_entries, list):
                step_idx += 1
                continue

            for entry in step_entries:
                if not isinstance(entry, dict):
                    continue
                role = entry.get("role", "")

                if role == "assistant":
                    content = entry.get("content", "")
                    # Try to extract tool_calls from the content
                    tool_calls = None
                    if isinstance(content, list):
                        tool_calls = _extract_tool_calls_from_result(content)
                        content_str = "" if tool_calls else _serialize(content)
                    elif isinstance(content, str):
                        content_str = content
                    else:
                        content_str = _serialize(content)

                    step = {
                        "role": "agent",
                        "content": content_str,
                        "timestamp": None,
                        "tool_calls": tool_calls,
                        "observation": None,
                        "metadata": {},
                    }
                    if entry.get("reasoning_content"):
                        step["metadata"]["reasoning_content"] = entry["reasoning_content"]
                    steps.append(step)

                elif role == "tool":
                    steps.append({
                        "role": "tool",
                        "content": str(entry.get("content", "")),
                        "metadata": {},
                    })

                # Skip handler_log, inference_input, etc.

            step_idx += 1

    # Determine final answer from the last result
    final_answer = ""
    if result:
        # result is list of turns, each turn is list of step responses
        last_turn = result[-1] if isinstance(result, list) else result
        if isinstance(last_turn, list) and last_turn:
            last_step = last_turn[-1]
            if isinstance(last_step, str):
                final_answer = last_step
            elif isinstance(last_step, list):
                tc = _extract_tool_calls_from_result(last_step)
                final_answer = _serialize(tc) if tc else _serialize(last_step)
            else:
                final_answer = _serialize(last_step)
        elif isinstance(last_turn, str):
            final_answer = last_turn

    return _build_trajectory(
        entry_id=entry_id,
        category=category,
        model_name=model_name,
        steps=steps,
        final_answer=final_answer,
        result_dict=result_dict,
    )


# ---------------------------------------------------------------------------
# Common builder
# ---------------------------------------------------------------------------

def _build_trajectory(
    entry_id: str,
    category: str,
    model_name: str,
    steps: list[dict],
    final_answer: str,
    result_dict: dict,
) -> dict:
    """Assemble the final Trajectory dict."""
    # Compute total tokens
    input_tokens = result_dict.get("input_token_count", 0)
    output_tokens = result_dict.get("output_token_count", 0)

    # For multi-turn, these are nested lists — flatten and sum
    def _sum_nested(val):
        if isinstance(val, (int, float)):
            return val
        if isinstance(val, list):
            return sum(_sum_nested(v) for v in val)
        return 0

    total_input = _sum_nested(input_tokens)
    total_output = _sum_nested(output_tokens)
    total_cached = _sum_nested(result_dict.get("cached_token_count", 0))

    return {
        "id": entry_id,
        "task_name": f"bfcl_v4/{category}/{entry_id}",
        "agent": model_name,
        "steps": steps,
        "final_answer": final_answer,
        "reward": 0.0,
        "benchmark": "bfcl_v4",
        "outcome": "",
        "source_format": "bfcl-inference-log",
        "task_id": entry_id,
        "metadata": {
            "category": category,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cached_tokens": total_cached,
            "total_tokens": total_input + total_output,
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bfcl_result_to_trajectory(
    result_dict: dict,
    model_name: str,
    test_entry: dict | None = None,
) -> dict:
    """
    Convert a single BFCL result dict into a Trajectory-schema dict.

    Args:
        result_dict: The dict written by BFCL generation (id, result, inference_log, ...).
        model_name: The model registry name.
        test_entry: The original test entry (optional, for user question context).

    Returns:
        A dict conforming to the Trajectory schema.
    """
    if _is_multi_turn(result_dict):
        return _convert_multi_turn(result_dict, model_name, test_entry)
    else:
        return _convert_single_turn(result_dict, model_name, test_entry)


def write_trajectory(trajectory: dict, output_dir: Path):
    """Write a single trajectory dict to output_dir/{id}.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{trajectory['id']}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False, default=str)
