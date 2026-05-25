"""
SWE-bench trajectory converter — transforms `mini-swe-agent-1.1` format
trajectory files into the unified Trajectory schema.

Input format:
    {
      "info": {"model_stats": {...}, "exit_status": "Submitted", ...},
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "THOUGHT: ...", "tool_calls": [...]},
        {"role": "tool", "content": "<returncode>0</returncode><output>...</output>"},
        ...
        {"role": "exit", "content": "..."}
      ],
      "trajectory_format": "mini-swe-agent-1.1",
      "instance_id": "django__django-10914"
    }
"""

from __future__ import annotations

import logging
import uuid

from skilllens.schema.trajectory import Step, Trajectory

logger = logging.getLogger(__name__)

# Mapping from SWE-bench message roles to our unified role enum
_SWE_ROLE_MAP: dict[str, str] = {
    "system": "system",
    "user": "user",
    "assistant": "agent",
    "tool": "tool",
}


def _extract_task_name(instance_id: str) -> str:
    """Derive a human-readable task name from an instance_id.

    Example: "django__django-10914" → "django-10914"
    """
    if not instance_id:
        return ""
    # instance_id format: "repo__repo-issue_number"
    # e.g. "django__django-10914" → split on "__" → ["django", "django-10914"]
    parts = instance_id.split("__")
    if len(parts) >= 2:
        return parts[-1]  # "django-10914"
    return instance_id


def convert_swe_bench_trajectory(
    data: dict,
    *,
    source_path: str = "",
    outcome: str = "",
) -> Trajectory:
    """Convert a SWE-bench (mini-swe-agent-1.1) trajectory dict to unified Trajectory.

    Parameters
    ----------
    data : dict
        Parsed JSON content of a ``.traj.json`` file.
    source_path : str
        Original file path for provenance tracking.
    outcome : str
        "resolved" / "unresolved" — typically obtained from evaluation results.

    Returns
    -------
    Trajectory
        Unified trajectory object.
    """
    instance_id = data.get("instance_id", "")
    traj_format = data.get("trajectory_format", "mini-swe-agent-1.1")
    info = data.get("info", {})
    messages = data.get("messages", [])

    # Extract model name from config
    model_name = ""
    config = info.get("config", {})
    if isinstance(config, dict):
        model_cfg = config.get("model", {})
        if isinstance(model_cfg, dict):
            model_name = model_cfg.get("model_name", "")

    exit_status = info.get("exit_status", "")

    # Convert messages to Steps, merging tool responses into previous agent step
    steps: list[Step] = []
    final_answer = ""
    for msg in messages:
        role_raw = msg.get("role", "")
        content = msg.get("content") or ""  # Handle None content
        tool_calls_raw = msg.get("tool_calls")

        # exit message → extract as final_answer, skip as step
        if role_raw == "exit":
            final_answer = content
            continue

        role = _SWE_ROLE_MAP.get(role_raw, "agent")

        # Normalize tool_calls format
        tool_calls = None
        if tool_calls_raw:
            tool_calls = []
            for tc in tool_calls_raw:
                # SWE-bench format: {"function": {"arguments": "...", "name": "..."}, "id": "...", "type": "function"}
                func = tc.get("function", {})
                tool_calls.append({
                    "function_name": func.get("name", ""),
                    "arguments": func.get("arguments", ""),
                    "call_id": tc.get("id", ""),
                })

        if role_raw == "tool" and steps:
            # Merge tool response into the most recent agent step's observation
            last_step = steps[-1]
            if last_step.role == "agent":
                existing_obs = last_step.observation or ""
                separator = "\n---\n" if existing_obs else ""
                last_step.observation = existing_obs + separator + content
                continue  # Don't create a separate step

        step = Step(
            role=role,
            content=content,
            tool_calls=tool_calls,
            metadata={"original_role": role_raw} if role_raw != role else {},
        )
        steps.append(step)

    # Build trajectory
    task_name = _extract_task_name(instance_id)
    traj_id = instance_id or str(uuid.uuid4())

    # Determine reward from outcome: resolved → 1.0, anything else → 0.0
    reward = 1.0 if outcome == "resolved" else 0.0

    return Trajectory(
        id=traj_id,
        task_name=task_name,
        agent=model_name,
        steps=steps,
        final_answer=final_answer,
        reward=reward,
        benchmark="swebench",
        outcome=outcome,
        source_format=traj_format,
        task_id=instance_id,
        metadata={
            "source_path": source_path,
            "exit_status": exit_status,
            "model_stats": info.get("model_stats", {}),
        },
    )
