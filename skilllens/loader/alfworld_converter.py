"""
ALFWorld trajectory converter — transforms ALFWorld trajectory JSON
into the unified Trajectory schema.

Input format (one JSON file per environment per round):
    {
      "env_id": 0,
      "test_round": 0,
      "model": "gpt-5.4",
      "dataset": "eval_in_distribution",
      "task_type": "look_at_obj_in_light",
      "gamefile": "/path/to/game.tw-pddl",
      "initial_observation": "You are in the middle of a room...",
      "steps": [
        {
          "step": 0,
          "prompt": "...",           # full prompt with observation + admissible actions
          "model_response": "...",   # full response with <think> and <action> tags
          "reasoning": "...",        # extracted reasoning text
          "action": "go to desk 1", # extracted action string
          "env_feedback": "You arrive at desk 1. On the desk 1, you see ...",
          "reward": 0.0,
          "done": false,
          "usage": {
            "prompt_tokens": 373,
            "completion_tokens": 207,
            "total_tokens": 580,
            "reasoning_tokens": 151
          }
        },
        ...
      ],
      "success": true,
      "total_steps": 17
    }
"""

from __future__ import annotations

import logging

from skilllens.schema.trajectory import Step, Trajectory

logger = logging.getLogger(__name__)


def convert_alfworld_trajectory(
    data: dict,
    *,
    source_path: str = "",
    round_num: int | None = None,
) -> Trajectory:
    """Convert an ALFWorld trajectory entry to unified Trajectory.

    Parameters
    ----------
    data : dict
        Parsed JSON from a single ``env_XXX.json`` file.
    source_path : str
        Original file path for provenance tracking.
    round_num : int | None
        Override the round number. If None, uses ``data["test_round"]``.

    Returns
    -------
    Trajectory
        Unified trajectory object.
    """
    env_id = data.get("env_id", 0)
    env_id_str = f"env_{env_id:03d}"
    model = data.get("model", "")
    task_type = data.get("task_type", "")
    dataset = data.get("dataset", "")
    gamefile = data.get("gamefile", "")
    initial_observation = data.get("initial_observation", "")
    total_steps = data.get("total_steps", 0)
    success = data.get("success", False)
    raw_steps = data.get("steps", [])

    if round_num is None:
        round_num = data.get("test_round", 0)

    # --- Build Steps ---
    steps: list[Step] = []

    for s in raw_steps:
        prompt = s.get("prompt", "") or ""
        model_response = s.get("model_response", "") or ""
        reasoning = s.get("reasoning", "") or ""
        action = s.get("action", "") or ""
        env_feedback = s.get("env_feedback", "") or ""
        reward = s.get("reward", 0.0)
        done = s.get("done", False)
        usage = s.get("usage", {})
        step_num = s.get("step", 0)

        # 1. User step: the prompt containing observation + admissible actions
        steps.append(Step(
            role="user",
            content=prompt,
        ))

        # 2. Agent step: model response + action as tool_call + env feedback as observation
        tool_calls = None
        if action:
            tool_calls = [{
                "function_name": "act",
                "arguments": action,
            }]

        step_metadata: dict = {
            "step": step_num,
            "reward": reward,
            "done": done,
        }
        if reasoning:
            step_metadata["reasoning"] = reasoning
        if usage:
            step_metadata["usage"] = usage

        steps.append(Step(
            role="agent",
            content=model_response,
            tool_calls=tool_calls,
            observation=env_feedback,
            metadata=step_metadata,
        ))

    # --- Determine outcome ---
    outcome = "resolved" if success else "unresolved"
    reward_val = 1.0 if success else 0.0

    # --- Final answer ---
    # The last action taken by the agent
    final_answer = ""
    if raw_steps:
        final_answer = raw_steps[-1].get("action", "") or ""

    # --- Build Trajectory ---
    traj_id = f"{env_id_str}_r{round_num}"

    return Trajectory(
        id=traj_id,
        task_name=task_type,
        agent=model,
        steps=steps,
        final_answer=final_answer,
        reward=reward_val,
        benchmark="alfworld",
        outcome=outcome,
        source_format="alfworld-react",
        task_id=env_id_str,
        metadata={
            "source_path": source_path,
            "task_type": task_type,
            "gamefile": gamefile,
            "dataset": dataset,
            "initial_observation": initial_observation,
            "total_steps": total_steps,
            "round": round_num,
            "env_id": env_id,
        },
    )
