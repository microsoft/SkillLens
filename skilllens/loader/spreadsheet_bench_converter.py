"""
SpreadsheetBench trajectory converter — transforms the SpreadsheetBench
``multi_react_exec`` format trajectory entries into the unified Trajectory schema.

Input format (one JSON object per line in ``trajectory.jsonl``):
    {
      "id": "472-15",
      "instruction": "...",
      "instruction_type": "Sheet-Level Manipulation" | "Cell-Level Manipulation",
      "answer_position": "B2",
      "spreadsheet_path": "spreadsheet/472-15",
      "model": "gpt-5.4",
      "setting": "react_exec",
      "max_turn_num": 5,
      "total_turns": 2,
      "output_generated": true,
      "solution": "...",               # final code (if successful)
      "total_time_sec": 28.98,
      "trajectory": [                  # structured turn-by-turn data
        {
          "turn": 1,
          "llm_response": "...",       # full assistant response (may contain code + text)
          "code_extracted": true,
          "code": "...",               # extracted Python code
          "exec_result": "...",        # stdout / traceback from execution
          "exec_success": true,
          "output_generated": false
        },
        ...
      ],
      "conversation": [               # flat string list: [system, assistant, exec_result, ...]
        "You are a spreadsheet expert ...",
        "```python\\n...\\n```\\n...",
        "Sheet title: Sheet1\\n...",
        ...
      ]
    }

Eval result format (``eval_result.json`` — a JSON array):
    [
      {
        "id": "472-15",
        "instruction_type": "Sheet-Level Manipulation",
        "test_case_results": [1],      # list of 0/1 per test case
        "soft_restriction": 1.0,
        "hard_restriction": 1          # 1 = pass, 0 = fail
      },
      ...
    ]
"""

from __future__ import annotations

import json
import logging
import re
import uuid

from skilllens.schema.trajectory import Step, Trajectory

logger = logging.getLogger(__name__)

# Regex to extract ```python ... ``` blocks from LLM responses
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

# Minimal heuristic patterns that indicate a string is likely Python code
_CODE_INDICATORS = (
    "import ", "from ", "def ", "class ", "print(", "for ", "if ",
    "while ", "with ", "open(", ".load", "pd.", "os.", "return",
    ".read", "openpyxl", "pandas", "xlsxwriter", "= ", "()",
)


def _extract_code(code_field: str, llm_response: str) -> str | None:
    """Extract actual executable code from raw trajectory turn data.

    The SpreadsheetBench ``code`` field is unreliable — in some cases it
    contains natural-language text or JSON wrappers rather than actual code.

    Strategy (in priority order):
    1. If ``llm_response`` contains a ```python code block, extract it.
    2. If ``code_field`` is a JSON ``{"stdin": "..."}`` wrapper, unwrap it.
    3. If ``code_field`` contains recognisable code patterns, use it as-is.
    4. Otherwise return ``None`` (the turn has no executable code).
    """
    # 1. Try extracting from markdown code block in the full LLM response
    blocks = _CODE_BLOCK_RE.findall(llm_response)
    if blocks:
        # Pick the last *non-empty* block — LLM responses often contain
        # empty ```output ... ``` blocks after the main code block.
        for block in reversed(blocks):
            code = block.strip()
            if code:
                return code
        # All blocks were empty — fall through to other strategies

    # 2. Try JSON stdin wrapper  e.g. {"stdin": "import ..."}
    if code_field.startswith('{"stdin"'):
        try:
            parsed = json.loads(code_field)
            stdin_code = parsed.get("stdin", "")
            if stdin_code and any(kw in stdin_code for kw in _CODE_INDICATORS):
                return stdin_code.strip()
        except (json.JSONDecodeError, TypeError):
            pass

    # 3. Check if the code field itself looks like real code
    if code_field and any(kw in code_field for kw in _CODE_INDICATORS):
        return code_field.strip()

    # 4. Not code — return None so caller doesn't create a tool_call
    return None


def _determine_outcome(
    entry_id: str,
    eval_map: dict[str, dict],
) -> tuple[str, float]:
    """Determine outcome and reward from eval results.

    Returns
    -------
    (outcome, reward)
        outcome: "resolved" | "unresolved" | "error"
        reward:  1.0 for resolved, 0.0 otherwise
    """
    if not eval_map or entry_id not in eval_map:
        return "error", 0.0

    eval_entry = eval_map[entry_id]
    test_results = eval_entry.get("test_case_results", [])
    hard = eval_entry.get("hard_restriction", 0)

    if all(r == 1 for r in test_results) and hard == 1:
        return "resolved", 1.0
    else:
        return "unresolved", 0.0


def convert_spreadsheet_bench_trajectory(
    data: dict,
    *,
    source_path: str = "",
    eval_map: dict[str, dict] | None = None,
    outcome_override: str = "",
) -> Trajectory:
    """Convert a SpreadsheetBench trajectory entry to unified Trajectory.

    Parameters
    ----------
    data : dict
        One parsed JSON object from ``trajectory.jsonl``.
    source_path : str
        Original file path for provenance tracking.
    eval_map : dict
        Mapping from entry id to eval result dict.  Used to determine outcome.
    outcome_override : str
        If provided, use this outcome instead of computing from eval_map.

    Returns
    -------
    Trajectory
        Unified trajectory object.
    """
    entry_id = str(data.get("id", ""))
    model = data.get("model", "")
    instruction = data.get("instruction", "")
    instruction_type = data.get("instruction_type", "")
    answer_position = data.get("answer_position", "")
    total_turns = data.get("total_turns", 0)
    output_generated = data.get("output_generated", False)
    solution = data.get("solution", "")
    total_time_sec = data.get("total_time_sec", 0.0)
    conversation = data.get("conversation", [])
    trajectory_data = data.get("trajectory", [])

    # --- Build Steps from conversation + trajectory ---
    steps: list[Step] = []

    # Step 0: System prompt (conversation[0])
    if conversation:
        steps.append(Step(
            role="system",
            content=conversation[0],
        ))

    # Remaining conversation entries alternate: assistant (odd), exec_result (even)
    # The trajectory list has structured per-turn data that enriches this
    turn_idx = 0
    for conv_i in range(1, len(conversation)):
        conv_content = conversation[conv_i]

        if conv_i % 2 == 1:
            # Assistant response
            # Get corresponding turn metadata if available
            turn_meta = {}
            actual_code: str | None = None
            llm_response = conv_content  # conversation[odd] == llm_response

            if turn_idx < len(trajectory_data):
                t = trajectory_data[turn_idx]
                raw_code = t.get("code", "")
                raw_llm_response = t.get("llm_response", "") or llm_response
                turn_meta = {
                    "turn": t.get("turn", turn_idx + 1),
                    "code_extracted": t.get("code_extracted", False),
                    "llm_latency_sec": t.get("llm_latency_sec", 0.0),
                }
                # Use robust extraction instead of blindly trusting code field
                actual_code = _extract_code(raw_code, raw_llm_response)

            # Build tool_calls only when actual executable code was found
            tool_calls = None
            if actual_code:
                tool_calls = [{
                    "function_name": "execute_python",
                    "arguments": actual_code,
                }]

            steps.append(Step(
                role="agent",
                content=conv_content,
                tool_calls=tool_calls,
                metadata=turn_meta,
            ))

        else:
            # Execution result — merge into the previous agent step as observation
            exec_meta = {}
            if turn_idx < len(trajectory_data):
                t = trajectory_data[turn_idx]
                exec_meta = {
                    "exec_success": t.get("exec_success", True),
                    "exec_latency_sec": t.get("exec_latency_sec", 0.0),
                    "output_generated": t.get("output_generated", False),
                }
                turn_idx += 1  # advance turn counter after processing both parts

            # Merge as observation into the last agent step
            if steps and steps[-1].role == "agent":
                steps[-1].observation = conv_content
                steps[-1].metadata.update(exec_meta)
            else:
                # Fallback: add as a separate tool step
                steps.append(Step(
                    role="tool",
                    content=conv_content,
                    metadata=exec_meta,
                ))

    # --- Determine outcome ---
    if outcome_override:
        outcome = outcome_override
        reward = 1.0 if outcome == "resolved" else 0.0
    else:
        outcome, reward = _determine_outcome(entry_id, eval_map or {})

    # --- Final answer ---
    # SpreadsheetBench evaluates by comparing the generated xlsx file,
    # not the code itself. There is no textual "final answer".
    final_answer = ""

    # --- Build Trajectory ---
    traj_id = entry_id or str(uuid.uuid4())

    return Trajectory(
        id=traj_id,
        task_name=f"{instruction_type}: {answer_position}" if answer_position else instruction_type,
        agent=model,
        steps=steps,
        final_answer=final_answer,
        reward=reward,
        benchmark="spreadsheetbench",
        outcome=outcome,
        source_format="spreadsheetbench-react-exec",
        task_id=entry_id,
        metadata={
            "source_path": source_path,
            "instruction_type": instruction_type,
            "answer_position": answer_position,
            "spreadsheet_path": data.get("spreadsheet_path", ""),
            "total_turns": total_turns,
            "max_turn_num": data.get("max_turn_num", 0),
            "output_generated": output_generated,
            "total_time_sec": total_time_sec,
            "setting": data.get("setting", ""),
        },
    )
