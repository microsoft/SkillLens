"""
Core inference loop for SpreadsheetBench with optional skill augmentation.

Wraps the original SpreadsheetBench inference_multiple.py logic but adds:
- Skill tool parsing (```skill``` blocks) when a skill_provider is present
- Structured trajectory recording with action_type: "skill_tool" vs "code_exec"
- Proper interleaving of skill consultation and code execution turns

The module imports SpreadsheetBench native helpers (code_exec, prompt_format,
llm_api, jupyter_kernel_cli) via sys.path manipulation, keeping them out of
this package's dependency tree.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
import threading
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Import SpreadsheetBench native modules
# ---------------------------------------------------------------------------

from skilllens.benchmarks.spreadsheetbench.inference.llm_api import get_llm_response
from skilllens.benchmarks.spreadsheetbench.inference.code_exec import get_exec_client, extract_code, exec_code
from skilllens.benchmarks.spreadsheetbench.inference.prompt_format import (
    PROMPT_FORMAT_SINGLE,
    PROMPT_DF_RCT_FORMAT,
    PROMPT_NO_DF_RCT_FORMAT,
)

# ---------------------------------------------------------------------------
# Skill imports (from this package)
# ---------------------------------------------------------------------------

from skilllens.inference.skill_provider import ReadOnlySkillProvider
from skilllens.inference.spreadsheetbench.skill_tools import (
    extract_skill_call,
    has_skill_call,
    build_skill_prompt_section,
    build_single_skill_prompt_section,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment error detection
# ---------------------------------------------------------------------------

# Patterns that indicate transient environment issues (not model bugs)
_ENV_ERROR_PATTERNS = [
    "PermissionError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "ConnectionError",
    "[Errno 13]",           # Permission denied
    "[Errno 111]",          # Connection refused
    "OSError: [Errno 28]",  # No space left on device
    "BlockingIOError",
    "BrokenPipeError",
    "docker",               # Docker-related errors
    "Internal Server Error", # HTTP 500 from code_exec service
]


def _is_env_error(exec_result: str) -> bool:
    """Check if an execution result indicates a transient environment error
    (not a bug in the model's code)."""
    for pattern in _ENV_ERROR_PATTERNS:
        if pattern in exec_result:
            return True
    return False

# Thread-safe lock for writing to shared output files
write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gen_file_content(input_file: str, row_num: int) -> str:
    """Read first N rows of each sheet and format as text."""
    excel_file = pd.ExcelFile(input_file)
    sheet_names = excel_file.sheet_names
    excel_data = {}

    for sheet_name in sheet_names:
        df = excel_file.parse(sheet_name)
        n = row_num if df.shape[0] > row_num else df.shape[0]
        excel_data[sheet_name] = df.head(n).to_string()

    final_str = ""
    for sheet_name, sheet_str in excel_data.items():
        final_str += f"Sheet Name: {sheet_name}\n"
        final_str += sheet_str + "\n"
        final_str += "-" * 50 + "\n"

    return final_str


def _build_prompt(data: dict, opt: Any, dataset_path: str) -> tuple[str, str, str]:
    """Build the task prompt and resolve file paths.

    Returns:
        (prompt, output_path_local, find_input_path)
    """
    data_id = data["spreadsheet_path"].lstrip("spreadsheet/")

    # Support both naming conventions: verified uses _init.xlsx, original uses _input.xlsx
    init_file = f"1_{data_id}_init.xlsx"
    input_file = f"1_{data_id}_input.xlsx"
    init_path = f"{dataset_path}/{data['spreadsheet_path']}/{init_file}"
    if os.path.exists(init_path):
        file_name = init_file
    else:
        file_name = input_file

    input_path = f"/mnt/data/{data['spreadsheet_path']}/{file_name}"
    # Use run_tag (if provided) to create unique output paths per experiment,
    # preventing Docker output directory collisions between different skill sets.
    run_tag = getattr(opt, "run_tag", f"multi_{opt.setting}_{opt.model}")
    output_path = f"/mnt/data/outputs/{run_tag}/1_{data_id}_output.xlsx"
    output_path_local = output_path.replace("/mnt/data", dataset_path)
    find_input_path = f"{dataset_path}/{data['spreadsheet_path']}/{file_name}"

    format_vars = {
        "instruction": data["instruction"],
        "spreadsheet_path": input_path,
        "instruction_type": data["instruction_type"],
        "answer_position": data["answer_position"],
        "max_turn_num": opt.max_turn_num,
        "output_path": output_path,
    }

    if opt.setting == "row_exec":
        file_content = gen_file_content(find_input_path, opt.row)
        format_vars["spreadsheet_content"] = file_content
        prompt = PROMPT_FORMAT_SINGLE.format_map(format_vars)
    elif opt.setting == "react_exec":
        prompt = PROMPT_NO_DF_RCT_FORMAT.format_map(format_vars)
    elif opt.setting == "row_react_exec":
        file_content = gen_file_content(find_input_path, opt.row)
        format_vars["spreadsheet_content"] = file_content
        prompt = PROMPT_DF_RCT_FORMAT.format_map(format_vars)
    else:
        raise ValueError(f"Unknown setting: {opt.setting}")

    return prompt, output_path_local, find_input_path


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------


def _cleanup_containers(conv_id: str):
    """Stop and remove Docker containers for a completed sample."""
    try:
        # Remove the main container and any retry containers (_r1, _r2, etc.)
        result = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"name=conv-{conv_id}"],
            capture_output=True, text=True, timeout=10,
        )
        container_ids = result.stdout.strip().split()
        if container_ids:
            subprocess.run(
                ["docker", "rm", "-f"] + container_ids,
                capture_output=True, timeout=15,
            )
    except Exception:
        pass  # Best-effort cleanup, don't fail the sample


def process_single_sample(
    data: dict,
    opt: Any,
    dataset_path: str,
    skill_provider: Optional[ReadOnlySkillProvider] = None,
) -> dict:
    """Process one sample: multi-round LLM inference + code execution.

    When ``skill_provider`` is not None, the model can use ```skill``` blocks
    to query the skill library.  Skill turns do NOT count toward the code
    execution turn budget (``max_turn_num``), but a separate
    ``max_skill_turns`` cap prevents infinite skill loops.

    If a Docker/environment error is detected during code execution, the
    entire sample is retried from scratch (up to ``max_sample_retries``
    times) to avoid wasting LLM turn budget on transient infra issues.

    Returns a detailed trajectory record with structured per-turn logs.
    """
    max_sample_retries = getattr(opt, "max_sample_retries", 3)

    for sample_attempt in range(1, max_sample_retries + 1):
        result = _run_sample_once(
            data, opt, dataset_path, skill_provider, sample_attempt,
        )
        if result is None:
            # _run_sample_once returns None only on unrecoverable error
            return None

        # Check if any turn had a Docker/env error
        had_env_error = False
        for t in result.get("trajectory", []):
            er = t.get("exec_result", "")
            if _is_env_error(er):
                had_env_error = True
                break

        if not had_env_error:
            return result

        # Collect the env error details for logging
        env_errors = [
            t.get("exec_result", "")[:200]
            for t in result.get("trajectory", [])
            if _is_env_error(t.get("exec_result", ""))
        ]
        # Docker error detected — clean up and retry the whole sample
        logger.warning(
            "ENV_RETRY | sample=%s | attempt=%d/%d | errors=%s | retrying entire sample...",
            data["id"], sample_attempt, max_sample_retries, env_errors,
        )
        run_tag = getattr(opt, "run_tag", "")
        base_conv_id = f"EVAL_{run_tag}_{data['id']}" if run_tag else f"EVAL_{opt.model}_{data['id']}"
        # Clean up both the base conv_id and the attempt-suffixed one
        conv_id = base_conv_id if sample_attempt == 1 else f"{base_conv_id}_r{sample_attempt}"
        _cleanup_containers(conv_id)
        time.sleep(3 * sample_attempt)  # back off before retry

    # All retries exhausted, return last result as-is
    logger.error(
        "ENV_FAIL | sample=%s | exhausted all %d retries, returning last result with env errors.",
        data["id"], max_sample_retries,
    )
    return result


def _run_sample_once(
    data: dict,
    opt: Any,
    dataset_path: str,
    skill_provider: Optional[ReadOnlySkillProvider],
    attempt: int,
) -> Optional[dict]:
    """Run a single attempt of one sample. Returns trajectory dict or None."""
    sample_start_time = time.time()
    max_skill_turns = getattr(opt, "max_skill_turns", 5)

    try:
        prompt, output_path_local, _ = _build_prompt(data, opt, dataset_path)

        # If skill mode is on, append skill library section to the prompt
        if skill_provider is not None:
            if skill_provider.is_single_skill:
                # Single skill: inline its content directly — no tool protocol
                prompt = (
                    prompt.rstrip()
                    + "\n\n"
                    + build_single_skill_prompt_section(
                        skill_provider.get_single_skill()
                    )
                )
                # Disable tool-based skill interaction for this sample
                active_skill_provider = None
            else:
                prompt = (
                    prompt.rstrip()
                    + "\n\n"
                    + build_skill_prompt_section(skill_provider.skill_count)
                )
                active_skill_provider = skill_provider
        else:
            active_skill_provider = None

        data_id = data["spreadsheet_path"].lstrip("spreadsheet/")

        # Each sample gets its own conv_id — include run_tag to avoid
        # container collisions when multiple experiments run in parallel.
        # On retry attempts (attempt > 1), append suffix so the Tornado API
        # creates a fresh kernel instead of reusing the stale mapping from
        # the previous (killed) container.
        run_tag = getattr(opt, "run_tag", "")
        base_conv_id = f"EVAL_{run_tag}_{data['id']}" if run_tag else f"EVAL_{opt.model}_{data['id']}"
        conv_id = base_conv_id if attempt == 1 else f"{base_conv_id}_r{attempt}"
        client = get_exec_client(opt.code_exec_url, conv_id)

        # Multi-round conversation
        messages = [prompt]
        trajectory = []
        response = ""
        final_code = ""
        output_generated = False
        total_turns = 0
        total_skill_turns = 0

        code_turn_idx = 0  # counts code execution turns (capped at max_turn_num)

        while code_turn_idx < opt.max_turn_num:
            total_turns += 1
            turn_record = {"turn": total_turns}

            # --- LLM call ---
            llm_start = time.time()
            response, usage_info = get_llm_response(messages, opt, return_usage=True)
            llm_elapsed = time.time() - llm_start
            messages.append(response)

            turn_record["llm_response"] = response
            turn_record["llm_latency_sec"] = round(llm_elapsed, 2)
            turn_record["usage"] = usage_info

            # --- Check for skill tool call first (if skill mode is on) ---
            skill_call = None
            if active_skill_provider is not None:
                skill_call = extract_skill_call(response)

            if skill_call is not None and total_skill_turns < max_skill_turns:
                # Skill tool call — dispatch and continue without counting
                # toward code execution turns
                fn_name, fn_args = skill_call
                skill_start = time.time()
                skill_result = active_skill_provider.dispatch(fn_name, fn_args)
                skill_elapsed = time.time() - skill_start

                total_skill_turns += 1

                turn_record["action_type"] = "skill_tool"
                turn_record["skill_function"] = fn_name
                turn_record["skill_arguments"] = fn_args
                turn_record["skill_result"] = skill_result
                turn_record["skill_latency_sec"] = round(skill_elapsed, 4)

                # Add skill result as the next user message
                skill_msg = (
                    f"**Skill tool result** (`{fn_name}`):\n"
                    f"```json\n{skill_result}\n```\n\n"
                    "Now proceed with your analysis. "
                    "Use ```python``` code blocks when you're ready to write code."
                )
                messages.append(skill_msg)

                logger.info(
                    "Skill call: %s(%s) for sample %s (skill turn %d)",
                    fn_name,
                    json.dumps(fn_args, ensure_ascii=False)[:80],
                    data["id"],
                    total_skill_turns,
                )

                trajectory.append(turn_record)
                continue  # Don't count toward code turns

            elif skill_call is not None and total_skill_turns >= max_skill_turns:
                # Exceeded skill turn budget — tell model to proceed with code
                turn_record["action_type"] = "skill_tool_budget_exceeded"
                budget_msg = (
                    "You have reached the maximum number of skill tool calls. "
                    "Please proceed with writing Python code to solve the task."
                )
                messages.append(budget_msg)
                trajectory.append(turn_record)
                continue  # Don't count toward code turns either

            # --- Code extraction & execution (normal code turn) ---
            code_turn_idx += 1
            turn_record["action_type"] = "code_exec"

            code = extract_code(response)
            code_extracted = bool(code and code.strip())
            turn_record["code_extracted"] = code_extracted
            turn_record["code"] = code if code_extracted else ""

            exec_start = time.time()
            if code_extracted:
                try:
                    exec_result = exec_code(client, code)
                    exec_success = True
                except Exception as e:
                    exec_result = f"Error occur when running code: {str(e)}"
                    exec_success = False
            else:
                exec_result = (
                    "No Python code was found in your response. "
                    "Please provide your code inside a ```python ... ``` block."
                )
                exec_success = False
            exec_elapsed = time.time() - exec_start
            messages.append(exec_result)

            turn_record["exec_result"] = exec_result
            turn_record["exec_success"] = exec_success
            turn_record["exec_latency_sec"] = round(exec_elapsed, 2)

            # Check if output file was generated
            output_generated = os.path.exists(output_path_local)
            turn_record["output_generated"] = output_generated

            trajectory.append(turn_record)

            if code_extracted:
                final_code = code

            if output_generated:
                # Copy xlsx to results directory (use output_dir if provided)
                results_xlsx_dir = getattr(opt, "_results_xlsx_dir", None)
                if results_xlsx_dir is None:
                    results_xlsx_dir = os.path.abspath(
                        f"{dataset_path}/../results/{opt.dataset}/multi_{opt.setting}_{opt.model}/xlsx"
                    )
                os.makedirs(results_xlsx_dir, exist_ok=True)
                try:
                    shutil.copy2(output_path_local, results_xlsx_dir)
                except Exception:
                    pass
                break

        sample_elapsed = time.time() - sample_start_time

        conv_result = {
            # --- Task metadata ---
            "id": data["id"],
            "instruction": data["instruction"],
            "instruction_type": data["instruction_type"],
            "answer_position": data["answer_position"],
            "spreadsheet_path": data["spreadsheet_path"],
            # --- Run config ---
            "model": opt.model,
            "setting": opt.setting,
            "max_turn_num": opt.max_turn_num,
            "skill_augmented": skill_provider is not None,
            # --- Result summary ---
            "total_turns": total_turns,
            "code_turns": code_turn_idx,
            "skill_turns": total_skill_turns,
            "output_generated": output_generated,
            "solution": final_code,
            "total_time_sec": round(sample_elapsed, 2),
            # --- Detailed trajectory ---
            "trajectory": trajectory,
            # --- Legacy flat conversation (for backward compatibility) ---
            "conversation": messages,
        }

        # Clean up Docker containers for this sample
        _cleanup_containers(conv_id)

        return conv_result

    except Exception as e:
        sample_elapsed = time.time() - sample_start_time
        error_msg = traceback.format_exc()
        logger.error("Error processing sample %s: %s", data["id"], e)

        results_dir = os.path.abspath(
            f"{dataset_path}/../results/{opt.dataset}/multi_{opt.setting}_{opt.model}"
        )
        os.makedirs(results_dir, exist_ok=True)
        with write_lock:
            with open(f"{results_dir}/error_log.jsonl", "a+") as f:
                f.write(
                    json.dumps(
                        {
                            "id": data["id"],
                            "error": str(e),
                            "traceback": error_msg,
                            "data": data,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        # Clean up Docker containers even on error
        _cleanup_containers(conv_id)

        return {
            "id": data["id"],
            "instruction": data.get("instruction", ""),
            "instruction_type": data["instruction_type"],
            "answer_position": data.get("answer_position", ""),
            "spreadsheet_path": data.get("spreadsheet_path", ""),
            "model": opt.model,
            "setting": opt.setting,
            "max_turn_num": opt.max_turn_num,
            "skill_augmented": skill_provider is not None,
            "total_turns": 0,
            "code_turns": 0,
            "skill_turns": 0,
            "output_generated": False,
            "solution": "",
            "total_time_sec": round(sample_elapsed, 2),
            "trajectory": [],
            "conversation": "",
            "error": str(e),
        }
