#!/usr/bin/env python3

"""
Skill-augmented SpreadsheetBench batch runner.

Usage:
    # Baseline (no skill) — equivalent to inference_multiple.py
    python -m skilllens.inference.spreadsheetbench.run \
        --model gpt-5-mini \
        --setting react_exec \
        --dataset spreadsheetbench_verified_400 \
        --api_key $AZURE_API_KEY \
        --base_url $AZURE_ENDPOINT \
        --num_workers 16

    # Skill-augmented
    python -m skilllens.inference.spreadsheetbench.run \
        --model gpt-5-mini \
        --setting react_exec \
        --dataset spreadsheetbench_verified_400 \
        --skill_set path/to/skill_set.json \
        --api_key $AZURE_API_KEY \
        --base_url $AZURE_ENDPOINT \
        --num_workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from skilllens.inference.skill_provider import ReadOnlySkillProvider
from skilllens.inference.spreadsheetbench.process import (
    process_single_sample,
    write_lock,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Locate SpreadsheetBench data directory
# ---------------------------------------------------------------------------

_SKILLLENS_ROOT = Path(__file__).resolve().parents[3]  # SkillLens-code/
_SB_ROOT = None  # must be set via --data_root


def parse_args():
    parser = argparse.ArgumentParser(
        description="SpreadsheetBench inference with optional skill augmentation."
    )

    # --- model / API ---
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--api_key", type=str, default="", help="Azure API key")
    parser.add_argument("--base_url", type=str, default="", help="Azure endpoint URL")
    parser.add_argument(
        "--api_version",
        type=str,
        default="2025-04-01-preview",
        help="Azure OpenAI API version",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default=None,
        help="Reasoning effort level: low, medium, high",
    )

    # --- task / data ---
    parser.add_argument(
        "--setting",
        type=str,
        default="react_exec",
        help="Setting: row_exec, react_exec, row_react_exec",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="spreadsheetbench_verified_400",
        help="Dataset name (subdirectory under SpreadsheetBench/data/)",
    )
    parser.add_argument(
        "--dataset_json",
        type=str,
        default="dataset.json",
        help="Dataset JSON. Either a bare filename (resolved inside the dataset "
             "directory) or a path (absolute or repo-relative) to any JSON file "
             "containing the sample list. The JSON may be either a list of samples "
             "or a wrapper {\"items\": [...]} (e.g. testset_v1.json).",
    )
    parser.add_argument("--max_turn_num", type=int, default=5, help="Max code execution turns")
    parser.add_argument(
        "--max_samples", type=int, default=0,
        help="If >0, only run the first N samples (after loading dataset_json). "
             "0 = run all samples in the dataset (default).",
    )
    parser.add_argument(
        "--row", type=int, default=5, help="Number of rows in prompt (for row_exec / row_react_exec)"
    )

    # --- backend ---
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="API backend: None (auto-detect Azure) or 'vllm' for vLLM/OpenAI-compatible",
    )

    # --- execution ---
    parser.add_argument(
        "--code_exec_url",
        type=str,
        default="http://localhost:8081/execute",
        help="Code execution Docker API URL",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Unique tag for Docker output path. If empty, auto-generated from output_dir basename.",
    )

    # --- skill augmentation ---
    parser.add_argument(
        "--skill_set",
        type=str,
        default="",
        help="Path to skill_set.json. Empty = baseline (no skill)",
    )
    parser.add_argument(
        "--max_skill_turns",
        type=int,
        default=5,
        help="Max number of skill tool calls per sample",
    )

    # --- multi-round ---
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=1,
        help="Number of independent inference rounds (for variance estimation). "
             "Each round produces its own trajectory_rN.jsonl and eval_result_rN.json.",
    )

    # --- output ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Custom output directory. If empty, auto-generated under SpreadsheetBench/results/",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="",
        help="Override SpreadsheetBench root directory (data + evaluation). "
             "If empty, uses the default auto-detected path.",
    )

    return parser.parse_args()


def main():
    global _SB_ROOT
    opt = parse_args()
    logger.info("Config: %s", vars(opt))

    # ---- Override data root if specified ----
    if opt.data_root:
        _SB_ROOT = Path(opt.data_root)
        logger.info("Using custom data root: %s", _SB_ROOT)
    if _SB_ROOT is None:
        logger.error("--data_root is required: path to SpreadsheetBench data directory")
        raise SystemExit(1)

    # ---- Skill provider ----
    skill_provider: ReadOnlySkillProvider | None = None
    if opt.skill_set:
        skill_path = Path(opt.skill_set)
        if not skill_path.exists():
            logger.error("Skill set file not found: %s", skill_path)
            sys.exit(1)
        skill_provider = ReadOnlySkillProvider.from_file(skill_path)
        logger.info(
            "Skill mode ON: loaded %d skills from %s",
            skill_provider.skill_count,
            skill_path,
        )
    else:
        logger.info("Skill mode OFF: running baseline (no skills)")

    # ---- Determine output directory ----
    if opt.output_dir:
        results_dir = Path(opt.output_dir)
    else:
        suffix = "_skill" if skill_provider else ""
        results_dir = Path(
            f"inference_output/spreadsheetbench/{opt.setting}_{opt.model}{suffix}"
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Results will be saved to %s", results_dir)

    # ---- Dataset ----
    dataset_path = str(_SB_ROOT / "data" / opt.dataset)
    # Resolve dataset_json: bare filename → inside dataset dir; otherwise treat as
    # a path (absolute or relative to CWD). Either form may wrap samples under
    # {"items": [...]} (the testset_v1.json convention) or be a plain list.
    if os.sep in opt.dataset_json or os.path.isabs(opt.dataset_json):
        dataset_json = opt.dataset_json
    else:
        dataset_json = os.path.join(dataset_path, opt.dataset_json)
    if not os.path.exists(dataset_json):
        logger.error("Dataset not found: %s", dataset_json)
        sys.exit(1)
    with open(dataset_json, "r") as fp:
        dataset = json.load(fp)
    if isinstance(dataset, dict) and "items" in dataset:
        dataset = dataset["items"]
    if opt.max_samples and opt.max_samples > 0:
        dataset = dataset[: opt.max_samples]
        logger.info("Truncated dataset to first %d samples (via --max_samples)", len(dataset))
    logger.info("Loaded %d samples from %s", len(dataset), dataset_json)

    # ---- run_tag base ----
    if not opt.run_tag:
        opt.run_tag = results_dir.name

    # ---- Save run metadata ----
    run_meta = {
        "model": opt.model,
        "setting": opt.setting,
        "dataset": opt.dataset,
        "dataset_json": opt.dataset_json,
        "max_turn_num": opt.max_turn_num,
        "num_workers": opt.num_workers,
        "num_rounds": opt.num_rounds,
        "skill_set": opt.skill_set or None,
        "skill_count": skill_provider.skill_count if skill_provider else 0,
        "max_skill_turns": opt.max_skill_turns,
        "reasoning_effort": opt.reasoning_effort,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (results_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False)
    )

    # ---- Add log file handler ----
    log_path = results_dir / "run.log"
    file_handler = logging.FileHandler(str(log_path), mode="a")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logger.info("Log file: %s", log_path)

    # ---- Run rounds ----
    num_rounds = opt.num_rounds
    round_results = []

    for round_i in range(1, num_rounds + 1):
        if num_rounds > 1:
            logger.info("=" * 60)
            logger.info("ROUND %d / %d", round_i, num_rounds)
            logger.info("=" * 60)

        eval_result = _run_one_round(
            opt, dataset, dataset_path, results_dir, skill_provider, round_i, num_rounds,
        )
        if eval_result is not None:
            round_results.append((round_i, eval_result))

    # ---- Print cross-round summary ----
    if len(round_results) > 1:
        _print_round_summary(round_results, results_dir)

    logger.info("All %d round(s) completed.", num_rounds)


def _run_one_round(
    opt,
    dataset: list[dict],
    dataset_path: str,
    results_dir: Path,
    skill_provider: ReadOnlySkillProvider | None,
    round_i: int,
    num_rounds: int,
):
    """Run one independent inference round and return eval results (or None)."""
    import copy

    # Per-round opt copy so we can modify run_tag without affecting other rounds
    round_opt = copy.copy(opt)

    # ---- Per-round file names ----
    if num_rounds > 1:
        round_suffix = f"_r{round_i}"
        traj_filename = f"trajectory_r{round_i}.jsonl"
        eval_filename = f"eval_result_r{round_i}.json"
        xlsx_subdir = f"xlsx_r{round_i}"
        round_opt.run_tag = f"{opt.run_tag}_r{round_i}"
    else:
        round_suffix = ""
        traj_filename = "trajectory.jsonl"
        eval_filename = "eval_result.json"
        xlsx_subdir = "xlsx"
        # run_tag stays as-is

    xlsx_dir = results_dir / xlsx_subdir
    xlsx_dir.mkdir(exist_ok=True)

    # Docker output directory — per-round isolation
    output_file_path = os.path.join(dataset_path, "outputs", round_opt.run_tag)
    os.makedirs(output_file_path, exist_ok=True)
    try:
        os.chmod(output_file_path, 0o777)
    except PermissionError:
        logger.warning(
            "Cannot chmod %s (owned by Docker user). "
            "If PermissionError occurs during execution, "
            "manually run: chmod 777 %s",
            output_file_path, output_file_path,
        )

    round_opt._results_xlsx_dir = str(xlsx_dir)

    # ---- Resume: skip completed samples ----
    output_jsonl = str(results_dir / traj_filename)
    completed_ids: set[str] = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r") as fp:
            for line in fp:
                try:
                    rec = json.loads(line)
                    completed_ids.add(rec["id"])
                except Exception:
                    pass
        logger.info("Resuming round %d: %d samples already completed.", round_i, len(completed_ids))

    remaining = [d for d in dataset if d["id"] not in completed_ids]
    logger.info(
        "Round %d — Total: %d, Remaining: %d, Workers: %d",
        round_i, len(dataset), len(remaining), round_opt.num_workers,
    )

    if not remaining:
        logger.info("Round %d: all samples already completed.", round_i)
    else:
        # ---- Run inference ----
        pbar = tqdm(total=len(remaining), desc=f"Inference R{round_i}")

        with ThreadPoolExecutor(max_workers=round_opt.num_workers) as executor:
            futures = {
                executor.submit(
                    process_single_sample,
                    data,
                    round_opt,
                    dataset_path,
                    skill_provider,
                ): data
                for data in remaining
            }

            for future in as_completed(futures):
                conv_result = future.result()
                if conv_result is not None:
                    with write_lock:
                        with open(output_jsonl, "a+") as fp:
                            fp.write(json.dumps(conv_result, ensure_ascii=False) + "\n")
                pbar.update(1)

        pbar.close()

    # ---- Save skill usage summary ----
    if skill_provider is not None:
        summary = {
            "skill_count": skill_provider.skill_count,
            "total_tool_calls": len(skill_provider.call_history),
            "call_history": skill_provider.call_history,
        }
        skill_summary_name = f"skill_usage_summary{round_suffix}.json"
        (results_dir / skill_summary_name).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
        logger.info(
            "Round %d skill usage: %d tool calls", round_i, len(skill_provider.call_history)
        )

    logger.info("Round %d inference done. Results saved to %s", round_i, output_jsonl)

    # ---- Token usage & cost summary ----
    try:
        total_prompt = 0
        total_completion = 0
        total_total = 0
        with open(output_jsonl, "r") as fp:
            for line in fp:
                rec = json.loads(line)
                for turn in rec.get("trajectory", []):
                    u = turn.get("usage")
                    if u:
                        total_prompt += u.get("prompt_tokens", 0)
                        total_completion += u.get("completion_tokens", 0)
                        total_total += u.get("total_tokens", 0)
        cost_summary = {
            "round": round_i,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_total,
        }
        cost_file = results_dir / f"token_usage_r{round_i}.json"
        cost_file.write_text(json.dumps(cost_summary, indent=2))
        logger.info(
            "Round %d token usage: prompt=%d, completion=%d, total=%d",
            round_i, total_prompt, total_completion, total_total,
        )
    except Exception as e:
        logger.warning("Failed to compute token usage for round %d: %s", round_i, e)

    # ---- Auto evaluation ----
    logger.info("Starting evaluation for round %d...", round_i)
    try:
        eval_result = _run_evaluation(
            dataset, dataset_path, str(xlsx_dir), str(results_dir), eval_filename, round_i,
        )
        return eval_result
    except Exception as e:
        logger.error("Evaluation for round %d failed: %s", round_i, e)
        return None


def _print_round_summary(round_results, results_dir):
    """Print cross-round summary with mean ± std."""
    import numpy as np
    from collections import defaultdict

    logger.info("=" * 60)
    logger.info("CROSS-ROUND SUMMARY (%d rounds)", len(round_results))
    logger.info("=" * 60)

    all_hard = []
    all_soft = []
    type_hard = defaultdict(list)

    for round_i, eval_result in round_results:
        hard_scores = [r["hard_restriction"] for r in eval_result]
        soft_scores = [r["soft_restriction"] for r in eval_result]
        hard_acc = np.mean(hard_scores) * 100
        soft_acc = np.mean(soft_scores) * 100
        all_hard.append(hard_acc)
        all_soft.append(soft_acc)
        logger.info("  Round %d: Hard=%.1f%%, Soft=%.1f%%", round_i, hard_acc, soft_acc)

        for r in eval_result:
            type_hard[r["instruction_type"]].append(r["hard_restriction"])

    logger.info("  ----- Aggregate -----")
    logger.info(
        "  Hard accuracy: %.1f%% ± %.1f%%",
        np.mean(all_hard), np.std(all_hard),
    )
    logger.info(
        "  Soft accuracy: %.1f%% ± %.1f%%",
        np.mean(all_soft), np.std(all_soft),
    )

    # Per-type per-round breakdown
    # Collect per-round, per-type accuracy
    type_per_round = defaultdict(list)
    for round_i, eval_result in round_results:
        round_type = defaultdict(list)
        for r in eval_result:
            round_type[r["instruction_type"]].append(r["hard_restriction"])
        for itype, scores in round_type.items():
            type_per_round[itype].append(np.mean(scores) * 100)

    for itype in sorted(type_per_round.keys()):
        accs = type_per_round[itype]
        logger.info(
            "  %s: %.1f%% ± %.1f%%",
            itype, np.mean(accs), np.std(accs),
        )

    # Save summary
    summary = {
        "num_rounds": len(round_results),
        "hard_mean": float(np.mean(all_hard)),
        "hard_std": float(np.std(all_hard)),
        "soft_mean": float(np.mean(all_soft)),
        "soft_std": float(np.std(all_soft)),
        "per_round_hard": {str(r): h for (r, _), h in zip(round_results, all_hard)},
        "per_type": {
            itype: {"mean": float(np.mean(accs)), "std": float(np.std(accs))}
            for itype, accs in type_per_round.items()
        },
    }
    (results_dir / "round_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    logger.info("Round summary saved to %s", results_dir / "round_summary.json")
    logger.info("=" * 60)


def _run_evaluation(dataset, dataset_path, xlsx_dir, results_dir, eval_filename="eval_result.json", round_i=1):
    """Run OJ-style evaluation comparing output xlsx with golden xlsx.

    Returns the list of per-sample eval dicts so the caller can aggregate
    across rounds.
    """
    import numpy as np
    from collections import defaultdict

    # Import evaluation function from benchmarks/
    from skilllens.benchmarks.spreadsheetbench.evaluation.evaluation import compare_workbooks

    # Detect verified vs original dataset
    sample_id = dataset[0]["id"]
    is_verified = os.path.exists(
        os.path.join(dataset_path, "spreadsheet", str(sample_id), f"1_{sample_id}_golden.xlsx")
    )
    num_test_cases = 1 if is_verified else 3
    gt_suffix = "_golden.xlsx" if is_verified else "_answer.xlsx"

    eval_results = []
    type_results = defaultdict(list)

    for data in dataset:
        sid = data["id"]
        test_case_scores = []
        for tc in range(1, num_test_cases + 1):
            gt_path = os.path.join(dataset_path, "spreadsheet", str(sid), f"{tc}_{sid}{gt_suffix}")
            proc_path = os.path.join(xlsx_dir, f"{tc}_{sid}_output.xlsx")
            try:
                result, _ = compare_workbooks(
                    gt_path, proc_path, data["instruction_type"], data["answer_position"]
                )
            except Exception:
                result = False
            test_case_scores.append(int(result))

        hard = 0 if 0 in test_case_scores else 1
        soft = sum(test_case_scores) / len(test_case_scores)
        eval_results.append({
            "id": sid,
            "instruction_type": data["instruction_type"],
            "test_case_results": test_case_scores,
            "soft_restriction": soft,
            "hard_restriction": hard,
        })
        type_results[data["instruction_type"]].append(hard)

    # Save eval results
    eval_path = os.path.join(results_dir, eval_filename)
    with open(eval_path, "w") as fp:
        json.dump(eval_results, fp, indent=2, ensure_ascii=False)

    # Print summary
    hard_scores = [r["hard_restriction"] for r in eval_results]
    soft_scores = [r["soft_restriction"] for r in eval_results]
    total = len(eval_results)
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS (Round %d)", round_i)
    logger.info("=" * 60)
    logger.info("Total samples: %d", total)
    logger.info("Hard accuracy: %.2f%% (%d/%d)", np.mean(hard_scores) * 100, sum(hard_scores), total)
    logger.info("Soft accuracy: %.2f%%", np.mean(soft_scores) * 100)
    for itype in sorted(type_results.keys()):
        scores = type_results[itype]
        logger.info(
            "  %s: %.1f%% (%d/%d)",
            itype, np.mean(scores) * 100, sum(scores), len(scores),
        )
    logger.info("Eval results saved to %s", eval_path)
    logger.info("=" * 60)

    return eval_results


if __name__ == "__main__":
    main()
