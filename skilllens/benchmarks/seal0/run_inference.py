"""Parallel inference runner with checkpoint resume and pass@k support"""
import argparse
import json
import os
import re


def _json_default(o):
    """Fallback serializer for json.dump/dumps (handles sets, etc.)."""
    if isinstance(o, set):
        return list(o)
    return str(o)
import math
import sys
import time
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

from skilllens.benchmarks.seal0.agent import ReActAgent


def normalize_question_text(question: Any) -> str:
    if isinstance(question, str):
        return question.strip()
    if question is None:
        return ""
    return str(question).strip()


def extract_question_from_item(item: Dict[str, Any]) -> str:
    question = item.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()

    messages = item.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            candidate = content.strip()
            if not candidate:
                continue
            if "User:" in candidate:
                try:
                    candidate = candidate.split("User:", 1)[1].strip()
                except Exception:
                    pass
            if candidate:
                return candidate

    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiteResearcher Parallel Inference Runner")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--flat_output", action="store_true", help="Write files directly into --output without model/dataset/runs subdirs")
    parser.add_argument("--dataset", type=str, default="gaia")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--presence_penalty", type=float, default=1.1)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--roll_out_count", type=int, default=3)
    parser.add_argument("--total_splits", type=int, default=1)
    parser.add_argument("--worker_split", type=int, default=1)
    parser.add_argument("--max_questions", type=int, default=0,
                        help="Cap the number of questions (0 = no cap)")
    parser.add_argument("--experiment_name", type=str, default="",
                        help="Optional experiment subdirectory name")
    parser.add_argument("--ckpt", type=str, default="",
                        help="Checkpoint directory")
    parser.add_argument("--ckpt_file", type=str, default="",
                        help="Checkpoint filename for this run")
    parser.add_argument("--no_resume", action="store_true", default=True,
                        help="Skip checkpoint resume, always run fresh (default: True)")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from previous checkpoints")
    args = parser.parse_args()

    # --resume explicitly overrides --no_resume
    if args.resume:
        args.no_resume = False

    model = args.model
    output_base = args.output
    roll_out_count = args.roll_out_count
    total_splits = args.total_splits
    worker_split = args.worker_split

    if worker_split < 1 or worker_split > total_splits:
        raise ValueError(f"Invalid worker_split {worker_split}; expected 1..{total_splits}")

    model_name = os.path.basename(model.rstrip('/'))

    experiment_name_raw = args.experiment_name.strip()
    experiment_name_sanitized = ""
    if experiment_name_raw:
        experiment_name_sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", experiment_name_raw)
        if not experiment_name_sanitized:
            experiment_name_sanitized = "experiment"

    # Build output directory hierarchy
    # With --flat_output: write directly into output_base (no model/dataset/runs nesting)
    # Without: output_base / [experiment] / {model}_sglang / {dataset_stem} / runs
    if getattr(args, "flat_output", False):
        runs_dir = output_base
        dataset_dir = output_base
    else:
        experiment_root = output_base
        if experiment_name_sanitized:
            experiment_root = os.path.join(output_base, experiment_name_sanitized)

        model_dir = os.path.join(experiment_root, f"{model_name}_sglang")

        dataset_identifier = args.dataset.rstrip('/\\')
        dataset_basename = os.path.basename(dataset_identifier)
        dataset_stem, _ = os.path.splitext(dataset_basename)
        if not dataset_stem:
            dataset_stem = dataset_basename

        dataset_dir = os.path.join(model_dir, dataset_stem)
        runs_dir = os.path.join(dataset_dir, "runs")

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    if args.ckpt:
        Path(args.ckpt).mkdir(parents=True, exist_ok=True)

    run_started_at = datetime.now()
    timestamp = run_started_at.strftime("%Y%m%d_%H%M%S")
    split_suffix = ""
    if total_splits > 1:
        split_suffix = f"_split{worker_split}of{total_splits}"
    experiment_filename = f"experiment_{timestamp}{split_suffix}.json"
    experiment_path = os.path.join(runs_dir, experiment_filename)

    # Load dataset
    data_filepath = f"{args.dataset}"
    try:
        if data_filepath.endswith(".json"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                raise ValueError("Input JSON must be a list of objects.")
        elif data_filepath.endswith(".jsonl"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported file extension. Please use .json or .jsonl files.")
    except FileNotFoundError:
        sys.exit(1)
    except (json.JSONDecodeError, ValueError):
        sys.exit(1)

    dataset_items_original = len(items)
    if args.max_questions and args.max_questions > 0:
        capped_count = min(args.max_questions, dataset_items_original)
        items = items[:capped_count]

    dataset_items_total = len(items)
    if dataset_items_total == 0:
        sys.exit(0)

    total_loaded = dataset_items_total

    items_per_split_before_resume = math.ceil(dataset_items_total / total_splits) if total_splits > 0 else dataset_items_total
    start_idx_before_resume = (worker_split - 1) * items_per_split_before_resume
    end_idx_before_resume = min(worker_split * items_per_split_before_resume, dataset_items_total)

    # Resume: load processed questions
    def load_processed_questions(rollout_idx: int) -> Set[str]:
        processed: Set[str] = set()
        search_dirs = []
        if args.ckpt and os.path.isdir(args.ckpt):
            search_dirs.append(args.ckpt)
        search_dirs.extend([dataset_dir, runs_dir])
        for directory in search_dirs:
            if not os.path.isdir(directory):
                continue
            for fname in os.listdir(directory):
                fpath = os.path.join(directory, fname)
                if not os.path.isfile(fpath):
                    continue

                if fname.endswith(".json"):
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            payload = json.load(f)
                    except Exception:
                        continue

                    records: List[Dict[str, Any]] = []
                    if isinstance(payload, dict):
                        if isinstance(payload.get("records"), list):
                            records = [e for e in payload["records"] if isinstance(e, dict)]
                        elif isinstance(payload.get("iterations"), list):
                            records = [e for e in payload["iterations"] if isinstance(e, dict)]
                    elif isinstance(payload, list):
                        records = [e for e in payload if isinstance(e, dict)]

                    for item in records:
                        rollout_val = item.get("rollout_idx")
                        if rollout_val not in (rollout_idx, None):
                            continue
                        if item.get("error"):
                            continue
                        question = normalize_question_text(item.get("question"))
                        if not question:
                            continue
                        processed.add(question)

                elif fname.endswith(".jsonl"):
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            for line in f:
                                try:
                                    data = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                if data.get("rollout_idx") != rollout_idx:
                                    continue
                                if data.get("error"):
                                    continue
                                question = normalize_question_text(data.get("question"))
                                if question:
                                    processed.add(question)
                    except Exception:
                        continue

        return processed

    processed_queries_per_rollout = {}
    if args.no_resume:
        processed_queries_per_rollout = {
            rollout_idx: set()
            for rollout_idx in range(1, roll_out_count + 1)
        }
    else:
        processed_queries_per_rollout = {
            rollout_idx: load_processed_questions(rollout_idx)
            for rollout_idx in range(1, roll_out_count + 1)
        }

    # Filter already completed
    filtered_items: List[Dict[str, Any]] = []
    seen_questions_global: Set[str] = set()
    completed_questions = 0
    for item in items:
        question_text = normalize_question_text(extract_question_from_item(item))
        if not question_text:
            continue
        if question_text in seen_questions_global:
            continue
        seen_questions_global.add(question_text)

        all_rollouts_completed = True
        for rollout_idx in range(1, roll_out_count + 1):
            if question_text not in processed_queries_per_rollout[rollout_idx]:
                all_rollouts_completed = False
                break

        if all_rollouts_completed:
            completed_questions += 1
            continue

        item = item.copy()
        item["question"] = question_text
        filtered_items.append(item)

    items_after_resume_filter_total = len(filtered_items)

    print(f"Loaded: {total_loaded}")
    print(f"Already completed: {completed_questions}")
    print(f"Remaining after dedup: {items_after_resume_filter_total}")

    if items_after_resume_filter_total == 0:
        sys.exit(0)

    items_per_split_after_resume = math.ceil(items_after_resume_filter_total / total_splits) if total_splits > 0 else items_after_resume_filter_total
    start_idx_after_resume = (worker_split - 1) * items_per_split_after_resume
    end_idx_after_resume = min(worker_split * items_per_split_after_resume, items_after_resume_filter_total)
    items = filtered_items[start_idx_after_resume:end_idx_after_resume]
    items_in_split_after_resume = len(items)

    if items_in_split_after_resume == 0:
        sys.exit(0)

    # Build task list
    tasks_to_run_all: List[Dict[str, Any]] = []
    per_rollout_task_counts = {i: 0 for i in range(1, roll_out_count + 1)}

    default_ports = [6001]
    env_ports = os.environ.get("PLANNING_PORTS", "")
    planning_ports = []
    if env_ports.strip():
        for token in re.split(r"[\s,;:]+", env_ports.strip()):
            if not token:
                continue
            try:
                planning_ports.append(int(token))
            except ValueError:
                continue
    if not planning_ports:
        planning_ports = default_ports

    planning_rr_idx = 0
    question_to_ports = {}
    for rollout_idx in range(1, roll_out_count + 1):
        processed_queries = processed_queries_per_rollout[rollout_idx]
        for item in items:
            question = normalize_question_text(item.get("question"))
            if not question:
                continue
            if question in processed_queries:
                continue
            if question not in question_to_ports:
                planning_port = planning_ports[planning_rr_idx % len(planning_ports)]
                question_to_ports[question] = planning_port
                planning_rr_idx += 1
            planning_port = question_to_ports[question]
            tasks_to_run_all.append({
                "item": item.copy(),
                "rollout_idx": rollout_idx,
                "planning_port": planning_port,
            })
            per_rollout_task_counts[rollout_idx] += 1

    llm_cfg = {
        'model': model,
        'generate_cfg': {
            'max_input_tokens': 320000,
            'max_retries': 10,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'presence_penalty': args.presence_penalty
        },
    }

    test_agent = ReActAgent(llm=llm_cfg)

    results_by_rollout = {i: [] for i in range(1, roll_out_count + 1)}
    results_locks = {i: threading.Lock() for i in range(1, roll_out_count + 1)}
    iteration_counters = {i: 0 for i in range(1, roll_out_count + 1)}

    if not tasks_to_run_all:
        sys.exit(0)

    # Incremental save: write each result as it completes
    incremental_path = os.path.join(runs_dir, f"incremental_{timestamp}{split_suffix}.jsonl")
    incremental_lock = threading.Lock()

    batch_start_time = time.time()
    total_tasks = len(tasks_to_run_all)
    completed_count = 0
    use_progress = os.environ.get("SHOW_PROGRESS", "true").strip().lower() != "false"

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    pbar = None
    error_count = 0
    # Show total progress including already-completed items
    total_with_completed = total_tasks + completed_questions
    if use_progress and has_tqdm:
        pbar = tqdm(
            total=total_with_completed,
            initial=completed_questions,
            desc=f"{model}",
            unit="q",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
    else:
        print(f"Processing {total_tasks} tasks ({completed_questions} already done)")

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool_executor:
        future_to_task = {
            pool_executor.submit(
                test_agent._run,
                task,
                model
            ): task for task in tasks_to_run_all
        }
        for future in as_completed(future_to_task):
            task_info = future_to_task[future]
            rollout_idx = task_info["rollout_idx"]
            item_payload = task_info.get("item", {})
            base_payload = {
                "rollout_idx": rollout_idx,
                "planning_port": task_info.get("planning_port"),
                "question_id": (
                    task_info.get("question_id")
                    or item_payload.get("question_id")
                    or item_payload.get("id")
                ),
            }
            try:
                result = future.result()
                entry = {**base_payload, **result}
            except Exception as exc:
                import traceback
                error_count += 1
                tqdm.write(f"  ✗ {base_payload.get('question_id','?')}: {type(exc).__name__}: {str(exc)[:120]}")
                entry = {**base_payload, "error": str(exc)}
            with results_locks[rollout_idx]:
                iteration_counters[rollout_idx] += 1
                entry["iteration_index"] = iteration_counters[rollout_idx]
                results_by_rollout[rollout_idx].append(entry)
            # Incremental save: append to JSONL immediately
            with incremental_lock:
                with open(incremental_path, "a", encoding="utf-8") as inc_f:
                    inc_f.write(json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n")
            completed_count += 1
            if pbar:
                # Show judge result + error count in progress bar
                correct = sum(1 for ro in results_by_rollout.values() for e in ro if e.get('judge',{}).get('correct'))
                verdict = entry.get("judge", {}).get("verdict", "")
                postfix = f"✓{correct}/{completed_count}"
                if error_count > 0:
                    postfix += f" err:{error_count}/{args.max_workers}"
                if verdict:
                    postfix += f" [{verdict}]"
                pbar.set_postfix_str(postfix)
                pbar.update(1)
            elif use_progress and not has_tqdm:
                print(f"  [{completed_count}/{total_tasks}] done", flush=True)
    if pbar:
        pbar.close()
    batch_end_time = time.time()
    print(f"Batch time: {batch_end_time - batch_start_time:.2f}s")

    run_completed_at = datetime.now()
    total_records = sum(len(entries) for entries in results_by_rollout.values())

    batch_metadata: Dict[str, Any] = {
        "model": model,
        "dataset": args.dataset,
        "dataset_items_original": dataset_items_original,
        "dataset_items_after_limit": dataset_items_total,
        "items_in_split_after_resume": items_in_split_after_resume,
        "items_remaining_total_after_resume": items_after_resume_filter_total,
        "rollout_count": roll_out_count,
        "timestamp": run_started_at.isoformat(),
        "completed_at": run_completed_at.isoformat(),
        "run_id": timestamp,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty,
        "max_workers": args.max_workers,
        "total_splits": total_splits,
        "worker_split": worker_split,
        "planning_ports": planning_ports,
        "max_questions": args.max_questions,
        "experiment_name": experiment_name_raw,
        "tasks_requested": len(tasks_to_run_all),
        "records_collected": total_records,
        "duration_seconds": (run_completed_at - run_started_at).total_seconds(),
    }

    ckpt_filename = args.ckpt_file.strip() if args.ckpt_file else ""
    if not ckpt_filename:
        ckpt_filename = f"batch_{worker_split:03d}_of_{total_splits}_{timestamp}.json"
    batch_metadata["ckpt_file"] = ckpt_filename
    batch_metadata["ckpt_directory"] = args.ckpt

    batch_records: List[Dict[str, Any]] = []
    for rollout_idx in range(1, roll_out_count + 1):
        entries = results_by_rollout[rollout_idx]
        if not entries:
            continue
        entries_sorted = sorted(entries, key=lambda item: item.get("iteration_index", 0))
        batch_records.extend(entries_sorted)

    # Judge summary
    judge_counts = {"total": 0, "correct": 0, "incorrect": 0, "skipped": 0, "errors": 0}
    for record in batch_records:
        judge_info = record.get("judge")
        if isinstance(judge_info, dict):
            status = judge_info.get("status")
            if status == "ok":
                judge_counts["total"] += 1
                correct_val = judge_info.get("correct")
                if isinstance(correct_val, bool):
                    is_correct = correct_val
                elif isinstance(correct_val, (int, float)):
                    is_correct = bool(correct_val)
                elif isinstance(correct_val, str):
                    is_correct = correct_val.strip().lower() in {"true", "1", "yes"}
                else:
                    is_correct = False
                if is_correct:
                    judge_counts["correct"] += 1
                else:
                    judge_counts["incorrect"] += 1
            else:
                judge_counts["skipped"] += 1
                if status == "error":
                    judge_counts["errors"] += 1
        else:
            judge_counts["skipped"] += 1

    judge_total_attempted = len(batch_records)
    judge_accuracy = judge_counts["correct"] / judge_total_attempted if judge_total_attempted else None
    batch_metadata["judge_summary"] = {
        "total": judge_counts["total"],
        "correct": judge_counts["correct"],
        "incorrect": judge_counts["incorrect"],
        "skipped": judge_counts["skipped"],
        "errors": judge_counts["errors"],
        "accuracy": judge_accuracy,
    }

    batch_payload = {
        "metadata": batch_metadata,
        "records": batch_records,
    }

    if args.ckpt:
        ckpt_path = os.path.join(args.ckpt, ckpt_filename)
        Path(args.ckpt).mkdir(parents=True, exist_ok=True)
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(batch_payload, f, ensure_ascii=False, indent=2, default=_json_default)
    else:
        with open(experiment_path, "w", encoding="utf-8") as f:
            json.dump(batch_payload, f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"Results saved. Records: {total_records}")
    if judge_accuracy is not None:
        print(f"Judge accuracy: {judge_accuracy:.2%} ({judge_counts['correct']}/{judge_total_attempted}, judged_ok={judge_counts['total']}, skipped={judge_counts['skipped']})")
