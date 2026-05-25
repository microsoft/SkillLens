import argparse
import heapq
import multiprocessing as mp
import os
import queue
import shutil
import threading
import traceback
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
from pathlib import Path
from typing import Optional

from skilllens.benchmarks.bfcl.bfcl_eval.constants.eval_config import (
    PROJECT_ROOT,
    RESULT_FILE_PATTERN,
    RESULT_PATH,
    TEST_IDS_TO_GENERATE_PATH,
)
from skilllens.benchmarks.bfcl.bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from skilllens.benchmarks.bfcl.bfcl_eval.eval_checker.eval_runner_helper import load_file
from skilllens.benchmarks.bfcl.bfcl_eval.model_handler.base_handler import BaseHandler
from skilllens.benchmarks.bfcl.bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from skilllens.benchmarks.bfcl.bfcl_eval.model_handler.utils import get_retrying_counter
from skilllens.benchmarks.bfcl.bfcl_eval.utils import *
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    # Refer to model_choice for supported models.
    parser.add_argument("--model", type=str, default="gorilla-openfunctions-v2", nargs="+")
    # Refer to test_categories for supported categories.
    parser.add_argument("--test-category", type=str, default="all", nargs="+")

    # Parameters for the model that you want to test.
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--include-input-log", action="store_true", default=False)
    parser.add_argument("--exclude-state-log", action="store_true", default=False)
    parser.add_argument("--num-threads", required=False, type=int)
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--backend", default="vllm", type=str, choices=["vllm", "sglang"])
    parser.add_argument("--gpu-memory-utilization", default=0.9, type=float)
    parser.add_argument("--result-dir", default=None, type=str)
    parser.add_argument("--run-ids", action="store_true", default=False)
    parser.add_argument("--allow-overwrite", "-o", action="store_true", default=False)
    parser.add_argument(
        "--export-trajectory",
        action="store_true",
        default=False,
        help="Export Trajectory-format JSON files alongside BFCL results.",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        default=None,
        help="Directory to write trajectory files. Default: $BFCL_PROJECT_ROOT/trajectories/{model}/",
    )
    parser.add_argument(
        "--skip-server-setup",
        action="store_true",
        default=False,
        help="Skip vLLM/SGLang server setup and use existing endpoint specified by the LOCAL_SERVER_ENDPOINT and LOCAL_SERVER_PORT environment variables.",
    )
    # Optional local model path
    parser.add_argument(
        "--local-model-path",
        type=str,
        default=None,
        help="Specify the path to a local directory containing the model's config/tokenizer/weights for fully offline inference. Use this only if the model weights are stored in a location other than the default HF_HOME directory.",
    )
    parser.add_argument(
        "--lora-modules",
        type=str,
        default=None,
        nargs="*",
        help="Specify the path to the LoRA modules for vLLM backend in name=\"path\" format. Can be specified multiple times.",
    )
    parser.add_argument(
        "--enable-lora",
        action="store_true",
        default=False,
        help="Enable LoRA for vLLM backend.",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=None,
        help="Specify the maximum LoRA rank for vLLM backend.",
    )
    parser.add_argument(
        "--test-round",
        type=int,
        default=None,
        help="Round number for trajectory output. Appends -r{N} to trajectory dir name.",
    )
    parser.add_argument(
        "--exclude-ids-file",
        type=str,
        default=None,
        help="Path to JSON file with 'excluded_ids' list. These test case IDs will be skipped entirely.",
    )
    parser.add_argument(
        "--include-ids-file",
        type=str,
        default=None,
        help="Path to JSON file with 'ids' list. Only these test case IDs will be run.",
    )
    args = parser.parse_args()
    print(f"Parsed arguments: {args}")

    return args


def build_handler(model_name, temperature):
    config = MODEL_CONFIG_MAPPING[model_name]
    handler = config.model_handler(
        model_name=config.model_name,
        temperature=temperature,
        registry_name=model_name,
        is_fc_model=config.is_fc_model,
    )
    return handler


def get_involved_test_entries(test_category_args, run_ids):
    all_test_categories, all_test_entries_involved = [], []
    if run_ids:
        all_test_categories, all_test_entries_involved = load_test_entries_from_id_file(
            TEST_IDS_TO_GENERATE_PATH
        )

    else:
        all_test_categories = parse_test_category_argument(test_category_args)
        for test_category in all_test_categories:
            all_test_entries_involved.extend(load_dataset_entry(test_category))

    return (
        all_test_categories,
        all_test_entries_involved,
    )


def collect_test_cases(args, model_name, all_test_categories, all_test_entries_involved):
    model_name_dir = model_name.replace("/", "_")
    model_result_dir = args.result_dir / model_name_dir

    # Load excluded IDs if provided
    exclude_ids = set()
    if getattr(args, "exclude_ids_file", None):
        with open(args.exclude_ids_file, "r") as f:
            exclude_data = json.load(f)
        exclude_ids = set(exclude_data.get("excluded_ids", []))
        if exclude_ids:
            before = len(all_test_entries_involved)
            all_test_entries_involved = [
                tc for tc in all_test_entries_involved if tc["id"] not in exclude_ids
            ]
            tqdm.write(f"🚫 Excluded {before - len(all_test_entries_involved)} test cases via {args.exclude_ids_file}")

    # Load included IDs if provided (only run these)
    if getattr(args, "include_ids_file", None):
        with open(args.include_ids_file, "r") as f:
            include_data = json.load(f)
        include_ids = set(include_data.get("ids", []))
        if include_ids:
            before = len(all_test_entries_involved)
            all_test_entries_involved = [
                tc for tc in all_test_entries_involved if tc["id"] in include_ids
            ]
            tqdm.write(f"✅ Included {len(all_test_entries_involved)}/{before} test cases via {args.include_ids_file}")

    existing_result = []
    for test_category in all_test_categories:
        # TODO: Simplify the handling of memory prerequisite entries/categories
        result_file_paths = [
            model_result_dir
            / get_directory_structure_by_category(test_category)
            / get_file_name_by_category(test_category, is_result_file=True)
        ]
        if is_memory(test_category):
            # Memory test cases have the pre-requisite entries in a separate file
            result_file_paths.append(
                model_result_dir
                / get_directory_structure_by_category(test_category)
                / get_file_name_by_category(f"{test_category}_prereq", is_result_file=True)
            )

        for file_path in result_file_paths:
            if file_path.exists():
                # Not allowing overwrite, we will load the existing results
                if not args.allow_overwrite:
                    existing_result.extend(load_file(file_path))
                # Allow overwrite and not running specific test ids, we will delete the existing result file before generating new results
                elif not args.run_ids:
                    file_path.unlink()
                # Allow overwrite and running specific test ids, we will do nothing here
                else:
                    pass

        if is_memory(test_category):
            # We also need to special handle the pre-requisite entries and the snapshot result for memory test cases
            snapshot_folder = model_result_dir / "memory_snapshot" / test_category
            if snapshot_folder.exists():
                if not args.allow_overwrite:
                    pass
                elif not args.run_ids:
                    shutil.rmtree(snapshot_folder)
                else:
                    # TODO: If run_ids and id involes prereq entries, we should just delete those snapshot files
                    # It's not implemented yet, but it won't affect the accuracy, as those files will be overwritten anyway (assume generation success)
                    pass

    # Filter out error results so they get retried on resume
    def _is_error_entry(entry):
        r = entry.get("result", "")
        return isinstance(r, str) and r.startswith("Error during inference:")

    successful_results = [e for e in existing_result if not _is_error_entry(e)]
    error_results = [e for e in existing_result if _is_error_entry(e)]
    existing_ids = set(entry["id"] for entry in successful_results)

    # Remove error entries from result files so retried results can be appended cleanly
    if error_results and not args.allow_overwrite:
        error_ids = set(e["id"] for e in error_results)
        for test_category in all_test_categories:
            result_file_paths = [
                model_result_dir
                / get_directory_structure_by_category(test_category)
                / get_file_name_by_category(test_category, is_result_file=True)
            ]
            if is_memory(test_category):
                result_file_paths.append(
                    model_result_dir
                    / get_directory_structure_by_category(test_category)
                    / get_file_name_by_category(f"{test_category}_prereq", is_result_file=True)
                )
            for file_path in result_file_paths:
                if file_path.exists():
                    entries = load_file(file_path)
                    cleaned = [e for e in entries if e["id"] not in error_ids]
                    if len(cleaned) < len(entries):
                        with open(file_path, "w") as f:
                            for e in cleaned:
                                f.write(json.dumps(e) + "\n")

    test_cases_to_generate = [
        test_case
        for test_case in all_test_entries_involved
        if test_case["id"] not in existing_ids
    ]

    num_skipped = len(all_test_entries_involved) - len(test_cases_to_generate)
    if num_skipped > 0:
        tqdm.write(f"⏩ Skipping {num_skipped} already-completed test cases, {len(test_cases_to_generate)} remaining")
    if error_results:
        tqdm.write(f"🔄 Retrying {len(error_results)} previously-failed test cases")

    # Skip format sensitivity test cases for FC models
    if (
        any(is_format_sensitivity(test_category) for test_category in all_test_categories)
        and MODEL_CONFIG_MAPPING[model_name].is_fc_model
    ):
        test_cases_to_generate = [
            test_case
            for test_case in test_cases_to_generate
            if not is_format_sensitivity(test_case["id"])
        ]

    test_cases_to_generate = clean_up_memory_prereq_entries(test_cases_to_generate)
    # TODO: Should we move these to the load_dataset_entry function?
    test_cases_to_generate = populate_initial_settings_for_memory_test_cases(
        test_cases_to_generate, model_result_dir
    )
    test_cases_to_generate = populate_initial_settings_for_web_search_test_cases(
        test_cases_to_generate
    )

    return sorted(test_cases_to_generate, key=sort_key), num_skipped


def multi_threaded_inference(handler, test_case, include_input_log, exclude_state_log, error_counter=None):

    assert type(test_case["function"]) is list

    try:
        result, metadata = handler.inference(
            test_case, include_input_log, exclude_state_log
        )
    except Exception as e:
        if error_counter is not None:
            with error_counter["lock"]:
                error_counter["total"] += 1
                err_type = type(e).__name__
                error_counter["by_type"][err_type] = error_counter["by_type"].get(err_type, 0) + 1

        if type(e).__name__ != "BadRequestError":
            tqdm.write(f"  ✗ {test_case['id']}: {type(e).__name__}: {str(e)[:120]}")

        result = f"Error during inference: {str(e)}"
        metadata = {"traceback": traceback.format_exc()}
        # Attach last message history for error debugging (set by base_handler)
        last_messages = getattr(e, "_last_messages", None)
        if last_messages is not None:
            metadata["last_messages"] = last_messages

    result_to_write = {
        "id": test_case["id"],
        "result": result,
        **metadata,
    }

    return result_to_write


def generate_results(args, model_name, test_cases_total, num_skipped=0):
    handler = build_handler(model_name, args.temperature)

    # Trajectory export setup
    trajectory_dir = None
    if args.export_trajectory:
        from skilllens.benchmarks.bfcl.bfcl_eval.trajectory_exporter import bfcl_result_to_trajectory, write_trajectory

        if args.trajectory_dir:
            trajectory_dir = Path(args.trajectory_dir)
        else:
            dir_name = model_name.replace("/", "_")
            if getattr(args, "test_round", None) is not None:
                dir_name = f"{dir_name}-r{args.test_round}"
            trajectory_dir = PROJECT_ROOT / "trajectories" / dir_name
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        tqdm.write(f"📝 Trajectory export enabled → {trajectory_dir}")

    # Build a lookup from test_case id -> test_case for trajectory context
    id_to_test_case_for_traj = {tc["id"]: tc for tc in test_cases_total} if trajectory_dir else {}

    if isinstance(handler, OSSHandler):
        handler: OSSHandler
        is_oss_model = True
        # For OSS models, if the user didn't explicitly set the number of threads,
        # we default to 100 threads to speed up the inference.
        num_threads = (
            args.num_threads
            if args.num_threads is not None
            else LOCAL_SERVER_MAX_CONCURRENT_REQUEST
        )
    else:
        handler: BaseHandler
        is_oss_model = False
        num_threads = args.num_threads if args.num_threads is not None else 1

    # Use a separate thread to write the results to the file to avoid concurrent IO issues
    def _writer():
        """Consume result dicts from the queue and write them with exclusive access."""
        while True:
            item = write_queue.get()
            if item is None:
                break
            handler.write(item, result_dir=args.result_dir, update_mode=args.run_ids)

            # Export trajectory if enabled
            if trajectory_dir is not None:
                try:
                    test_entry = id_to_test_case_for_traj.get(item["id"])
                    traj = bfcl_result_to_trajectory(item, model_name, test_entry)
                    write_trajectory(traj, trajectory_dir)
                except Exception as e:
                    tqdm.write(f"⚠️ Trajectory export failed for {item.get('id', '?')}: {e}")

                # Write _err.json with last message history for failed samples
                result_val = item.get("result", "")
                if isinstance(result_val, str) and result_val.startswith("Error during inference:"):
                    last_msgs = item.get("last_messages")
                    if last_msgs is not None:
                        err_data = {
                            "id": item["id"],
                            "error": result_val,
                            "last_messages": last_msgs,
                        }
                        err_file = trajectory_dir / f"{item['id']}_err.json"
                        try:
                            with open(err_file, "w", encoding="utf-8") as ef:
                                json.dump(err_data, ef, ensure_ascii=False, indent=2, default=str)
                        except Exception as e:
                            tqdm.write(f"⚠️ Error log write failed for {item.get('id', '?')}: {e}")

            write_queue.task_done()

    write_queue: queue.Queue = queue.Queue()

    writer_thread = threading.Thread(target=_writer, daemon=True)
    writer_thread.start()

    try:
        if is_oss_model:
            handler.spin_up_local_server(
                num_gpus=args.num_gpus,
                gpu_memory_utilization=args.gpu_memory_utilization,
                backend=args.backend,
                skip_server_setup=args.skip_server_setup,
                local_model_path=args.local_model_path,
                lora_modules=args.lora_modules,
                enable_lora=args.enable_lora,
                max_lora_rank=args.max_lora_rank,
            )

        # ───── dependency bookkeeping ──────────────────────────────
        dependencies = {
            test_case["id"]: set(test_case.get("depends_on", []))
            for test_case in test_cases_total
        }
        children_of = defaultdict(list)
        for test_case in test_cases_total:
            for dependency_id in test_case.get("depends_on", []):
                children_of[dependency_id].append(test_case["id"])

        id_to_test_case = {test_case["id"]: test_case for test_case in test_cases_total}

        ready_queue = [
            (sort_key(id_to_test_case[test_case_id]), test_case_id)
            for test_case_id, dependency_ids in dependencies.items()
            if not dependency_ids
        ]
        heapq.heapify(ready_queue)
        in_flight: dict[Future, str] = {}  # future -> test_case_id
        completed = set()
        error_counter = {"total": 0, "by_type": {}, "lock": threading.Lock()}
        retrying_counter = get_retrying_counter()
        with retrying_counter["lock"]:
            retrying_counter["count"] = 0
            retrying_counter["count_content_filter"] = 0
            retrying_counter["count_retrying"] = 0
            retrying_counter["count_timeout"] = 0
        grand_total = len(test_cases_total) + num_skipped
        completed_count = num_skipped
        success_count = num_skipped  # skipped (resumed) samples count as successes

        def _update_postfix():
            parts = []
            # Success rate as float
            if completed_count > 0:
                parts.append(f"✓ {success_count / completed_count:.2f}")
            throttled = retrying_counter["count"]
            cf = retrying_counter["count_content_filter"]
            retrying = retrying_counter["count_retrying"]
            timeout = retrying_counter["count_timeout"]
            if throttled > 0:
                detail = f"⏳ {throttled}/{num_threads} throttled"
                if cf > 0:
                    detail += f" (filter:{cf})"
                parts.append(detail)
            if timeout > 0:
                parts.append(f"⏱ {timeout}/{num_threads} timeout")
            if retrying > 0:
                parts.append(f"🔄 {retrying}/{num_threads} retrying")
            pbar.set_postfix_str(", ".join(parts) if parts else "")

        with ThreadPoolExecutor(max_workers=num_threads) as pool, tqdm(
            total=grand_total,
            initial=num_skipped,
            desc=f"{model_name}",
            position=0,
            leave=True,
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        ) as pbar:

            # seed initial ready tasks
            while ready_queue and len(in_flight) < num_threads:
                _, test_case_id = heapq.heappop(ready_queue)
                test_case = id_to_test_case[test_case_id]
                future = pool.submit(
                    multi_threaded_inference,
                    handler,
                    test_case,
                    args.include_input_log,
                    args.exclude_state_log,
                    error_counter,
                )
                in_flight[future] = test_case_id

            # main scheduler loop
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    test_case_id = in_flight.pop(future)
                    result_dict = future.result()

                    # Enqueue the result for the writer thread to handle file IO
                    write_queue.put(result_dict)

                    # Track success rate
                    completed_count += 1
                    result_val = result_dict.get("result", "")
                    if not (isinstance(result_val, str) and result_val.startswith("Error during inference:")):
                        success_count += 1

                    # Update progress bar right after inference completes
                    pbar.update()
                    _update_postfix()
                    completed.add(test_case_id)

                    # unlock children
                    for child_id in children_of[test_case_id]:
                        dependencies[child_id].discard(test_case_id)
                        if not dependencies[child_id]:
                            heapq.heappush(
                                ready_queue,
                                (sort_key(id_to_test_case[child_id]), child_id),
                            )

                # refill the pool up to max_workers
                while ready_queue and len(in_flight) < num_threads:
                    _, test_case_id = heapq.heappop(ready_queue)
                    test_case = id_to_test_case[test_case_id]
                    future = pool.submit(
                        multi_threaded_inference,
                        handler,
                        test_case,
                        args.include_input_log,
                        args.exclude_state_log,
                        error_counter,
                    )
                    in_flight[future] = test_case_id

    finally:
        # Signal writer thread to finish and wait for it
        write_queue.put(None)
        writer_thread.join()

        if is_oss_model:
            handler.shutdown_local_server()


def main(args):

    # Note: The following environment variables are needed for the memory vector store implementation
    # Otherwise you get segfault or huggingface tokenizer warnings
    # disable HuggingFace tokenizers’ thread pool
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # limit all OpenMP/MKL threads to 1
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # use spawn method for multiprocessing
    mp.set_start_method("spawn", force=True)

    if type(args.model) is not list:
        args.model = [args.model]
    if type(args.test_category) is not list:
        args.test_category = [args.test_category]

    (
        all_test_categories,
        all_test_entries_involved,
    ) = get_involved_test_entries(args.test_category, args.run_ids)

    for model_name in args.model:
        if model_name not in MODEL_CONFIG_MAPPING:
            raise ValueError(
                f"Unknown model_name '{model_name}'.\n"
                "• For officially supported models, please refer to `SUPPORTED_MODELS.md`.\n"
                "• For running new models, please refer to `README.md` and `CONTRIBUTING.md`."
            )
    tqdm.write(f"Generating results for {args.model}")
    if args.run_ids:
        tqdm.write("Running specific test cases. Ignoring `--test-category` argument.")
    else:
        tqdm.write(f"Running full test cases for categories: {all_test_categories}.")

    if any(is_format_sensitivity(test_category) for test_category in all_test_categories):
        for model_name in args.model:
            if MODEL_CONFIG_MAPPING[model_name].is_fc_model:
                tqdm.write(
                    "⚠️ Warning: Format sensitivity test cases are only supported for prompting (non-FC) models. "
                    f"Since {model_name} is a FC model based on its config, the format sensitivity test cases will be skipped."
                )

    if args.result_dir is not None:
        args.result_dir = PROJECT_ROOT / args.result_dir
    else:
        args.result_dir = RESULT_PATH

    for model_name in args.model:
        test_cases_total, num_skipped = collect_test_cases(
            args,
            model_name,
            all_test_categories,
            deepcopy(all_test_entries_involved),
        )

        if len(test_cases_total) == 0:
            tqdm.write(
                f"✅ All selected test cases have been previously generated for {model_name}. No new test cases to generate."
            )
        else:
            generate_results(args, model_name, test_cases_total, num_skipped=num_skipped)
            # Sort the result files by id at the end
            for model_result_json in args.result_dir.rglob(RESULT_FILE_PATTERN):
                sort_file_content_by_id(model_result_json)
