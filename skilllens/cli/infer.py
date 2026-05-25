"""CLI sub-command: infer — unified inference interface for all benchmarks."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

logger = logging.getLogger("skilllens")


def _add_infer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", "-b", type=str, required=True,
                        choices=["alfworld", "swebench", "spreadsheetbench", "bfcl", "seal0"],
                        help="Benchmark to run.")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Model name (e.g. gpt-5.4, gpt-4o).")
    parser.add_argument("--provider", type=str, default="azure",
                        choices=["azure", "openai", "vllm"],
                        help="API provider (default: azure).")
    parser.add_argument("--endpoint", type=str, default="",
                        help="API endpoint URL.")
    parser.add_argument("--api-key", type=str, default="",
                        help="API key. Empty = use Managed Identity (Azure).")
    parser.add_argument("--num-samples", "-n", type=int, default=0,
                        help="Number of samples to evaluate (0 = full split, the default). "
                             "Use a small N for smoke tests.")
    parser.add_argument("--num-rounds", "-r", type=int, default=1,
                        help="Number of evaluation rounds (default: 1; use 3 for variance estimation).")
    parser.add_argument("--workers", "-w", type=int, default=16,
                        help="Number of parallel workers (default: 16).")
    parser.add_argument("--skill-set", type=str, default="",
                        help="Path to skill_set.json. Empty = baseline.")
    parser.add_argument("--reasoning-effort", type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help="Reasoning effort for reasoning-capable models (default: medium). "
                             "Honored by benchmarks that pass it to the LLM (ALFWorld, SpreadsheetBench, BFCL).")
    parser.add_argument("--max-turn-num", type=int, default=10,
                        help="Max ReAct/code-exec turns per sample (default: 10). "
                             "Honored by SpreadsheetBench; other benchmarks have their own caps.")
    parser.add_argument("--output-dir", "-o", type=str, default="",
                        help="Output directory. Default: inference_output/{benchmark}/...")
    parser.add_argument("--benchmark-args", nargs="*", default=[],
                        help="Extra benchmark-specific args as KEY=VALUE pairs.")
    parser.add_argument("--verbose", "-v", action="store_true")


def _parse_extra_args(pairs: list[str]) -> dict[str, list[str]]:
    """Parse KEY=VALUE pairs into a dict of lists.

    Repeated keys accumulate (used by SWE-bench's repeatable `-c` option).
    The dispatcher emits ``--k v`` for every value in each list.
    """
    result: dict[str, list[str]] = {}
    for pair in pairs:
        if "=" in pair:
            k, v = pair.split("=", 1)
            result.setdefault(k, []).append(v)
    return result


def _pop_single(extra: dict[str, list[str]], key: str, default: str) -> str:
    """Pop a single-valued extra arg (first value if list)."""
    if key in extra:
        vals = extra.pop(key)
        return vals[0] if vals else default
    return default


def _emit_extras(extra: dict[str, list[str]]) -> list[str]:
    """Flatten extras dict back into a list of ``--key value`` argv tokens.

    Each value in a list becomes its own ``--key value`` pair, preserving
    repeated options like ``-c key=val -c other=val2``. Single-char keys
    are emitted with a single dash (``-c``) to match short-option style.
    """
    out: list[str] = []
    for k, vals in extra.items():
        flag = f"-{k}" if len(k) == 1 else f"--{k}"
        for v in vals:
            out += [flag, v]
    return out


def cmd_infer(args: argparse.Namespace) -> None:
    benchmark = args.benchmark
    extra = _parse_extra_args(args.benchmark_args or [])

    # Build default output dir
    if args.output_dir:
        output_dir = args.output_dir
    else:
        tag = "with_skill" if args.skill_set else "baseline"
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_model = args.model.replace("/", "_")
        output_dir = f"inference_output/{benchmark}/{safe_model}_{tag}_{ts}"

    logger.info("Benchmark: %s | Model: %s | Rounds: %d | Samples: %d | Workers: %d",
                benchmark, args.model, args.num_rounds, args.num_samples, args.workers)
    logger.info("Output: %s", output_dir)
    if args.skill_set:
        logger.info("Skill set: %s", args.skill_set)

    if benchmark == "alfworld":
        _run_alfworld(args, output_dir, extra)
    elif benchmark == "swebench":
        _run_swebench(args, output_dir, extra)
    elif benchmark == "spreadsheetbench":
        _run_spreadsheetbench(args, output_dir, extra)
    elif benchmark == "bfcl":
        _run_bfcl(args, output_dir, extra)
    elif benchmark == "seal0":
        _run_seal0(args, output_dir, extra)


# ---------------------------------------------------------------------------
# ALFWorld
# ---------------------------------------------------------------------------

def _run_alfworld(args: argparse.Namespace, output_dir: str, extra: dict) -> None:
    """Dispatch to ALFWorld runner."""
    argv = [
        "--backend", args.provider,
        "--deployment", args.model,
        "--env_num", str(args.num_samples),
        "--test_times", str(args.num_rounds),
        "--max_workers", str(args.workers),
        "--output_dir", output_dir,
    ]
    if args.endpoint:
        argv += ["--azure_endpoint", args.endpoint]
    if args.api_key:
        argv += ["--api_key", args.api_key]
    if args.skill_set:
        argv += ["--skill_set", args.skill_set]

    # Pass through extra args
    argv += _emit_extras(extra)

    # Import and run
    from skilllens.inference.alfworld.run import parse_args as alf_parse, main as alf_main
    old_argv = sys.argv
    sys.argv = ["alfworld_run"] + argv
    try:
        alf_main()
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# SWE-bench
# ---------------------------------------------------------------------------

def _run_swebench(args: argparse.Namespace, output_dir: str, extra: dict) -> None:
    """Dispatch to SWE-bench runner, looping for num_rounds."""
    # SWE-bench uses litellm model format
    if args.provider == "azure" and not args.model.startswith("azure/"):
        model = f"azure/{args.model}"
    else:
        model = args.model

    # Set endpoint env var for litellm
    if args.endpoint:
        os.environ["AZURE_OPENAI_ENDPOINT"] = args.endpoint

    subset = _pop_single(extra, "subset", "lite")
    split = _pop_single(extra, "split", "dev")

    for round_idx in range(1, args.num_rounds + 1):
        round_output = output_dir if args.num_rounds == 1 else f"{output_dir}/round_{round_idx}"
        logger.info("SWE-bench round %d/%d → %s", round_idx, args.num_rounds, round_output)

        argv = [
            "--model", model,
            "--subset", subset,
            "--split", split,
            "--workers", str(args.workers),
            "--output", round_output,
        ]
        if args.num_samples and args.num_samples > 0:
            argv += ["--slice", f"0:{args.num_samples}"]
        if args.skill_set:
            argv += ["--skill-set", args.skill_set]

        argv += _emit_extras(extra)

        # SWE-bench uses typer, invoke via sys.argv
        old_argv = sys.argv
        sys.argv = ["swebench_run"] + argv
        try:
            from skilllens.inference.swebench.run import app
            app(standalone_mode=False)
        finally:
            sys.argv = old_argv

    if args.num_rounds > 1:
        logger.info("SWE-bench: %d rounds completed. Results in %s/round_*/", args.num_rounds, output_dir)


# ---------------------------------------------------------------------------
# SpreadsheetBench
# ---------------------------------------------------------------------------

def _run_spreadsheetbench(args: argparse.Namespace, output_dir: str, extra: dict) -> None:
    """Dispatch to SpreadsheetBench runner.

    Defaults match the layout produced by
    `skilllens/benchmarks/spreadsheetbench/README.md` setup (dataset under
    data/test_pool/spreadsheetbench/sb_root, sb-api on localhost:8081, the
    200 held-out ids in testset_v1.json).
    """
    setting = _pop_single(extra, "setting", "react_exec")
    data_root = _pop_single(extra, "data_root", "data/test_pool/spreadsheetbench/sb_root")
    dataset_json = _pop_single(
        extra, "dataset_json", "data/test_pool/spreadsheetbench/testset_v1.json"
    )
    code_exec_url = _pop_single(extra, "code_exec_url", "http://localhost:8081/execute")

    argv = [
        "--model", args.model,
        "--backend", args.provider,
        "--setting", setting,
        "--data_root", data_root,
        "--dataset_json", dataset_json,
        "--code_exec_url", code_exec_url,
        "--max_turn_num", str(args.max_turn_num),
        "--max_samples", str(args.num_samples),
        "--num_workers", str(args.workers),
        "--num_rounds", str(args.num_rounds),
        "--reasoning_effort", args.reasoning_effort,
        "--output_dir", output_dir,
    ]
    if args.endpoint:
        argv += ["--base_url", args.endpoint]
    if args.api_key:
        argv += ["--api_key", args.api_key]
    if args.skill_set:
        argv += ["--skill_set", args.skill_set]

    argv += _emit_extras(extra)

    old_argv = sys.argv
    sys.argv = ["spreadsheetbench_run"] + argv
    try:
        from skilllens.inference.spreadsheetbench.run import main as ssb_main
        ssb_main()
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# BFCL
# ---------------------------------------------------------------------------

def _run_bfcl(args: argparse.Namespace, output_dir: str, extra: dict) -> None:
    """Dispatch to BFCL runner.

    BFCL doesn't expose a native "first N" flag — it filters by an
    explicit id list. To make `--num-samples N` work uniformly, we read
    the include-ids-file (defaulting to the held-out test split when not
    given), slice the first N ids, and write a temp manifest that BFCL
    consumes via --include-ids-file.
    """
    include_file = _pop_single(extra, "include-ids-file", "")

    if args.num_samples and args.num_samples > 0:
        import json as _json
        import tempfile as _tempfile
        src = include_file or "data/test_pool/bfcl/testset_v1.json"
        ids = _json.loads(open(src, encoding="utf-8").read()).get("ids", [])
        ids = ids[: args.num_samples]
        fd, tmp = _tempfile.mkstemp(prefix="bfcl_include_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            _json.dump({"ids": ids}, f)
        include_file = tmp
        logger.info("BFCL: sliced %d ids from %s → %s", len(ids), src, tmp)

    argv = [
        "--model", args.model,
        "--num-rounds", str(args.num_rounds),
        "--num-threads", str(args.workers),
        "--output-dir", output_dir,
        "--reasoning-effort", args.reasoning_effort,
    ]
    if include_file:
        argv += ["--include-ids-file", include_file]
    if args.skill_set:
        argv += ["--skill-set", args.skill_set]

    # Pass through extra args
    argv += _emit_extras(extra)

    from skilllens.inference.bfcl.run import main as bfcl_main
    bfcl_main(argv)


# ---------------------------------------------------------------------------
# SEAL-0
# ---------------------------------------------------------------------------

def _run_seal0(args: argparse.Namespace, output_dir: str, extra: dict) -> None:
    """Dispatch to SEAL-0 (LiteResearcher) runner."""
    dataset = _pop_single(extra, "dataset", "")

    argv = [
        "--model", args.model,
        "--num-rounds", str(args.num_rounds),
        "--max-workers", str(args.workers),
        "--output-dir", output_dir,
    ]
    if args.num_samples and args.num_samples > 0:
        argv += ["--max-questions", str(args.num_samples)]
    if args.reasoning_effort:
        argv += ["--reasoning-effort", args.reasoning_effort]
    if dataset:
        argv += ["--dataset", dataset]
    if args.skill_set:
        argv += ["--skill-set", args.skill_set]

    # Pass through extra args
    argv += _emit_extras(extra)

    from skilllens.inference.seal0.run import main as seal0_main
    seal0_main(argv)
