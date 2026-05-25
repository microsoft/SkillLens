"""CLI sub-command: extract."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from skilllens.config import build_config
from skilllens.client.openai_client import LLMClient
from skilllens.loader.trajectory_loader import TrajectoryLoader

logger = logging.getLogger("skilllens")


def _add_extract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to YAML configuration file.")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path to trajectory file or directory (overrides config).")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (overrides config).")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a config value (e.g. --set llm.model=gpt-4o).")
    parser.add_argument("--filter-outcome", choices=["resolved", "unresolved"], default=None,
                        help="Only use trajectories with this outcome.")
    parser.add_argument("--max-trajectories", type=int, default=0,
                        help="Limit number of trajectories (0 = all).")
    parser.add_argument("--include-feedback", action="store_true", default=None,
                        help="Include outcome/reward in extraction prompt.")
    parser.add_argument("--no-feedback", dest="include_feedback", action="store_false",
                        help="Exclude outcome/reward from extraction prompt.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose (DEBUG) logging.")


async def _run_extract(args: argparse.Namespace) -> None:
    """Execute the extraction pipeline."""
    # Build config
    cli_overrides: dict = {}
    for item in args.set:
        if "=" in item:
            k, v = item.split("=", 1)
            cli_overrides[k] = v
    if args.input:
        cli_overrides["input.path"] = args.input
    if args.output:
        cli_overrides["output.path"] = args.output
    # Apply --include-feedback / --no-feedback
    if getattr(args, "include_feedback", None) is not None:
        cli_overrides["extraction.include_feedback"] = str(args.include_feedback).lower()

    cfg = build_config(yaml_path=args.config, cli_overrides=cli_overrides)
    logger.debug("Resolved config:\n%s", cfg.model_dump_json(indent=2))

    # Load trajectories
    input_path = Path(cfg.input.path)
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    if input_path.is_file():
        # Check if the file is an experience_pool bundle ({trajectories: [...]})
        with open(input_path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "trajectories" in raw and isinstance(raw["trajectories"], list):
            # Input pool format — load each trajectory from the embedded list
            from skilllens.schema.trajectory import TrajectorySet
            trajectories = []
            for traj_data in raw["trajectories"]:
                trajectories.append(TrajectoryLoader.from_generic(traj_data))
            trajectory_set = TrajectorySet(trajectories=trajectories, source=str(input_path))
        elif isinstance(raw, list):
            # Flat list of trajectories — e.g. [traj1, traj2, ...]
            from skilllens.schema.trajectory import TrajectorySet
            trajectories = []
            for traj_data in raw:
                trajectories.append(TrajectoryLoader.from_generic(traj_data))
            trajectory_set = TrajectorySet(trajectories=trajectories, source=str(input_path))
        else:
            # Single trajectory file
            traj = TrajectoryLoader.load_from_file(input_path)
            from skilllens.schema.trajectory import TrajectorySet
            trajectory_set = TrajectorySet(trajectories=[traj], source=str(input_path))
    else:
        trajectory_set = TrajectoryLoader.load_from_directory(input_path)

    logger.info("Loaded %d trajectories from %s",
                len(trajectory_set.trajectories), cfg.input.path)

    # Optional outcome filter
    if args.filter_outcome:
        trajectory_set.trajectories = [
            t for t in trajectory_set.trajectories
            if t.outcome == args.filter_outcome
        ]
        logger.info("Filtered to %d %s trajectories",
                     len(trajectory_set.trajectories), args.filter_outcome)

    # Optional limit
    if args.max_trajectories > 0:
        trajectory_set.trajectories = trajectory_set.trajectories[:args.max_trajectories]
        logger.info("Limiting to %d trajectories", len(trajectory_set.trajectories))

    if not trajectory_set.trajectories:
        logger.error("No trajectories loaded — nothing to do.")
        sys.exit(1)

    # Build LLM client
    client = LLMClient(cfg.llm)

    # Instantiate extraction method
    if cfg.extraction.method == "sequential":
        from skilllens.extraction.sequential import SequentialExtraction
        extractor = SequentialExtraction(
            client,
            max_skills=cfg.extraction.max_skills,
            max_skill_chars=cfg.extraction.max_skill_chars,
            max_total_chars=cfg.extraction.max_total_chars,
            include_feedback=cfg.extraction.include_feedback,
            max_tool_rounds=cfg.extraction.max_tool_rounds,
            shuffle_trajectories=cfg.extraction.shuffle_trajectories,
            snapshot_interval=cfg.extraction.snapshot_interval,
            output_dir=cfg.output.path,
        )
    elif cfg.extraction.method == "parallel":
        from skilllens.extraction.parallel import ParallelExtraction
        extractor = ParallelExtraction(
            client,
            batch_size=cfg.extraction.batch_size,
            merge_group_size=cfg.extraction.merge_group_size,
            max_concurrency=cfg.extraction.max_concurrency,
            max_skills=cfg.extraction.max_skills,
            max_skill_chars=cfg.extraction.max_skill_chars,
            max_total_chars=cfg.extraction.max_total_chars,
            include_feedback=cfg.extraction.include_feedback,
            max_tool_rounds=cfg.extraction.max_tool_rounds,
            shuffle_trajectories=cfg.extraction.shuffle_trajectories,
            max_modes_per_trajectory=cfg.extraction.max_modes_per_trajectory,
            meta_skill_guidance=cfg.extraction.meta_skill_guidance,
            output_dir=cfg.output.path,
        )
    else:
        logger.error("Unknown extraction method: %s", cfg.extraction.method)
        sys.exit(1)

    # ----- Determine output directory -----
    input_source = str(input_path)
    input_tag = Path(input_source).stem
    input_tag = input_tag.replace("/", "_").replace(" ", "_")

    extractor_model_slug = cfg.llm.model.replace("/", "_").replace(" ", "_")
    timestamp = datetime.now().strftime("%m%d_%H%M")

    method = cfg.extraction.method
    if method == "parallel":
        bs = cfg.extraction.batch_size
        gs = cfg.extraction.merge_group_size
        method_tag = f"parallel_b{bs}_g{gs}"
    else:
        method_tag = method

    run_name = f"{method_tag}_{extractor_model_slug}_{input_tag}_{timestamp}"
    benchmark = cfg.input.benchmark
    run_dir = Path(cfg.output.path) / benchmark / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler so logs are also written to run_dir/run.log
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)

    # Update extractor's output_dir so extraction_log / store_history land here
    if hasattr(extractor, "output_dir"):
        extractor.output_dir = str(run_dir)

    # Run extraction
    logger.info("Starting extraction (method=%s) …", cfg.extraction.method)
    logger.info("Run directory: %s", run_dir)
    skill_set = await extractor.extract(trajectory_set)
    logger.info("Extraction complete — %d skills produced.", len(skill_set.skills))

    # ----- Attach input metadata -----
    skill_set.metadata["input"] = {
        "path": input_source,
        "filter_outcome": args.filter_outcome,
        "max_trajectories": args.max_trajectories,
        "trajectory_count_used": len(trajectory_set.trajectories),
    }

    # ----- Save skill_set.json -----
    out_file = run_dir / "skill_set.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(skill_set.model_dump(), f, indent=2, ensure_ascii=False)
    logger.info("Skill set saved to %s", out_file)


def cmd_extract(args: argparse.Namespace) -> None:
    asyncio.run(_run_extract(args))
