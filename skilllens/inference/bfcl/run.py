"""
BFCL v4 inference runner — multi-turn function calling evaluation.

Wraps the embedded bfcl_eval package to run generate + evaluate for
multi-turn categories, with optional skill injection via --skill_set.

Usage:
    # Baseline
    python -m skilllens infer --benchmark bfcl --model gpt-5.4 --num-rounds 3

    # With skill augmentation
    python -m skilllens infer --benchmark bfcl --model gpt-5.4 \\
        --skill-set output/skill_set.json

Prerequisites:
    - The BFCL test data is fetched automatically by the embedded bfcl_eval package.
    - Azure Managed Identity or OPENAI_API_KEY must be configured.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

logger = logging.getLogger("skilllens.inference.bfcl")

# Path to the embedded bfcl_eval package
BFCL_EVAL_ROOT = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "bfcl"

# Repo-root .env (where AZURE_DEPLOYMENT_MAP / AZURE_CLIENT_ID live).
# Loaded eagerly because the bfcl subprocess loads its .env from
# BFCL_PROJECT_ROOT/.env (which is the per-run trajectory dir) and therefore
# never sees the real credentials. We seed os.environ here so they propagate
# through env=os.environ.copy() into the child process.
REPO_ROOT_ENV = Path(__file__).resolve().parents[3] / ".env"
if load_dotenv is not None and REPO_ROOT_ENV.exists():
    load_dotenv(dotenv_path=str(REPO_ROOT_ENV), override=False)

# Multi-turn categories used in SkillLens experiments
CATEGORIES = [
    "multi_turn_long_context",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
]


def _render_skill_to_file(skill_set_path: str, skill_index: int = 0) -> str:
    """Render a skill from skill_set.json to a temporary text file for injection.

    Returns the path to the temporary file containing the rendered skill text.
    """
    with open(skill_set_path, encoding="utf-8") as f:
        data = json.load(f)

    skills = data.get("skills", [])
    if skill_index < 0 or skill_index >= len(skills):
        raise ValueError(f"Skill index {skill_index} out of range (0..{len(skills)-1})")

    skill = skills[skill_index]
    parts = [f"# Skill: {skill['name']}"]
    if skill.get("description"):
        parts.append(f"\n{skill['description']}")
    if skill.get("body"):
        parts.append(f"\n{skill['body']}")

    text = "\n".join(parts)

    fd, path = tempfile.mkstemp(suffix=".txt", prefix="skill_inject_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)

    logger.info("Skill rendered to %s (%d chars)", path, len(text))
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="BFCL v4 multi-turn inference")
    parser.add_argument("--model", type=str, required=True,
                        help="Model deployment name (e.g. gpt-5.4)")
    parser.add_argument("--num-rounds", type=int, default=3,
                        help="Number of evaluation rounds (default: 3)")
    parser.add_argument("--num-threads", type=int, default=10,
                        help="Number of parallel threads (default: 10)")
    parser.add_argument("--skill-set", type=str, default="",
                        help="Path to skill_set.json for skill-augmented runs")
    parser.add_argument("--skill-index", type=int, default=0,
                        help="Index of skill to inject (default: 0)")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Output directory for trajectories")
    parser.add_argument("--test-category", type=str, default=",".join(CATEGORIES),
                        help="Test categories (comma-separated)")
    parser.add_argument("--reasoning-effort", type=str, default="",
                        help="Reasoning effort (low/medium/high)")
    parser.add_argument("--include-ids-file", type=str, default="",
                        help="JSON file with {ids: [...]} to include")
    parser.add_argument("--exclude-ids-file", type=str, default="",
                        help="JSON file with {ids: [...]} to exclude")
    return parser.parse_args(argv)


def run_bfcl_generate(args, round_num: int, traj_dir: str):
    """Run bfcl generate for one round using the embedded package."""
    # Build model name (append -FC for function calling)
    bfcl_model = f"{args.model}-FC"

    # Construct the command to run bfcl_eval as a module
    cmd = [
        sys.executable, "-m", "skilllens.benchmarks.bfcl.bfcl_eval",
        "generate",
        "--model", bfcl_model,
        "--test-category", args.test_category,
        "--num-threads", str(args.num_threads),
        "--export-trajectory",
        "--trajectory-dir", traj_dir,
        "--test-round", str(round_num),
        "-o",  # allow overwrite
    ]

    if args.exclude_ids_file:
        cmd += ["--exclude-ids-file", args.exclude_ids_file]
    if args.include_ids_file:
        cmd += ["--include-ids-file", args.include_ids_file]

    # Skill injection via environment variable
    env = os.environ.copy()
    env["BFCL_PROJECT_ROOT"] = os.path.abspath(traj_dir)
    if args.skill_set:
        # Render skill from JSON to plain text for injection
        skill_text_file = _render_skill_to_file(args.skill_set, args.skill_index)
        env["SKILL_INJECT_FILE"] = skill_text_file

    if args.reasoning_effort:
        env["REASONING_EFFORT"] = args.reasoning_effort

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, env=env, capture_output=False)
    return result.returncode == 0


def run_bfcl_evaluate(args, traj_dir: str):
    """Run bfcl evaluate on the results."""
    bfcl_model = f"{args.model}-FC"

    cmd = [
        sys.executable, "-m", "skilllens.benchmarks.bfcl.bfcl_eval",
        "evaluate",
        "--model", bfcl_model,
        "--test-category", args.test_category,
        "--partial-eval",
    ]

    env = os.environ.copy()
    env["BFCL_PROJECT_ROOT"] = os.path.abspath(traj_dir)

    logger.info("Running evaluation: %s", " ".join(cmd))
    result = subprocess.run(cmd, env=env, capture_output=False)
    return result.returncode == 0


def main(argv=None):
    args = parse_args(argv)

    # Build output directory
    if args.output_dir:
        output_base = args.output_dir
    else:
        safe_model = args.model.replace("/", "_")
        effort_tag = f"-{args.reasoning_effort}" if args.reasoning_effort else ""
        skill_tag = "-skill" if args.skill_set else ""
        output_base = f"inference_output/bfcl/{safe_model}{effort_tag}{skill_tag}"

    os.makedirs(output_base, exist_ok=True)

    logger.info("BFCL v4 inference: model=%s, rounds=%d, output=%s",
                args.model, args.num_rounds, output_base)

    for round_idx in range(args.num_rounds):
        traj_dir = os.path.join(output_base, f"round_{round_idx}")
        os.makedirs(traj_dir, exist_ok=True)

        logger.info("Round %d/%d → %s", round_idx + 1, args.num_rounds, traj_dir)
        ok = run_bfcl_generate(args, round_idx, traj_dir)
        if not ok:
            logger.error("Round %d generate failed", round_idx)
            continue

        # Run evaluation per round (PROJECT_ROOT must match generate)
        run_bfcl_evaluate(args, traj_dir)
    logger.info("Done. Results in %s", output_base)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
