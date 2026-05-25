"""
SEAL-0 inference runner — web research agent evaluation.

Wraps the embedded LiteResearcher agent to run parallel inference on SEAL-0
questions, with optional skill injection via --skill_set.

Usage:
    # Baseline
    python -m skilllens infer --benchmark seal0 --model gpt-5.4 --num-samples 50

    # With skill augmentation
    python -m skilllens infer --benchmark seal0 --model gpt-5.4 \\
        --skill-set output/skill_set.json

Prerequisites:
    - Search server running: python -m skilllens.benchmarks.seal0.search_server
    - Browser server running: python -m skilllens.benchmarks.seal0.browser_server
    - Dataset JSONL file (e.g. sealqa_seal_0.jsonl)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger("skilllens.inference.seal0")


def _render_skill_to_file(skill_set_path: str, skill_index: int = 0) -> str:
    """Render a skill from skill_set.json to a temporary text file for injection."""
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
    parser = argparse.ArgumentParser(description="SEAL-0 (LiteResearcher) inference")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (e.g. gpt-5.4)")
    _default_dataset = str(Path(__file__).resolve().parent.parent.parent
                           / "benchmarks" / "seal0" / "data" / "sealqa_seal_0.jsonl")
    parser.add_argument("--dataset", type=str, default=_default_dataset,
                        help="Path to dataset file (JSONL or JSON)")
    parser.add_argument("--num-rounds", type=int, default=1,
                        help="Number of rollouts per question (default: 1)")
    parser.add_argument("--max-workers", type=int, default=20,
                        help="Number of parallel workers (default: 20)")
    parser.add_argument("--max-questions", type=int, default=0,
                        help="Cap the dataset at first N questions (0 = full dataset, the default). "
                             "Useful for smoke tests.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--presence-penalty", type=float, default=1.1)
    parser.add_argument("--skill-set", type=str, default="",
                        help="Path to skill_set.json for skill injection")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Output directory for results")
    parser.add_argument("--reasoning-effort", type=str, default="",
                        help="Reasoning effort (low/medium/high)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Build output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        import time
        safe_model = args.model.replace("/", "_")
        skill_tag = "_skill" if args.skill_set else ""
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"inference_output/seal0/{safe_model}{skill_tag}_{ts}"

    os.makedirs(output_dir, exist_ok=True)

    # Set up environment for the agent
    env_updates = {}
    if args.skill_set:
        # Render skill from JSON to plain text for injection
        skill_text_file = _render_skill_to_file(args.skill_set)
        env_updates["SKILL_INJECT_FILE"] = skill_text_file
    if args.reasoning_effort:
        env_updates["REASONING_EFFORT"] = args.reasoning_effort

    # Default the agent's LLM provider to Azure when an Azure endpoint is
    # available in the environment. The embedded LiteResearcher agent reads
    # API_PROVIDER + AZURE_ENDPOINT (not the AZURE_OPENAI_* names) so we
    # translate here, leaving any explicit user override alone.
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    if azure_endpoint:
        env_updates.setdefault("API_PROVIDER", "azure")
        env_updates.setdefault("AZURE_ENDPOINT", azure_endpoint)
        api_ver = os.environ.get("AZURE_OPENAI_API_VERSION") or os.environ.get("AZURE_API_VERSION")
        if api_ver:
            env_updates.setdefault("AZURE_API_VERSION", api_ver)

    # Default the judge model to the inference model so judge_answer is
    # actually invoked (otherwise it short-circuits with "not configured").
    env_updates.setdefault("SUMMARY_MODEL_NAME", args.model)

    for k, v in env_updates.items():
        os.environ[k] = v

    logger.info("SEAL-0 inference: model=%s, dataset=%s, output=%s",
                args.model, args.dataset, output_dir)

    # Build argv for run_inference's argparse
    run_argv = [
        "--model", args.model,
        "--dataset", args.dataset,
        "--output", output_dir,
        "--flat_output",
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--presence_penalty", str(args.presence_penalty),
        "--max_workers", str(args.max_workers),
        "--roll_out_count", str(args.num_rounds),
    ]
    if args.max_questions and args.max_questions > 0:
        run_argv += ["--max_questions", str(args.max_questions)]
    if args.resume:
        run_argv.append("--resume")

    # Run via the embedded run_inference module
    old_argv = sys.argv
    sys.argv = ["seal0_run"] + run_argv
    try:
        # Import and execute the run_inference __main__ block
        import runpy
        runpy.run_module(
            "skilllens.benchmarks.seal0.run_inference",
            run_name="__main__",
            alter_sys=True,
        )
    finally:
        sys.argv = old_argv

    logger.info("Done. Results in %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
