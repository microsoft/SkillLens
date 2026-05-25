"""CLI sub-command: convert."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger("skilllens")

# Default root for experience pool data
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent   # SkillLens-code/
_POOL_ROOT = _REPO_ROOT / "data" / "experience_pool"


def _add_convert_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trajectory-dir", type=str, required=True,
                        help="Path to raw inference output directory.")
    parser.add_argument("--eval-result", type=str, default=None,
                        help="Path to evaluation results JSON (for outcome annotation).")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output experience pool JSON path. Defaults to "
                             "data/experience_pool/{benchmark}/{model-name}.json.")
    parser.add_argument("--benchmark", type=str, default="swebench",
                        help="Benchmark name (default: swebench).")
    parser.add_argument("--model-name", type=str, default="",
                        help="Model name to set in the agent field.")
    parser.add_argument("--filter-outcome", choices=["resolved", "unresolved"], default=None,
                        help="Only include trajectories with this outcome.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging.")


def cmd_convert(args: argparse.Namespace) -> None:
    from skilllens.loader.batch_converter import convert_all, print_convert_summary

    traj_dir = Path(args.trajectory_dir)
    eval_path = Path(args.eval_result) if args.eval_result else None

    if not traj_dir.exists():
        logger.error("Trajectory directory does not exist: %s", traj_dir)
        sys.exit(1)
    if eval_path and not eval_path.exists():
        logger.error("Evaluation result file does not exist: %s", eval_path)
        sys.exit(1)

    # Convert to unified schema in a temp dir, then package into pool JSON
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        stats = convert_all(
            traj_dir=traj_dir,
            eval_path=eval_path,
            output_dir=tmp,
            model_name=args.model_name,
            filter_outcome=args.filter_outcome,
            benchmark=args.benchmark,
        )

        print_convert_summary(stats, tmp)

        if stats["failed"] > 0:
            logger.warning("%d trajectories failed to convert.", stats["failed"])

        # Collect all converted trajectories
        trajs = []
        for f in sorted(tmp.glob("*.json")):
            trajs.append(json.loads(f.read_text(encoding="utf-8")))

        if not trajs:
            logger.error("No trajectories converted. Aborting.")
            sys.exit(1)

        # Build experience pool
        pool = {
            "trajectories": trajs,
            "source": str(traj_dir),
            "metadata": {
                "benchmark": args.benchmark,
                "model": args.model_name or traj_dir.name,
                "count": len(trajs),
                "resolved": sum(1 for t in trajs if t.get("outcome") == "resolved"),
                "unresolved": sum(1 for t in trajs if t.get("outcome") == "unresolved"),
            },
        }

        # Determine output path
        if args.output:
            out_path = Path(args.output)
        else:
            model_slug = args.model_name or traj_dir.name
            out_path = _POOL_ROOT / args.benchmark / f"{model_slug}.json"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(pool, indent=2, ensure_ascii=False), encoding="utf-8")

        resolved = pool["metadata"]["resolved"]
        unresolved = pool["metadata"]["unresolved"]
        logger.info(
            "Experience pool: %d trajectories (%d resolved, %d unresolved) → %s",
            len(trajs), resolved, unresolved, out_path,
        )
