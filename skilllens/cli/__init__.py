"""
CLI package for SkillLens.

Sub-commands
------------
    extract    Run skill extraction on trajectory data  (default)
    convert    Batch-convert raw benchmark trajectories to unified format
    infer      Run benchmark inference with optional skill augmentation
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (if present) before any config access
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)

from skilllens.cli.extract import _add_extract_args, cmd_extract
from skilllens.cli.convert import _add_convert_args, cmd_convert
from skilllens.cli.infer import _add_infer_args, cmd_infer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skilllens",
        description="SkillLens -- extract reusable skills from agent trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # extract (default)
    p_extract = subparsers.add_parser(
        "extract",
        help="Run skill extraction on trajectory data.",
    )
    _add_extract_args(p_extract)

    # convert
    p_convert = subparsers.add_parser(
        "convert",
        help="Batch-convert raw benchmark trajectories to unified format.",
    )
    _add_convert_args(p_convert)

    # infer
    p_infer = subparsers.add_parser(
        "infer",
        help="Run benchmark inference with optional skill augmentation.",
    )
    _add_infer_args(p_infer)

    # Also add extract args at top-level for backward compat (no sub-command)
    _add_extract_args(parser)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Logging
    verbose = getattr(args, "verbose", False)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    command = args.command

    # Backward compat: no sub-command -> extract
    if command is None:
        if args.input:
            command = "extract"
        else:
            parser.print_help()
            return

    if command == "extract":
        cmd_extract(args)
    elif command == "convert":
        cmd_convert(args)
    elif command == "infer":
        cmd_infer(args)
    else:
        parser.print_help()
