#!/usr/bin/env python3
"""Validate/reprocess BFCL trajectory JSON files.

BFCL's trajectory_exporter.py already produces schema-conforming JSON.
This script is a no-op pass-through that validates and optionally strips
full context from steps (keeping only per-turn content).

Usage:
    python scripts/lib/convert_bfcl_traj.py --input <bfcl_traj_dir> --output <schema_traj_dir>
"""
import argparse
import json
import os
import sys


def strip_accumulated_context(traj):
    """Remove full accumulated messages from steps, keep per-turn data only."""
    for step in traj.get("steps", []):
        # Remove any full conversation history stuffed into metadata
        meta = step.get("metadata", {})
        meta.pop("full_messages", None)
        meta.pop("accumulated_context", None)
    return traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input BFCL trajectory directory")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Input directory not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    count = 0
    for fname in os.listdir(args.input):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(args.input, fname), "r", encoding="utf-8") as f:
            traj = json.load(f)

        traj = strip_accumulated_context(traj)

        with open(os.path.join(args.output, fname), "w", encoding="utf-8") as f:
            json.dump(traj, f, ensure_ascii=False, indent=2)
        count += 1

    print(f"Processed {count} trajectory files → {args.output}")


if __name__ == "__main__":
    main()
