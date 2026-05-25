"""
Batch converter — convert a directory of raw benchmark trajectories
to unified Trajectory JSON files, optionally annotating outcomes from
an evaluation-result file.

Supports multiple benchmark formats:
  - swebench (mini-swe-agent-1.1)
  - spreadsheetbench (multi_react_exec)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from skilllens.loader.swe_bench_converter import convert_swe_bench_trajectory
from skilllens.loader.spreadsheet_bench_converter import (
    convert_spreadsheet_bench_trajectory,
)
from skilllens.loader.alfworld_converter import (
    convert_alfworld_trajectory,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_eval_results_swebench(eval_path: Path) -> tuple[set[str], set[str]]:
    """Load SWE-bench evaluation results and return ``(resolved_ids, unresolved_ids)``."""
    with open(eval_path, encoding="utf-8") as f:
        data = json.load(f)

    resolved = set(data.get("resolved_ids", []))
    unresolved = set(data.get("unresolved_ids", []))
    logger.info(
        "SWE-bench eval results: %d resolved, %d unresolved",
        len(resolved),
        len(unresolved),
    )
    return resolved, unresolved


def load_eval_results_spreadsheet(eval_path: Path) -> dict[str, dict]:
    """Load SpreadsheetBench evaluation results as a mapping from id to entry."""
    with open(eval_path, encoding="utf-8") as f:
        data = json.load(f)

    eval_map: dict[str, dict] = {}
    for entry in data:
        entry_id = str(entry.get("id", ""))
        if entry_id:
            eval_map[entry_id] = entry

    resolved = sum(
        1 for e in eval_map.values()
        if all(r == 1 for r in e.get("test_case_results", []))
        and e.get("hard_restriction", 0) == 1
    )
    logger.info(
        "SpreadsheetBench eval results: %d resolved, %d unresolved (total %d)",
        resolved,
        len(eval_map) - resolved,
        len(eval_map),
    )
    return eval_map


def find_trajectory_files(traj_dir: Path, benchmark: str = "swebench") -> list[Path]:
    """Find trajectory files under *traj_dir* based on benchmark type."""
    if benchmark == "spreadsheetbench":
        # SpreadsheetBench uses a single trajectory.jsonl file
        jsonl = traj_dir / "trajectory.jsonl"
        if jsonl.exists():
            return [jsonl]
        # Fallback: look for any .jsonl files
        files = sorted(traj_dir.glob("*.jsonl"))
        logger.info("Found %d .jsonl files in %s", len(files), traj_dir)
        return files
    elif benchmark == "alfworld":
        # ALFWorld: individual env_XXX.json files per environment
        files = sorted(traj_dir.glob("env_*.json"))
        logger.info("Found %d env_*.json files in %s", len(files), traj_dir)
        return files
    else:
        # SWE-bench: individual .traj.json files
        files = sorted(traj_dir.glob("**/*.traj.json"))
        logger.info("Found %d trajectory files in %s", len(files), traj_dir)
        return files


# ------------------------------------------------------------------
# SWE-bench batch conversion
# ------------------------------------------------------------------

def _convert_all_swebench(
    traj_dir: Path,
    eval_path: Path | None,
    output_dir: Path,
    model_name: str = "",
    filter_outcome: str | None = None,
) -> dict:
    """Convert all SWE-bench trajectories."""
    resolved_ids: set[str] = set()
    unresolved_ids: set[str] = set()
    if eval_path:
        resolved_ids, unresolved_ids = load_eval_results_swebench(eval_path)

    traj_files = find_trajectory_files(traj_dir, benchmark="swebench")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_found": len(traj_files),
        "converted": 0,
        "resolved": 0,
        "unresolved": 0,
        "error": 0,
        "failed": 0,
        "skipped_by_filter": 0,
    }

    for traj_file in traj_files:
        try:
            with open(traj_file, encoding="utf-8") as f:
                data = json.load(f)

            instance_id = data.get(
                "instance_id", traj_file.stem.replace(".traj", "")
            )

            if instance_id in resolved_ids:
                outcome = "resolved"
            elif instance_id in unresolved_ids:
                outcome = "unresolved"
            else:
                outcome = "error"

            if filter_outcome and outcome != filter_outcome:
                stats["skipped_by_filter"] += 1
                continue

            traj = convert_swe_bench_trajectory(
                data, source_path=str(traj_file), outcome=outcome,
            )

            if model_name:
                traj.agent = model_name

            out_file = output_dir / f"{instance_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(traj.model_dump(), f, indent=2, ensure_ascii=False)

            stats["converted"] += 1
            if outcome == "resolved":
                stats["resolved"] += 1
            elif outcome == "unresolved":
                stats["unresolved"] += 1
            else:
                stats["error"] += 1

        except Exception:
            logger.warning("Failed to convert %s", traj_file, exc_info=True)
            stats["failed"] += 1

    return stats


# ------------------------------------------------------------------
# SpreadsheetBench batch conversion
# ------------------------------------------------------------------

def _convert_all_spreadsheet(
    traj_dir: Path,
    eval_path: Path | None,
    output_dir: Path,
    model_name: str = "",
    filter_outcome: str | None = None,
) -> dict:
    """Convert all SpreadsheetBench trajectories from a trajectory.jsonl file."""
    eval_map: dict[str, dict] = {}
    if eval_path:
        eval_map = load_eval_results_spreadsheet(eval_path)

    traj_files = find_trajectory_files(traj_dir, benchmark="spreadsheetbench")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_found": 0,
        "converted": 0,
        "resolved": 0,
        "unresolved": 0,
        "error": 0,
        "failed": 0,
        "skipped_by_filter": 0,
    }

    for traj_file in traj_files:
        with open(traj_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                stats["total_found"] += 1

                try:
                    data = json.loads(line)
                    entry_id = str(data.get("id", f"line_{line_num}"))

                    traj = convert_spreadsheet_bench_trajectory(
                        data,
                        source_path=str(traj_file),
                        eval_map=eval_map,
                    )

                    if model_name:
                        traj.agent = model_name

                    outcome = traj.outcome

                    if filter_outcome and outcome != filter_outcome:
                        stats["skipped_by_filter"] += 1
                        continue

                    out_file = output_dir / f"{entry_id}.json"
                    with open(out_file, "w", encoding="utf-8") as fout:
                        json.dump(traj.model_dump(), fout, indent=2, ensure_ascii=False)

                    stats["converted"] += 1
                    if outcome == "resolved":
                        stats["resolved"] += 1
                    elif outcome == "unresolved":
                        stats["unresolved"] += 1
                    else:
                        stats["error"] += 1

                except Exception:
                    logger.warning(
                        "Failed to convert line %d in %s",
                        line_num, traj_file, exc_info=True,
                    )
                    stats["failed"] += 1

    return stats


# ------------------------------------------------------------------
# ALFWorld batch conversion
# ------------------------------------------------------------------

def _convert_all_alfworld(
    traj_dir: Path,
    eval_path: Path | None,
    output_dir: Path,
    model_name: str = "",
    filter_outcome: str | None = None,
) -> dict:
    """Convert all ALFWorld trajectories from a directory of env_XXX.json files."""
    traj_files = find_trajectory_files(traj_dir, benchmark="alfworld")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_found": len(traj_files),
        "converted": 0,
        "resolved": 0,
        "unresolved": 0,
        "error": 0,
        "failed": 0,
        "skipped_by_filter": 0,
    }

    for traj_file in traj_files:
        try:
            with open(traj_file, encoding="utf-8") as f:
                data = json.load(f)

            traj = convert_alfworld_trajectory(
                data,
                source_path=str(traj_file),
            )

            if model_name:
                traj.agent = model_name

            outcome = traj.outcome

            if filter_outcome and outcome != filter_outcome:
                stats["skipped_by_filter"] += 1
                continue

            env_id_str = f"env_{data['env_id']:03d}"
            out_file = output_dir / f"{env_id_str}.json"
            with open(out_file, "w", encoding="utf-8") as fout:
                json.dump(traj.model_dump(), fout, indent=2, ensure_ascii=False)

            stats["converted"] += 1
            if outcome == "resolved":
                stats["resolved"] += 1
            elif outcome == "unresolved":
                stats["unresolved"] += 1
            else:
                stats["error"] += 1

        except Exception:
            logger.warning("Failed to convert %s", traj_file, exc_info=True)
            stats["failed"] += 1

    return stats


# ------------------------------------------------------------------
# BFCL batch conversion
# ------------------------------------------------------------------

def _load_bfcl_eval_outcomes(traj_dir: Path) -> dict[str, bool]:
    """Load BFCL evaluation outcomes from score and result files.

    Score files only contain FAILED entries (valid=False). Entries present in
    result files but absent from score files are PASSED (resolved).

    Returns mapping from entry id to correctness (True=resolved, False=unresolved).
    """
    # Collect all failed IDs from score files
    failed_ids: set[str] = set()
    score_files = sorted(traj_dir.glob("**/score/**/*_score.json"))
    for sf in score_files:
        try:
            with open(sf, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i == 0:
                        continue  # skip summary line
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    eid = entry.get("id", "")
                    if eid:
                        failed_ids.add(eid)
        except Exception:
            logger.warning("Failed to read score file %s", sf, exc_info=True)

    # Collect all IDs from result files
    all_ids: set[str] = set()
    result_files = sorted(traj_dir.glob("**/result/**/*_result.json"))
    for rf in result_files:
        try:
            with open(rf, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    eid = d.get("id", "")
                    if eid:
                        all_ids.add(eid)
        except Exception:
            logger.warning("Failed to read result file %s", rf, exc_info=True)

    passed_ids = all_ids - failed_ids

    # Build outcome map
    outcome_map: dict[str, bool] = {}
    for eid in passed_ids:
        outcome_map[eid] = True
    for eid in failed_ids:
        outcome_map[eid] = False

    if outcome_map:
        correct = sum(1 for v in outcome_map.values() if v)
        logger.info("BFCL eval outcomes: %d entries (%d correct, %d failed)",
                    len(outcome_map), correct, len(failed_ids))
    return outcome_map


def _convert_all_bfcl(
    traj_dir: Path,
    output_dir: Path,
    model_name: str = "",
) -> dict:
    """Convert BFCL trajectory JSON files to the unified experience-pool schema.

    The per-instance ``<test_id>.json`` files emitted by the BFCL
    trajectory exporter are already schema-conforming (id, steps,
    final_answer, …). This function:
      1. loads each one directly,
      2. skips ``*_err.json`` sidecars (Azure content-filter / errored
         turns that never produced a usable trajectory),
      3. backfills ``outcome`` + ``reward`` from the BFCL score files
         (resolved iff entry is absent from the failed list),
      4. optionally overrides the ``agent`` field with ``model_name``.
    """
    # Load evaluation outcomes from the score files (failed_ids vs all_ids)
    score_map = _load_bfcl_eval_outcomes(traj_dir)

    # Find trajectory files: top-level *.json under traj_dir, excluding the
    # score/ and result/ sub-trees and the *_err.json sidecars.
    files = sorted(
        p for p in traj_dir.glob("**/*.json")
        if "score" not in p.parts
        and "result" not in p.parts
        and not p.name.endswith("_err.json")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_found": 0,
        "converted": 0,
        "resolved": 0,
        "unresolved": 0,
        "error": 0,
        "failed": 0,
        "skipped_by_filter": 0,
    }

    for traj_file in files:
        stats["total_found"] += 1
        try:
            with open(traj_file, encoding="utf-8") as f:
                traj = json.load(f)
        except Exception:
            logger.warning("Failed to read %s", traj_file, exc_info=True)
            stats["failed"] += 1
            continue

        if not isinstance(traj, dict) or "steps" not in traj:
            logger.warning("Skipping %s: not a trajectory-schema JSON", traj_file.name)
            stats["failed"] += 1
            continue

        traj_id = traj.get("id") or traj.get("task_id") or traj_file.stem

        # Override agent name if caller passed one (raw exporter writes
        # the FC model tag, e.g. ``gpt-5.4-FC``).
        if model_name:
            traj["agent"] = model_name

        # Backfill outcome/reward from score map. If the id is not in
        # the score map (e.g. eval not yet run, or content-filter cases
        # that never made it into result files), leave outcome unset.
        if traj_id in score_map:
            is_correct = score_map[traj_id]
            traj["outcome"] = "resolved" if is_correct else "unresolved"
            traj["reward"] = 1.0 if is_correct else 0.0

        out_file = output_dir / f"{traj_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(traj, f, ensure_ascii=False, indent=2)

        stats["converted"] += 1
        outcome = traj.get("outcome")
        if outcome == "resolved":
            stats["resolved"] += 1
        elif outcome == "unresolved":
            stats["unresolved"] += 1
        else:
            stats["error"] += 1

    return stats


# ------------------------------------------------------------------
# SEAL-0 batch conversion
# ------------------------------------------------------------------

def _convert_all_seal0(
    traj_dir: Path,
    output_dir: Path,
    model_name: str = "",
) -> dict:
    """Convert SEAL-0 (LiteResearcher) JSONL output to per-question trajectory JSON."""
    from skilllens.loader.seal0_converter import convert_record, sanitize_filename

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all JSONL files
    jsonl_files = sorted(traj_dir.glob("**/*.jsonl"))

    stats = {
        "total_found": 0,
        "converted": 0,
        "resolved": 0,
        "unresolved": 0,
        "error": 0,
        "failed": 0,
        "skipped_by_filter": 0,
    }

    records: dict[str, dict] = {}
    for fpath in jsonl_files:
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = record.get("question", "")
                if q:
                    records[q] = record

    stats["total_found"] = len(records)

    for question, record in records.items():
        try:
            tid, traj = convert_record(
                record,
                model=model_name or "unknown",
                benchmark="seal0",
                dataset_name="sealqa_seal_0",
            )
            fname = sanitize_filename(tid) + ".json"
            out_file = output_dir / fname
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(traj, f, ensure_ascii=False, indent=2)
            stats["converted"] += 1
        except Exception:
            logger.warning("Failed to convert seal0 record: %s...", question[:50], exc_info=True)
            stats["failed"] += 1

    return stats


# ------------------------------------------------------------------
# Unified entry-point
# ------------------------------------------------------------------

def convert_all(
    traj_dir: Path,
    eval_path: Path | None,
    output_dir: Path,
    model_name: str = "",
    filter_outcome: str | None = None,
    benchmark: str = "swebench",
) -> dict:
    """Convert all trajectories under *traj_dir* and write unified JSON.

    Parameters
    ----------
    traj_dir : Path
        Directory containing raw trajectory files.
    eval_path : Path | None
        Evaluation results JSON — used for outcome annotation.
    output_dir : Path
        Where to write the unified ``{instance_id}.json`` files.
    model_name : str
        If given, overrides the ``agent`` field in every output trajectory.
    filter_outcome : str | None
        ``"resolved"`` / ``"unresolved"`` — skip trajectories that don't
        match.  ``None`` keeps all.
    benchmark : str
        Benchmark type: ``"swebench"`` or ``"spreadsheetbench"``.

    Returns
    -------
    dict
        Conversion statistics.
    """
    if benchmark == "spreadsheetbench":
        return _convert_all_spreadsheet(
            traj_dir, eval_path, output_dir, model_name, filter_outcome,
        )
    elif benchmark == "alfworld":
        return _convert_all_alfworld(
            traj_dir, eval_path, output_dir, model_name, filter_outcome,
        )
    elif benchmark == "bfcl":
        return _convert_all_bfcl(
            traj_dir, output_dir, model_name,
        )
    elif benchmark == "seal0":
        return _convert_all_seal0(
            traj_dir, output_dir, model_name,
        )
    else:
        # Default to swebench
        return _convert_all_swebench(
            traj_dir, eval_path, output_dir, model_name, filter_outcome,
        )


def print_convert_summary(stats: dict, output_dir: Path) -> None:
    """Print a human-readable conversion summary to stdout."""
    print("\n" + "=" * 50)
    print("Conversion Summary")
    print("=" * 50)
    print(f"  Total files found:      {stats['total_found']}")
    print(f"  Successfully converted: {stats['converted']}")
    print(f"    - Resolved:           {stats['resolved']}")
    print(f"    - Unresolved:         {stats['unresolved']}")
    print(f"    - Error:              {stats['error']}")
    print(f"  Skipped (filter):       {stats['skipped_by_filter']}")
    print(f"  Failed:                 {stats['failed']}")
    print(f"  Output directory:       {output_dir}")
    print("=" * 50)


# ------------------------------------------------------------------
# Manifest
# ------------------------------------------------------------------

_MANIFEST_NAME = "manifest.json"


def update_manifest(
    data_root: Path,
    benchmark: str,
    model_name: str,
    stats: dict,
    source_dir: str | None = None,
    eval_result: str | None = None,
) -> Path:
    """Create or update ``data/trajectories/manifest.json``.

    The manifest records every converted dataset so that downstream tools
    (loader, extraction CLI) can discover available data without
    scanning the filesystem.

    Returns the path to the manifest file.
    """
    manifest_path = data_root / _MANIFEST_NAME

    # Load existing manifest
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        data_root.mkdir(parents=True, exist_ok=True)
        manifest = {"datasets": {}}

    key = f"{benchmark}/{model_name}"
    manifest["datasets"][key] = {
        "benchmark": benchmark,
        "model": model_name,
        "path": f"{benchmark}/{model_name}",
        "total": stats.get("converted", 0),
        "resolved": stats.get("resolved", 0),
        "unresolved": stats.get("unresolved", 0),
        "unknown_outcome": stats.get("error", 0),
        "source_dir": source_dir or "",
        "eval_result": eval_result or "",
        "converted_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("Manifest updated: %s", manifest_path)
    return manifest_path
