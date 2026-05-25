"""
Trajectory loader — reads trajectory data from files/directories and converts
various formats (ATIF-v1.5, generic JSON) into the unified Trajectory schema.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from skilllens.loader.swe_bench_converter import convert_swe_bench_trajectory
from skilllens.loader.spreadsheet_bench_converter import (
    convert_spreadsheet_bench_trajectory,
)

from skilllens.schema.trajectory import Step, Trajectory, TrajectorySet

logger = logging.getLogger(__name__)

# Mapping from ATIF-v1.5 source field to our unified role enum
_ATIF_ROLE_MAP: dict[str, str] = {
    "system": "system",
    "user": "user",
    "agent": "agent",
    "tool": "tool",
}


class TrajectoryLoader:
    """Load and convert trajectory data from various sources."""

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_from_file(path: str | Path) -> Trajectory:
        """Load a single trajectory JSON file (auto-detecting format)."""
        path = Path(path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Detect SWE-bench mini-swe-agent format
        if data.get("trajectory_format", "").startswith("mini-swe-agent"):
            return TrajectoryLoader.from_swe_bench(data, source_path=str(path))
        # Detect SpreadsheetBench format (has 'trajectory' list + 'conversation' list)
        elif (
            isinstance(data.get("trajectory"), list)
            and isinstance(data.get("conversation"), list)
            and "instruction" in data
        ):
            return TrajectoryLoader.from_spreadsheet_bench(data, source_path=str(path))
        elif data.get("schema_version", "").startswith("ATIF"):
            return TrajectoryLoader.from_atif(data, source_path=str(path))
        return TrajectoryLoader.from_generic(data, source_path=str(path))

    @staticmethod
    def load_from_directory(
        path: str | Path,
        pattern: str | None = None,
    ) -> TrajectorySet:
        """Recursively load all matching trajectory files from *path*.

        If *pattern* is not given, tries multiple glob patterns to cover
        SWE-bench (``*.traj.json``), unified (``*.json``), and legacy
        (``*trajectory*.json``) formats.
        """
        root = Path(path)

        if pattern is not None:
            files = sorted(root.glob(pattern))
        else:
            # Try multiple patterns and de-duplicate
            seen: set[Path] = set()
            files: list[Path] = []
            for pat in ("**/*.traj.json", "**/*trajectory*.json", "**/*.json"):
                for fp in sorted(root.glob(pat)):
                    if fp not in seen:
                        seen.add(fp)
                        files.append(fp)

        logger.info("Found %d trajectory files in %s", len(files), root)

        trajectories: list[Trajectory] = []
        for fp in files:
            try:
                traj = TrajectoryLoader.load_from_file(fp)
                trajectories.append(traj)
            except Exception:
                logger.warning("Failed to load %s", fp, exc_info=True)

        return TrajectorySet(
            trajectories=trajectories,
            source=str(root),
        )

    # ------------------------------------------------------------------
    # Format converters
    # ------------------------------------------------------------------

    @staticmethod
    def from_swe_bench(data: dict, *, source_path: str = "", outcome: str = "") -> Trajectory:
        """Convert a SWE-bench (mini-swe-agent) JSON object to the unified Trajectory model."""
        return convert_swe_bench_trajectory(data, source_path=source_path, outcome=outcome)

    @staticmethod
    def from_spreadsheet_bench(data: dict, *, source_path: str = "") -> Trajectory:
        """Convert a SpreadsheetBench JSON object to the unified Trajectory model."""
        return convert_spreadsheet_bench_trajectory(data, source_path=source_path)

    @staticmethod
    def from_atif(data: dict, *, source_path: str = "") -> Trajectory:
        """Convert an ATIF-v1.5 JSON object to the unified Trajectory model.

        ATIF-v1.5 field mapping
        -----------------------
        - ``steps[i].source``   → ``role`` (via ``_ATIF_ROLE_MAP``)
        - ``steps[i].message``  → ``content``
        - ``steps[i].tool_calls`` → ``tool_calls``
        - ``steps[i].observation.results`` → ``observation`` (joined text)
        - ``steps[i].timestamp`` → ``timestamp``
        """
        agent_info = data.get("agent", {})
        agent_name = agent_info.get("name", "")
        model_name = agent_info.get("model_name", "")
        agent_label = f"{agent_name}/{model_name}" if model_name else agent_name

        steps: list[Step] = []
        for raw_step in data.get("steps", []):
            source = raw_step.get("source", "agent")
            role = _ATIF_ROLE_MAP.get(source, "agent")

            # Build observation string from nested results list
            observation: str | None = None
            obs_data = raw_step.get("observation")
            if isinstance(obs_data, dict):
                results = obs_data.get("results", [])
                if results:
                    observation = "\n---\n".join(
                        r.get("content", "") for r in results if isinstance(r, dict)
                    )

            step = Step(
                role=role,
                content=raw_step.get("message", ""),
                timestamp=raw_step.get("timestamp"),
                tool_calls=raw_step.get("tool_calls"),
                observation=observation,
                metadata={
                    k: v
                    for k, v in raw_step.items()
                    if k not in ("source", "message", "timestamp", "tool_calls", "observation")
                },
            )
            steps.append(step)

        traj_id = data.get("session_id", str(uuid.uuid4()))

        # Try to infer task_name from source path
        task_name = ""
        if source_path:
            parts = Path(source_path).parts
            # heuristic: path contains  …/<task_name>__<hash>/agent/trajectory.json
            for part in reversed(parts):
                if "__" in part and part != "agent":
                    task_name = part.split("__")[0]
                    break

        return Trajectory(
            id=traj_id,
            task_name=task_name,
            agent=agent_label,
            steps=steps,
            metadata={
                "schema_version": data.get("schema_version", ""),
                "source_path": source_path,
            },
        )

    @staticmethod
    def from_generic(data: dict, *, source_path: str = "") -> Trajectory:
        """Load a trajectory from a generic JSON dict.

        Expected top-level keys: ``id``, ``task_name``, ``agent``, ``steps``,
        ``reward``, ``metadata``.  Each step should have at least ``role`` and
        ``content``.
        """
        steps: list[Step] = []
        for raw_step in data.get("steps", []):
            steps.append(
                Step(
                    role=raw_step.get("role", "agent"),
                    content=raw_step.get("content", ""),
                    timestamp=raw_step.get("timestamp"),
                    tool_calls=raw_step.get("tool_calls"),
                    observation=raw_step.get("observation"),
                    metadata=raw_step.get("metadata", {}),
                )
            )

        return Trajectory(
            id=data.get("id", str(uuid.uuid4())),
            task_name=data.get("task_name", ""),
            agent=data.get("agent", ""),
            steps=steps,
            final_answer=data.get("final_answer", ""),
            reward=data.get("reward", 0.0),
            benchmark=data.get("benchmark", ""),
            outcome=data.get("outcome", ""),
            source_format=data.get("source_format", ""),
            task_id=data.get("task_id", ""),
            metadata={
                **(data.get("metadata", {})),
                "source_path": source_path,
            },
        )
