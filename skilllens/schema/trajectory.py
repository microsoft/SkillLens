"""
Trajectory data models — universal representation for agent trajectories.

Inspired by ADP (Agent Data Protocol) but kept minimal for this benchmark's needs.
Supports conversion from ATIF-v1.5 and generic JSON formats.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Step(BaseModel):
    """A single step within an agent trajectory."""

    role: Literal["system", "user", "agent", "tool"]
    content: str = ""
    timestamp: str | None = None
    tool_calls: list[dict] | None = None  # Agent's tool invocations
    observation: str | None = None  # Tool/environment return value
    metadata: dict = Field(default_factory=dict)


class Trajectory(BaseModel):
    """A complete agent trajectory for one task execution."""

    id: str
    task_name: str = ""
    agent: str = ""  # Agent name or model identifier
    steps: list[Step] = Field(default_factory=list)
    final_answer: str = ""  # Final submission (patch, answer, etc.)
    reward: float = 0.0  # 1.0 = resolved, 0.0 = any form of failure
    benchmark: str = ""  # "swebench", "spreadsheetbench", "alfworld", "bfcl", "seal0"
    outcome: str = ""  # "resolved" / "unresolved" / "error"
    source_format: str = ""  # "mini-swe-agent-1.1", "atif-v1.5", "generic"
    task_id: str = ""  # Benchmark original task id, e.g. "django__django-10914"
    metadata: dict = Field(default_factory=dict)


class TrajectorySet(BaseModel):
    """A collection of trajectories serving as input to an extraction method."""

    trajectories: list[Trajectory] = Field(default_factory=list)
    source: str = ""  # Data provenance identifier
    metadata: dict = Field(default_factory=dict)
