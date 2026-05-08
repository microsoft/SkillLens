"""
Mode data models — structured representations for success/failure patterns.

Used by the mode_based (map-reduce) extraction method. Each ModeSet contains
success_modes and failure_modes extracted from agent trajectories, which are
hierarchically merged before being synthesised into skills.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Mode(BaseModel):
    """A single success or failure pattern extracted from a trajectory.

    - **success**: A strategy, methodology, or decision pattern that
      contributed to the agent successfully completing the task.
    - **failure**: An error pattern, pitfall, or anti-pattern that
      caused or contributed to the agent failing the task.
    """

    type: Literal["success", "failure"]
    pattern: str  # Short name for the pattern (e.g. "incremental-validation")
    description: str  # 1-3 sentences describing the pattern
    evidence: str = ""  # Brief evidence from the trajectory
    source_trajectory_ids: list[str] = Field(default_factory=list)


class ModeSet(BaseModel):
    """A collection of modes — the output of a map or reduce step.

    The same data structure is used for:
    - Map output (modes from a single trajectory)
    - Intermediate reduce output (merged modes from a group)
    - Final reduce input (merged modes ready for skill synthesis)
    """

    success_modes: list[Mode] = Field(default_factory=list)
    failure_modes: list[Mode] = Field(default_factory=list)
    source_trajectory_ids: list[str] = Field(default_factory=list)
    summary: str = ""
