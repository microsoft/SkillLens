"""
Abstract base class for all skill extraction methods.
"""

from __future__ import annotations

import abc

from skilllens.client.openai_client import LLMClient
from skilllens.schema.skill import SkillSet
from skilllens.schema.trajectory import TrajectorySet


class ExtractionMethod(abc.ABC):
    """Interface that every extraction method must implement."""

    def __init__(self, client: LLMClient, **kwargs):
        self.client = client
        self.config = kwargs

    @abc.abstractmethod
    async def extract(self, trajectory_set: TrajectorySet) -> SkillSet:
        """Run extraction and return the resulting skill set."""
        ...
