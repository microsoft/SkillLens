"""Skill extraction methods."""

from .base import ExtractionMethod

__all__ = ["ExtractionMethod"]

# Lazy imports for extraction methods:
# SequentialExtraction → from skilllens.extraction.sequential import SequentialExtraction
# ParallelExtraction → from skilllens.extraction.parallel import ParallelExtraction
