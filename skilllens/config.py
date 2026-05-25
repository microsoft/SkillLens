"""
Configuration management — loads settings from YAML files, environment
variables, and CLI overrides.

Priority (highest → lowest): CLI args > environment variables > YAML file > defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from skilllens.client.openai_client import LLMConfig


class ExtractionConfig(BaseModel):
    """Parameters governing the extraction process."""

    method: str = "parallel"  # "sequential" | "parallel"

    # --- NaiveTool method parameters ---
    max_skills: int = 1
    max_skill_chars: int = 3000
    max_total_chars: int = 0          # 0 = no total limit
    include_feedback: bool = True
    max_tool_rounds: int = 20
    shuffle_trajectories: bool = False

    # --- Parallel (map-reduce) method parameters ---
    batch_size: int = 8               # 0 = all trajectories in one batch
    merge_group_size: int = 4         # proposals per reduce group
    max_concurrency: int = 8          # max parallel API calls

    # --- Mode-based method parameters ---
    max_modes_per_trajectory: int = 3  # max modes extracted per trajectory
    meta_skill_guidance: str = ""      # optional guidance text injected into extraction prompts


class InputConfig(BaseModel):
    """Settings for input data."""

    path: str = ""
    format: str = "auto"  # auto | atif | generic
    benchmark: str = "swebench"  # benchmark name, used as output subdirectory


class OutputConfig(BaseModel):
    """Settings for output data."""

    path: str = "./output"
    save_intermediate: bool = True


class AppConfig(BaseModel):
    """Top-level application configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict:
    """Read a YAML config file and return its contents as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_env(raw: dict) -> dict:
    """Override selected fields from environment variables."""
    env_map = {
        "AZURE_OPENAI_API_KEY": ("llm", "api_key"),
        "AZURE_OPENAI_ENDPOINT": ("llm", "azure_endpoint"),
        "OPENAI_API_KEY": ("llm", "api_key"),
        "OPENAI_BASE_URL": ("llm", "base_url"),
        "LLM_MODEL": ("llm", "model"),
    }
    for env_var, key_path in env_map.items():
        value = os.environ.get(env_var)
        if value:
            section, key = key_path
            raw.setdefault(section, {})[key] = value
    return raw


def build_config(
    yaml_path: str | Path | None = None,
    cli_overrides: dict | None = None,
) -> AppConfig:
    """Construct an ``AppConfig`` by merging YAML → env → CLI overrides."""
    raw: dict = {}

    if yaml_path and Path(yaml_path).is_file():
        raw = load_yaml(yaml_path)

    raw = _apply_env(raw)

    # Apply CLI-level overrides (flat dotted keys like "llm.model")
    if cli_overrides:
        for dotted_key, value in cli_overrides.items():
            parts = dotted_key.split(".")
            target = raw
            for p in parts[:-1]:
                target = target.setdefault(p, {})
            target[parts[-1]] = value

    return AppConfig(**raw)
