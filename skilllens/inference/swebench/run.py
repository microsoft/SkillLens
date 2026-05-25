#!/usr/bin/env python3

"""
Skill-augmented SWE-bench batch runner.

This is a thin wrapper around mini-swe-agent's ``swebench.py`` batch runner.
When ``--skill-set`` is provided, it swaps in :class:`SkillAwareLitellmModel`
and :class:`SkillAwareAgent` so the LLM has read-only access to a skill
library during problem solving.  Without ``--skill-set`` the behaviour is
identical to running ``mini-extra swebench`` directly.

Usage:
    python -m skilllens.inference.swebench.run \
        --skill-set path/to/skill_set.json \
        --model azure/gpt-5-mini \
        --subset lite --split test \
        --workers 8 \
        --output ./inference_output/gpt5mini_with_skill_gpt54 \
        -c swebench.yaml \
        -c model.model_kwargs.temperature=1
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import logging
import random
import re
import threading
import time
import traceback
from pathlib import Path

import typer
from rich.live import Live

# ---------------------------------------------------------------------------
# mini-swe-agent imports — we reuse as much as possible
# ---------------------------------------------------------------------------
from skilllens.benchmarks.swebench.minisweagent.agents.default import DefaultAgent
from skilllens.benchmarks.swebench.minisweagent.config import get_config_from_spec
from skilllens.benchmarks.swebench.minisweagent.models import get_model
from skilllens.benchmarks.swebench.minisweagent.run.benchmarks.swebench import (
    DEFAULT_CONFIG_FILE,
    DATASET_MAPPING,
    ProgressTrackingAgent,
    filter_instances,
    get_sb_environment,
    remove_from_preds_file,
    update_preds_file,
)
from skilllens.benchmarks.swebench.minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from skilllens.benchmarks.swebench.minisweagent.utils.log import add_file_handler, logger
from skilllens.benchmarks.swebench.minisweagent.utils.serialize import UNSET, recursive_merge

# ---------------------------------------------------------------------------
# skilllens imports
# ---------------------------------------------------------------------------
from skilllens.inference.skill_provider import ReadOnlySkillProvider
from skilllens.inference.swebench.skill_agent import (
    SkillAwareAgent,
    SkillAwareLitellmModel,
    build_skill_section,
    build_single_skill_section,
)

skill_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill-aware ProgressTrackingAgent
# ---------------------------------------------------------------------------

class SkillProgressTrackingAgent(SkillAwareAgent):
    """SkillAwareAgent with progress-bar updates (mirrors swebench.py's ProgressTrackingAgent)."""

    def __init__(
        self,
        *args,
        progress_manager: RunBatchProgressManager,
        instance_id: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        self.progress_manager.update_instance_status(
            self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})"
        )
        return super().step()


# ---------------------------------------------------------------------------
# Instance processing
# ---------------------------------------------------------------------------

_OUTPUT_FILE_LOCK = threading.Lock()


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    skill_provider: ReadOnlySkillProvider | None = None,
) -> None:
    """Process a single SWEBench instance — with optional skill augmentation."""
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    # Clean up stale state from previous (possibly failed) runs
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    # ---- build config copy (avoid mutation across threads) ----
    config = copy.deepcopy(config)

    # ---- inject skill section into system_template when skill mode is on ----
    if skill_provider is not None:
        agent_cfg = config.setdefault("agent", {})
        original_sys = agent_cfg.get("system_template", "")
        if skill_provider.is_single_skill:
            # Single skill: inline content directly, no tool protocol
            agent_cfg["system_template"] = (
                original_sys.rstrip()
                + "\n\n"
                + build_single_skill_section(skill_provider.get_single_skill())
            )
        else:
            agent_cfg["system_template"] = (
                original_sys.rstrip() + "\n\n" + build_skill_section(skill_provider.skill_count)
            )

    # ---- create model ----
    if skill_provider is not None and not skill_provider.is_single_skill:
        # Multiple skills: use tool-based interaction
        model = SkillAwareLitellmModel(
            skill_provider=skill_provider, **config.get("model", {})
        )
    else:
        # No skills or single skill (inlined): standard model
        model = get_model(config=config.get("model", {}))

    task = instance["problem_statement"]

    progress_manager.on_instance_start(instance_id)
    progress_manager.update_instance_status(instance_id, "Pulling/starting environment")

    agent = None
    exit_status = None
    result = None
    extra_info = {}

    try:
        env = get_sb_environment(config, instance)

        if skill_provider is not None and not skill_provider.is_single_skill:
            # Multiple skills: use SkillAware agent (handles skill tool calls)
            agent = SkillProgressTrackingAgent(
                model,
                env,
                progress_manager=progress_manager,
                instance_id=instance_id,
                **config.get("agent", {}),
            )
        else:
            # No skills or single skill (inlined): standard agent
            agent = ProgressTrackingAgent(
                model,
                env,
                progress_manager=progress_manager,
                instance_id=instance_id,
                **config.get("agent", {}),
            )

        info = agent.run(task)
        exit_status = info.get("exit_status")
        result = info.get("submission")
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, ""
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    finally:
        if agent is not None:
            traj_path = instance_dir / f"{instance_id}.traj.json"
            extra_save_data = {
                "info": {
                    "exit_status": exit_status,
                    "submission": result,
                    **extra_info,
                },
                "instance_id": instance_id,
            }
            # Record skill provider info in trajectory for later analysis
            if skill_provider is not None:
                extra_save_data["info"]["skill_provider"] = {
                    "skill_count": skill_provider.skill_count,
                    "call_history": skill_provider.call_history,
                }
            agent.save(traj_path, extra_save_data)
            logger.info(f"Saved trajectory to '{traj_path}'")
        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
        progress_manager.on_instance_end(instance_id, exit_status)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_HELP_TEXT = """Run SWE-bench with optional skill-library augmentation.

[not dim]
Mirrors `mini-extra swebench` but adds [bold green]--skill-set[/bold green] for skill injection.
Without --skill-set, behaviour is identical to the upstream runner.
[/not dim]
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

[bold red]IMPORTANT:[/bold red] [red]If you set this option, the default config file will not be used.[/red]
So you need to explicitly set it e.g., with [bold green]-c swebench.yaml <other options>[/bold green]

Multiple configs will be recursively merged.

Examples:

[bold green]-c swebench.yaml -c model.model_kwargs.temperature=0.5[/bold green]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    # --- skill-specific options ---
    skill_set: str = typer.Option("", "--skill-set", help="Path to skill_set.json. Empty = baseline (no skill)"),
    # --- standard swebench options (mirrored from mini-swe-agent) ---
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    instance_ids: str = typer.Option("", "--instance-ids", help="Path to file with instance IDs (one per line or JSON with 'instance_ids' key). Overrides --filter/--slice.", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help=_CONFIG_SPEC_HELP_TEXT, rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type (docker, singularity, ...)", rich_help_panel="Advanced"),
) -> None:
    # fmt: on

    # ---- skill provider ----
    skill_provider: ReadOnlySkillProvider | None = None
    if skill_set:
        skill_path = Path(skill_set)
        if not skill_path.exists():
            typer.echo(f"Error: skill set file not found: {skill_path}", err=True)
            raise typer.Exit(code=1)
        skill_provider = ReadOnlySkillProvider.from_file(skill_path)
        skill_logger.info(
            "Skill mode ON: loaded %d skills from %s", skill_provider.skill_count, skill_path
        )
    else:
        skill_logger.info("Skill mode OFF: running baseline (no skills)")

    # ---- output dir ----
    if not output:
        import time as _time
        output = f"inference_output/swebench/{subset}_{split}_{_time.strftime('%Y%m%d_%H%M%S')}"
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(output_path / "minisweagent.log")

    # ---- save run metadata ----
    run_meta = {
        "skill_set": skill_set or None,
        "skill_count": skill_provider.skill_count if skill_provider else 0,
        "model": model,
        "subset": subset,
        "split": split,
        "instance_ids_file": instance_ids or None,
        "workers": workers,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_path / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))

    # ---- load dataset ----
    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    # ---- filter by instance-ids file (if provided) ----
    if instance_ids:
        ids_path = Path(instance_ids)
        if not ids_path.exists():
            typer.echo(f"Error: instance IDs file not found: {ids_path}", err=True)
            raise typer.Exit(code=1)
        text = ids_path.read_text()
        # Support both plain text (one ID per line) and JSON with 'instance_ids' key
        if ids_path.suffix == ".json":
            id_set = set(json.loads(text)["instance_ids"])
        else:
            id_set = {line.strip() for line in text.splitlines() if line.strip()}
        before = len(instances)
        instances = [inst for inst in instances if inst["instance_id"] in id_set]
        logger.info(f"Instance-IDs filter: {before} -> {len(instances)} instances (from {ids_path})")
    # Apply filter/slice/shuffle (works on either full dataset or instance-ids subset)
    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)

    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    # ---- build config ----
    logger.info(f"Building agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    })
    config = recursive_merge(*configs)

    # ---- run ----
    progress_manager = RunBatchProgressManager(
        len(instances), output_path / f"exit_statuses_{time.time()}.yaml"
    )

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    with Live(progress_manager.render_group, refresh_per_second=4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_instance,
                    instance,
                    output_path,
                    config,
                    progress_manager,
                    skill_provider,
                ): instance["instance_id"]
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)

    # ---- save skill usage summary ----
    if skill_provider is not None:
        summary = {
            "skill_count": skill_provider.skill_count,
            "total_tool_calls": len(skill_provider.call_history),
            "call_history": skill_provider.call_history,
        }
        (output_path / "skill_usage_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
        skill_logger.info(
            "Skill usage: %d tool calls across run", len(skill_provider.call_history)
        )

    # ---- SWE-bench evaluation ----
    preds_path = output_path / "preds.json"
    if preds_path.exists():
        logger.info("Starting SWE-bench evaluation...")
        try:
            from swebench.harness.run_evaluation import main as run_swebench_eval

            instance_id_list = [inst["instance_id"] for inst in instances]
            # Run eval from inside output_path so that swebench harness writes
            # its logs/ directory there instead of polluting the repo root.
            import os as _os
            _orig_cwd = _os.getcwd()
            _os.chdir(str(output_path.resolve()))
            try:
              run_swebench_eval(
                dataset_name=dataset_path,
                split=split,
                instance_ids=instance_id_list,
                predictions_path=str(preds_path.name),
                max_workers=workers,
                force_rebuild=False,
                cache_level="env",
                clean=False,
                open_file_limit=4096,
                run_id=output_path.name,
                timeout=1800,
                namespace=None,
                rewrite_reports=False,
                modal=False,
                report_dir=".",
            )
            finally:
              _os.chdir(_orig_cwd)
            logger.info("SWE-bench evaluation completed. Reports saved by swebench harness.")
        except ImportError:
            logger.warning(
                "swebench package not installed — skipping evaluation. "
                "Install with: pip install -e /path/to/swebench"
            )
        except Exception:
            logger.error("SWE-bench evaluation failed:\n%s", traceback.format_exc())
    else:
        logger.warning("No preds.json found — skipping evaluation.")


if __name__ == "__main__":
    app()
