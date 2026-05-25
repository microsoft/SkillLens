"""
Mode-based (Map-Reduce) extraction — processes trajectories by extracting
success/failure modes and hierarchically merging them into skills.

Architecture per batch:
  1. MAP   (parallel):  Each trajectory → independent LLM → ModeSet (JSON)
     - Resolved trajectories → success modes
     - Unresolved trajectories → failure modes
  2. REDUCE (hierarchical):
     - Intermediate layers: groups of ModeSets → LLM merge → ModeSet (JSON)
     - Final layer: remaining ModeSets → LLM tool-calling → SkillStore ops
  3. Batches are processed sequentially; store state carries across batches.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from skilllens.client.openai_client import LLMClient
from skilllens.extraction.base import ExtractionMethod
from skilllens.extraction.skill_store import SkillStore
from skilllens.prompts.parallel_map_modes import (
    build_mode_map_system_prompt,
    build_mode_map_user_message,
)
from skilllens.prompts.parallel_reduce import (
    build_final_reduce_system_prompt,
    build_final_reduce_user_message,
    build_intermediate_reduce_system_prompt,
    build_intermediate_reduce_user_message,
)
from skilllens.schema.modes import Mode, ModeSet
from skilllens.schema.skill import SkillSet
from skilllens.schema.trajectory import Trajectory, TrajectorySet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

_JSON_RETRY_PROMPT = (
    "Your previous response was not valid JSON. Please output ONLY a single "
    "JSON object with the schema described in the system prompt. "
    "Do NOT include any text before or after the JSON. "
    "Do NOT use markdown fences — output raw JSON only."
)

_MAX_JSON_RETRIES = 3


def parse_mode_set(response_text: str) -> ModeSet | None:
    """Try to parse a ModeSet from LLM output text.

    Returns the parsed ``ModeSet`` on success, or ``None`` if parsing failed.
    """
    # Try fenced code block
    match = re.search(r"```(?:json)?\s*\n(.*?)```", response_text, re.DOTALL)
    json_str = match.group(1).strip() if match else response_text.strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(
            "JSON parse failed (len=%d). First 200 chars: %s",
            len(response_text),
            response_text[:200],
        )
        return None

    try:
        if isinstance(data, dict):
            # Parse modes from either success_modes/failure_modes keys
            success_modes = []
            failure_modes = []

            for m in data.get("success_modes", []):
                if isinstance(m, dict):
                    m.setdefault("type", "success")
                    success_modes.append(Mode(**m))
                elif isinstance(m, Mode):
                    success_modes.append(m)

            for m in data.get("failure_modes", []):
                if isinstance(m, dict):
                    m.setdefault("type", "failure")
                    failure_modes.append(Mode(**m))
                elif isinstance(m, Mode):
                    failure_modes.append(m)

            # Also handle a flat "modes" list
            for m in data.get("modes", []):
                if isinstance(m, dict):
                    mode = Mode(**m)
                    if mode.type == "success":
                        success_modes.append(mode)
                    else:
                        failure_modes.append(mode)

            return ModeSet(
                success_modes=success_modes,
                failure_modes=failure_modes,
                summary=data.get("summary", ""),
            )
    except Exception as e:
        logger.warning("Failed to parse mode set structure: %s", e)

    return None


# ---------------------------------------------------------------------------
# ParallelExtraction
# ---------------------------------------------------------------------------

class ParallelExtraction(ExtractionMethod):
    """Mode-based parallel skill extraction.

    Extracts success/failure modes from trajectories, merges them
    hierarchically, and synthesises them into skills.

    Parameters (via ``**kwargs``):
        batch_size:               Trajectories per batch (0 = all in one batch).
        merge_group_size:         ModeSets per reduce group.
        max_concurrency:          Max parallel API calls.
        max_skills:               Skill count upper limit.
        max_skill_chars:          Per-skill char limit.
        max_total_chars:          Total char budget (0 = unlimited).
        include_feedback:         Include outcome/reward in map prompt.
        max_tool_rounds:          Max ReAct rounds for final reduce.
        shuffle_trajectories:     Randomize trajectory order.
        max_modes_per_trajectory: Max modes per trajectory in map phase.
        output_dir:               Directory for logs.
    """

    def __init__(self, client: LLMClient, **kwargs):
        super().__init__(client, **kwargs)
        self.batch_size: int = kwargs.get("batch_size", 0)
        self.merge_group_size: int = kwargs.get("merge_group_size", 4)
        self.max_concurrency: int = kwargs.get("max_concurrency", 8)
        self.max_skills: int = kwargs.get("max_skills", 10)
        self.max_skill_chars: int = kwargs.get("max_skill_chars", 2000)
        self.max_total_chars: int = kwargs.get("max_total_chars", 0)
        self.include_feedback: bool = kwargs.get("include_feedback", True)
        self.max_tool_rounds: int = kwargs.get("max_tool_rounds", 20)
        self.shuffle_trajectories: bool = kwargs.get("shuffle_trajectories", False)
        self.max_modes_per_trajectory: int = kwargs.get("max_modes_per_trajectory", 3)
        self.meta_skill_guidance: str = kwargs.get("meta_skill_guidance", "")
        self.output_dir: str = kwargs.get("output_dir", "./output")

        # Accumulated statistics
        self._stats: dict = {
            "map_llm_calls": 0,
            "map_tokens_in": 0,
            "map_tokens_out": 0,
            "map_failures": 0,
            "map_success_trajs": 0,
            "map_failure_trajs": 0,
            "reduce_intermediate_llm_calls": 0,
            "reduce_intermediate_tokens_in": 0,
            "reduce_intermediate_tokens_out": 0,
            "reduce_intermediate_failures": 0,
            "reduce_final_llm_calls": 0,
            "reduce_final_tokens_in": 0,
            "reduce_final_tokens_out": 0,
            "reduce_final_tool_calls": 0,
        }

    # ==================================================================
    # Public API
    # ==================================================================

    async def extract(self, trajectory_set: TrajectorySet) -> SkillSet:
        store = SkillStore(
            max_skills=self.max_skills,
            max_skill_chars=self.max_skill_chars,
            max_total_chars=self.max_total_chars,
        )

        trajectories = list(trajectory_set.trajectories)
        if self.shuffle_trajectories:
            random.shuffle(trajectories)

        total = len(trajectories)
        resolved = sum(1 for t in trajectories if t.outcome == "resolved")
        unresolved = total - resolved

        # batch_size=0 means all in one batch
        bs = self.batch_size if self.batch_size > 0 else total
        batches = [trajectories[i:i + bs] for i in range(0, total, bs)]

        map_log: list[dict] = []
        reduce_log: list[dict] = []

        logger.info(
            "=" * 60 + "\n"
            "  Extraction starting\n"
            "  Method         : parallel (map-reduce)\n"
            "  Trajectories   : %d (resolved=%d, unresolved=%d)\n"
            "  Batch size     : %s\n"
            "  Merge group    : %d\n"
            "  Max concurrency: %d\n"
            "  Max modes/traj : %d\n"
            "  Max skills     : %d\n"
            "  Max chars/skill: %s\n"
            "  Model          : %s\n"
            + "=" * 60,
            total, resolved, unresolved,
            bs if self.batch_size > 0 else f"{total} (all)",
            self.merge_group_size,
            self.max_concurrency,
            self.max_modes_per_trajectory,
            self.max_skills,
            f"{self.max_skill_chars:,}" if self.max_skill_chars else "unlimited",
            self.client.config.model,
        )

        t_start = time.time()

        for batch_idx, batch in enumerate(batches):
            t_batch_start = time.time()
            batch_resolved = sum(1 for t in batch if t.outcome == "resolved")
            logger.info(
                "-" * 50 + "\n"
                "  Batch %d/%d (%d trajectories: %d resolved, %d unresolved)\n"
                "  Store: %d skills, %d chars",
                batch_idx + 1, len(batches), len(batch),
                batch_resolved, len(batch) - batch_resolved,
                store.skill_count, store.total_chars,
            )

            # MAP PHASE
            store_snapshot = store.snapshot()
            mode_sets = await self._map_phase(
                batch, store_snapshot, batch_idx, map_log
            )

            # REDUCE PHASE
            await self._reduce_phase(
                mode_sets, store, store_snapshot, batch_idx, reduce_log
            )

            t_batch_elapsed = time.time() - t_batch_start
            logger.info(
                "  Batch %d/%d done in %s — Store: %d skills, %d chars",
                batch_idx + 1, len(batches),
                _format_duration(t_batch_elapsed),
                store.skill_count, store.total_chars,
            )

            # Incremental save
            self._save_map_log(map_log)
            self._save_reduce_log(reduce_log)
            self._save_store_history(store.history)

        # Final save
        self._save_map_log(map_log)
        self._save_reduce_log(reduce_log)
        self._save_store_history(store.history)

        # Build SkillSet
        extraction_config = {
            "method": "parallel",
            "batch_size": self.batch_size,
            "merge_group_size": self.merge_group_size,
            "max_concurrency": self.max_concurrency,
            "max_skills": self.max_skills,
            "max_skill_chars": self.max_skill_chars,
            "max_total_chars": self.max_total_chars,
            "include_feedback": self.include_feedback,
            "max_tool_rounds": self.max_tool_rounds,
            "shuffle_trajectories": self.shuffle_trajectories,
            "max_modes_per_trajectory": self.max_modes_per_trajectory,
            "total_trajectories": total,
            "total_batches": len(batches),
            "resolved_trajectories": resolved,
            "unresolved_trajectories": unresolved,
        }

        skill_set = store.to_skill_set(
            extractor_model=self.client.config.model,
            extraction_method="parallel",
            extraction_config=extraction_config,
        )

        # Attach stats
        total_elapsed = time.time() - t_start
        final_stats = dict(self._stats)
        final_stats["total_tokens_in"] = (
            final_stats["map_tokens_in"]
            + final_stats["reduce_intermediate_tokens_in"]
            + final_stats["reduce_final_tokens_in"]
        )
        final_stats["total_tokens_out"] = (
            final_stats["map_tokens_out"]
            + final_stats["reduce_intermediate_tokens_out"]
            + final_stats["reduce_final_tokens_out"]
        )
        final_stats["total_llm_calls"] = (
            final_stats["map_llm_calls"]
            + final_stats["reduce_intermediate_llm_calls"]
            + final_stats["reduce_final_llm_calls"]
        )
        final_stats["wall_clock_seconds"] = round(total_elapsed, 2)
        skill_set.metadata["extraction_stats"] = final_stats
        skill_set.metadata["trajectory_summaries"] = store.summaries

        # Final log
        logger.info(
            "=" * 60 + "\n"
            "  Extraction complete\n"
            "  Total time          : %s\n"
            "  Trajectories        : %d (resolved=%d, unresolved=%d)\n"
            "  Batches             : %d\n"
            "  Skills produced     : %d (%d chars)\n"
            "  Map calls           : %d (tokens: %d in + %d out)\n"
            "  Intermediate reduce : %d (tokens: %d in + %d out)\n"
            "  Final reduce        : %d (tokens: %d in + %d out, %d tool calls)\n"
            "  Total tokens        : %d in + %d out = %d\n"
            + "=" * 60,
            _format_duration(total_elapsed),
            total, resolved, unresolved,
            len(batches),
            store.skill_count, store.total_chars,
            final_stats["map_llm_calls"],
            final_stats["map_tokens_in"], final_stats["map_tokens_out"],
            final_stats["reduce_intermediate_llm_calls"],
            final_stats["reduce_intermediate_tokens_in"],
            final_stats["reduce_intermediate_tokens_out"],
            final_stats["reduce_final_llm_calls"],
            final_stats["reduce_final_tokens_in"],
            final_stats["reduce_final_tokens_out"],
            final_stats["reduce_final_tool_calls"],
            final_stats["total_tokens_in"], final_stats["total_tokens_out"],
            final_stats["total_tokens_in"] + final_stats["total_tokens_out"],
        )

        return skill_set

    # ==================================================================
    # MAP PHASE
    # ==================================================================

    async def _map_phase(
        self,
        batch: list[Trajectory],
        store_snapshot: dict,
        batch_idx: int,
        log: list[dict],
    ) -> list[ModeSet]:
        """Run map sub-agents in parallel. Each trajectory → ModeSet."""
        semaphore = asyncio.Semaphore(self.max_concurrency)

        # Pre-build system prompts for both outcome types
        success_system = build_mode_map_system_prompt(
            max_modes_per_trajectory=self.max_modes_per_trajectory,
            max_skills=self.max_skills,
            max_skill_chars=self.max_skill_chars,
            max_total_chars=self.max_total_chars,
            is_success=True,
            meta_skill_guidance=self.meta_skill_guidance,
        )
        failure_system = build_mode_map_system_prompt(
            max_modes_per_trajectory=self.max_modes_per_trajectory,
            max_skills=self.max_skills,
            max_skill_chars=self.max_skill_chars,
            max_total_chars=self.max_total_chars,
            is_success=False,
            meta_skill_guidance=self.meta_skill_guidance,
        )

        t_map_start = time.time()

        async def process_one(traj: Trajectory, traj_idx: int) -> ModeSet:
            async with semaphore:
                traj_id = traj.task_id or traj.id
                is_success = traj.outcome == "resolved"
                mode_label = "success" if is_success else "failure"

                logger.info(
                    "    Map [%d/%d] %s (%s) ...",
                    traj_idx + 1, len(batch), traj_id, mode_label,
                )

                system = success_system if is_success else failure_system
                user_msg = build_mode_map_user_message(
                    traj, store_snapshot,
                    include_feedback=self.include_feedback,
                )
                messages: list[dict] = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ]

                t0 = time.time()
                total_usage: dict = {"tokens_in": 0, "tokens_out": 0}
                ms: ModeSet | None = None

                for attempt in range(1, _MAX_JSON_RETRIES + 1):
                    response_text, usage = await self.client.chat_with_usage(
                        messages
                    )
                    total_usage["tokens_in"] += usage.get("tokens_in", 0)
                    total_usage["tokens_out"] += usage.get("tokens_out", 0)

                    ms = parse_mode_set(response_text)
                    if ms is not None:
                        break

                    if attempt < _MAX_JSON_RETRIES:
                        logger.warning(
                            "    Map [%d/%d] %s — JSON parse failed "
                            "(attempt %d/%d), retrying...",
                            traj_idx + 1, len(batch), traj_id,
                            attempt, _MAX_JSON_RETRIES,
                        )
                        messages.append(
                            {"role": "assistant", "content": response_text}
                        )
                        messages.append(
                            {"role": "user", "content": _JSON_RETRY_PROMPT}
                        )

                if ms is None:
                    logger.warning(
                        "    Map [%d/%d] %s — JSON parse failed after %d "
                        "attempts, using empty ModeSet",
                        traj_idx + 1, len(batch), traj_id,
                        _MAX_JSON_RETRIES,
                    )
                    ms = ModeSet()

                elapsed = time.time() - t0
                ms.source_trajectory_ids = [traj_id]

                # Track resolved/unresolved counts
                if is_success:
                    self._stats["map_success_trajs"] += 1
                else:
                    self._stats["map_failure_trajs"] += 1

                self._stats["map_llm_calls"] += 1
                self._stats["map_tokens_in"] += total_usage["tokens_in"]
                self._stats["map_tokens_out"] += total_usage["tokens_out"]

                n_success = len(ms.success_modes)
                n_failure = len(ms.failure_modes)

                log.append({
                    "batch_idx": batch_idx,
                    "trajectory_id": traj_id,
                    "trajectory_outcome": traj.outcome,
                    "mode_type": mode_label,
                    "success_modes_count": n_success,
                    "failure_modes_count": n_failure,
                    "success_modes": [m.model_dump() for m in ms.success_modes],
                    "failure_modes": [m.model_dump() for m in ms.failure_modes],
                    "summary": ms.summary,
                    "tokens_in": total_usage["tokens_in"],
                    "tokens_out": total_usage["tokens_out"],
                    "elapsed_seconds": round(elapsed, 2),
                })

                logger.info(
                    "    Map [%d/%d] %s done in %.1fs — "
                    "%d success + %d failure modes, tokens=%d+%d",
                    traj_idx + 1, len(batch), traj_id, elapsed,
                    n_success, n_failure,
                    total_usage["tokens_in"], total_usage["tokens_out"],
                )
                return ms

        results = await asyncio.gather(
            *[process_one(t, i) for i, t in enumerate(batch)],
            return_exceptions=True,
        )

        # Graceful degradation
        cleaned: list[ModeSet] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                traj_id = batch[i].task_id or batch[i].id
                logger.warning("    Map FAILED for %s: %s", traj_id, r)
                self._stats["map_failures"] += 1
                cleaned.append(
                    ModeSet(source_trajectory_ids=[traj_id])
                )
            else:
                cleaned.append(r)

        t_map_elapsed = time.time() - t_map_start
        total_success = sum(len(ms.success_modes) for ms in cleaned)
        total_failure = sum(len(ms.failure_modes) for ms in cleaned)
        logger.info(
            "  Map phase done in %s — %d/%d succeeded, "
            "%d success modes + %d failure modes",
            _format_duration(t_map_elapsed),
            len(batch) - self._stats["map_failures"],
            len(batch),
            total_success, total_failure,
        )

        return cleaned

    # ==================================================================
    # REDUCE PHASE
    # ==================================================================

    async def _reduce_phase(
        self,
        mode_sets: list[ModeSet],
        store: SkillStore,
        store_snapshot: dict,
        batch_idx: int,
        log: list[dict],
    ):
        """Hierarchical reduce: intermediate layers → JSON, final → tool-calling."""
        G = self.merge_group_size
        current_level = mode_sets
        level = 0

        # Intermediate reduce layers
        while len(current_level) > G:
            level += 1
            groups = [
                current_level[i:i + G]
                for i in range(0, len(current_level), G)
            ]
            logger.info(
                "  Reduce level %d: %d mode sets → %d groups (group_size=%d)",
                level, len(current_level), len(groups), G,
            )

            t_level_start = time.time()

            merged_results = await asyncio.gather(
                *[
                    self._intermediate_reduce(group, store_snapshot)
                    for group in groups
                ],
                return_exceptions=True,
            )

            merged: list[ModeSet] = []
            for i, r in enumerate(merged_results):
                if isinstance(r, Exception):
                    logger.warning(
                        "  Reduce level %d group %d FAILED: %s — "
                        "using first ModeSet as fallback",
                        level, i + 1, r,
                    )
                    self._stats["reduce_intermediate_failures"] += 1
                    merged.append(groups[i][0] if groups[i] else ModeSet())
                else:
                    merged.append(r)

            current_level = merged

            t_level_elapsed = time.time() - t_level_start
            total_success = sum(len(ms.success_modes) for ms in current_level)
            total_failure = sum(len(ms.failure_modes) for ms in current_level)
            logger.info(
                "  Reduce level %d done in %s — %d merged sets, "
                "%d success + %d failure modes",
                level, _format_duration(t_level_elapsed),
                len(current_level), total_success, total_failure,
            )

            log.append({
                "batch_idx": batch_idx,
                "level": level,
                "type": "intermediate",
                "input_count": len(mode_sets),
                "groups": len(groups),
                "output_count": len(current_level),
                "total_success_modes": total_success,
                "total_failure_modes": total_failure,
                "elapsed_seconds": round(t_level_elapsed, 2),
                "merged_mode_sets": [
                    {
                        "source_trajectory_ids": ms.source_trajectory_ids,
                        "summary": ms.summary,
                        "success_modes": [m.model_dump() for m in ms.success_modes],
                        "failure_modes": [m.model_dump() for m in ms.failure_modes],
                    }
                    for ms in current_level
                ],
            })

        # Final reduce
        logger.info(
            "  Final reduce: %d mode sets → tool-calling",
            len(current_level),
        )

        t_final_start = time.time()

        store.set_trajectory_context(
            batch_idx,
            f"batch_{batch_idx}_final_reduce",
        )

        await self._final_reduce(current_level, store)

        # Retry up to 3 times if final reduce produced 0 skills
        for retry_i in range(3):
            if store.skill_count > 0:
                break
            logger.warning(
                "  Final reduce produced 0 skills — retry %d/3 with fresh conversation",
                retry_i + 1,
            )
            store.reset_for_retry()
            await self._final_reduce(current_level, store)

        t_final_elapsed = time.time() - t_final_start
        logger.info(
            "  Final reduce done in %s — Store: %d skills, %d chars",
            _format_duration(t_final_elapsed),
            store.skill_count, store.total_chars,
        )

        log.append({
            "batch_idx": batch_idx,
            "level": level + 1,
            "type": "final",
            "input_count": len(current_level),
            "elapsed_seconds": round(t_final_elapsed, 2),
            "skill_count_after": store.skill_count,
            "total_chars_after": store.total_chars,
            "input_mode_sets": [
                {
                    "source_trajectory_ids": ms.source_trajectory_ids,
                    "summary": ms.summary,
                    "success_modes": [m.model_dump() for m in ms.success_modes],
                    "failure_modes": [m.model_dump() for m in ms.failure_modes],
                }
                for ms in current_level
            ],
        })

    async def _intermediate_reduce(
        self,
        group: list[ModeSet],
        store_snapshot: dict,
    ) -> ModeSet:
        """Intermediate reduce: LLM merges multiple ModeSets → one (JSON)."""
        system = build_intermediate_reduce_system_prompt(
            max_modes_per_type=self.max_modes_per_trajectory * 3,
        )
        user = build_intermediate_reduce_user_message(group, store_snapshot)
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        total_usage: dict = {"tokens_in": 0, "tokens_out": 0}
        merged: ModeSet | None = None

        for attempt in range(1, _MAX_JSON_RETRIES + 1):
            response_text, usage = await self.client.chat_with_usage(messages)
            total_usage["tokens_in"] += usage.get("tokens_in", 0)
            total_usage["tokens_out"] += usage.get("tokens_out", 0)

            merged = parse_mode_set(response_text)
            if merged is not None:
                break

            if attempt < _MAX_JSON_RETRIES:
                logger.warning(
                    "  Intermediate reduce — JSON parse failed "
                    "(attempt %d/%d), retrying...",
                    attempt, _MAX_JSON_RETRIES,
                )
                messages.append(
                    {"role": "assistant", "content": response_text}
                )
                messages.append(
                    {"role": "user", "content": _JSON_RETRY_PROMPT}
                )

        if merged is None:
            logger.warning(
                "  Intermediate reduce — JSON parse failed after %d "
                "attempts, using empty ModeSet",
                _MAX_JSON_RETRIES,
            )
            merged = ModeSet()

        # Collect all source trajectory IDs
        all_traj_ids: list[str] = []
        for ms in group:
            all_traj_ids.extend(ms.source_trajectory_ids)
        merged.source_trajectory_ids = all_traj_ids

        self._stats["reduce_intermediate_llm_calls"] += 1
        self._stats["reduce_intermediate_tokens_in"] += total_usage["tokens_in"]
        self._stats["reduce_intermediate_tokens_out"] += total_usage["tokens_out"]

        return merged

    async def _final_reduce(
        self,
        mode_sets: list[ModeSet],
        store: SkillStore,
    ):
        """Final reduce: LLM synthesises modes into skills via tools."""
        system = build_final_reduce_system_prompt(
            max_skills=self.max_skills,
            max_skill_chars=self.max_skill_chars,
            max_total_chars=self.max_total_chars,
            meta_skill_guidance=self.meta_skill_guidance,
        )
        user = build_final_reduce_user_message(mode_sets)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        final_text, full_msgs, round_stats = await self.client.chat_with_tools(
            messages,
            tools=store.tool_schemas,
            tool_handler=store.dispatch,
            max_rounds=self.max_tool_rounds,
            finish_tool_name="finish_extraction",
        )

        self._stats["reduce_final_llm_calls"] += 1
        self._stats["reduce_final_tokens_in"] += round_stats["tokens_in"]
        self._stats["reduce_final_tokens_out"] += round_stats["tokens_out"]
        self._stats["reduce_final_tool_calls"] += round_stats["tool_calls_count"]

    # ==================================================================
    # Log saving
    # ==================================================================

    def _save_map_log(self, log: list[dict]) -> None:
        """Save per-trajectory map modes."""
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "map_modes.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for entry in log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Map modes saved: %s (%d entries)", path, len(log))

    def _save_reduce_log(self, log: list[dict]) -> None:
        """Save reduce phase logs."""
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "reduce_log.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for entry in log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Reduce log saved: %s (%d entries)", path, len(log))

    def _save_store_history(self, history: list[dict]) -> None:
        """Save the full store operation history."""
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "store_history.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for entry in history:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Store history saved: %s (%d operations)", path, len(history))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h{m:02d}m{s:02d}s"
