"""
Naive tool-use extraction — a simple ReAct loop where the LLM uses tools
to manage an in-memory skill store.

For each trajectory the LLM gets a fresh conversation (system + user message)
but the SkillStore state persists across trajectories.
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from skilllens.client.openai_client import LLMClient
from skilllens.extraction.base import ExtractionMethod
from skilllens.extraction.skill_store import SkillStore
from skilllens.prompts.sequential import (
    SYSTEM_PROMPT,
    build_system_prompt,
    build_user_message,
)
from skilllens.schema.skill import SkillSet
from skilllens.schema.trajectory import TrajectorySet

logger = logging.getLogger(__name__)


class SequentialExtraction(ExtractionMethod):
    """Simple ReAct-loop skill extraction using tool calls.

    The LLM operates on a ``SkillStore`` via 6 tools:
    ``list_skills``, ``view_skill``, ``add_skill``, ``update_skill``,
    ``delete_skill``, ``finish_extraction``.

    Parameters (passed via ``**kwargs``):
        max_skills:           Skill count upper limit (default 10).
        max_skill_chars:      Per-skill char limit for desc+content (default 2000).
        max_total_chars:      Total char budget across all skills (0 = unlimited).
        include_feedback:     Include outcome/reward in prompt (default True).
        max_tool_rounds:      Max ReAct rounds per trajectory (default 20).
        shuffle_trajectories: Randomize trajectory order (default False).
        snapshot_interval:    Save snapshot every N trajectories (0 = off).
        output_dir:           Directory for logs and snapshots (default "./output").
    """

    def __init__(self, client: LLMClient, **kwargs):
        super().__init__(client, **kwargs)
        self.max_skills: int = kwargs.get("max_skills", 10)
        self.max_skill_chars: int = kwargs.get("max_skill_chars", 2000)
        self.max_total_chars: int = kwargs.get("max_total_chars", 0)
        self.include_feedback: bool = kwargs.get("include_feedback", True)
        self.max_tool_rounds: int = kwargs.get("max_tool_rounds", 20)
        self.shuffle_trajectories: bool = kwargs.get("shuffle_trajectories", False)
        self.snapshot_interval: int = kwargs.get("snapshot_interval", 0)
        self.output_dir: str = kwargs.get("output_dir", "./output")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def extract(self, trajectory_set: TrajectorySet) -> SkillSet:
        store = SkillStore(
            max_skills=self.max_skills,
            max_skill_chars=self.max_skill_chars,
            max_total_chars=self.max_total_chars,
        )

        # Build system prompt with concrete constraint values
        system_prompt = build_system_prompt(
            max_skills=self.max_skills,
            max_skill_chars=self.max_skill_chars,
            max_total_chars=self.max_total_chars,
        )

        trajectories = list(trajectory_set.trajectories)
        if self.shuffle_trajectories:
            random.shuffle(trajectories)

        total = len(trajectories)
        extraction_log: list[dict] = []
        extraction_trajectories: list[dict] = []  # full agent conversations

        logger.info(
            "="*60 + "\n"
            "  Extraction starting\n"
            "  Method       : sequential\n"
            "  Trajectories : %d\n"
            "  Max skills   : %d\n"
            "  Max chars/skill: %s\n"
            "  Max total chars: %s\n"
            "  Model        : %s\n"
            + "="*60,
            total, self.max_skills,
            f"{self.max_skill_chars:,}" if self.max_skill_chars else "unlimited",
            f"{self.max_total_chars:,}" if self.max_total_chars else "unlimited",
            self.client.config.model,
        )

        t_extraction_start = time.time()

        for i, traj in enumerate(trajectories):
            # Set trajectory context for store history logging
            store.set_trajectory_context(i, traj.task_id or traj.id)

            # Progress & ETA
            traj_id = traj.task_id or traj.id
            if i > 0:
                elapsed = time.time() - t_extraction_start
                avg_per_traj = elapsed / i
                eta_secs = avg_per_traj * (total - i)
                eta_str = _format_duration(eta_secs)
                elapsed_str = _format_duration(elapsed)
            else:
                elapsed_str = "0s"
                eta_str = "?"

            logger.info(
                "-"*50 + "\n"
                "  [%d/%d] %s (outcome=%s)\n"
                "  Elapsed: %s | ETA: %s | Skills: %d",
                i + 1, total, traj_id, traj.outcome or "?",
                elapsed_str, eta_str, store.skill_count,
            )

            t_traj_start = time.time()

            # Fresh conversation for each trajectory; store state persists
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_message(
                    traj, i, total,
                    include_feedback=self.include_feedback,
                )},
            ]

            final_text, full_msgs, round_stats = (
                await self.client.chat_with_tools(
                    messages,
                    tools=store.tool_schemas,
                    tool_handler=store.dispatch,
                    max_rounds=self.max_tool_rounds,
                    finish_tool_name="finish_extraction",
                )
            )

            # Build log entry
            log_entry = {
                "trajectory_index": i,
                "trajectory_id": traj.task_id or traj.id,
                "outcome": traj.outcome,
                "tool_calls_count": round_stats["tool_calls_count"],
                "rounds": round_stats["rounds"],
                "finish_reason": round_stats["finish_reason"],
                "tokens_in": round_stats["tokens_in"],
                "tokens_out": round_stats["tokens_out"],
                "skill_count_after": store.skill_count,
                "total_chars_after": store.total_chars,
            }
            extraction_log.append(log_entry)

            # Record full agent conversation (extraction trajectory)
            extraction_trajectories.append({
                "trajectory_index": i,
                "trajectory_id": traj.task_id or traj.id,
                "outcome": traj.outcome,
                "messages": full_msgs,
                "stats": round_stats,
                "skill_count_after": store.skill_count,
            })

            traj_elapsed = time.time() - t_traj_start
            logger.info(
                "  ✓ [%d/%d] Done in %s — "
                "%d tool calls, %d rounds, finish=%s, "
                "tokens=%d+%d, skills=%d (%d chars)",
                i + 1, total, _format_duration(traj_elapsed),
                round_stats["tool_calls_count"],
                round_stats["rounds"],
                round_stats["finish_reason"],
                round_stats["tokens_in"],
                round_stats["tokens_out"],
                store.skill_count,
                store.total_chars,
            )

            # Snapshot
            if self.snapshot_interval > 0 and (i + 1) % self.snapshot_interval == 0:
                self._save_snapshot(store, i + 1)

        # Save extraction log
        self._save_extraction_log(extraction_log)

        # Save full extraction trajectories (agent conversations)
        self._save_extraction_trajectories(extraction_trajectories)

        # Build final SkillSet
        extraction_config = {
            "max_skills": self.max_skills,
            "max_skill_chars": self.max_skill_chars,
            "max_total_chars": self.max_total_chars,
            "include_feedback": self.include_feedback,
            "max_tool_rounds": self.max_tool_rounds,
            "shuffle_trajectories": self.shuffle_trajectories,
            "total_trajectories": total,
        }

        skill_set = store.to_skill_set(
            extractor_model=self.client.config.model,
            extraction_config=extraction_config,
        )

        # Attach extraction stats to metadata (keep concise in skill_set.json)
        total_tool_calls = sum(e["tool_calls_count"] for e in extraction_log)
        total_tokens_in = sum(e["tokens_in"] for e in extraction_log)
        total_tokens_out = sum(e["tokens_out"] for e in extraction_log)
        skill_set.metadata["extraction_stats"] = {
            "total_tool_calls": total_tool_calls,
            "total_tokens_in": total_tokens_in,
            "total_tokens_out": total_tokens_out,
            "total_trajectories_processed": total,
            "finish_reasons": {
                reason: sum(
                    1 for e in extraction_log if e["finish_reason"] == reason
                )
                for reason in set(e["finish_reason"] for e in extraction_log)
            },
        }
        # LLM's per-trajectory summaries (from finish_extraction calls) — compact
        skill_set.metadata["trajectory_summaries"] = store.summaries

        # store_history is potentially large — save to a separate file
        self._save_store_history(store.history)

        # Final summary
        total_elapsed = time.time() - t_extraction_start
        total_tool_calls_final = sum(e["tool_calls_count"] for e in extraction_log)
        logger.info(
            "="*60 + "\n"
            "  Extraction complete\n"
            "  Total time      : %s\n"
            "  Trajectories    : %d\n"
            "  Skills produced : %d (%d chars)\n"
            "  Total tool calls: %d\n"
            "  Total tokens    : %d in + %d out = %d\n"
            "  Avg time/traj   : %s\n"
            + "="*60,
            _format_duration(total_elapsed),
            total,
            store.skill_count, store.total_chars,
            total_tool_calls_final,
            total_tokens_in, total_tokens_out,
            total_tokens_in + total_tokens_out,
            _format_duration(total_elapsed / total) if total > 0 else "?",
        )

        return skill_set

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _save_snapshot(self, store: SkillStore, after_n: int) -> None:
        """Save an intermediate SkillStore snapshot."""
        out_dir = Path(self.output_dir) / "snapshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        snapshot = store.snapshot()
        snapshot["after_trajectories"] = after_n
        path = out_dir / f"after_{after_n:04d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        logger.info("Snapshot saved: %s", path)

    def _save_extraction_log(self, log: list[dict]) -> None:
        """Save the per-trajectory extraction log."""
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "extraction_log.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for entry in log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Extraction log saved: %s (%d entries)", path, len(log))

    def _save_store_history(self, history: list[dict]) -> None:
        """Save the full store operation history to a separate file."""
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "store_history.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for entry in history:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Store history saved: %s (%d operations)", path, len(history))

    def _save_extraction_trajectories(self, trajectories: list[dict]) -> None:
        """Save the full extraction agent conversations.

        Each line is a JSON object representing one input trajectory's
        extraction session, including the complete message history
        (system, user, assistant with tool_calls, tool results).
        This allows post-hoc analysis of how the extractor reasoned
        and operated on the skill store.
        """
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "extraction_trajectories.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for entry in trajectories:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(
            "Extraction trajectories saved: %s (%d entries)",
            path, len(trajectories),
        )


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
