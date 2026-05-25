#!/usr/bin/env python3
"""Convert LiteResearcher incremental JSONL → per-question Trajectory JSON."""
import argparse
import json
import os
import re
import hashlib
from datetime import datetime


def make_id(question_id, question):
    if question_id:
        return str(question_id)
    return hashlib.md5(question.encode()).hexdigest()[:12]


def sanitize_filename(s):
    return re.sub(r'[^\w\-.]', '_', str(s))[:120]


def convert_record(record, model, benchmark, dataset_name):
    question = record.get("question", "")
    question_id = record.get("question_id") or ""
    tid = make_id(question_id, question)

    steps = []

    # System prompt — from raw messages[0] if available
    messages = record.get("messages", [])
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        steps.append({
            "role": "system",
            "content": messages[0]["content"],
            "metadata": {"type": "system_prompt"},
        })

    # User question
    if question:
        steps.append({
            "role": "user",
            "content": question,
            "metadata": {"type": "task_input"},
        })

    # Convert tool_interactions → Step pairs (agent + tool)
    for interaction in record.get("tool_interactions", []):
        tool_call = interaction.get("tool_call")
        tool_call_raw = interaction.get("tool_call_raw", "")
        response = interaction.get("response", "")

        # Agent step with tool call
        agent_step = {
            "role": "agent",
            "content": tool_call_raw.strip() if not tool_call else "",
            "tool_calls": [tool_call] if tool_call else None,
            "observation": None,
            "metadata": {},
        }
        steps.append(agent_step)

        # Tool response step
        if response:
            tool_step = {
                "role": "tool",
                "content": "",
                "observation": response,
                "metadata": {},
            }
            steps.append(tool_step)

    # Final agent step with answer
    final_answer = record.get("final_answer", record.get("prediction", ""))
    if final_answer:
        steps.append({
            "role": "agent",
            "content": final_answer,
            "observation": None,
            "metadata": {"is_final_answer": True},
        })

    # Judge info
    judge = record.get("judge", {})
    correct = judge.get("correct", False)
    if isinstance(correct, str):
        correct = correct.strip().lower() in ("true", "1", "yes")

    trajectory = {
        "id": tid,
        "task_name": f"{benchmark}/{dataset_name}/{tid}",
        "agent": model,
        "steps": steps,
        "final_answer": final_answer,
        "reward": 1.0 if correct else 0.0,
        "benchmark": benchmark,
        "outcome": "resolved" if correct else "unresolved",
        "source_format": "lite-researcher",
        "task_id": tid,
        "metadata": {
            "question": question,
            "reference_answer": record.get("answer", ""),
            "judge": judge,
            "total_time": record.get("total_time"),
            "token_stats": record.get("token_stats"),
            "termination": record.get("termination"),
            "rollout_idx": record.get("rollout_idx"),
        },
    }
    return tid, trajectory


def main():
    parser = argparse.ArgumentParser(description="Convert LiteResearcher output to Trajectory schema")
    parser.add_argument("--input", required=True, help="Input directory containing incremental JSONL files")
    parser.add_argument("--output", required=True, help="Output directory for per-question JSON")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--benchmark", default="sealqa", help="Benchmark name")
    parser.add_argument("--dataset", default="", help="Dataset name (e.g. seal_0)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Collect all records from JSONL files, deduplicate by question (keep latest)
    records = {}
    input_dir = args.input
    jsonl_files = []

    # Search for JSONL in input dir and subdirs
    for root, dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if f.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, f))

    for fpath in jsonl_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = record.get("question", "")
                if q:
                    records[q] = record

    print(f"Found {len(records)} unique questions from {len(jsonl_files)} JSONL files")

    converted = 0
    for question, record in records.items():
        tid, traj = convert_record(record, args.model, args.benchmark, args.dataset)
        fname = sanitize_filename(tid) + ".json"
        outpath = os.path.join(args.output, fname)
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(traj, f, ensure_ascii=False, indent=2)
        converted += 1

    print(f"Converted {converted} trajectories → {args.output}")


if __name__ == "__main__":
    main()
