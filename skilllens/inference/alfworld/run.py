"""
ALFWorld Inference Evaluation with optional skill augmentation.

This module replicates the original eval_alfworld_api_noray.py logic exactly,
adding only:
  - --skill_set argument for loading a skill set
  - --max_skill_turns for bounding multi-skill tool interactions per step
  - Skill-augmented prompt building via prompt.build_prompt_with_skill()
  - Multi-skill tool protocol handling (```skill``` blocks in model responses)

When --skill_set is NOT provided, behavior is identical to the original script:
same prompt templates, same API call logic, same action projection, same results.
"""

import os
import re
import yaml
import argparse
import numpy as np
import time
import json
import logging
import glob
import threading
import concurrent.futures
from datetime import datetime
from collections import defaultdict

from openai import AzureOpenAI, OpenAI

from skilllens.inference.alfworld.prompt import (
    build_prompt,
    build_prompt_with_skill,
)
from skilllens.inference.skill_provider import ReadOnlySkillProvider
from skilllens.inference.spreadsheetbench.skill_tools import (
    extract_skill_call,
    has_skill_call,
)

# Global lock for TextWorld env load/reset (tatsu parser is not thread-safe)
_env_load_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Path setup — identical to original
# ---------------------------------------------------------------------------

ALF_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "benchmarks", "alfworld", "config_tw.yaml",
)
ALF_CONFIG_PATH = os.path.normpath(ALF_CONFIG_PATH)


def load_config(path):
    assert os.path.exists(path), f"Config not found: {path}"
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Collect game files (lightweight, no env init) — identical to original
# ---------------------------------------------------------------------------

def collect_game_files(env_num, eval_dataset="eval_out_of_distribution", seed=42):
    """Return a list of game file paths, shuffled by seed."""
    from skilllens.benchmarks.alfworld.alfred_tw_env import AlfredTWEnv

    config = load_config(ALF_CONFIG_PATH)
    base = AlfredTWEnv(config, train_eval=eval_dataset)
    game_files = list(base.game_files)
    if env_num > 0:
        game_files = game_files[:env_num]

    rng = np.random.RandomState(seed)
    rng.shuffle(game_files)
    return game_files, config


# ---------------------------------------------------------------------------
# Single-env factory — identical to original
# ---------------------------------------------------------------------------

def make_single_env(config):
    """Create a single TextWorld environment instance."""
    from skilllens.benchmarks.alfworld.alfred_tw_env import (
        AlfredDemangler, AlfredInfos,
    )
    import textworld
    from textworld.envs.wrappers import Filter, GenericEnvironment, Limit

    request_infos = textworld.EnvInfos(
        won=True, admissible_commands=True, extras=["gamefile"]
    )
    max_episode_steps = config["dagger"]["training"]["max_nb_steps_per_episode"]

    env = GenericEnvironment(request_infos)
    env = Limit(env, max_episode_steps=max_episode_steps)
    env = AlfredDemangler(env, shuffle=False)
    env = AlfredInfos(env)
    env = Filter(env)
    return env


# ---------------------------------------------------------------------------
# Action projection — identical to original (lines 124-139)
# ---------------------------------------------------------------------------

TASKS = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


def project_action(raw_response, has_reasoning=False):
    """Extract action from <action>...</action>, return (action_str, is_valid).
    has_reasoning: if True, skip <think> tag check (reasoning came from API field).
    """
    text = raw_response.lower()
    start = text.find("<action>")
    end = text.find("</action>")
    if start == -1 or end == -1:
        return text[-30:], False
    action = text[start + len("<action>"):end].strip()
    if not has_reasoning:
        if "<think>" not in raw_response.lower() or "</think>" not in raw_response.lower():
            return action, False
    if re.search(r'[\u4e00-\u9fff]', raw_response):
        return action, False
    return action, True


# ---------------------------------------------------------------------------
# Helpers — identical to original
# ---------------------------------------------------------------------------

def extract_action_text(model_response):
    match = re.search(r'<action>(.*?)</action>', model_response, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_think_text(model_response):
    match = re.search(r'<think>(.*?)</think>', model_response, re.DOTALL)
    return match.group(1).strip() if match else None


def get_task_type(gamefile):
    for task in TASKS:
        if task in gamefile:
            return task
    return "other"


def extract_task_description(obs_text):
    marker = "Your task is to: "
    idx = obs_text.find(marker)
    if idx != -1:
        return obs_text[idx + len(marker):].strip()
    return ""


def save_trajectory(trajectory, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# API call with retry — identical to original (lines 230-279)
# ---------------------------------------------------------------------------

def call_api(client, prompt, args):
    """Call Azure OpenAI API. Returns (content, reasoning, usage_info)."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            api_kwargs = dict(
                model=args.deployment,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=args.max_completion_tokens,
            )
            if args.temperature is not None:
                api_kwargs["temperature"] = args.temperature
            if args.reasoning_effort is not None:
                api_kwargs["reasoning_effort"] = args.reasoning_effort
            response = client.chat.completions.create(**api_kwargs)
            content = response.choices[0].message.content or ""
            reasoning_text = ""
            msg = response.choices[0].message
            extras = getattr(msg, "model_extra", None) or {}
            if extras.get("reasoning"):
                reasoning_text = extras["reasoning"]
            elif hasattr(msg, "reasoning_content") and msg.reasoning_content:
                reasoning_text = msg.reasoning_content
            else:
                reasoning_text = extract_think_text(content) or ""
            usage_info = {}
            if response.usage:
                usage_info["prompt_tokens"] = response.usage.prompt_tokens
                usage_info["completion_tokens"] = response.usage.completion_tokens
                usage_info["total_tokens"] = response.usage.total_tokens
                if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
                    details = response.usage.completion_tokens_details
                    usage_info["reasoning_tokens"] = getattr(details, "reasoning_tokens", 0) or 0
                else:
                    usage_info["reasoning_tokens"] = 0
            return content.strip(), reasoning_text, usage_info
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 2 ** attempt + np.random.uniform(0, 1)
                time.sleep(wait)
                continue
            if attempt == max_retries - 1:
                logging.warning(f"  API error after {max_retries} retries: {e}")
            else:
                logging.warning(f"  API error (attempt {attempt+1}): {e}")
    return "<action>look</action>", "", {}


# ---------------------------------------------------------------------------
# Call API with multi-skill tool protocol support
# ---------------------------------------------------------------------------

def call_api_with_skills(client, prompt, args, skill_provider, max_skill_turns=3):
    """Call API, handling multi-skill tool interactions if needed.

    If the model returns a ```skill``` block instead of an <action> tag,
    dispatch the skill call, append the result, and re-query the model.

    Returns (content, reasoning, usage_info, skill_interactions)
    where skill_interactions is a list of dicts recording each skill call.
    """
    skill_interactions = []
    messages_text = prompt  # accumulate as a single string for stateless API

    for skill_turn in range(max_skill_turns + 1):  # +1 for final action turn
        content, reasoning, usage_info = call_api(client, messages_text, args)

        # Check for skill call
        if skill_provider is not None:
            skill_call = extract_skill_call(content)
        else:
            skill_call = None

        if skill_call is not None and skill_turn < max_skill_turns:
            fn_name, fn_args = skill_call
            skill_result = skill_provider.dispatch(fn_name, fn_args)

            skill_interactions.append({
                "skill_turn": skill_turn,
                "function": fn_name,
                "arguments": fn_args,
                "result": skill_result,
            })

            # Append skill result and re-query
            skill_msg = (
                f"\n\n**Skill tool result** (`{fn_name}`):\n"
                f"```json\n{skill_result}\n```\n\n"
                "Now proceed to take an action. "
                "Use <think> and <action> tags as instructed."
            )
            messages_text = messages_text + "\n\nAssistant: " + content + "\n\n" + skill_msg
            continue

        elif skill_call is not None and skill_turn >= max_skill_turns:
            # Budget exceeded — append warning and let model try again
            budget_msg = (
                "\n\nYou have reached the maximum number of skill tool calls. "
                "Please take an action now using <action> tags."
            )
            messages_text = messages_text + "\n\nAssistant: " + content + "\n\n" + budget_msg
            # One more API call to get the action
            content, reasoning, usage_info = call_api(client, messages_text, args)

        # No skill call or budget exceeded — return
        return content, reasoning, usage_info, skill_interactions

    # Should not reach here, but just in case
    return content, reasoning, usage_info, skill_interactions


# ---------------------------------------------------------------------------
# Run a single environment end-to-end
# ---------------------------------------------------------------------------

def run_single_env(env_idx, gamefile, config, client, args, traj_dir, test_idx,
                   progress_counter, total_envs, lock,
                   skill_provider=None, max_skill_turns=3):
    """
    Load one env, run it for up to max_steps, save trajectory, release env.
    Thread-safe: each call is fully independent.

    When skill_provider is None, behavior is identical to the original script.
    """
    env = make_single_env(config)
    with _env_load_lock:
        env.load(gamefile)
        obs_text, infos = env.reset()

    admissible = infos.get("admissible_commands", [])
    task_desc = extract_task_description(obs_text)
    task_type = get_task_type(gamefile)

    # Trajectory record
    trajectory = {
        "env_id": env_idx,
        "test_round": test_idx,
        "model": args.deployment,
        "dataset": args.eval_dataset,
        "task_type": task_type,
        "gamefile": gamefile,
        "initial_observation": obs_text,
        "steps": [],
        "success": False,
        "total_steps": 0,
    }
    if skill_provider is not None:
        trajectory["skill_set"] = args.skill_set

    history_records = []
    won = False
    done = False

    # Determine if multi-skill tool protocol is active
    has_tool_protocol = (
        skill_provider is not None
        and not skill_provider.is_single_skill
    )
    # For multi-skill, we need the active provider for dispatch
    active_skill_provider = skill_provider if has_tool_protocol else None

    for step_idx in range(args.max_steps):
        # Build prompt (with skill injection if applicable)
        prompt, _ = build_prompt_with_skill(
            obs_text, admissible, task_desc, history_records,
            init=(step_idx == 0),
            skill_provider=skill_provider,
        )

        # Call API (with or without skill tool protocol)
        if active_skill_provider is not None:
            model_response, reasoning_text, usage_info, skill_interactions = (
                call_api_with_skills(
                    client, prompt, args,
                    active_skill_provider, max_skill_turns,
                )
            )
        else:
            model_response, reasoning_text, usage_info = call_api(client, prompt, args)
            skill_interactions = []

        # Project action
        has_reasoning = bool(reasoning_text and "<think>" not in model_response.lower())
        action, is_valid = project_action(model_response, has_reasoning=has_reasoning)

        # Step env
        with _env_load_lock:
            next_obs, reward, done, next_infos = env.step(action)
        next_admissible = next_infos.get("admissible_commands", [])

        # Record step
        step_record = {
            "step": step_idx,
            "prompt": prompt,
            "model_response": model_response,
            "reasoning": reasoning_text,
            "action": extract_action_text(model_response),
            "env_feedback": next_obs,
            "reward": float(reward),
            "done": bool(done),
            "usage": usage_info,
        }
        if skill_interactions:
            step_record["skill_interactions"] = skill_interactions
        trajectory["steps"].append(step_record)

        # Store history
        history_records.append({"text_obs": obs_text, "action": action})

        # Check done
        if done:
            won = bool(next_infos.get("won", False))
            trajectory["success"] = won
            trajectory["total_steps"] = step_idx + 1
            break

        # Advance
        obs_text = next_obs
        admissible = next_admissible

    if not done:
        trajectory["total_steps"] = args.max_steps

    # Save trajectory
    traj_fp = os.path.join(traj_dir, f"env_{env_idx:03d}.json")
    save_trajectory(trajectory, traj_fp)

    # Log progress
    with lock:
        progress_counter[0] += 1
        done_count = progress_counter[0]
        total_steps = len(trajectory["steps"])
        total_prompt_tokens = sum(s.get("usage", {}).get("prompt_tokens", 0) for s in trajectory["steps"])
        total_completion_tokens = sum(s.get("usage", {}).get("completion_tokens", 0) for s in trajectory["steps"])
        total_reasoning_tokens = sum(s.get("usage", {}).get("reasoning_tokens", 0) for s in trajectory["steps"])
        status = "WON" if won else ("DONE" if done else "MAX_STEPS")
        logging.info(
            f"  [{done_count:3d}/{total_envs}] env_{env_idx:03d} | {status} | "
            f"steps={total_steps} | task={task_type} | "
            f"tokens(p={total_prompt_tokens}, c={total_completion_tokens}, r={total_reasoning_tokens})"
        )

    env.close()
    return env_idx, trajectory


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ALFWorld API Evaluation with optional skill augmentation"
    )
    parser.add_argument("--backend", type=str, default="azure",
                        choices=["azure", "vllm", "gemini"])
    parser.add_argument("--azure_endpoint", type=str, default=None)
    parser.add_argument("--api_key", type=str, default="none")
    parser.add_argument("--vllm_base_url", type=str, default=None)
    parser.add_argument("--deployment", type=str, required=True)
    parser.add_argument("--api_version", type=str, default="2025-04-01-preview")
    parser.add_argument("--env_num", type=int, default=134)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Set to -1 to use model default")
    parser.add_argument("--eval_dataset", type=str, default="eval_out_of_distribution")
    parser.add_argument("--test_times", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_completion_tokens", type=int, default=8192)
    parser.add_argument("--reasoning_effort", type=str, default=None,
                        choices=["none", "minimal", "low", "medium", "high", "xhigh"])
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--resume_dir", type=str, default=None)
    # Skill augmentation arguments
    parser.add_argument("--skill_set", type=str, default=None,
                        help="Path to skill_set.json for skill-augmented evaluation")
    parser.add_argument("--max_skill_turns", type=int, default=3,
                        help="Max skill tool calls per env step (multi-skill mode)")
    parser.add_argument("--exp_tag", type=str, default=None,
                        help="Custom tag to identify setting in output dir name, e.g. "
                             "'ext_gpt-5.4_pool_qwen_eval_qwen'")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory for trajectories and logs")
    args = parser.parse_args()
    if args.temperature < 0:
        args.temperature = None
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load skill provider if specified
    skill_provider = None
    if args.skill_set:
        skill_provider = ReadOnlySkillProvider.from_file(args.skill_set)
        skill_info = (
            f"Skill set: {args.skill_set} "
            f"({skill_provider.skill_count} skills, "
            f"single={skill_provider.is_single_skill})"
        )
    else:
        skill_info = "No skill set (baseline mode)"

    # Resume or create new trajectory directory
    if args.resume_dir:
        traj_dir = args.resume_dir
        assert os.path.isdir(traj_dir), f"Resume dir not found: {traj_dir}"
    elif args.output_dir:
        traj_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = args.deployment.replace("/", "_")
        if args.exp_tag:
            dir_name = f"{args.exp_tag}_{timestamp}"
        else:
            skill_tag = "_skill" if args.skill_set else ""
            dir_name = f"{safe_name}{skill_tag}_{timestamp}"
        traj_dir = os.path.join(
            "inference_output/alfworld",
            dir_name,
        )
    os.makedirs(traj_dir, exist_ok=True)

    os.makedirs("logs/alfworld", exist_ok=True)
    log_fp = os.path.join(traj_dir, "eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.FileHandler(log_fp, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info(f"Args: {vars(args)}")
    logging.info(f"Trajectory directory: {traj_dir}")
    logging.info(skill_info)

    # Init API client
    if args.backend == "azure":
        azure_kwargs = dict(
            api_version=args.api_version,
            azure_endpoint=args.azure_endpoint,
        )
        # Use Managed Identity if no api_key and AZURE_CLIENT_ID is set
        if (not args.api_key or args.api_key == "none") and os.environ.get("AZURE_CLIENT_ID"):
            from azure.identity import ManagedIdentityCredential, get_bearer_token_provider
            cred = ManagedIdentityCredential(client_id=os.environ["AZURE_CLIENT_ID"])
            azure_kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                cred, "https://cognitiveservices.azure.com/.default"
            )
        else:
            azure_kwargs["api_key"] = args.api_key
        client = AzureOpenAI(**azure_kwargs)
    elif args.backend == "gemini":
        base_url = args.vllm_base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
        client = OpenAI(
            api_key=args.api_key,
            base_url=base_url.rstrip("/"),
        )
    else:
        client = OpenAI(
            base_url=args.vllm_base_url,
            api_key=args.api_key,
        )

    # Collect game files
    logging.info(f"Collecting game files (n={args.env_num}, dataset={args.eval_dataset})")
    game_files, tw_config = collect_game_files(args.env_num, args.eval_dataset, args.seed)
    num_envs = len(game_files)
    args.env_num = num_envs
    logging.info(f"Total game files: {num_envs}, max_workers: {args.max_workers}")

    overall_success_rates = []
    task_success_history = defaultdict(list)

    for test_idx in range(args.test_times):
        logging.info(f"\n========== Test round {test_idx + 1}/{args.test_times} ==========")
        start_time = time.time()

        round_dir = traj_dir if args.test_times == 1 else os.path.join(traj_dir, f"round_{test_idx}")
        if args.test_times > 1:
            os.makedirs(round_dir, exist_ok=True)

        # Resume support
        completed_envs = set()
        for existing in glob.glob(os.path.join(round_dir, "env_*.json")):
            env_id = int(os.path.basename(existing).replace("env_", "").replace(".json", ""))
            completed_envs.add(env_id)
        if completed_envs:
            logging.info(f"  Resuming: {len(completed_envs)} envs already completed, skipping them")

        progress_counter = [len(completed_envs)]
        lock = threading.Lock()

        all_trajectories = [None] * num_envs
        for env_id in completed_envs:
            traj_fp = os.path.join(round_dir, f"env_{env_id:03d}.json")
            with open(traj_fp) as f:
                all_trajectories[env_id] = json.load(f)

        envs_to_run = [i for i in range(num_envs) if i not in completed_envs]
        logging.info(f"  Envs to run: {len(envs_to_run)}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            for i in envs_to_run:
                f = executor.submit(
                    run_single_env,
                    env_idx=i,
                    gamefile=game_files[i],
                    config=tw_config,
                    client=client,
                    args=args,
                    traj_dir=round_dir,
                    test_idx=test_idx,
                    progress_counter=progress_counter,
                    total_envs=num_envs,
                    lock=lock,
                    skill_provider=skill_provider,
                    max_skill_turns=args.max_skill_turns,
                )
                futures[f] = i

            for future in concurrent.futures.as_completed(futures):
                idx, traj = future.result()
                all_trajectories[idx] = traj

        # Aggregate results
        overall_success = np.array([t["success"] for t in all_trajectories], dtype=bool)
        task_success_cnt = defaultdict(int)
        task_total_cnt = defaultdict(int)

        for t in all_trajectories:
            task_type = t["task_type"]
            task_total_cnt[task_type] += 1
            if t["success"]:
                task_success_cnt[task_type] += 1

        round_sr = overall_success.mean()
        overall_success_rates.append(round_sr)

        # Token usage summary
        round_prompt_tokens = 0
        round_completion_tokens = 0
        round_reasoning_tokens = 0
        for traj in all_trajectories:
            for step in traj["steps"]:
                u = step.get("usage", {})
                round_prompt_tokens += u.get("prompt_tokens", 0)
                round_completion_tokens += u.get("completion_tokens", 0)
                round_reasoning_tokens += u.get("reasoning_tokens", 0)

        logging.info(f"\n  Round {test_idx + 1} overall success rate: {round_sr:.4f}")
        logging.info(f"  Round {test_idx + 1} total tokens: prompt={round_prompt_tokens}, "
                     f"completion={round_completion_tokens}, reasoning={round_reasoning_tokens}")
        for task in TASKS + ["other"]:
            if task_total_cnt.get(task, 0) > 0:
                rate = task_success_cnt[task] / task_total_cnt[task]
                task_success_history[task].append(rate)
                logging.info(f"    {task:<40s}: {rate:.4f} ({task_success_cnt[task]}/{task_total_cnt[task]})")

        logging.info(f"  Time: {time.time() - start_time:.1f}s")

    # Final summary
    logging.info("\n=============== Final Summary ===============")
    logging.info(f"Model: {args.deployment}")
    logging.info(f"Dataset: {args.eval_dataset}")
    logging.info(f"Overall success rate: {np.mean(overall_success_rates):.4f} +/- {np.std(overall_success_rates):.4f}")

    for task in TASKS:
        if task_success_history.get(task):
            logging.info(f"  {task:<40s}: {np.mean(task_success_history[task]):.4f}")

    results = {
        "model": args.deployment,
        "eval_dataset": args.eval_dataset,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "max_completion_tokens": args.max_completion_tokens,
        "reasoning_effort": args.reasoning_effort,
        "env_num": args.env_num,
        "max_workers": args.max_workers,
        "overall_success_rate_mean": float(np.mean(overall_success_rates)),
        "per_round_success_rates": [float(x) for x in overall_success_rates],
        "task_success_rates": {
            task: {"mean": float(np.mean(task_success_history[task])),
                   "per_round": [float(x) for x in task_success_history[task]]}
            for task in TASKS if task_success_history.get(task)
        },
    }
    if args.skill_set:
        results["skill_set"] = args.skill_set
        results["skill_count"] = skill_provider.skill_count
        results["max_skill_turns"] = args.max_skill_turns

    summary_fp = os.path.join(traj_dir, "summary.json")
    with open(summary_fp, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {summary_fp}")
    logging.info(f"Trajectories saved to {traj_dir}")


if __name__ == "__main__":
    main()
