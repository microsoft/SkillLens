"""LiteResearcher ReAct Agent - multi-turn reasoning with search and visit tools."""

import ast
import json
import os
import re
import time
import asyncio
import random
import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import json5
except ImportError:
    json5 = None

import aiohttp
from openai import OpenAI, AzureOpenAI, APIError, APIConnectionError, APITimeoutError
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# Models that use reasoning and do NOT support temperature / top_p / stop.
_REASONING_MODELS = {"o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"}


def _is_reasoning_model(model: str) -> bool:
    model_lower = model.lower()
    return any(rm in model_lower for rm in _REASONING_MODELS)


def _is_gemini_model(model: str) -> bool:
    if os.environ.get("GEMINI_USE_OPENAI", "").strip():
        return False
    return "gemini" in model.lower()

# ---------------------------------------------------------------------------
# Client factory: supports both OpenAI-compatible and Azure OpenAI backends.
# Controlled by the API_PROVIDER env var ("openai" or "azure").
# ---------------------------------------------------------------------------

_cached_token_provider = None


def _get_azure_token_provider():
    """Lazy-init a token provider for Azure Managed Identity."""
    global _cached_token_provider
    if _cached_token_provider is None:
        from azure.identity import ManagedIdentityCredential, get_bearer_token_provider
        client_id = os.environ.get("AZURE_CLIENT_ID", "")
        credential = ManagedIdentityCredential(client_id=client_id)
        _cached_token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
    return _cached_token_provider


def make_openai_client(timeout: float = 600.0) -> OpenAI:
    """Create an OpenAI-compatible client based on API_PROVIDER."""
    provider = os.environ.get("API_PROVIDER", "openai").strip().lower()
    if provider == "azure":
        return AzureOpenAI(
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            azure_ad_token_provider=_get_azure_token_provider(),
            api_version=os.environ.get("AZURE_API_VERSION", "2025-04-01-preview"),
            max_retries=3,
            timeout=timeout,
        )
    else:
        api_key = os.environ.get("SGLANG_API_KEY", "EMPTY")
        api_base = os.environ.get("SGLANG_API_BASE", "http://127.0.0.1:6001/v1")
        return OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)


def make_judge_client(timeout: float = 300.0) -> OpenAI:
    """Create a client for the judge / summary model.

    Supports a separate endpoint for judging via AZURE_JUDGE_ENDPOINT.
    This is needed when the inference model and judge model are on
    different Azure deployments (e.g. gpt-5.4-mini on searchagent5,
    judge model on Azure endpoint).
    """
    provider = os.environ.get("JUDGE_API_PROVIDER",
                              os.environ.get("API_PROVIDER", "openai")).strip().lower()
    if provider == "azure":
        endpoint = os.environ.get("AZURE_JUDGE_ENDPOINT",
                                  os.environ.get("AZURE_ENDPOINT", ""))
        api_version = os.environ.get("AZURE_JUDGE_API_VERSION",
                                     os.environ.get("AZURE_API_VERSION", "2025-04-01-preview"))
        return AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=_get_azure_token_provider(),
            api_version=api_version,
            max_retries=3,
            timeout=timeout,
        )
    else:
        visit_api_base = os.environ.get("VISIT_API_BASE", "").strip()
        if visit_api_base:
            base_url = visit_api_base
        else:
            summary_ports_str = os.environ.get("SUMMARY_PORTS", "")
            if summary_ports_str:
                ports = []
                for token in summary_ports_str.replace(',', ' ').replace(';', ' ').split():
                    try:
                        ports.append(int(token))
                    except ValueError:
                        pass
                base_url = f"http://127.0.0.1:{ports[0]}/v1" if ports else ""
            else:
                base_url = os.environ.get("API_BASE", "").strip()
        api_key = os.environ.get("API_KEY", "EMPTY")
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

from skilllens.benchmarks.seal0.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_SIMPLE, JUDGE_PROMPT, JUDGE_PROMPT_XBENCH

# ---------------------------------------------------------------------------
# Gemini native client and context cache management
# ---------------------------------------------------------------------------

_gemini_client = None
_gemini_cache_name = None


def _get_gemini_client():
    """Lazy-init a Gemini native client."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY must be set for Gemini models")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _ensure_gemini_cache(model: str, system_prompt: str, skill_text: str) -> str | None:
    """Create a Gemini cached content if skill is large enough. Returns cache name or None."""
    global _gemini_cache_name
    if _gemini_cache_name is not None:
        return _gemini_cache_name if _gemini_cache_name != "" else None

    combined = ""
    if system_prompt:
        combined += system_prompt
    if skill_text:
        combined += "\n\n" + skill_text

    # Need enough content for cache (min 4096 tokens ~ 500+ chars of real content)
    if len(combined) < 500:
        _gemini_cache_name = ""  # mark as attempted
        return None

    try:
        from google.genai.types import Content, Part, CreateCachedContentConfig
        client = _get_gemini_client()
        cache = client.caches.create(
            model=model,
            config=CreateCachedContentConfig(
                contents=[Content(parts=[Part(text=combined)], role="user")],
            )
        )
        _gemini_cache_name = cache.name
        import logging
        logging.getLogger(__name__).info(f"Gemini context cache created: {cache.name}")

        import atexit
        def _cleanup():
            try:
                client.caches.delete(name=cache.name)
            except Exception:
                pass
        atexit.register(_cleanup)

        return _gemini_cache_name
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to create Gemini cache: {e}")
        _gemini_cache_name = ""  # mark as attempted
        return None


# FastAPI service URLs
SEARCH_SERVER_URL = os.environ.get("SEARCH_SERVER_URL", "http://127.0.0.1:8001")
BROWSER_SERVER_URL = os.environ.get("BROWSER_SERVER_URL", "http://127.0.0.1:8002")
TOOL_SERVER_TIMEOUT = int(os.environ.get("TOOL_SERVER_TIMEOUT", 300))


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


class ReActAgent:
    """Multi-turn ReAct agent that calls search/visit tools via HTTP services."""

    def __init__(self, llm=None, **kwargs):
        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        self._api_usage = {"input": 0, "output": 0, "cached": 0}

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_loads(payload: str) -> Optional[Dict]:
        if json5 is not None:
            try:
                return json5.loads(payload)
            except Exception:
                pass
        try:
            return json.loads(payload)
        except Exception:
            pass
        try:
            val = ast.literal_eval(payload)
            if isinstance(val, dict):
                return val
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Answer formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_answer_markup(answer: str) -> str:
        if not isinstance(answer, str):
            return ""
        cleaned = answer.strip()
        if not cleaned:
            return ""
        lower = cleaned.lower()
        if lower.startswith("<answer") and "</answer>" in lower:
            return cleaned
        if "\n" in cleaned:
            return f"<answer>\n{cleaned}\n</answer>"
        return f"<answer>{cleaned}</answer>"

    @staticmethod
    def _extract_tool_interactions(messages: List[Dict]) -> List[Dict]:
        interactions: List[Dict] = []
        for idx, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if "<tool_call>" not in content or "</tool_call>" not in content:
                continue
            tool_call_raw = content.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
            parsed_payload = ReActAgent._safe_loads(tool_call_raw)
            tool_response = None
            if idx + 1 < len(messages):
                next_msg = messages[idx + 1]
                if (
                    next_msg.get("role") == "user"
                    and "<tool_response>" in next_msg.get("content", "")
                    and "</tool_response>" in next_msg.get("content", "")
                ):
                    tool_response = (
                        next_msg["content"].split("<tool_response>", 1)[1]
                        .split("</tool_response>", 1)[0]
                        .strip()
                    )
            interactions.append({
                "tool_call_raw": tool_call_raw,
                "tool_call": parsed_payload,
                "response": tool_response,
            })
        return interactions

    # ------------------------------------------------------------------
    # Judge
    # ------------------------------------------------------------------

    def judge_answer(self, question: str, reference: str, prediction: str, data_path: str = "") -> Dict[str, Any]:
        summary_enabled = os.environ.get("SUMMARY_ENABLE", "true").lower() != "false"
        is_xbench = "xbench" in data_path.lower() if data_path else False
        provider = os.environ.get("JUDGE_API_PROVIDER",
                                  os.environ.get("API_PROVIDER", "openai")).strip().lower()

        model_name = os.environ.get("SUMMARY_MODEL_NAME") or os.environ.get("SUMMARY_MODEL_PATH", "")

        if not summary_enabled:
            return {"status": "skipped", "reason": "disabled", "correct": None,
                    "verdict": None, "reference": reference, "prediction": prediction}

        # For openai provider, still need base_url check
        if provider != "azure":
            visit_api_base = os.environ.get("VISIT_API_BASE", "").strip()
            if visit_api_base:
                base_url = visit_api_base
            else:
                summary_ports_str = os.environ.get("SUMMARY_PORTS", "")
                if summary_ports_str:
                    ports = []
                    for token in summary_ports_str.replace(',', ' ').replace(';', ' ').split():
                        try:
                            ports.append(int(token))
                        except ValueError:
                            pass
                    base_url = f"http://127.0.0.1:{ports[0]}/v1" if ports else ""
                else:
                    base_url = os.environ.get("API_BASE", "").strip()
            if not base_url or not model_name:
                return {"status": "skipped", "reason": "not configured", "correct": None,
                        "verdict": None, "reference": reference, "prediction": prediction}
        else:
            if not model_name:
                return {"status": "skipped", "reason": "not configured", "correct": None,
                        "verdict": None, "reference": reference, "prediction": prediction}

        if is_xbench:
            prompt_text = JUDGE_PROMPT_XBENCH.format(
                question=str(question).strip(), reference=str(reference).strip(),
                prediction=str(prediction).strip())
        else:
            prompt_text = JUDGE_PROMPT.format(
                question=str(question).strip(), reference=str(reference).strip(),
                prediction=str(prediction).strip())

        timeout = float(os.environ.get("SUMMARY_SERVER_TIMEOUT", 300))
        client = make_judge_client(timeout=timeout)
        try:
            max_tokens = 512 if is_xbench else 128
            judge_token_param = "max_completion_tokens" if provider == "azure" else "max_tokens"
            judge_reasoning = os.environ.get("JUDGE_REASONING_EFFORT", "").strip()
            judge_temp = os.environ.get("JUDGE_TEMPERATURE", "").strip()

            create_kwargs = dict(
                model=model_name,
                messages=[{"role": "user", "content": prompt_text}],
            )
            create_kwargs[judge_token_param] = max_tokens

            if judge_reasoning:
                create_kwargs["reasoning_effort"] = judge_reasoning

            # temperature/top_p not supported when reasoning_effort is set
            if not judge_reasoning:
                create_kwargs["temperature"] = float(judge_temp) if judge_temp else 0.0
                create_kwargs["top_p"] = 1.0

            response = client.chat.completions.create(**create_kwargs)
            raw_content = response.choices[0].message.content or "" if response.choices else ""

            if is_xbench:
                normalized = raw_content.strip()
                has_error = "结论" in normalized and "错误" in normalized.split("结论")[-1]
                has_correct = "结论" in normalized and "正确" in normalized.split("结论")[-1]
                is_correct = (not has_error and has_correct) if (has_error or has_correct) else ("正确" in normalized and "错误" not in normalized)
            else:
                normalized = raw_content.strip().lower()
                is_correct = "correct" in normalized and "incorrect" not in normalized

            return {"status": "ok", "correct": is_correct, "verdict": "CORRECT" if is_correct else "INCORRECT",
                    "raw": raw_content, "reference": reference, "prediction": prediction}
        except Exception as exc:
            return {"status": "error", "error": str(exc), "correct": None,
                    "verdict": None, "reference": reference, "prediction": prediction}

    # ------------------------------------------------------------------
    # LLM server interaction
    # ------------------------------------------------------------------

    def _call_gemini_native(self, msgs, max_tokens=10000):
        """Call Gemini model using native SDK with context cache support."""
        from google.genai.types import Content, Part, GenerateContentConfig, ThinkingConfig

        client = _get_gemini_client()

        # Convert OpenAI messages to Gemini format
        system_parts = []
        contents = []
        for msg in msgs:
            role = msg.get("role", "")
            text = msg.get("content", "")
            if role == "system":
                system_parts.append(text)
            elif role == "user":
                contents.append(Content(role="user", parts=[Part(text=text)]))
            elif role == "assistant":
                contents.append(Content(role="model", parts=[Part(text=text)]))

        # Build config
        reasoning_effort = os.environ.get("REASONING_EFFORT", "").strip()
        config = GenerateContentConfig(
            thinking_config=ThinkingConfig(include_thoughts=True),
        )
        if not reasoning_effort:
            config.temperature = self.llm_generate_cfg.get('temperature', 0.6)
            config.top_p = self.llm_generate_cfg.get('top_p', 0.95)

        # Try to use cache
        skill_text = ""
        skill_file = os.environ.get("SKILL_INJECT_FILE", "")
        if skill_file and os.path.isfile(skill_file):
            with open(skill_file, "r", encoding="utf-8") as f:
                skill_text = f.read().strip()
        prompt_text = ""
        prompt_file = os.environ.get("PROMPT_INJECT_FILE", "")
        if prompt_file and os.path.isfile(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()

        system_prompt = "\n".join(system_parts)
        inject_for_cache = "\n\n".join(filter(None, [system_prompt, prompt_text, skill_text]))

        cache_name = _ensure_gemini_cache(self.model, system_prompt, "\n\n".join(filter(None, [prompt_text, skill_text])))
        if cache_name:
            config.cached_content = cache_name
        else:
            if system_prompt:
                config.system_instruction = system_prompt

        config.max_output_tokens = max_tokens

        max_attempts = 5
        last_error = None
        for attempt in range(max_attempts):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                # Accumulate usage
                if response.usage_metadata:
                    um = response.usage_metadata
                    self._api_usage["input"] += getattr(um, 'prompt_token_count', 0) or 0
                    self._api_usage["output"] += getattr(um, 'candidates_token_count', 0) or 0
                    self._api_usage["cached"] += getattr(um, 'cached_content_token_count', 0) or 0

                # Extract text
                text_parts = []
                if response.candidates and response.candidates[0].content:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'thought') and part.thought:
                            continue
                        if part.text:
                            text_parts.append(part.text)
                content = "".join(text_parts).strip()
                if content:
                    return content
                last_error = "empty response from Gemini"
            except Exception as e:
                last_error = str(e)

            wait_time = min(1 * (2 ** (attempt + 1)) + random.uniform(0, 1), 60)
            print(f"[Gemini Retry] attempt={attempt+1}/{max_attempts}, wait={wait_time:.1f}s, error={str(last_error)[:200]}")
            time.sleep(wait_time)

        return f"ERROR: max retries ({max_attempts}) exceeded. Last error: {str(last_error)[:300]}"

    def call_server(self, msgs, planning_port, max_tries=3):
        # Route Gemini models to native SDK for context caching support
        if _is_gemini_model(self.model):
            model_max_length = int(os.environ.get("MAIN_MAX_MODEL_LEN", 90000))
            try:
                prompt_tokens = self.count_tokens(msgs)
            except Exception:
                prompt_tokens = sum(len(str(msg.get("content", ""))) for msg in msgs) // 4
            available_tokens = model_max_length - prompt_tokens - 1000
            max_tokens = max(512, min(available_tokens, 10000))
            return self._call_gemini_native(msgs, max_tokens=max_tokens)

        provider = os.environ.get("API_PROVIDER", "openai").strip().lower()
        if provider == "azure":
            client = make_openai_client(timeout=600.0)
        else:
            api_key = os.environ.get("SGLANG_API_KEY", "EMPTY")
            api_base = os.environ.get("SGLANG_API_BASE")
            if not api_base:
                api_base = f"http://127.0.0.1:{planning_port}/v1"
            client = OpenAI(api_key=api_key, base_url=api_base, timeout=600.0)
        model_max_length = int(os.environ.get("MAIN_MAX_MODEL_LEN", 90000))

        try:
            prompt_tokens = self.count_tokens(msgs)
        except Exception:
            prompt_tokens = sum(len(str(msg.get("content", ""))) for msg in msgs) // 4

        available_tokens = model_max_length - prompt_tokens - 1000
        max_tokens = max(512, min(available_tokens, 10000))

        last_error = None
        empty_response_count = 0
        is_reasoning = _is_reasoning_model(self.model)
        max_attempts = 5
        attempt = 0
        while attempt < max_attempts:
            try:
                token_param = "max_completion_tokens" if provider == "azure" else "max_tokens"
                create_kwargs = dict(
                    model=self.model,
                    messages=msgs,
                )
                create_kwargs[token_param] = max_tokens

                # Reasoning effort (low/medium/high) — supported by reasoning
                # models AND gpt-5.x series
                reasoning_effort = os.environ.get("REASONING_EFFORT", "").strip()
                if reasoning_effort:
                    create_kwargs["reasoning_effort"] = reasoning_effort

                # Always set temperature=1 for all providers
                create_kwargs["temperature"] = self.llm_generate_cfg.get('temperature', 1.0)

                if provider != "azure" and not is_reasoning:
                    create_kwargs["stop"] = ["\n<tool_response>", "<tool_response>"]
                if provider != "azure":
                    # These params are only supported by vLLM/SGLang-style servers
                    if not is_reasoning:
                        create_kwargs.update(
                            frequency_penalty=self.llm_generate_cfg.get('frequency_penalty', 0.0),
                            logprobs=True,
                            presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1),
                        )
                    extra_body = {
                        "top_k": self.llm_generate_cfg.get('top_k', 20),
                        "min_p": self.llm_generate_cfg.get('min_p', 0.0),
                        "repetition_penalty": self.llm_generate_cfg.get('repetition_penalty', 1.0),
                    }
                    # Enable thinking/reasoning for Qwen3.5 models served via sglang
                    if "qwen" in self.model.lower():
                        extra_body["enable_thinking"] = True
                    create_kwargs["extra_body"] = extra_body
                chat_response = client.chat.completions.create(**create_kwargs)
                # Accumulate API token usage
                if hasattr(chat_response, 'usage') and chat_response.usage:
                    u = chat_response.usage
                    self._api_usage["input"] += getattr(u, 'prompt_tokens', 0) or 0
                    self._api_usage["output"] += getattr(u, 'completion_tokens', 0) or 0
                    ptd = getattr(u, 'prompt_tokens_details', None)
                    self._api_usage["cached"] += getattr(ptd, 'cached_tokens', 0) or 0
                content = chat_response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                empty_response_count += 1
                if empty_response_count >= 5:
                    return "ERROR: model returned empty response 5 times consecutively"
                last_error = "empty response from model (content is None or blank)"
            except APIError as e:
                last_error = str(e)
                if "context_length" in last_error.lower() or "maximum context length" in last_error.lower():
                    return f"LENGTH_LIMIT_ERROR: prompt={prompt_tokens}, limit={model_max_length}"
            except (APIConnectionError, APITimeoutError) as e:
                last_error = str(e)
            except Exception as e:
                last_error = str(e)

            attempt += 1
            wait_time = min(1 * (2 ** attempt) + random.uniform(0, 1), 60)
            print(f"[Retry] attempt={attempt}/{max_attempts}, wait={wait_time:.1f}s, error={str(last_error)[:200]}")
            time.sleep(wait_time)

        return f"ERROR: max retries ({max_attempts}) exceeded. Last error: {str(last_error)[:300]}"

    def count_tokens(self, messages):
        if AutoTokenizer is None:
            # Rough estimate: ~4 chars per token
            return sum(len(str(m.get("content", ""))) for m in messages) // 4
        tokenizer_path = os.environ.get("TOKEN_COUNT_MODEL_PATH", self.llm_local_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(full_prompt, return_tensors="pt")
        return len(tokens["input_ids"][0])

    @staticmethod
    def count_messages_tokens_with_template(messages: List[Dict], tokenizer_path: str = None) -> Dict[str, int]:
        import warnings
        if tokenizer_path is None:
            tokenizer_path = os.environ.get("TOKEN_COUNT_MODEL_PATH", "")
        result = {"total_tokens": 0, "system_tokens": 0, "user_tokens": 0,
                  "assistant_tokens": 0, "tokenizer_path": tokenizer_path}
        if not tokenizer_path:
            result["error"] = "TOKEN_COUNT_MODEL_PATH not set"
            return result
        if AutoTokenizer is None:
            result["error"] = "transformers not installed"
            return result
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                tokens = tokenizer(full_prompt, return_tensors="pt")
                result["total_tokens"] = len(tokens["input_ids"][0])
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if content:
                    msg_tokens = len(tokenizer.encode(content, add_special_tokens=False))
                    if role == "system":
                        result["system_tokens"] += msg_tokens
                    elif role == "user":
                        result["user_tokens"] += msg_tokens
                    elif role == "assistant":
                        result["assistant_tokens"] += msg_tokens
        except Exception as e:
            result["error"] = str(e)
        return result

    # ------------------------------------------------------------------
    # Tool calling via HTTP
    # ------------------------------------------------------------------

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        return asyncio.run(self._async_call_tool(tool_name, tool_args))

    async def _async_call_tool(self, tool_name: str, tool_args: dict) -> str:
        timeout = aiohttp.ClientTimeout(total=TOOL_SERVER_TIMEOUT)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if tool_name == "search":
                    payload = {"query": tool_args.get("query", [])}
                    async with session.post(f"{SEARCH_SERVER_URL}/search", json=payload) as resp:
                        data = await resp.json()
                        return data["result"] if data.get("success") else f"[Search] Error: {data.get('error')}"
                elif tool_name == "visit":
                    payload = {"url": tool_args.get("url", ""), "goal": tool_args.get("goal", "")}
                    async with session.post(f"{BROWSER_SERVER_URL}/browse", json=payload) as resp:
                        data = await resp.json()
                        return data["result"] if data.get("success") else f"[Visit] Error: {data.get('error')}"
                else:
                    return f"Error: Tool {tool_name} not found"
        except asyncio.TimeoutError:
            return f"Error: Tool {tool_name} timeout after {TOOL_SERVER_TIMEOUT}s"
        except Exception as e:
            return f"Error: Tool {tool_name} service error: {str(e)}"

    # ------------------------------------------------------------------
    # Result building
    # ------------------------------------------------------------------

    def _build_result(self, question, answer, messages, termination,
                      prediction=None, question_id=None, data_path="",
                      total_time=None, turn_times=None, error=False) -> Dict[str, Any]:
        final_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                final_message = str(msg.get("content", ""))
                break

        if prediction is not None:
            final_answer = prediction.strip() if isinstance(prediction, str) else ""
        else:
            final_answer = final_message.strip() if final_message else ""

        answer_text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        markup = self._format_answer_markup(final_answer)

        result: Dict[str, Any] = {
            "question": question, "answer": answer, "messages": messages,
            "prediction": final_answer, "prediction_tagged": markup,
            "termination": termination, "final_answer": final_answer,
            "final_answer_markup": markup, "final_message": final_message,
            "question_id": question_id,
        }
        if total_time is not None:
            result["total_time"] = total_time
        if turn_times is not None:
            result["turn_times"] = turn_times

        result["token_stats"] = self.count_messages_tokens_with_template(messages)
        result["api_usage"] = dict(self._api_usage)
        result["tool_interactions"] = self._extract_tool_interactions(messages)

        if error:
            result["error"] = termination
            result["judge"] = {"status": "error", "correct": False, "verdict": "ERROR",
                               "reference": answer_text, "prediction": final_answer}
        elif final_answer:
            result["judge"] = self.judge_answer(question, answer_text, final_answer, data_path)
        else:
            result["judge"] = {"status": "no_answer", "correct": False, "verdict": "INCORRECT",
                               "reference": answer_text, "prediction": final_answer}
        return result

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def _run(self, data, model, **kwargs) -> Dict[str, Any]:
        self.model = model
        item_payload = data.get("item", {})
        question_id = data.get("question_id") or item_payload.get("question_id") or item_payload.get("id")
        data_path = data.get("data_path", "")

        try:
            question = item_payload["question"]
        except Exception:
            raw_msg = ""
            msgs = item_payload.get("messages") or []
            if len(msgs) > 1 and isinstance(msgs[1], dict):
                raw_msg = msgs[1].get("content", "")
            elif msgs and isinstance(msgs[0], dict):
                raw_msg = msgs[0].get("content", "")
            question = raw_msg.split("User:", 1)[1].strip() if "User:" in raw_msg else raw_msg
        if not isinstance(question, str):
            question = str(question)

        start_time = time.time()
        planning_port = data["planning_port"]
        answer = item_payload.get("answer", "")

        model_max_ctx = int(os.environ.get("MAIN_MAX_MODEL_LEN", 90000))
        max_timeout = int(os.environ.get("MAX_TIMEOUT_SECONDS", 9000))
        max_calls = int(os.environ.get("MAX_LLM_CALL_PER_RUN", 100))

        system_prompt = SYSTEM_PROMPT + today_date()
        # For Gemini models, skill/prompt will be handled by the native cache path
        _using_gemini = _is_gemini_model(self.model)
        # Inject additional prompt if configured via PROMPT_INJECT_FILE env var
        prompt_file = os.environ.get("PROMPT_INJECT_FILE")
        if prompt_file and os.path.isfile(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
            if prompt_text:
                system_prompt += "\n\n" + prompt_text
        messages = [{"role": "system", "content": system_prompt}]
        # Inject skill if configured via SKILL_INJECT_FILE env var
        # (skip for Gemini — handled by context cache in _call_gemini_native)
        if not _using_gemini:
            skill_file = os.environ.get("SKILL_INJECT_FILE")
            if skill_file and os.path.isfile(skill_file):
                with open(skill_file, "r", encoding="utf-8") as f:
                    skill_text = f.read().strip()
                if skill_text:
                    messages.append({"role": "user", "content": skill_text})
                    messages.append({"role": "assistant", "content": "I understand the skill instructions and will apply them."})
        messages.append({"role": "user", "content": question})

        calls_left = max_calls
        round_idx = 0
        termination = ""
        turn_times: List[Dict[str, Any]] = []

        while calls_left > 0:
            # Timeout check
            if time.time() - start_time > max_timeout:
                return self._build_result(question, answer, messages,
                    f"timeout_after_{max_timeout}s", "No answer found (timeout)",
                    question_id, data_path, time.time() - start_time, turn_times,
                    error=True)

            round_idx += 1
            calls_left -= 1
            turn_start = time.time()

            content = self.call_server(messages, planning_port)
            llm_time = time.time() - turn_start

            if '<tool_response>' in content:
                content = content[:content.find('<tool_response>')]

            # If model generated both <tool_call> and <answer> in one message
            # (happens when stop tokens aren't supported, e.g. Azure reasoning),
            # truncate before <answer> — the answer is hallucinated
            # before seeing the tool result.
            if '<tool_call>' in content and '<answer>' in content:
                tc_end = content.find('</tool_call>')
                ans_start = content.find('<answer>')
                if tc_end != -1 and tc_end < ans_start:
                    # Has closing tag: truncate after </tool_call>
                    content = content[:tc_end + len('</tool_call>')]
                elif tc_end == -1:
                    # No closing tag (e.g. gpt-5.4): truncate before <answer>
                    content = content[:ans_start].rstrip()

            messages.append({"role": "assistant", "content": content.strip()})

            norm = content.strip().lower()

            # Fatal errors → terminate
            if norm.startswith("error:"):
                turn_times.append({"turn": round_idx, "llm_time": llm_time, "tool_time": 0.0,
                                   "total_time": time.time() - turn_start, "action": "api_error"})
                return self._build_result(question, answer, messages, "api_error",
                    content.strip(), question_id, data_path, time.time() - start_time, turn_times,
                    error=True)

            if norm.startswith("length_limit_error"):
                turn_times.append({"turn": round_idx, "llm_time": llm_time, "tool_time": 0.0,
                                   "total_time": time.time() - turn_start, "action": "length_error"})
                return self._build_result(question, answer, messages, "length_limit_exceeded",
                    "Context length exceeded", question_id, data_path, time.time() - start_time, turn_times,
                    error=True)

            if "server_error:" in norm:
                turn_times.append({"turn": round_idx, "llm_time": llm_time, "tool_time": 0.0,
                                   "total_time": time.time() - turn_start, "action": "server_error"})
                return self._build_result(question, answer, messages, "server_error",
                    "Server error", question_id, data_path, time.time() - start_time, turn_times,
                    error=True)

            # Tool call
            tool_time = 0.0
            if '<tool_call>' in content and '</tool_call>' in content:
                tc_str = content.split('<tool_call>')[1].split('</tool_call>')[0]
                tc_start = time.time()
                try:
                    tc = self._safe_loads(tc_str)
                    result = self.custom_call_tool(tc.get('name', ''), tc.get('arguments', {}))
                except Exception:
                    result = 'Error: Invalid tool call JSON.'
                tool_time = time.time() - tc_start
                messages.append({"role": "user", "content": "<tool_response>\n" + result + "\n</tool_response>"})

            # Answer found
            if '<answer>' in content and '</answer>' in content:
                turn_times.append({"turn": round_idx, "llm_time": llm_time, "tool_time": tool_time,
                                   "total_time": time.time() - turn_start, "action": "answer"})
                termination = "answer"
                break

            # Max calls exceeded
            if calls_left <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            turn_times.append({"turn": round_idx, "llm_time": llm_time, "tool_time": tool_time,
                               "total_time": time.time() - turn_start,
                               "action": "tool_call" if '<tool_call>' in content else "thinking"})

            # Context length check → send reminder
            try:
                token_count = self.count_tokens(messages)
            except Exception:
                token_count = 0

            if token_count > model_max_ctx:
                messages[-1]['content'] = (
                    "You have now reached the maximum context length. "
                    "Stop making tool calls and provide your best answer in this format: "
                    "<think>your final thinking</think>\n<answer>your answer</answer>"
                )
                final_start = time.time()
                content = self.call_server(messages, planning_port)
                messages.append({"role": "assistant", "content": content.strip()})

                if '<answer>' in content and '</answer>' in content:
                    prediction = content.split('<answer>')[1].split('</answer>')[0]
                    termination = 'answer_at_context_limit'
                else:
                    prediction = content.strip()
                    termination = 'context_limit_no_format'

                turn_times.append({"turn": round_idx + 1, "llm_time": time.time() - final_start,
                                   "tool_time": 0.0, "total_time": time.time() - final_start,
                                   "action": "context_limit_reminder"})
                return self._build_result(question, answer, messages, termination,
                    prediction, question_id, data_path, time.time() - start_time, turn_times)

        # Post-loop: extract answer (supports partial responses)
        last = messages[-1]['content'] if messages else ""
        if '<answer>' in last:
            parts = last.split('<answer>')
            if len(parts) > 1:
                a = parts[1]
                prediction = a.split('</answer>')[0] if '</answer>' in a else a.strip()
                termination = termination or 'answer'
            else:
                prediction = 'No answer found.'
                termination = 'answer_not_found'
        else:
            prediction = 'No answer found.'
            termination = 'exceed_max_turns' if calls_left == 0 else 'answer_not_found'

        return self._build_result(question, answer, messages, termination,
            prediction, question_id, data_path, time.time() - start_time, turn_times)
