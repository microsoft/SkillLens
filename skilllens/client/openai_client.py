"""
Unified LLM client supporting Azure OpenAI, vLLM, and standard OpenAI endpoints.

Uses the official ``openai`` Python SDK's **Responses API** (``/v1/responses``)
for all calls.  This gives access to reasoning summaries on reasoning models
(gpt-5-mini, o1, o3, …) and also works with vLLM (v0.8.x+).

Includes exponential-backoff retry logic and a ReAct-style tool-use loop.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Callable, Literal

from openai import AsyncAzureOpenAI, AsyncOpenAI, BadRequestError, AuthenticationError
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# Models that use reasoning and do NOT support temperature / top_p params.
# They only accept temperature=1 (default), so we simply omit it.
_REASONING_MODELS = {"gpt-5-mini", "o1", "o1-mini", "o1-preview", "o3", "o3-mini"}


def _is_reasoning_model(model: str) -> bool:
    """Check if a model name corresponds to a reasoning model."""
    model_lower = model.lower()
    return any(rm in model_lower for rm in _REASONING_MODELS)


def _is_retryable(exc: BaseException) -> bool:
    """Return True only for transient/retryable errors (not 400/401/403)."""
    if isinstance(exc, (BadRequestError, AuthenticationError)):
        return False
    return True


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    """Configuration for LLM API access."""

    provider: Literal["openai", "azure", "vllm", "gemini"] = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = ""

    # Azure-specific
    azure_endpoint: str = ""
    api_version: str = "2025-04-01-preview"

    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 4096
    max_retries: int = 3
    timeout: int = 600

    # Reasoning parameters (Responses API)
    reasoning_effort: str = ""  # "", "low", "medium", "high"

    # API style: "responses" (default) or "chat" (Chat Completions API fallback)
    api_style: Literal["responses", "chat"] = "responses"


# ---------------------------------------------------------------------------
# Helpers — format conversion
# ---------------------------------------------------------------------------

def _messages_to_responses_input(
    messages: list[dict],
) -> tuple[str, list[dict]]:
    """Convert Chat Completions messages to Responses API format.

    Returns ``(instructions, conversation_items)`` where *instructions*
    comes from the system message and *conversation_items* are the
    remaining user/assistant/tool messages.
    """
    instructions = ""
    conversation: list[dict] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            instructions = content or ""
        elif role == "user":
            conversation.append({"role": "user", "content": content})
        elif role == "assistant":
            # Pass-through — shouldn't appear in initial messages
            if content:
                conversation.append({"role": "assistant", "content": content})
        elif role == "tool":
            # Shouldn't appear in initial messages either
            conversation.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": content or "",
            })

    return instructions, conversation


def _convert_tool_schemas(tools: list[dict]) -> list[dict]:
    """Convert Chat Completions tool schemas to Responses API format.

    ``{type, function: {name, description, parameters}}``
    → ``{type, name, description, parameters}``
    """
    converted = []
    for tool in tools:
        func = tool.get("function", {})
        converted.append({
            "type": "function",
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {}),
        })
    return converted


def _clean_output_for_input(output_items) -> list[dict]:
    """Clean response output items so they can be fed back as input.

    - ``reasoning`` → skip (output-only)
    - ``function_call`` → keep {type, call_id, name, arguments}
    - ``message`` → convert to {role: "assistant", content: text}
    """
    cleaned: list[dict] = []
    for item in output_items:
        d = item.model_dump() if hasattr(item, "model_dump") else item
        t = d.get("type", "")
        if t == "reasoning":
            continue
        elif t == "function_call":
            fc_item = {
                "type": "function_call",
                "call_id": d["call_id"],
                "name": d["name"],
                "arguments": d["arguments"],
            }
            # Preserve Gemini thought_signature
            if d.get("thought_signature"):
                fc_item["thought_signature"] = d["thought_signature"]
            cleaned.append(fc_item)
        elif t == "message":
            texts = [
                c["text"]
                for c in d.get("content", [])
                if c.get("type") == "output_text"
            ]
            if texts:
                cleaned.append({
                    "role": "assistant",
                    "content": "".join(texts),
                })
    return cleaned


# ---------------------------------------------------------------------------
# Chat Completions → Responses API adapter
# ---------------------------------------------------------------------------

class _ChatToResponsesAdapter:
    """Wraps a Chat Completions response to look like a Responses API response.

    This allows upper-layer code (chat, chat_with_tools) to work unchanged.
    """

    def __init__(self, chat_response):
        self._raw = chat_response
        self.output = self._build_output()
        self.usage = self._build_usage()

    def _build_output(self) -> list:
        """Convert choices[0].message to Responses-API-style output items."""
        items = []
        msg = self._raw.choices[0].message

        # Text content → message item
        if msg.content:
            items.append(_SimpleNamespace(
                type="message",
                content=[_SimpleNamespace(type="output_text", text=msg.content)],
            ))

        # Tool calls → function_call items
        if msg.tool_calls:
            for tc in msg.tool_calls:
                fc_kwargs = dict(
                    type="function_call",
                    call_id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                # Preserve Gemini thought_signature for multi-turn function calling
                extra = getattr(tc, "extra_content", None)
                if extra and isinstance(extra, dict):
                    sig = extra.get("google", {}).get("thought_signature")
                    if sig:
                        fc_kwargs["thought_signature"] = sig
                items.append(_SimpleNamespace(**fc_kwargs))

        return items

    def _build_usage(self):
        """Map Chat Completions usage to Responses-API-style usage."""
        u = self._raw.usage
        if not u:
            return None
        return _SimpleNamespace(
            input_tokens=getattr(u, "prompt_tokens", 0),
            output_tokens=getattr(u, "completion_tokens", 0),
            output_tokens_details=None,
        )


class _SimpleNamespace:
    """Minimal namespace that supports attribute access and model_dump()."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, _SimpleNamespace):
                result[k] = v.model_dump()
            elif isinstance(v, list):
                result[k] = [
                    x.model_dump() if isinstance(x, _SimpleNamespace) else x
                    for x in v
                ]
            else:
                result[k] = v
        return result


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Unified async LLM client using the Responses API."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = self._build_client()

    # -- factory ----------------------------------------------------------

    def _build_client(self) -> AsyncOpenAI | AsyncAzureOpenAI:
        if self.config.provider == "azure":
            import os
            azure_kwargs: dict = {
                "azure_endpoint": self.config.azure_endpoint,
                "api_version": self.config.api_version,
                "timeout": self.config.timeout,
            }
            # Use Managed Identity if AZURE_CLIENT_ID is set and no explicit api_key
            if not self.config.api_key and os.environ.get("AZURE_CLIENT_ID"):
                from azure.identity import ManagedIdentityCredential, get_bearer_token_provider
                cred = ManagedIdentityCredential(client_id=os.environ["AZURE_CLIENT_ID"])
                azure_kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                    cred, "https://cognitiveservices.azure.com/.default"
                )
                logger.info("Azure client using ManagedIdentity (auto-refresh)")
            else:
                azure_kwargs["api_key"] = self.config.api_key
            return AsyncAzureOpenAI(**azure_kwargs)
        if self.config.provider == "gemini":
            base_url = self.config.base_url or "https://generativelanguage.googleapis.com/v1beta/openai"
            return AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=base_url.rstrip("/"),
                timeout=self.config.timeout,
            )
        # Both "openai" and "vllm" use the standard AsyncOpenAI client;
        # the only difference is the base_url.
        kwargs: dict = {
            "api_key": self.config.api_key or "EMPTY",
            "timeout": self.config.timeout,
        }
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return AsyncOpenAI(**kwargs)

    # -- reasoning params -------------------------------------------------

    @property
    def _use_chat_completions(self) -> bool:
        """Whether to use Chat Completions API instead of Responses API."""
        return self.config.api_style == "chat"

    def _build_reasoning_params(self) -> dict | None:
        """Build the ``reasoning`` parameter for the Responses API."""
        params: dict = {"summary": "detailed"}
        if self.config.reasoning_effort:
            params["effort"] = self.config.reasoning_effort
        return params

    # -- public API -------------------------------------------------------

    async def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        """Send a single-turn request via the Responses API.

        Returns the assistant's text content.
        """
        text, _usage = await self.chat_with_usage(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return text

    async def chat_with_usage(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> tuple[str, dict]:
        """Send a single-turn request via the Responses API.

        Returns ``(text, usage)`` where *usage* is a dict with keys
        ``tokens_in``, ``tokens_out``, and ``reasoning_tokens``.
        """
        instructions, conversation = _messages_to_responses_input(messages)
        response = await self._responses_with_retry(
            conversation=conversation,
            instructions=instructions,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Extract text from output
        text_parts: list[str] = []
        for item in response.output:
            if hasattr(item, "type") and item.type == "message":
                for c in (item.content or []):
                    if hasattr(c, "type") and c.type == "output_text":
                        text_parts.append(c.text)

        # Extract usage info
        usage_info: dict = {"tokens_in": 0, "tokens_out": 0, "reasoning_tokens": 0}
        usage = getattr(response, "usage", None)
        if usage:
            usage_info["tokens_in"] = getattr(usage, "input_tokens", 0)
            usage_info["tokens_out"] = getattr(usage, "output_tokens", 0)
            details = getattr(usage, "output_tokens_details", None)
            if details:
                usage_info["reasoning_tokens"] = getattr(
                    details, "reasoning_tokens", 0
                )

        return "".join(text_parts), usage_info

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_handler: Callable[[str, dict], str],
        *,
        max_rounds: int = 20,
        finish_tool_name: str = "finish_extraction",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, list[dict], dict]:
        """Run a ReAct-style tool-use loop via the Responses API.

        Parameters
        ----------
        messages : list[dict]
            Initial messages (system + user) in Chat Completions format.
        tools : list[dict]
            Tool definitions in Chat Completions format
            (``{type, function: {name, …}}``).  Automatically converted
            to Responses API format internally.
        tool_handler : Callable[[str, dict], str]
            ``(function_name, arguments_dict) → result_string``.
        max_rounds : int
            Maximum number of LLM calls (safety limit).
        finish_tool_name : str
            When the LLM calls this tool the loop terminates.
        temperature : float | None
            Override for this call.
        max_tokens : int | None
            Override for this call.

        Returns
        -------
        tuple[str, list[dict], dict]
            ``(final_text, full_messages, stats)`` where *full_messages*
            preserves the Chat Completions message format (with an extra
            ``reasoning_summary`` field on assistant messages) for
            downstream consumers and logging.
        """
        # Convert to Responses API formats
        instructions, conversation = _messages_to_responses_input(messages)
        resp_tools = _convert_tool_schemas(tools)

        # Keep a Chat-Completions-style message log for the caller
        msgs = list(messages)

        stats = {
            "tool_calls_count": 0,
            "rounds": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "reasoning_tokens": 0,
            "finish_reason": "unknown",
        }
        final_text = ""

        for _round in range(max_rounds):
            stats["rounds"] += 1
            logger.info("  Round %d/%d …", stats["rounds"], max_rounds)

            response = await self._responses_with_retry(
                conversation=conversation,
                instructions=instructions,
                tools=resp_tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # -- collect usage --
            usage = getattr(response, "usage", None)
            if usage:
                stats["tokens_in"] += getattr(usage, "input_tokens", 0)
                stats["tokens_out"] += getattr(usage, "output_tokens", 0)
                details = getattr(usage, "output_tokens_details", None)
                if details:
                    stats["reasoning_tokens"] += getattr(
                        details, "reasoning_tokens", 0
                    )

            # -- parse output items --
            reasoning_summaries: list[str] = []
            function_calls: list[dict] = []  # raw dicts
            message_text = ""

            for item in response.output:
                d = item.model_dump() if hasattr(item, "model_dump") else item
                item_type = d.get("type", "")

                if item_type == "reasoning":
                    for s in d.get("summary") or []:
                        txt = s.get("text", "")
                        if txt:
                            reasoning_summaries.append(txt)

                elif item_type == "function_call":
                    function_calls.append(d)

                elif item_type == "message":
                    for c in d.get("content") or []:
                        if c.get("type") == "output_text":
                            message_text += c.get("text", "")

            # -- build Chat-Completions-style assistant message for log --
            assistant_msg: dict = {
                "role": "assistant",
                "content": message_text or None,
            }
            if reasoning_summaries:
                assistant_msg["reasoning_summary"] = (
                    "\n\n".join(reasoning_summaries)
                )
            if function_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": fc["call_id"],
                        "type": "function",
                        "function": {
                            "name": fc["name"],
                            "arguments": fc["arguments"],
                        },
                    }
                    for fc in function_calls
                ]
            msgs.append(assistant_msg)

            # -- extend conversation for next round --
            conversation.extend(_clean_output_for_input(response.output))

            # -- no tool calls → model decided to stop --
            if not function_calls:
                final_text = message_text
                stats["finish_reason"] = "stop"
                break

            # -- process tool calls --
            finished = False
            for fc in function_calls:
                fn_name = fc["name"]
                try:
                    fn_args = json.loads(fc["arguments"])
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}

                result = tool_handler(fn_name, fn_args)
                stats["tool_calls_count"] += 1

                # Append tool result to Chat-Completions-style log
                msgs.append({
                    "role": "tool",
                    "tool_call_id": fc["call_id"],
                    "content": result,
                })

                # Append tool result to Responses API conversation
                conversation.append({
                    "type": "function_call_output",
                    "call_id": fc["call_id"],
                    "output": result,
                })

                # Log
                result_preview = result[:120].replace("\n", " ")
                logger.info(
                    "    → %s(%s) → %s",
                    fn_name,
                    json.dumps(fn_args, ensure_ascii=False)[:80],
                    result_preview,
                )
                logger.debug(
                    "Tool call full result: %s(%s) → %s",
                    fn_name,
                    json.dumps(fn_args, ensure_ascii=False),
                    result,
                )

                if fn_name == finish_tool_name:
                    finished = True

            if finished:
                final_text = message_text
                stats["finish_reason"] = "finish_tool"
                break
        else:
            stats["finish_reason"] = "max_rounds"
            logger.warning("Tool loop exhausted max_rounds=%d", max_rounds)

        logger.info(
            "Tool loop done: rounds=%d, tool_calls=%d, "
            "tokens_in=%d, tokens_out=%d, reasoning=%d, finish=%s",
            stats["rounds"],
            stats["tool_calls_count"],
            stats["tokens_in"],
            stats["tokens_out"],
            stats["reasoning_tokens"],
            stats["finish_reason"],
        )

        return final_text, msgs, stats

    # -- internals --------------------------------------------------------

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=120),
        retry=retry_if_exception(_is_retryable),
        before_sleep=lambda rs: logger.warning(
            "LLM call failed (attempt %d): %s — retrying …",
            rs.attempt_number,
            rs.outcome.exception() if rs.outcome else "unknown",
        ),
    )
    async def _responses_with_retry(
        self,
        conversation: list[dict],
        instructions: str = "",
        tools: list[dict] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Low-level API call with retry.

        Uses Responses API by default; falls back to Chat Completions API
        for providers that don't support it (e.g. gemini).
        """
        if self._use_chat_completions:
            return await self._chat_completions_call(
                conversation, instructions, tools,
                temperature=temperature, max_tokens=max_tokens,
            )
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        params: dict = {
            "model": self.config.model,
            "input": conversation,
        }

        if instructions:
            params["instructions"] = instructions

        if tools:
            params["tools"] = tools

        # Reasoning parameters
        reasoning = self._build_reasoning_params()
        if reasoning:
            params["reasoning"] = reasoning

        # Temperature — reasoning models only support default (1.0)
        if not _is_reasoning_model(self.config.model):
            params["temperature"] = temp

        # Max output tokens
        if tokens:
            params["max_output_tokens"] = tokens

        t0 = time.time()
        response = await self._client.responses.create(**params)
        elapsed = time.time() - t0

        usage = getattr(response, "usage", None)
        reasoning_tokens = 0
        if usage:
            details = getattr(usage, "output_tokens_details", None)
            if details:
                reasoning_tokens = getattr(details, "reasoning_tokens", 0)

        logger.info(
            "LLM response: model=%s tokens_in=%s tokens_out=%s "
            "reasoning=%s time=%.1fs",
            self.config.model,
            getattr(usage, "input_tokens", "?") if usage else "?",
            getattr(usage, "output_tokens", "?") if usage else "?",
            reasoning_tokens,
            elapsed,
        )

        return response

    async def _chat_completions_call(
        self,
        conversation: list[dict],
        instructions: str = "",
        tools: list[dict] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Chat Completions API fallback for providers without Responses API.

        Converts inputs/outputs so the caller sees the same interface as
        ``_responses_with_retry`` returning a Responses-API-like object.
        """
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        # Build Chat Completions messages from Responses API conversation
        messages: list[dict] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})

        for item in conversation:
            if item.get("role") in ("user", "assistant"):
                messages.append({"role": item["role"], "content": item.get("content", "")})
            elif item.get("type") == "function_call":
                # Assistant's tool call
                tc_dict = {
                    "id": item["call_id"],
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "arguments": item["arguments"],
                    },
                }
                # Pass back Gemini thought_signature if present
                if item.get("thought_signature"):
                    tc_dict["extra_content"] = {
                        "google": {"thought_signature": item["thought_signature"]}
                    }
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc_dict],
                })
            elif item.get("type") == "function_call_output":
                # Tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": item["call_id"],
                    "content": item.get("output", ""),
                })

        params: dict = {
            "model": self.config.model,
            "messages": messages,
        }

        if tools:
            # tools are already in Responses API format; convert back to Chat format
            chat_tools = []
            for t in tools:
                chat_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {}),
                    },
                })
            params["tools"] = chat_tools

        if not _is_reasoning_model(self.config.model):
            params["temperature"] = temp

        if self.config.reasoning_effort:
            params["reasoning_effort"] = self.config.reasoning_effort

        if tokens:
            params["max_completion_tokens"] = tokens

        t0 = time.time()
        response = await self._client.chat.completions.create(**params)
        elapsed = time.time() - t0

        # Wrap into a Responses-API-like object
        usage = response.usage
        logger.info(
            "LLM response (chat): model=%s tokens_in=%s tokens_out=%s time=%.1fs",
            self.config.model,
            getattr(usage, "prompt_tokens", "?") if usage else "?",
            getattr(usage, "completion_tokens", "?") if usage else "?",
            elapsed,
        )

        return _ChatToResponsesAdapter(response)
