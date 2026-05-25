import time
import logging
from typing import List
from openai import AzureOpenAI, OpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def _build_client(opt):
    """Build OpenAI client based on backend type.

    Supports:
      - Azure OpenAI (default): requires --base_url with *.azure.com
      - vLLM / OpenAI-compatible: use --backend vllm, --base_url http://host:port/v1
    """
    backend = getattr(opt, 'backend', None)

    if backend == 'vllm':
        # vLLM exposes an OpenAI-compatible API
        base_url = opt.base_url.rstrip('/')
        if not base_url.endswith('/v1'):
            base_url += '/v1'
        return OpenAI(
            api_key=getattr(opt, 'api_key', None) or 'EMPTY',
            base_url=base_url,
        )
    elif backend == 'gemini':
        # Gemini OpenAI-compatible API
        base_url = getattr(opt, 'base_url', '') or 'https://generativelanguage.googleapis.com/v1beta/openai/'
        return OpenAI(
            api_key=opt.api_key,
            base_url=base_url.rstrip('/'),
        )
    else:
        # Default: Azure OpenAI
        import os
        azure_kwargs = {
            "azure_endpoint": opt.base_url,
            "api_version": getattr(opt, 'api_version', '2025-04-01-preview'),
        }
        # Use Managed Identity if no api_key and AZURE_CLIENT_ID is set
        if not getattr(opt, 'api_key', None) and os.environ.get("AZURE_CLIENT_ID"):
            from azure.identity import ManagedIdentityCredential, get_bearer_token_provider
            cred = ManagedIdentityCredential(client_id=os.environ["AZURE_CLIENT_ID"])
            azure_kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                cred, "https://cognitiveservices.azure.com/.default"
            )
        else:
            azure_kwargs["api_key"] = opt.api_key
        return AzureOpenAI(**azure_kwargs)


def get_llm_response(messages: List[str], opt, max_retries=5, return_usage=False):
    client = _build_client(opt)
    formatted_messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": messages[i]}
        for i in range(len(messages))
    ]

    kwargs = {
        "messages": formatted_messages,
        "model": opt.model,
    }
    reasoning_effort = getattr(opt, 'reasoning_effort', None)
    # For vLLM / reasoning models, ensure sufficient max_completion_tokens
    backend = getattr(opt, 'backend', None)
    if backend == 'vllm':
        kwargs["max_completion_tokens"] = getattr(opt, 'max_completion_tokens', 16384)

    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    for attempt in range(1, max_retries + 1):
        try:
            start_time = time.time()
            chat_completion = client.chat.completions.create(**kwargs)
            elapsed = time.time() - start_time

            content = chat_completion.choices[0].message.content
            # Some reasoning models (e.g. Qwen3.5) may return content=None
            # if thinking exhausted the token budget
            if content is None:
                content = ""
                logger.warning(
                    f"API OK but content=None (reasoning model budget exhausted?) | "
                    f"model={opt.model}"
                )
            usage = chat_completion.usage
            logger.info(
                f"API OK | model={opt.model} | attempt={attempt} | "
                f"latency={elapsed:.1f}s | "
                f"prompt_tokens={usage.prompt_tokens} | "
                f"completion_tokens={usage.completion_tokens} | "
                f"total_tokens={usage.total_tokens} | "
                f"response_len={len(content)}"
            )
            if return_usage:
                return content, {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
            return content

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            # Check if it's a rate limit error (429)
            is_rate_limit = '429' in error_msg or 'rate' in error_msg.lower()
            # Check if it's a server error (5xx)
            is_server_error = any(code in error_msg for code in ['500', '502', '503', '504'])

            if is_rate_limit or is_server_error:
                # Exponential backoff: 10s, 20s, 40s, 80s, 160s
                wait_time = 10 * (2 ** (attempt - 1))
                logger.warning(
                    f"API RETRY | model={opt.model} | attempt={attempt}/{max_retries} | "
                    f"error={error_type}: {error_msg[:200]} | "
                    f"waiting={wait_time}s"
                )
                if attempt < max_retries:
                    time.sleep(wait_time)
                    continue

            # Non-retryable error or max retries exceeded
            logger.error(
                f"API FAIL | model={opt.model} | attempt={attempt}/{max_retries} | "
                f"error={error_type}: {error_msg[:300]}"
            )
            raise
