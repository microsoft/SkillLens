"""
Shared factory that builds either an ``OpenAI`` or ``AzureOpenAI`` client
depending on the ``OPENAI_CLIENT_TYPE`` environment variable.

Supported values for ``OPENAI_CLIENT_TYPE``:
  - ``"azure"``  (default) – uses Azure Managed Identity auth
  - ``"openai"`` – uses the standard OpenAI API key auth

When using Azure, the model-to-endpoint mapping is read from the
``AZURE_DEPLOYMENT_MAP`` environment variable (JSON), so you never need
to touch this code when endpoints change.
"""

import json
import os

from openai import OpenAI

# Cache the token provider so we only create it once
_token_provider = None
# Cache the parsed deployment map
_deployment_map_cache = None


def _get_deployment_map() -> dict:
    """Lazily load and cache AZURE_DEPLOYMENT_MAP from env (set after dotenv loads)."""
    global _deployment_map_cache
    if _deployment_map_cache is None:
        raw = os.getenv("AZURE_DEPLOYMENT_MAP", "{}")
        try:
            _deployment_map_cache = json.loads(raw)
        except json.JSONDecodeError:
            _deployment_map_cache = {}
    return _deployment_map_cache


def _get_azure_token_provider():
    global _token_provider
    if _token_provider is None:
        from azure.identity import ManagedIdentityCredential, get_bearer_token_provider

        client_id = os.getenv(
            "AZURE_MANAGED_IDENTITY_CLIENT_ID",
            os.getenv("AZURE_CLIENT_ID", ""),
        )
        credential = ManagedIdentityCredential(client_id=client_id)
        _token_provider = get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default",
        )
    return _token_provider


def _resolve_azure_model(model_name: str):
    """
    Given a BFCL model name like ``gpt-5.2-2025-12-11`` or ``gpt-4o-mini-2024-07-18``,
    strip the date suffix and look up the Azure endpoint from ``AZURE_DEPLOYMENT_MAP``.

    Returns (azure_endpoint, api_version, deployment_name).
    """
    deployment_map = _get_deployment_map()

    # Try progressively shorter prefixes to match.
    # e.g. "gpt-5.2-2025-12-11" → "gpt-5.2-2025-12-11", "gpt-5.2-2025-12", "gpt-5.2-2025", "gpt-5.2"
    parts = model_name.split("-")
    for end in range(len(parts), 0, -1):
        candidate = "-".join(parts[:end])
        if candidate in deployment_map:
            entry = deployment_map[candidate]
            return entry["endpoint"], entry["api_version"], candidate

    # Fallback: default endpoint
    default_endpoint = os.getenv(
        "AZURE_OPENAI_ENDPOINT", ""
    )
    default_api_version = os.getenv("AZURE_API_VERSION", "2025-04-01-preview")
    return default_endpoint, default_api_version, model_name


def _build_azure_client(model_name: str):
    from openai import AzureOpenAI

    endpoint, api_version, _deployment = _resolve_azure_model(model_name)
    token_provider = _get_azure_token_provider()

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
        max_retries=5,
    )
    return client


def _build_openai_client():
    kwargs = {}
    if api_key := os.getenv("OPENAI_API_KEY"):
        kwargs["api_key"] = api_key
    if base_url := os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = base_url
    if headers_env := os.getenv("OPENAI_DEFAULT_HEADERS"):
        kwargs["default_headers"] = json.loads(headers_env)
    return OpenAI(**kwargs)


def _is_openai_model(model_name: str) -> bool:
    """Check if the model name looks like an OpenAI model (gpt-*, o3*, o4*)."""
    lower = model_name.lower()
    return any(lower.startswith(p) for p in ("gpt-", "o3", "o4"))


def build_openai_client(model_name: str):
    """
    Build and return an OpenAI-compatible client and the deployment/model name
    to use in API calls.

    Returns ``(client, api_model_name)`` where *api_model_name* is the Azure
    deployment name when using Azure, or the original *model_name* for OpenAI.

    Reads ``OPENAI_CLIENT_TYPE`` from environment:
      - ``"azure"``  (default) → ``AzureOpenAI`` with Managed Identity
      - ``"openai"`` → standard ``OpenAI``

    Non-OpenAI models (e.g. DeepSeek, Qwen) always get a standard OpenAI
    client regardless of the setting, since subclass handlers will override
    ``self.client`` anyway.
    """
    client_type = os.getenv("OPENAI_CLIENT_TYPE", "azure").lower()

    if client_type == "azure" and _is_openai_model(model_name):
        _endpoint, _api_version, deployment = _resolve_azure_model(model_name)
        client = _build_azure_client(model_name)
        return client, deployment
    elif client_type == "openai" or not _is_openai_model(model_name):
        return _build_openai_client(), model_name
    else:
        raise ValueError(
            f"Unknown OPENAI_CLIENT_TYPE='{client_type}'. "
            "Supported values: 'azure' (default), 'openai'."
        )
