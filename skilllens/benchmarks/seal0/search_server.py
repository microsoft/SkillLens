"""FastAPI server for Search tool (Google Serper / Microsoft AI)"""
import os
import json
import time
import http.client
import logging
import uuid
import requests as _requests
from datetime import datetime
from typing import List, Union
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ==============================================================================
# Configuration via environment variables
# ==============================================================================
# SEARCH_PROVIDER     - "serper" (default) or "microsoft"
# SERPER_KEY_ID       - Serper API Key (required when provider=serper)
# MICROSOFT_SEARCH_API_KEY - Microsoft AI API Key (required when provider=microsoft)
# MICROSOFT_SEARCH_API_URL - Microsoft AI search endpoint (default below)
# SEARCH_SERVER_PORT  - Server port (default: 8001)
# SEARCH_MAX_WORKERS  - Concurrent threads (default: 10)
# ==============================================================================

LOG_DIR = os.environ.get(
    "SEAL0_LOG_DIR",
    os.path.join(os.getcwd(), "logs", "seal0", "search"),
)
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Search Service")
SERPER_KEY = os.environ.get("SERPER_KEY_ID", "")
SEARCH_PROVIDER = os.environ.get("SEARCH_PROVIDER", "serper").strip().lower()
MICROSOFT_SEARCH_API_KEY = os.environ.get("MICROSOFT_SEARCH_API_KEY", "")
MICROSOFT_SEARCH_API_URL = os.environ.get(
    "MICROSOFT_SEARCH_API_URL",
    "https://api.microsoft.ai/v3/search/classic"
)

MAX_WORKERS = int(os.environ.get("SEARCH_MAX_WORKERS", 10))
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
FILTER_UNIFUNCS = os.environ.get("FILTER_UNIFUNCS_ENABLED", "false").lower() == "true"
logger.info(f"Search service configured with provider={SEARCH_PROVIDER}, max_workers={MAX_WORKERS}, filter_unifuncs={FILTER_UNIFUNCS}")


class SearchRequest(BaseModel):
    query: Union[str, List[str]]


class SearchResponse(BaseModel):
    success: bool
    result: str
    elapsed_ms: float
    error: str = ""


def contains_chinese(text: str) -> bool:
    return any('\u4E00' <= c <= '\u9FFF' for c in text)


def google_search_sync(query: str, req_id: str = "") -> str:
    """Perform single Google search via Serper API (sync)"""
    prefix = f"[{req_id}] " if req_id else ""
    is_chinese = contains_chinese(query)
    logger.info(f"{prefix}🔍 Search | query='{query}' | lang={'zh' if is_chinese else 'en'}")

    conn = http.client.HTTPSConnection("google.serper.dev")

    if is_chinese:
        payload = json.dumps({"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"})
    else:
        payload = json.dumps({"q": query, "location": "United States", "gl": "us", "hl": "en"})

    headers = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}

    for i in range(5):
        try:
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            logger.info(f"{prefix}  ✓ API success | attempt: {i+1}/5")
            break
        except Exception as e:
            if i == 4:
                logger.error(f"{prefix}  ✗ API failed | error: timeout")
                return f"Google search timeout for query: '{query}'"
            logger.warning(f"{prefix}  ⚠ API retry | attempt: {i+1}/5 | error: {str(e)[:50]}")
            time.sleep(0.5 * (2 ** i))
            conn = http.client.HTTPSConnection("google.serper.dev")

    data = json.loads(res.read().decode("utf-8"))

    if "organic" not in data:
        logger.warning(f"{prefix}  ⚠ No results")
        return f"No results found for '{query}'."

    snippets, idx = [], 0
    for page in data["organic"]:
        if FILTER_UNIFUNCS and "unifuncs.com" in page.get("link", ""):
            continue
        idx += 1
        date = f"\nDate published: {page['date']}" if "date" in page else ""
        source = f"\nSource: {page['source']}" if "source" in page else ""
        snippet = f"\n{page['snippet']}" if "snippet" in page else ""
        line = f"{idx}. [{page['title']}]({page['link']}){date}{source}\n{snippet}"
        line = line.replace("Your browser can't play this video.", "")
        snippets.append(line)

    result = f"A Google search for '{query}' found {len(snippets)} results:\n\n## Web Results\n" + "\n\n".join(snippets)
    logger.info(f"{prefix}✓ Search done | results: {len(snippets)} | chars: {len(result):,}")
    return result


async def google_search(query: str, req_id: str = "") -> str:
    """Async wrapper for google search"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, google_search_sync, query, req_id)


def microsoft_search_sync(query: str, req_id: str = "") -> str:
    """Perform search via Microsoft AI API (sync)"""
    prefix = f"[{req_id}] " if req_id else ""
    is_chinese = contains_chinese(query)
    lang = "zh" if is_chinese else "en"
    region = "CN" if is_chinese else "US"
    logger.info(f"{prefix}🔍 MS-Search | query='{query}' | lang={lang}")

    headers = {
        "host": "api.microsoft.ai",
        "x-apikey": MICROSOFT_SEARCH_API_KEY,
        "content-type": "application/json",
    }
    payload = {
        "query": query,
        "region": region,
        "language": lang,
        "maxResultsWeb": 10,
        "maxLength": 200,
        "contentFormat": "Passage",
        "responseFilter": ["webResults"],
    }

    for i in range(5):
        try:
            resp = _requests.post(MICROSOFT_SEARCH_API_URL, headers=headers,
                                  json=payload, timeout=30)
            data = resp.json()
            if "errorCode" in data:
                raise ValueError(f"API error: {data.get('errorCode')} - {data.get('userMessage', '')}")
            logger.info(f"{prefix}  ✓ MS API success | attempt: {i+1}/5")
            break
        except Exception as e:
            if i == 4:
                logger.error(f"{prefix}  ✗ MS API failed | error: {str(e)[:80]}")
                return f"Microsoft search failed for query: '{query}': {str(e)[:100]}"
            logger.warning(f"{prefix}  ⚠ MS API retry | attempt: {i+1}/5 | error: {str(e)[:50]}")
            time.sleep(0.5 * (2 ** i))

    web_results = data.get("webResults", [])
    if not web_results:
        logger.warning(f"{prefix}  ⚠ No results")
        return f"No results found for '{query}'."

    snippets = []
    for idx, page in enumerate(web_results, 1):
        url = page.get("url", "")
        if FILTER_UNIFUNCS and "unifuncs.com" in url:
            continue
        title = page.get("title", "")
        snippet_text = page.get("snippet", "")
        signals = page.get("answerSignals", {})
        date = ""
        if signals.get("publishedAt"):
            date = f"\nDate published: {signals['publishedAt']}"
        line = f"{idx}. [{title}]({url}){date}\n{snippet_text}"
        snippets.append(line)

    result = f"A search for '{query}' found {len(snippets)} results:\n\n## Web Results\n" + "\n\n".join(snippets)
    logger.info(f"{prefix}✓ MS-Search done | results: {len(snippets)} | chars: {len(result):,}")
    return result


async def microsoft_search(query: str, req_id: str = "") -> str:
    """Async wrapper for Microsoft AI search"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, microsoft_search_sync, query, req_id)


async def do_search(query: str, req_id: str = "") -> str:
    """Dispatch to the configured search provider"""
    if SEARCH_PROVIDER == "microsoft":
        return await microsoft_search(query, req_id)
    else:
        return await google_search(query, req_id)


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    req_id = str(uuid.uuid4())[:8]
    start = time.time()

    try:
        if SEARCH_PROVIDER == "serper" and not SERPER_KEY:
            raise ValueError("SERPER_KEY_ID not configured")
        if SEARCH_PROVIDER == "microsoft" and not MICROSOFT_SEARCH_API_KEY:
            raise ValueError("MICROSOFT_SEARCH_API_KEY not configured")

        queries = [req.query] if isinstance(req.query, str) else req.query
        logger.info(f"[{req_id}] 📨 Search request | provider={SEARCH_PROVIDER} | queries: {len(queries)}")

        tasks = [do_search(q, req_id) for q in queries]
        results = await asyncio.gather(*tasks)
        result = "\n=======\n".join(results)

        elapsed = (time.time() - start) * 1000
        logger.info(f"[{req_id}] ✅ Search done | {elapsed:.1f}ms | {len(result):,} chars")
        return SearchResponse(success=True, result=result, elapsed_ms=elapsed)

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        logger.error(f"[{req_id}] ❌ Search failed | {elapsed:.1f}ms | {str(e)}")
        return SearchResponse(success=False, result="", elapsed_ms=elapsed, error=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "search"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("SEARCH_SERVER_PORT", 8001))
    logger.info(f"Starting search server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
