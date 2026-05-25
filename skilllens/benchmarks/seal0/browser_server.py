"""FastAPI server for Browser/Visit tool (webpage fetching + LLM summarization)"""
import os
import re
import json
import time
import random
import uuid
import html
import logging
import requests
import tiktoken
import asyncio
import urllib.parse
from datetime import datetime
from typing import List, Union, Iterable
from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI, AsyncAzureOpenAI
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# ==============================================================================
# Configuration via environment variables
# ==============================================================================
# BROWSE_PROVIDER         - "scrapedo" (default) or "microsoft"
# SCRAPEDO_API_KEY        - ScrapeDo API Key (required when provider=scrapedo)
# SCRAPEDO_CUSTOM_WAIT_MS - ScrapeDo render wait time in ms (default: 2000)
# MICROSOFT_BROWSE_API_KEY - Microsoft AI API Key (required when provider=microsoft)
# MICROSOFT_BROWSE_API_URL - Microsoft AI browse endpoint (default below)
# BROWSER_SERVER_PORT     - Server port (default: 8002)
# BROWSER_MAX_WORKERS     - Concurrent threads (default: 5000)
# SUMMARY_PORTS           - Summary service ports, comma separated (default: 7001)
# SUMMARY_MODEL_NAME      - Summary model name
# ==============================================================================

LOG_DIR = os.environ.get(
    "SEAL0_LOG_DIR",
    os.path.join(os.getcwd(), "logs", "seal0", "browser"),
)
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Browser Service")

MAX_WORKERS = int(os.environ.get("BROWSER_MAX_WORKERS", 5000))
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

BROWSE_PROVIDER = os.environ.get("BROWSE_PROVIDER", "scrapedo").strip().lower()
logger.info(f"Browser service configured with provider={BROWSE_PROVIDER}, max_workers={MAX_WORKERS}")

_active_tasks = 0
_task_lock = threading.Lock()
_recent_logs = deque(maxlen=1000)
_log_lock = threading.Lock()

SCRAPEDO_API_KEY = os.environ.get("SCRAPEDO_API_KEY", "")
SCRAPEDO_CUSTOM_WAIT_MS = int(os.environ.get("SCRAPEDO_CUSTOM_WAIT_MS", 2000))
SCRAPEDO_MAX_RETRIES = 1

MICROSOFT_BROWSE_API_KEY = os.environ.get("MICROSOFT_BROWSE_API_KEY", "")
MICROSOFT_BROWSE_API_URL = os.environ.get(
    "MICROSOFT_BROWSE_API_URL",
    "https://api.microsoft.ai/v3/browse"
)

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""


# ==============================================================================
# HTML to Markdown conversion
# ==============================================================================

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def _join_markdown_chunks(chunks: Iterable[str]) -> str:
    filtered = [chunk for chunk in chunks if chunk]
    return "".join(filtered)


def _html_to_markdown_with_bs4(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "canvas"]):
        tag.decompose()

    def render_node(node) -> str:
        from bs4 import NavigableString, Tag

        if isinstance(node, NavigableString):
            return _normalize_whitespace(html.unescape(str(node))) + " "
        if not isinstance(node, Tag):
            return ""

        name = node.name.lower()

        if name == "br":
            return "\n"
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(name[1])
            content = _normalize_whitespace(_join_markdown_chunks(render_node(child) for child in node.children))
            return f"{'#' * level} {content}\n\n" if content else ""
        if name == "p":
            content = _normalize_whitespace(_join_markdown_chunks(render_node(child) for child in node.children))
            return f"{content}\n\n" if content else ""
        if name in {"ul", "ol"}:
            items = []
            for child in node.find_all("li", recursive=False):
                items.append(render_node(child))
            return "".join(items) + "\n"
        if name == "li":
            content = _normalize_whitespace(_join_markdown_chunks(render_node(child) for child in node.children))
            return f"- {content}\n" if content else ""
        if name == "a":
            content = _normalize_whitespace(_join_markdown_chunks(render_node(child) for child in node.children))
            href = node.get("href", "").strip()
            if content and href:
                return f"[{content}]({href})"
            return content
        if name in {"strong", "b"}:
            content = _normalize_whitespace(_join_markdown_chunks(render_node(child) for child in node.children))
            return f"**{content}**" if content else ""
        if name in {"em", "i"}:
            content = _normalize_whitespace(_join_markdown_chunks(render_node(child) for child in node.children))
            return f"*{content}*" if content else ""
        if name == "table":
            rows = node.find_all("tr")
            if not rows:
                return _join_markdown_chunks(render_node(child) for child in node.children)
            max_cols = 0
            for row in rows:
                cols = 0
                for cell in row.find_all(["th", "td"], recursive=False):
                    cols += int(cell.get("colspan", 1))
                max_cols = max(max_cols, cols)
            max_cols = max(max_cols, 1)
            md_rows = []
            for row in rows:
                cells = row.find_all(["th", "td"], recursive=False)
                cell_texts = []
                for cell in cells:
                    text = _normalize_whitespace(
                        _join_markdown_chunks(render_node(c) for c in cell.children)
                    ).replace("|", "\\|")
                    span = int(cell.get("colspan", 1))
                    cell_texts.append(text)
                    for _ in range(span - 1):
                        cell_texts.append("")
                while len(cell_texts) < max_cols:
                    cell_texts.append("")
                md_rows.append("| " + " | ".join(cell_texts[:max_cols]) + " |")
            if md_rows:
                separator = "| " + " | ".join(["---"] * max_cols) + " |"
                md_rows.insert(1, separator)
            return "\n".join(md_rows) + "\n\n"
        if name in {"thead", "tbody", "tfoot"}:
            return _join_markdown_chunks(render_node(child) for child in node.children)
        if name == "tr":
            return ""
        if name in {"td", "th"}:
            return _join_markdown_chunks(render_node(child) for child in node.children)
        return _join_markdown_chunks(render_node(child) for child in node.children)

    body = soup.body or soup
    markdown = _join_markdown_chunks(render_node(child) for child in body.children)
    cleaned = re.sub(r"\n{3,}", "\n\n", markdown)
    return cleaned.strip()


def _html_to_markdown_basic(raw_html: str) -> str:
    text = re.sub(r"<\s*(script|style|noscript|iframe)[^>]*>.*?<\s*/\s*\1\s*>", " ", raw_html, flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p>", "\n\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return _normalize_whitespace(text)


def html_to_markdown(raw_html: str) -> str:
    if not raw_html:
        return ""
    if BeautifulSoup is not None:
        try:
            return _html_to_markdown_with_bs4(raw_html)
        except Exception:
            pass
    return _html_to_markdown_basic(raw_html)


# ==============================================================================
# Request/Response models
# ==============================================================================

class BrowseRequest(BaseModel):
    url: Union[str, List[str]]
    goal: str


class BrowseResponse(BaseModel):
    success: bool
    result: str
    elapsed_ms: float
    error: str = ""


# ==============================================================================
# Helper functions
# ==============================================================================

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = encoding.decode(tokens[:max_tokens])
    logger.warning(f"⚠️  Content truncated: {len(tokens):,} → {max_tokens:,} tokens")
    return truncated


def get_max_content_tokens() -> int:
    return 95000


def get_summary_ports() -> List[int]:
    summary_ports_str = os.environ.get("SUMMARY_PORTS", "7001")
    summary_ports = []
    for token in summary_ports_str.replace(',', ' ').replace(';', ' ').split():
        try:
            summary_ports.append(int(token))
        except ValueError:
            pass
    return summary_ports if summary_ports else [7001]


def _is_chinese_url(url: str) -> bool:
    from urllib.parse import urlparse
    hostname = urlparse(url).hostname or ""
    if hostname.endswith('.cn') or hostname.endswith('.com.cn'):
        return True
    chinese_domains = ['baidu.com', 'zhihu.com', 'bilibili.com', 'csdn.net',
                       'jianshu.com', 'douban.com', 'sina.com', 'sohu.com',
                       'qq.com', 'taobao.com', 'jd.com', '163.com', 'weibo.com']
    return any(hostname.endswith(d) for d in chinese_domains)


def _has_mojibake(text: str, sample_size: int = 500) -> bool:
    sample = text[:sample_size]
    mojibake_count = len(re.findall(r'[åèçäéêëì]{1}[^\u4e00-\u9fff\w\s,.!?;:，。！？；：]{2,5}', sample))
    return mojibake_count >= 3


# ==============================================================================
# Page fetching
# ==============================================================================

def fetch_page_sync(url: str, req_id: str = "", start_time: float = None) -> str:
    """Fetch webpage via direct request first, fallback to Scrapedo API"""
    prefix = f"[{req_id}]" if req_id else ""
    if start_time is None:
        start_time = time.time()

    proxies = {
        "http": os.environ.get("http_proxy", ""),
        "https": os.environ.get("https_proxy", ""),
    }
    # Remove empty proxies
    proxies = {k: v for k, v in proxies.items() if v}

    # Chinese URLs: try Jina Reader first
    if _is_chinese_url(url) and 'r.jina.ai' not in url:
        try:
            jina_url = f"https://r.jina.ai/{url}"
            elapsed = time.time() - start_time
            logger.info(f"{prefix} 🔗 Jina-CN | {elapsed:.1f}s | {url[:60]}")
            jina_api_url = (
                f"https://api.scrape.do/?token={SCRAPEDO_API_KEY}"
                f"&url={urllib.parse.quote(jina_url)}"
            )
            jina_resp = requests.get(jina_api_url, timeout=120, proxies=proxies if proxies else None)
            if jina_resp.encoding and jina_resp.encoding.lower() in ('iso-8859-1', 'latin-1', 'ascii'):
                jina_resp.encoding = jina_resp.apparent_encoding or 'utf-8'
            if jina_resp.status_code == 200 and len(jina_resp.text) > 300:
                jina_body = html_to_markdown(jina_resp.text) if jina_resp.text.strip().startswith('<') else jina_resp.text
                if not _has_mojibake(jina_body):
                    elapsed = time.time() - start_time
                    logger.info(f"{prefix} ✅ Jina-CN | {elapsed:.1f}s | {len(jina_body):,} chars")
                    return jina_body
        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning(f"{prefix} ⚠️  Jina-CN | {elapsed:.1f}s | {str(e)[:80]}")

    # Direct request
    try:
        elapsed = time.time() - start_time
        logger.info(f"{prefix} 🌐 Direct | {elapsed:.1f}s")
        direct_resp = requests.get(url, timeout=60, proxies=proxies if proxies else None)
        if direct_resp.encoding and direct_resp.encoding.lower() in ('iso-8859-1', 'latin-1', 'ascii'):
            direct_resp.encoding = direct_resp.apparent_encoding or 'utf-8'
        if direct_resp.status_code == 200 and len(direct_resp.text) > 300:
            body = html_to_markdown(direct_resp.text)
            if not _has_mojibake(body):
                elapsed = time.time() - start_time
                logger.info(f"{prefix} ✅ Direct | {elapsed:.1f}s | {len(body):,} chars")
                return body
    except Exception as e:
        elapsed = time.time() - start_time
        logger.warning(f"{prefix} ⚠️ Direct | {elapsed:.1f}s | {str(e)}")

    # Scrapedo fallback
    for attempt in range(SCRAPEDO_MAX_RETRIES):
        elapsed = time.time() - start_time
        logger.info(f"{prefix} 🌐 Fetch  | {elapsed:.1f}s | attempt {attempt+1}/{SCRAPEDO_MAX_RETRIES}")
        try:
            api_url = (
                f"https://api.scrape.do/?token={SCRAPEDO_API_KEY}"
                f"&url={urllib.parse.quote(url)}"
                f"&customWait={SCRAPEDO_CUSTOM_WAIT_MS}&render=true"
            )
            resp = requests.get(api_url, timeout=120, proxies=proxies if proxies else None)
            if resp.encoding and resp.encoding.lower() in ('iso-8859-1', 'latin-1', 'ascii'):
                resp.encoding = resp.apparent_encoding or 'utf-8'
            if resp.status_code == 200 and len(resp.text) > 300:
                body = html_to_markdown(resp.text)
                if not _has_mojibake(body):
                    elapsed = time.time() - start_time
                    logger.info(f"{prefix} ✅ Fetch  | {elapsed:.1f}s | {len(body):,} chars")
                    return body
                raise ValueError("Content has mojibake encoding")
            raise ValueError(f"HTTP {resp.status_code}")
        except Exception as e:
            if attempt == SCRAPEDO_MAX_RETRIES - 1:
                elapsed = time.time() - start_time
                logger.warning(f"{prefix} ⚠️ Fetch | {elapsed:.1f}s | {str(e)[:80]}")
            else:
                time.sleep(1)

    # Jina fallback
    if 'r.jina.ai' not in url:
        try:
            jina_url = f"https://r.jina.ai/{url}"
            elapsed = time.time() - start_time
            logger.info(f"{prefix} 🔗 Jina  | fallback | {elapsed:.1f}s")
            jina_api_url = (
                f"https://api.scrape.do/?token={SCRAPEDO_API_KEY}"
                f"&url={urllib.parse.quote(jina_url)}"
            )
            jina_resp = requests.get(jina_api_url, timeout=120, proxies=proxies if proxies else None)
            if jina_resp.encoding and jina_resp.encoding.lower() in ('iso-8859-1', 'latin-1', 'ascii'):
                jina_resp.encoding = jina_resp.apparent_encoding or 'utf-8'
            if jina_resp.status_code == 200 and len(jina_resp.text) > 200:
                jina_body = html_to_markdown(jina_resp.text) if jina_resp.text.strip().startswith('<') else jina_resp.text
                elapsed = time.time() - start_time
                logger.info(f"{prefix} ✅ Jina  | fallback | {elapsed:.1f}s | {len(jina_body):,} chars")
                return jina_body
        except Exception as e:
            elapsed = time.time() - start_time
            logger.warning(f"{prefix} ⚠️ Jina  | fallback | {elapsed:.1f}s | {str(e)[:80]}")

    elapsed = time.time() - start_time
    logger.error(f"{prefix} ❌ Fetch  | all failed | {elapsed:.1f}s | url: {url[:120]}")
    return f"[fetch_error] All methods failed (direct/scrapedo/jina) for {url}"


async def fetch_page(url: str, req_id: str = "", start_time: float = None) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, fetch_page_sync, url, req_id, start_time)


def microsoft_browse_sync(url: str, req_id: str = "", start_time: float = None) -> str:
    """Fetch webpage content via Microsoft AI Browse API (sync)"""
    prefix = f"[{req_id}]" if req_id else ""
    if start_time is None:
        start_time = time.time()

    is_chinese = _is_chinese_url(url)
    lang = "zh" if is_chinese else "en"
    region = "CN" if is_chinese else "US"

    headers = {
        "host": "api.microsoft.ai",
        "x-apikey": MICROSOFT_BROWSE_API_KEY,
        "content-type": "application/json",
    }
    payload = {
        "url": url,
        "maxLength": 30000,
        "language": lang,
        "region": region,
        "contentFormat": "markdown",
    }

    for i in range(3):
        try:
            elapsed = time.time() - start_time
            logger.info(f"{prefix} 🌐 MS-Browse | {elapsed:.1f}s | attempt {i+1}/3 | {url[:80]}")
            resp = requests.post(MICROSOFT_BROWSE_API_URL, headers=headers,
                                 json=payload, timeout=60)
            data = resp.json()
            if "errorCode" in data:
                raise ValueError(f"API error: {data.get('errorCode')} - {data.get('userMessage', '')}")
            content = data.get("content", "")
            if content and len(content) > 100:
                elapsed = time.time() - start_time
                logger.info(f"{prefix} ✅ MS-Browse | {elapsed:.1f}s | {len(content):,} chars")
                return content
            raise ValueError(f"Content too short: {len(content)} chars")
        except Exception as e:
            if i == 2:
                elapsed = time.time() - start_time
                logger.error(f"{prefix} ❌ MS-Browse | {elapsed:.1f}s | {str(e)[:80]}")
                return f"[fetch_error] Microsoft browse failed for {url}: {str(e)[:100]}"
            time.sleep(1 * (2 ** i))

    return f"[fetch_error] Microsoft browse failed for {url}"


async def microsoft_browse(url: str, req_id: str = "", start_time: float = None) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, microsoft_browse_sync, url, req_id, start_time)


def clean_llm_response(content: str) -> str:
    if not content:
        return ""
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
    return content


# ==============================================================================
# LLM Summarization
# ==============================================================================

async def summarize_content(content: str, goal: str, req_id: str = "", start_time: float = None, max_retries: int = 2) -> dict:
    prefix = f"[{req_id}]" if req_id else ""
    if start_time is None:
        start_time = time.time()

    api_key = os.environ.get("API_KEY", "EMPTY")
    model_name = os.environ.get("SUMMARY_MODEL_NAME", "")

    provider = os.environ.get("API_PROVIDER", "openai").strip().lower()
    if provider == "azure":
        from azure.identity import ManagedIdentityCredential, get_bearer_token_provider
        _cid = os.environ.get("AZURE_CLIENT_ID", "")
        _credential = ManagedIdentityCredential(client_id=_cid)
        _token_provider = get_bearer_token_provider(_credential, "https://cognitiveservices.azure.com/.default")
        client = AsyncAzureOpenAI(
            azure_endpoint=os.environ.get("AZURE_JUDGE_ENDPOINT", os.environ.get("AZURE_ENDPOINT", "")),
            azure_ad_token_provider=_token_provider,
            api_version=os.environ.get("AZURE_JUDGE_API_VERSION", os.environ.get("AZURE_API_VERSION", "2025-04-01-preview")),
            max_retries=3,
            timeout=1200.0,
        )
    else:
        summary_ports = get_summary_ports()
        selected_port = random.choice(summary_ports)

        api_base = os.environ.get("VISIT_API_BASE")
        if not api_base:
            api_base = f"http://127.0.0.1:{selected_port}/v1"

        client = AsyncOpenAI(api_key=api_key, base_url=api_base, timeout=1200.0)
    prompt = EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
    MAX_SUMMARY_OUTPUT_TOKENS = 12000

    for attempt in range(max_retries):
        elapsed = time.time() - start_time
        logger.info(f"{prefix} 🤖 Summary | {elapsed:.1f}s | attempt {attempt+1}/{max_retries}")
        try:
            token_kwarg = {"max_completion_tokens": MAX_SUMMARY_OUTPUT_TOKENS} if provider == "azure" else {"max_tokens": MAX_SUMMARY_OUTPUT_TOKENS}
            resp = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                **token_kwarg,
            )
            raw = resp.choices[0].message.content or ""
            raw = clean_llm_response(raw)
            raw = raw.replace("```json", "").replace("```", "").strip()

            left, right = raw.find("{"), raw.rfind("}")
            if left != -1 and right != -1 and left <= right:
                raw = raw[left:right + 1]

            result = json.loads(raw)
            elapsed = time.time() - start_time
            logger.info(f"{prefix} ✅ Summary | {elapsed:.1f}s | done")
            return result
        except json.JSONDecodeError:
            elapsed = time.time() - start_time
            logger.error(f"{prefix} ❌ Summary | JSON error | {elapsed:.1f}s | raw[:300]: {raw[:300]}")
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{prefix} ❌ Summary | {elapsed:.1f}s | {str(e)[:200]}")

        if attempt == max_retries - 1:
            return {"evidence": "Failed to extract", "summary": "Content could not be processed"}

    return {"evidence": "Failed to extract", "summary": "Content could not be processed"}


# ==============================================================================
# URL Processing
# ==============================================================================

async def process_url(url: str, goal: str, req_id: str = "", start_time: float = None) -> str:
    global _active_tasks

    with _task_lock:
        _active_tasks += 1

    prefix = f"[{req_id}]" if req_id else ""
    if start_time is None:
        start_time = time.time()

    try:
        elapsed = time.time() - start_time
        logger.info(f"{prefix} 📨 Browse | provider={BROWSE_PROVIDER} | {elapsed:.1f}s | {url[:80]}")

        if BROWSE_PROVIDER == "microsoft":
            content = await microsoft_browse(url, req_id, start_time)
        else:
            content = await fetch_page(url, req_id, start_time)

        if content.startswith("[fetch_error]"):
            elapsed = time.time() - start_time
            logger.error(f"{prefix} ❌ Browse | {elapsed:.1f}s | fetch failed | url: {url[:120]}")
            result = f"The useful information in {url} for user goal {goal} as follows: \n\n"
            result += "Evidence in page: \nThe provided webpage content could not be accessed.\n\n"
            result += "Summary: \nThe webpage content could not be processed.\n\n"
            return result

        MIN_CONTENT_CHARS = 500
        if len(content.strip()) < MIN_CONTENT_CHARS:
            elapsed = time.time() - start_time
            logger.warning(f"{prefix} ⚠️ Browse | {elapsed:.1f}s | too short ({len(content.strip())} chars)")
            result = f"The useful information in {url} for user goal {goal} as follows: \n\n"
            result += "Evidence in page: \nThe webpage content is too short or could not be fully loaded.\n\n"
            result += "Summary: \nThe webpage content could not be processed.\n\n"
            return result

        max_content_tokens = get_max_content_tokens()
        encoding = tiktoken.get_encoding("cl100k_base")
        original_tokens = len(encoding.encode(content))
        content = truncate_to_tokens(content, max_content_tokens)

        parsed = await summarize_content(content, goal, req_id, start_time)

        evidence = str(parsed.get("evidence", "N/A"))
        summary = str(parsed.get("summary", "N/A"))

        result = f"The useful information in {url} for user goal {goal} as follows: \n\n"
        result += "Evidence in page: \n" + evidence + "\n\n"
        result += "Summary: \n" + summary + "\n\n"

        elapsed = time.time() - start_time
        logger.info(f"{prefix} ✅ Browse | {elapsed:.1f}s | {len(result):,} chars")
        return result
    finally:
        with _task_lock:
            _active_tasks = max(0, _active_tasks - 1)


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.post("/browse", response_model=BrowseResponse)
async def browse(req: BrowseRequest):
    req_id = str(uuid.uuid4())[:8]
    start = time.time()

    try:
        if BROWSE_PROVIDER == "scrapedo" and not SCRAPEDO_API_KEY:
            raise ValueError("SCRAPEDO_API_KEY not configured")
        if BROWSE_PROVIDER == "microsoft" and not MICROSOFT_BROWSE_API_KEY:
            raise ValueError("MICROSOFT_BROWSE_API_KEY not configured")

        urls = [req.url] if isinstance(req.url, str) else req.url
        logger.info(f"[{req_id}] 📨 API | urls: {len(urls)}")

        tasks = [process_url(url, req.goal, req_id, start) for url in urls]
        results = await asyncio.gather(*tasks)
        result = "\n=======\n".join(results)

        elapsed = (time.time() - start) * 1000
        logger.info(f"[{req_id}] ✅ API | {elapsed/1000:.1f}s | {len(result):,} chars")
        return BrowseResponse(success=True, result=result.strip(), elapsed_ms=elapsed)

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        logger.error(f"[{req_id}] ❌ API | {elapsed/1000:.1f}s | {str(e)}")
        return BrowseResponse(success=False, result="", elapsed_ms=elapsed, error=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "browser"}


@app.get("/status")
def get_status():
    with _task_lock:
        active_tasks = _active_tasks
    return {
        "status": "ok",
        "service": "browser",
        "active_tasks": active_tasks,
        "max_workers": MAX_WORKERS,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("BROWSER_SERVER_PORT", 8002))
    logger.info(f"Starting browser server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
