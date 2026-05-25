import os
import random
import time
from typing import Optional
from urllib.parse import urlparse

import html2text
import requests
from bs4 import BeautifulSoup

# SerpAPI is optional — only needed when SEARCH_PROVIDER=serpapi
try:
    from serpapi import GoogleSearch
except ImportError:
    GoogleSearch = None

ERROR_TEMPLATES = [
    "503 Server Error: Service Unavailable for url: {url}",
    "429 Client Error: Too Many Requests for url: {url}",
    "403 Client Error: Forbidden for url: {url}",
    (
        "HTTPSConnectionPool(host='{host}', port=443): Max retries exceeded with url: {path} "
        "(Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x{id1:x}>, "
        "'Connection to {host} timed out. (connect timeout=5)'))"
    ),
    "HTTPSConnectionPool(host='{host}', port=443): Read timed out. (read timeout=5)",
    (
        "Max retries exceeded with url: {path} "
        "(Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x{id2:x}>: "
        "Failed to establish a new connection: [Errno -2] Name or service not known'))"
    ),
]


class WebSearchAPI:
    def __init__(self):
        self._api_description = "This tool belongs to the Web Search API category. It provides functions to search the web and browse search results."
        self.show_snippet = True
        # Note: The following two random generators are used to simulate random errors, but that feature is not currently used
        # This one used to determine if we should simulate a random error
        # Outcome (True means simulate error): [True, False, True, True, False, True, True, True, False, False, True, True, False, True, False, False, False, False, False, True]
        self._random = random.Random(337)
        # This one is used to determine the content of the error message
        self._rng = random.Random(1053)

    def _load_scenario(self, initial_config: dict, long_context: bool = False):
        # We don't care about the long_context parameter here
        # It's there to match the signature of functions in the multi-turn evaluation code
        self.show_snippet = initial_config["show_snippet"]

    def search_engine_query(
        self,
        keywords: str,
        max_results: Optional[int] = 10,
        region: Optional[str] = "wt-wt",
    ) -> list:
        """
        This function queries the search engine for the provided keywords and region.

        Args:
            keywords (str): The keywords to search for.
            max_results (int, optional): The maximum number of search results to return. Defaults to 10.
            region (str, optional): The region to search in. Defaults to "wt-wt". Possible values include:
                - xa-ar for Arabia
                - xa-en for Arabia (en)
                - ar-es for Argentina
                - au-en for Australia
                - at-de for Austria
                - be-fr for Belgium (fr)
                - be-nl for Belgium (nl)
                - br-pt for Brazil
                - bg-bg for Bulgaria
                - ca-en for Canada
                - ca-fr for Canada (fr)
                - ct-ca for Catalan
                - cl-es for Chile
                - cn-zh for China
                - co-es for Colombia
                - hr-hr for Croatia
                - cz-cs for Czech Republic
                - dk-da for Denmark
                - ee-et for Estonia
                - fi-fi for Finland
                - fr-fr for France
                - de-de for Germany
                - gr-el for Greece
                - hk-tzh for Hong Kong
                - hu-hu for Hungary
                - in-en for India
                - id-id for Indonesia
                - id-en for Indonesia (en)
                - ie-en for Ireland
                - il-he for Israel
                - it-it for Italy
                - jp-jp for Japan
                - kr-kr for Korea
                - lv-lv for Latvia
                - lt-lt for Lithuania
                - xl-es for Latin America
                - my-ms for Malaysia
                - my-en for Malaysia (en)
                - mx-es for Mexico
                - nl-nl for Netherlands
                - nz-en for New Zealand
                - no-no for Norway
                - pe-es for Peru
                - ph-en for Philippines
                - ph-tl for Philippines (tl)
                - pl-pl for Poland
                - pt-pt for Portugal
                - ro-ro for Romania
                - ru-ru for Russia
                - sg-en for Singapore
                - sk-sk for Slovak Republic
                - sl-sl for Slovenia
                - za-en for South Africa
                - es-es for Spain
                - se-sv for Sweden
                - ch-de for Switzerland (de)
                - ch-fr for Switzerland (fr)
                - ch-it for Switzerland (it)
                - tw-tzh for Taiwan
                - th-th for Thailand
                - tr-tr for Turkey
                - ua-uk for Ukraine
                - uk-en for United Kingdom
                - us-en for United States
                - ue-es for United States (es)
                - ve-es for Venezuela
                - vn-vi for Vietnam
                - wt-wt for No region

        Returns:
            list: A list of search result dictionaries, each containing information such as:
            - 'title' (str): The title of the search result.
            - 'href' (str): The URL of the search result.
            - 'body' (str): A brief description or snippet from the search result.
        """
        provider = os.getenv("SEARCH_PROVIDER", "serper").lower()

        if provider == "serper":
            return self._search_via_serper(keywords, max_results, region)
        elif provider == "serpapi":
            return self._search_via_serpapi(keywords, max_results, region)
        elif provider == "microsoft":
            return self._search_via_microsoft(keywords, max_results, region)
        elif provider == "server":
            return self._search_via_server(keywords, max_results, region)
        else:
            raise ValueError(
                f"Unknown SEARCH_PROVIDER='{provider}'. Supported values: 'serper', 'serpapi', 'microsoft', 'server'."
            )

    # ------------------------------------------------------------------ #
    #  Serper (google.serper.dev) backend — default                       #
    # ------------------------------------------------------------------ #
    def _search_via_serper(
        self, keywords: str, max_results: int, region: str
    ) -> list:
        backoff = 2
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": os.getenv("SERPER_API_KEY", ""),
            "Content-Type": "application/json",
        }
        payload = {"q": keywords, "num": max_results}
        # Map BFCL region codes (e.g. "us-en") to Serper gl/hl params
        if region and region != "wt-wt":
            parts = region.split("-")
            payload["gl"] = parts[0]
            if len(parts) > 1:
                payload["hl"] = parts[1]

        while True:
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                if response.status_code == 429:
                    wait_time = backoff + random.uniform(0, backoff)
                    print(
                        "*" * 100
                        + f"\n❗️❗️ [WebSearchAPI] Received 429 from Serper. Retrying in {wait_time:.1f}s…"
                        + "*" * 100
                    )
                    time.sleep(wait_time)
                    backoff = min(backoff * 2, 120)
                    continue
                response.raise_for_status()
                search_results = response.json()
            except requests.exceptions.HTTPError:
                raise
            except Exception as e:
                print(
                    "*" * 100
                    + f"\n❗️❗️ [WebSearchAPI] Serper error: {e}"
                    + "*" * 100
                )
                return {"error": str(e)}
            break

        organic = search_results.get("organic", [])
        if not organic:
            return {
                "error": "Failed to retrieve the search results from server. Please try again later."
            }

        results = []
        for item in organic[:max_results]:
            entry = {"title": item.get("title", ""), "href": item.get("link", "")}
            if self.show_snippet:
                entry["body"] = item.get("snippet", "")
            results.append(entry)
        return results

    # ------------------------------------------------------------------ #
    #  Microsoft AI (api.microsoft.ai) backend                            #
    # ------------------------------------------------------------------ #
    def _search_via_microsoft(
        self, keywords: str, max_results: int, region: str
    ) -> list:
        backoff = 2
        api_url = os.getenv(
            "MICROSOFT_SEARCH_API_URL",
            "https://api.microsoft.ai/v3/search/classic",
        )
        api_key = os.getenv("MICROSOFT_SEARCH_API_KEY", "")
        headers = {
            "host": "api.microsoft.ai",
            "x-apikey": api_key,
            "content-type": "application/json",
        }

        # Map BFCL region codes (e.g. "us-en") to Microsoft region/language
        ms_region = "US"
        ms_lang = "en"
        if region and region != "wt-wt":
            parts = region.split("-")
            ms_region = parts[0].upper()
            if len(parts) > 1:
                ms_lang = parts[1]

        payload = {
            "query": keywords,
            "region": ms_region,
            "language": ms_lang,
            "maxResultsWeb": max_results,
            "maxLength": 200,
            "contentFormat": "Passage",
            "responseFilter": ["webResults"],
        }

        while True:
            try:
                response = requests.post(
                    api_url, headers=headers, json=payload, timeout=30
                )
                data = response.json()
                if data.get("errorCode") == "AuthUserThrottled":
                    wait_time = backoff + random.uniform(0, backoff)
                    print(
                        "*" * 100
                        + f"\n❗️❗️ [WebSearchAPI] Received rate limit from Microsoft AI. Retrying in {wait_time:.1f}s…"
                        + "*" * 100
                    )
                    time.sleep(wait_time)
                    backoff = min(backoff * 2, 120)
                    continue
                if "errorCode" in data:
                    return {"error": f"Microsoft AI error: {data.get('errorCode')} - {data.get('userMessage', '')}"}
                break
            except Exception as e:
                print(
                    "*" * 100
                    + f"\n❗️❗️ [WebSearchAPI] Microsoft AI error: {e}"
                    + "*" * 100
                )
                return {"error": str(e)}

        web_results = data.get("webResults", [])
        if not web_results:
            return {
                "error": "Failed to retrieve the search results from server. Please try again later."
            }

        results = []
        for item in web_results[:max_results]:
            entry = {"title": item.get("title", ""), "href": item.get("url", "")}
            if self.show_snippet:
                entry["body"] = item.get("snippet", "")
            results.append(entry)
        return results

    # ------------------------------------------------------------------ #
    #  Centralized server backend (POST to SEARCH_SERVER_URL/search)      #
    # ------------------------------------------------------------------ #
    def _search_via_server(
        self, keywords: str, max_results: int, region: str
    ) -> list:
        server_url = os.getenv("SEARCH_SERVER_URL", "http://127.0.0.1:8003")
        backoff = 2

        while True:
            try:
                response = requests.post(
                    f"{server_url}/search",
                    json={"query": keywords},
                    timeout=30,
                )
                data = response.json()
                if not data.get("success"):
                    return {"error": data.get("error", "Unknown server error")}
                break
            except Exception as e:
                if "429" in str(e) or "throttl" in str(e).lower():
                    wait_time = backoff + random.uniform(0, backoff)
                    print(f"❗️ [WebSearchAPI] Server throttled. Retrying in {wait_time:.1f}s…")
                    time.sleep(wait_time)
                    backoff = min(backoff * 2, 120)
                    continue
                return {"error": str(e)}

        # Parse the server's text result back into [{title, href, body}]
        result_text = data.get("result", "")
        results = []
        import re
        for match in re.finditer(
            r'\d+\.\s+\[([^\]]*)\]\(([^)]*)\)(?:\nDate published: [^\n]*)?\n(.*?)(?=\n\n\d+\.|$)',
            result_text, re.DOTALL
        ):
            entry = {"title": match.group(1), "href": match.group(2)}
            if self.show_snippet:
                entry["body"] = match.group(3).strip()
            results.append(entry)
            if len(results) >= max_results:
                break

        if not results:
            return {"error": "Failed to parse search results from server."}
        return results

    # ------------------------------------------------------------------ #
    #  SerpAPI (serpapi.com) backend — original implementation             #
    # ------------------------------------------------------------------ #
    def _search_via_serpapi(
        self, keywords: str, max_results: int, region: str
    ) -> list:
        if GoogleSearch is None:
            raise ImportError(
                "serpapi package is required when SEARCH_PROVIDER=serpapi. "
                "Install it with: pip install google-search-results"
            )

        backoff = 2
        params = {
            "engine": "duckduckgo",
            "q": keywords,
            "kl": region,
            "api_key": os.getenv("SERPAPI_API_KEY"),
        }

        while True:
            try:
                search = GoogleSearch(params)
                search_results = search.get_dict()
            except Exception as e:
                if "429" in str(e):
                    wait_time = backoff + random.uniform(0, backoff)
                    print(
                        "*" * 100
                        + f"\n❗️❗️ [WebSearchAPI] Received 429 from SerpAPI. Retrying in {wait_time:.1f}s…"
                        + "*" * 100
                    )
                    time.sleep(wait_time)
                    backoff = min(backoff * 2, 120)
                    continue
                else:
                    print(
                        "*" * 100
                        + f"\n❗️❗️ [WebSearchAPI] SerpAPI error: {e}"
                        + "*" * 100
                    )
                    return {"error": str(e)}

            if "error" in search_results and "429" in str(search_results["error"]):
                wait_time = backoff + random.uniform(0, backoff)
                print(
                    "*" * 100
                    + f"\n❗️❗️ [WebSearchAPI] Received 429 from SerpAPI. Retrying in {wait_time:.1f}s…"
                    + "*" * 100
                )
                time.sleep(wait_time)
                backoff = min(backoff * 2, 120)
                continue

            break

        if "organic_results" not in search_results:
            return {
                "error": "Failed to retrieve the search results from server. Please try again later."
            }

        search_results = search_results["organic_results"]
        results = []
        for result in search_results[:max_results]:
            entry = {"title": result["title"], "href": result["link"]}
            if self.show_snippet:
                entry["body"] = result["snippet"]
            results.append(entry)
        return results

    def fetch_url_content(self, url: str, mode: str = "raw") -> str:
        """
        This function retrieves content from the provided URL and processes it based on the selected mode.

        Args:
            url (str): The URL to fetch content from. Must start with 'http://' or 'https://'.
            mode (str, optional): The mode to process the fetched content. Defaults to "raw".
                Supported modes are:
                    - "raw": Returns the raw HTML content.
                    - "markdown": Converts raw HTML content to Markdown format for better readability, using html2text.
                    - "truncate": Extracts and cleans text by removing scripts, styles, and extraneous whitespace.
        """
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")

        try:
            # A header that mimics a browser request. This helps avoid 403 Forbidden errors.
            # TODO: Is this the best way to do this?
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/112.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Referer": "https://www.google.com/",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-User": "?1",
                "Sec-Fetch-Dest": "document",
            }
            response = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
            response.raise_for_status()

            # Note: Un-comment this when we want to simulate a random error
            # Flip a coin to simulate a random error
            # if self._random.random() < 0.95:
            #     return {"error": self._fake_requests_get_error_msg(url)}

            # Process the response based on the mode
            if mode == "raw":
                return {"content": response.text}

            elif mode == "markdown":
                converter = html2text.HTML2Text()
                markdown = converter.handle(response.text)
                return {"content": markdown}

            elif mode == "truncate":
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove scripts and styles
                for script_or_style in soup(["script", "style"]):
                    script_or_style.extract()

                # Extract and clean text
                text = soup.get_text(separator="\n", strip=True)
                return {"content": text}
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        except Exception as e:
            return {"error": f"An error occurred while fetching {url}: {str(e)}"}

    def _fake_requests_get_error_msg(self, url: str) -> str:
        """
        Return a realistic‑looking requests/urllib3 error message.
        """
        parsed = urlparse(url)

        context = {
            "url": url,
            "host": parsed.hostname or "unknown",
            "path": parsed.path or "/",
            "id1": self._rng.randrange(0x10000000, 0xFFFFFFFF),
            "id2": self._rng.randrange(0x10000000, 0xFFFFFFFF),
        }

        template = self._rng.choice(ERROR_TEMPLATES)

        return template.format(**context)
