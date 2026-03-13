from __future__ import annotations

from html import unescape
import re
from typing import override

import httpx
from pydantic import SecretStr

from agent.core.tools import Tool
from typing import cast

from .results import SearchResultItem, ToolError, WebSearchSuccess


class WebSearchTool(Tool):
    """
    Search the web and return top results.
    """

    def __init__(self, api_key: SecretStr | None = None):
        self._api_key: SecretStr = api_key or SecretStr("")

    @property
    @override
    def name(self) -> str:
        return "web_search"

    @property
    @override
    def description(self) -> str:
        return "Search the web and return top results."

    @property
    @override
    def parameters_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "count": {"type": "integer", "description": "Number of results", "default": 5},
            },
            "required": ["query"],
        }

    @override
    async def execute(self, query: str = "", count: int = 5, **kwargs: object) -> object:
        count = max(1, min(count, 20))
        if self._api_key.get_secret_value():
            return await self._search_brave(query=query, count=count)
        return await self._search_duckduckgo(query=query, count=count)

    async def _search_brave(self, query: str, count: int) -> object:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key.get_secret_value(),
        }
        params = {"q": query, "count": count}
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
            )
            if response.status_code >= 400:
                return ToolError(
                    error=f"Brave search failed: HTTP {response.status_code}",
                )
            raw_data: object = response.json()  # pyright: ignore[reportAny]
            if not isinstance(raw_data, dict):
                return ToolError(error="Unexpected response format")
            data = cast(dict[str, object], raw_data)
            raw_web = data.get("web")
            web_obj = cast(dict[str, object], raw_web) if isinstance(raw_web, dict) else {}
            raw_items = web_obj.get("results")
            items = cast(list[object], raw_items) if isinstance(raw_items, list) else []
            results: list[SearchResultItem] = []
            for item in items[:count]:
                if isinstance(item, dict):
                    entry = cast(dict[str, object], item)
                    results.append(
                        SearchResultItem(
                            title=str(entry.get("title") or ""),
                            url=str(entry.get("url") or ""),
                            snippet=str(entry.get("description") or ""),
                        )
                    )
            return WebSearchSuccess(
                provider="brave",
                query=query,
                results=results,
            )

    async def _search_duckduckgo(self, query: str, count: int) -> object:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            response = await client.post(
                "https://duckduckgo.com/html/",
                data={"q": query},
                headers={"User-Agent": "mochiagent"},
            )
            if response.status_code >= 400:
                return ToolError(
                    error=f"DuckDuckGo search failed: HTTP {response.status_code}",
                )

            html = response.text
            results: list[SearchResultItem] = []

            try:
                pattern = re.compile(
                    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                    re.IGNORECASE | re.DOTALL,
                )
                for match in pattern.finditer(html):
                    href = unescape(match.group(1))
                    title_raw = match.group(2)
                    title = unescape(re.sub(r"<[^>]+>", "", title_raw)).strip()
                    if href and title:
                        results.append(SearchResultItem(title=title, url=href, snippet=""))
                    if len(results) >= count:
                        break
            except Exception:
                return ToolError(
                    error="Failed to parse DuckDuckGo response HTML",
                )

            return WebSearchSuccess(
                provider="duckduckgo",
                query=query,
                results=results,
            )
