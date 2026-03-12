from __future__ import annotations

from html import unescape
import re
from typing import cast, override

import httpx

from agent.core.tools import Tool


class WebSearchTool(Tool):
    """
    Search the web and return top results.
    """

    def __init__(self, api_key: str = ""):
        self.api_key: str = api_key

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
        if self.api_key:
            return await self._search_brave(query=query, count=count)
        return await self._search_duckduckgo(query=query, count=count)

    async def _search_brave(self, query: str, count: int) -> object:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }
        params = {"q": query, "count": count}
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
            )
            if response.status_code >= 400:
                return {
                    "success": False,
                    "error": f"Brave search failed: HTTP {response.status_code}",
                }
            data = cast(dict[str, object], response.json())
            web_obj = cast(dict[str, object], data.get("web") or {})
            items = cast(list[object], web_obj.get("results") or [])
            results: list[dict[str, object]] = [
                {
                    "title": cast(str, cast(dict[str, object], item).get("title") or ""),
                    "url": cast(str, cast(dict[str, object], item).get("url") or ""),
                    "snippet": cast(str, cast(dict[str, object], item).get("description") or ""),
                }
                for item in items[:count]
            ]
            return {
                "success": True,
                "provider": "brave",
                "query": query,
                "results": results,
            }

    async def _search_duckduckgo(self, query: str, count: int) -> object:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                "https://duckduckgo.com/html/",
                data={"q": query},
                headers={"User-Agent": "mochiagent"},
            )
            if response.status_code >= 400:
                return {
                    "success": False,
                    "error": f"DuckDuckGo search failed: HTTP {response.status_code}",
                }

            html = response.text
            pattern = re.compile(
                r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                re.IGNORECASE | re.DOTALL,
            )
            results: list[dict[str, object]] = []
            for match in pattern.finditer(html):
                href = unescape(match.group(1))
                title_raw = match.group(2)
                title = unescape(re.sub(r"<[^>]+>", "", title_raw)).strip()
                results.append({"title": title, "url": href, "snippet": ""})
                if len(results) >= count:
                    break

            return {
                "success": True,
                "provider": "duckduckgo",
                "query": query,
                "results": results,
            }
