from __future__ import annotations

from html import unescape
import re
from typing import Any, Dict

import httpx

from agent.core.tools import Tool


class WebSearchTool(Tool):
    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web and return top results."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "count": {"type": "integer", "description": "Number of results", "default": 5},
            },
            "required": ["query"],
        }

    async def execute(self, query: str, count: int = 5) -> Any:
        count = max(1, min(count, 20))
        if self.api_key:
            return await self._search_brave(query=query, count=count)
        return await self._search_duckduckgo(query=query, count=count)

    async def _search_brave(self, query: str, count: int) -> Any:
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
            data = response.json()
            items = data.get("web", {}).get("results", [])
            results = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("description", ""),
                }
                for item in items[:count]
            ]
            return {
                "success": True,
                "provider": "brave",
                "query": query,
                "results": results,
            }

    async def _search_duckduckgo(self, query: str, count: int) -> Any:
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
            results = []
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
