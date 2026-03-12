from __future__ import annotations

from typing import override

import httpx

from agent.core.tools import Tool

from ._utils import truncate_text


class WebFetchTool(Tool):
    def __init__(self, max_chars: int = 20000):
        self.max_chars: int = max_chars

    @property
    @override
    def name(self) -> str:
        return "web_fetch"

    @property
    @override
    def description(self) -> str:
        return "Fetch content from URL."

    @property
    @override
    def parameters_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "HTTP/HTTPS URL"},
                "timeout": {
                    "type": "number",
                    "description": "Request timeout in seconds",
                    "default": 20,
                },
            },
            "required": ["url"],
        }

    @override
    async def execute(self, url: str = "", timeout: float = 20, **kwargs: object) -> object:
        if not (url.startswith("http://") or url.startswith("https://")):
            return {"success": False, "error": "Only http/https URLs are supported"}

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
            text = response.text
            output, truncated = truncate_text(text, self.max_chars)

            return {
                "success": response.status_code < 400,
                "url": str(response.url),
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "content": output,
                "truncated": truncated,
            }
