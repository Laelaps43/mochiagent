from __future__ import annotations

import ipaddress
import socket
from typing import override
from urllib.parse import urlparse

import httpx

from agent.core.tools import Tool

from ._utils import truncate_text
from .results import ToolError, WebFetchSuccess

_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _is_private_ip(host: str) -> bool:
    """Check if a hostname resolves to a private/reserved IP address."""
    try:
        addr_infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False
    for _family, _type, _proto, _canonname, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_loopback or ip.is_private or ip.is_reserved or ip.is_link_local:
            return True
        for network in _BLOCKED_NETWORKS:
            if ip in network:
                return True
    return False


class WebFetchTool(Tool):
    """
    Fetch content from URL.
    """

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
            return ToolError(error="Only http/https URLs are supported")

        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        if not hostname:
            return ToolError(error="Invalid URL: missing hostname")
        if _is_private_ip(hostname):
            return ToolError(error="Access to private/internal addresses is blocked")

        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=False, max_redirects=0
        ) as client:
            current_url = url
            response = await client.get(current_url)
            for _ in range(5):
                if not response.is_redirect:
                    break
                redirect_url = str(response.next_request.url) if response.next_request else ""
                if not redirect_url:
                    break
                redirect_parsed = urlparse(redirect_url)
                redirect_host = redirect_parsed.hostname or ""
                if redirect_host and _is_private_ip(redirect_host):
                    return ToolError(error="Redirect to private/internal address is blocked")
                response = await client.get(redirect_url)

            text = response.text[: self.max_chars * 2]
            output, truncated = truncate_text(text, self.max_chars)
            content_type: str = response.headers.get("content-type", "")  # pyright: ignore[reportAny]

            return WebFetchSuccess(
                success=response.status_code < 400,
                url=str(response.url),
                status_code=response.status_code,
                content_type=content_type,
                content=output,
                truncated=truncated,
            )
