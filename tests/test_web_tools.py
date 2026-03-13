from __future__ import annotations

from unittest.mock import AsyncMock, patch

from pydantic import SecretStr

from agent.common.tools.web_fetch_tool import WebFetchTool
from agent.common.tools.web_search_tool import WebSearchTool
from agent.common.tools.results import ToolError, WebFetchSuccess, WebSearchSuccess


class _FakeResponse:
    status_code: int
    text: str
    url: str
    headers: dict[str, str]
    _json: dict[str, object]
    next_request: object

    def __init__(
        self,
        status_code: int,
        text: str = "",
        json_data: dict[str, object] | None = None,
        url: str = "https://example.com",
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.url = url
        self.headers = {"content-type": "text/html"}
        self._json = json_data or {}
        self.next_request = None

    @property
    def is_redirect(self) -> bool:
        return self.status_code in (301, 302, 303, 307, 308)

    def json(self) -> dict[str, object]:
        return self._json


class _FakeAsyncClient:
    _response: _FakeResponse
    get: AsyncMock
    post: AsyncMock

    def __init__(self, response: _FakeResponse, **_kwargs: object) -> None:
        self._response = response
        self.get = AsyncMock(return_value=response)
        self.post = AsyncMock(return_value=response)

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        pass


async def test_web_fetch_invalid_url_returns_error() -> None:
    tool = WebFetchTool()
    result = await tool.execute(url="ftp://example.com")
    assert isinstance(result, ToolError)
    assert "http" in result.error.lower()


async def test_web_fetch_success_200() -> None:
    tool = WebFetchTool()
    fake_resp = _FakeResponse(200, text="<html>hello</html>")
    client = _FakeAsyncClient(fake_resp)
    with patch("httpx.AsyncClient", return_value=client):
        result = await tool.execute(url="https://example.com")
    assert isinstance(result, WebFetchSuccess)
    assert result.success is True
    assert result.status_code == 200
    assert "hello" in result.content


async def test_web_fetch_404_returns_failure() -> None:
    tool = WebFetchTool()
    fake_resp = _FakeResponse(404, text="Not Found", url="https://example.com/nope")
    client = _FakeAsyncClient(fake_resp)
    with patch("httpx.AsyncClient", return_value=client):
        result = await tool.execute(url="https://example.com/nope")
    assert isinstance(result, WebFetchSuccess)
    assert result.success is False
    assert result.status_code == 404


async def test_web_fetch_content_truncated() -> None:
    tool = WebFetchTool(max_chars=10)
    long_text = "A" * 1000
    fake_resp = _FakeResponse(200, text=long_text)
    client = _FakeAsyncClient(fake_resp)
    with patch("httpx.AsyncClient", return_value=client):
        result = await tool.execute(url="https://example.com")
    assert isinstance(result, WebFetchSuccess)
    assert result.truncated is True


async def test_web_fetch_content_not_truncated() -> None:
    tool = WebFetchTool(max_chars=10000)
    short_text = "short content"
    fake_resp = _FakeResponse(200, text=short_text)
    client = _FakeAsyncClient(fake_resp)
    with patch("httpx.AsyncClient", return_value=client):
        result = await tool.execute(url="https://example.com")
    assert isinstance(result, WebFetchSuccess)
    assert result.truncated is False
    assert result.content == short_text


async def test_web_search_uses_brave_when_api_key_set() -> None:
    tool = WebSearchTool(api_key=SecretStr("test-key"))
    brave_json: dict[str, object] = {
        "web": {
            "results": [
                {"title": "Result 1", "url": "https://r1.com", "description": "Snippet 1"},
                {"title": "Result 2", "url": "https://r2.com", "description": "Snippet 2"},
            ]
        }
    }
    fake_resp = _FakeResponse(200, json_data=brave_json)
    client = _FakeAsyncClient(fake_resp)
    with patch("httpx.AsyncClient", return_value=client):
        result = await tool.execute(query="python async", count=2)
    assert isinstance(result, WebSearchSuccess)
    assert result.provider == "brave"
    assert len(result.results) == 2
    assert result.results[0].title == "Result 1"


async def test_web_search_uses_duckduckgo_when_no_api_key() -> None:
    tool = WebSearchTool()
    html = (
        "<html><body>"
        '<a class="result__a" href="https://example.com/1">First Result</a>'
        '<a class="result__a" href="https://example.com/2">Second Result</a>'
        "</body></html>"
    )
    fake_resp = _FakeResponse(200, text=html)
    client = _FakeAsyncClient(fake_resp)
    with patch("httpx.AsyncClient", return_value=client):
        result = await tool.execute(query="test query", count=2)
    assert isinstance(result, WebSearchSuccess)
    assert result.provider == "duckduckgo"


async def test_web_search_brave_http_error() -> None:
    tool = WebSearchTool(api_key=SecretStr("test-key"))
    fake_resp = _FakeResponse(403, text="Forbidden")
    client = _FakeAsyncClient(fake_resp)
    with patch("httpx.AsyncClient", return_value=client):
        result = await tool.execute(query="blocked")
    assert isinstance(result, ToolError)
    assert "403" in result.error


async def test_web_search_duckduckgo_http_error() -> None:
    tool = WebSearchTool()
    fake_resp = _FakeResponse(503, text="Service Unavailable")
    client = _FakeAsyncClient(fake_resp)
    with patch("httpx.AsyncClient", return_value=client):
        result = await tool.execute(query="down")
    assert isinstance(result, ToolError)
    assert "503" in result.error


async def test_web_search_count_clamp_max() -> None:
    tool = WebSearchTool(api_key=SecretStr("k"))
    brave_json: dict[str, object] = {"web": {"results": []}}
    fake_resp = _FakeResponse(200, json_data=brave_json)
    client = _FakeAsyncClient(fake_resp)
    with patch("httpx.AsyncClient", return_value=client):
        result = await tool.execute(query="q", count=999)
    assert isinstance(result, WebSearchSuccess)
    assert len(result.results) <= 20


async def test_web_search_count_clamp_min() -> None:
    tool = WebSearchTool(api_key=SecretStr("k"))
    brave_json: dict[str, object] = {"web": {"results": []}}
    fake_resp = _FakeResponse(200, json_data=brave_json)
    client = _FakeAsyncClient(fake_resp)
    with patch("httpx.AsyncClient", return_value=client):
        result = await tool.execute(query="q", count=0)
    assert isinstance(result, WebSearchSuccess)


async def test_web_search_brave_empty_results() -> None:
    tool = WebSearchTool(api_key=SecretStr("key"))
    brave_json: dict[str, object] = {"web": {}}
    fake_resp = _FakeResponse(200, json_data=brave_json)
    client = _FakeAsyncClient(fake_resp)
    with patch("httpx.AsyncClient", return_value=client):
        result = await tool.execute(query="nothing", count=5)
    assert isinstance(result, WebSearchSuccess)
    assert len(result.results) == 0


async def test_web_search_duckduckgo_no_matches() -> None:
    tool = WebSearchTool()
    fake_resp = _FakeResponse(200, text="<html><body>no results here</body></html>")
    client = _FakeAsyncClient(fake_resp)
    with patch("httpx.AsyncClient", return_value=client):
        result = await tool.execute(query="zzz", count=3)
    assert isinstance(result, WebSearchSuccess)
    assert len(result.results) == 0


async def test_web_search_duckduckgo_count_limit() -> None:
    tool = WebSearchTool()
    links = "\n".join(
        f'<a class="result__a" href="https://example.com/{i}">Result {i}</a>' for i in range(10)
    )
    html = f"<html><body>{links}</body></html>"
    fake_resp = _FakeResponse(200, text=html)
    client = _FakeAsyncClient(fake_resp)
    with patch("httpx.AsyncClient", return_value=client):
        result = await tool.execute(query="many", count=3)
    assert isinstance(result, WebSearchSuccess)
    assert len(result.results) == 3
