"""Tool result models - Pydantic models for structured tool return values."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class ToolError(BaseModel):
    """All tools share this model for error returns."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    success: bool = False
    error: str


class ReadFileSuccess(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    success: bool = True
    path: str
    content: str
    truncated: bool
    size_bytes: int
    offset: int
    limit: int
    next_offset: int
    eof: bool


class WriteFileSuccess(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    success: bool = True
    path: str
    bytes_written: int
    append: bool


class EditFileSuccess(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    success: bool = True
    path: str
    replacements: int
    warning: str | None = None


class ListDirSuccess(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    success: bool = True
    path: str
    entries: list[str]
    truncated: bool


class ExecResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    success: bool
    exit_code: int | None
    output: str
    truncated: bool


class SearchResultItem(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    title: str
    url: str
    snippet: str


class WebFetchSuccess(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    success: bool
    url: str
    status_code: int
    content_type: str
    content: str
    truncated: bool


class WebSearchSuccess(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    success: bool = True
    provider: str
    query: str
    results: list[SearchResultItem]
