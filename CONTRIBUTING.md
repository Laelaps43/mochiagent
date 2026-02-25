# Contributing

Thanks for contributing to MochiAgent.

## Development Setup

```bash
uv sync --extra dev
```

## Local Checks

```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev pytest -q
```

## Pull Request Guidelines

1. Keep PR scope focused.
2. Include tests for behavior changes.
3. Explain design tradeoffs and migration impact.
4. Ensure CI is green before requesting review.
