#!/usr/bin/env python3
"""Validate the non-empty, NaN-free content of an OpenAI completion."""
from __future__ import annotations

import json
import math
import sys
from typing import Any


def completion_content(body: Any) -> str:
    """Return valid completion content or raise ``ValueError``."""
    if not isinstance(body, dict):
        raise ValueError("response is not a JSON object")
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("missing choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("completion content is empty")
    if "nan" in content.lower() or any(
        isinstance(value, float) and math.isnan(value) for value in _walk(body)
    ):
        raise ValueError("completion contains NaN; verify the pinned libr4d build")
    return content


def _walk(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)
    else:
        yield value


def main() -> int:
    try:
        content = completion_content(json.load(sys.stdin))
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"smoke validation failed: {exc}", file=sys.stderr)
        return 1
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
