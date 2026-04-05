from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


async def run_tavily_search(*, query: str, api_key: str | None) -> dict[str, Any]:
    """POST /search; возвращает JSON Tavily или объект с полем error."""
    if not api_key or not api_key.strip():
        return {"ok": False, "error": "TAVILY_API_KEY не задан в окружении"}
    q = query.strip()
    if not q:
        return {"ok": False, "error": "пустой query"}
    payload = {
        "api_key": api_key.strip(),
        "query": q,
        "max_results": 5,
        "search_depth": "basic",
    }
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(TAVILY_SEARCH_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.warning("Tavily HTTP %s: %s", e.response.status_code, e.response.text[:500])
        return {"ok": False, "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        logger.exception("Tavily request failed")
        return {"ok": False, "error": str(e)}
    return {"ok": True, "raw": data}


def format_tavily_for_tool(data: dict[str, Any]) -> str:
    """Сжать ответ Tavily в строку для tool result."""
    if not data.get("ok"):
        return json.dumps(data, ensure_ascii=False)
    raw = data.get("raw") or {}
    lines: list[str] = []
    if raw.get("answer"):
        lines.append(f"Краткий ответ: {raw['answer']}")
    for i, r in enumerate(raw.get("results") or [], 1):
        title = r.get("title") or ""
        url = r.get("url") or ""
        content = (r.get("content") or "")[:600]
        lines.append(f"{i}. {title}\n   URL: {url}\n   {content}")
    if not lines:
        return json.dumps({"ok": True, "note": "пустой результат"}, ensure_ascii=False)
    return "\n\n".join(lines)


def tavily_api_key_from_env() -> str | None:
    return os.environ.get("TAVILY_API_KEY", "").strip() or None
