"""Триггеры явных действий в тексте сообщения."""

from __future__ import annotations

# Поиск в интернете (Tavily) — только если пользователь набрал эту подстроку (регистр не важен).
WEB_SEARCH_TRIGGER = "найди"


def message_allows_web_search(text: str | None) -> bool:
    if not text:
        return False
    return WEB_SEARCH_TRIGGER.lower() in text.lower()
