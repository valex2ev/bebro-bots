"""Правила текстовых алиасов: только обращения к ботам, не @username людей и не «никнеймы» в стиле Telegram."""

from __future__ import annotations

import re
import sqlite3

# Telegram username: 5–32 символа, [a-zA-Z0-9_], с буквы; после нормализации в нижний регистр.
_ASCII_USERNAME = re.compile(r"^[a-z][a-z0-9_]{4,31}$")


def reject_reason_for_chat_alias(raw: str) -> str | None:
    """
    None — строку можно сохранить как алиас.
    Иначе — код причины отказа (для ответа инструмента).
    """
    key = raw.strip().lower()
    if not key:
        return "empty"
    if "@" in key:
        return "no_at_mentions"
    # Подстроки в стиле логина Telegram (ASCII, длина ≥8), чтобы не ловить короткие англ. слова.
    if _ASCII_USERNAME.fullmatch(key) and len(key) >= 8:
        return "no_telegram_username_shape"
    return None


def purge_username_like_aliases(conn: sqlite3.Connection) -> int:
    """Удалить из chat_aliases строки с @ и похожие на Telegram username (ASCII). Возвращает число удалённых строк."""
    rows = conn.execute("SELECT id, alias_text FROM chat_aliases").fetchall()
    deleted = 0
    for r in rows:
        tid = int(r["id"])
        t = str(r["alias_text"])
        if "@" in t:
            conn.execute("DELETE FROM chat_aliases WHERE id=?", (tid,))
            deleted += 1
            continue
        tl = t.strip().lower()
        if _ASCII_USERNAME.fullmatch(tl) and len(tl) >= 8:
            conn.execute("DELETE FROM chat_aliases WHERE id=?", (tid,))
            deleted += 1
    return deleted
