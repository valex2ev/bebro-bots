from __future__ import annotations

import random
from dataclasses import dataclass

from telegram import Message


@dataclass
class Engagement:
    """Решение: вызывать ли LLM и обязательно ли ответить текстом."""

    call_llm: bool
    force_reply: bool  # упоминание @бот — ожидаем ответ (не только реакция)


def _text_has_mention(text: str | None, bot_username: str) -> bool:
    if not text:
        return False
    u = bot_username.lstrip("@").lower()
    needle = f"@{u}"
    return needle.lower() in text.lower()


def _text_matches_chat_alias(text: str | None, aliases: list[str]) -> bool:
    if not text or not aliases:
        return False
    hay = text.lower()
    for a in aliases:
        if a in hay:
            return True
    return False


def _message_text_for_mention(message: Message) -> str | None:
    return message.text or message.caption


def decide_engagement(
    message: Message,
    *,
    bot_username: str,
    chat_aliases: list[str] | None = None,
    reply_probability: float = 0.15,
) -> Engagement:
    """@username — обязательный ответ; текстовый алиас чата (из БД) — то же.

    Иначе одна вероятность ``reply_probability`` из конфига (не из БД).
    """
    combined = _message_text_for_mention(message)
    if combined and _text_has_mention(combined, bot_username):
        return Engagement(call_llm=True, force_reply=True)
    aliases = chat_aliases or []
    if combined and _text_matches_chat_alias(combined, aliases):
        return Engagement(call_llm=True, force_reply=True)

    p = float(reply_probability)
    return Engagement(call_llm=random.random() < p, force_reply=False)
