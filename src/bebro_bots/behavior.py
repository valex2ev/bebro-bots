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


def _message_text_for_mention(message: Message) -> str | None:
    return message.text or message.caption


def decide_engagement(
    message: Message,
    *,
    bot_id: int,
    bot_username: str,
    settings: dict,
) -> Engagement:
    """@username — обязательный ответ; ответ на сообщение бота — по вероятности; иначе — базовая."""
    combined = _message_text_for_mention(message)
    if combined and _text_has_mention(combined, bot_username):
        return Engagement(call_llm=True, force_reply=True)

    reply_to_bot = bool(
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.id == bot_id
    )
    if reply_to_bot:
        p = float(settings.get("reply_to_bot_probability", 0.35))
        return Engagement(call_llm=random.random() < p, force_reply=False)

    p = float(settings.get("reply_probability", 0.15))
    return Engagement(call_llm=random.random() < p, force_reply=False)
