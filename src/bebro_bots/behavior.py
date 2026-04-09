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
    bot_id: int,
    bot_username: str,
    settings: dict,
    chat_aliases: list[str] | None = None,
    reply_probability: float = 0.15,
    reply_to_bot_probability: float = 0.35,
) -> Engagement:
    """@username — обязательный ответ; текстовый алиас чата (из БД) — то же; иначе — базовая.

    Базовые вероятности задаются в конфиге; при наличии ключей в settings (например после set_preference)
    используются значения из БД по чату.
    """
    combined = _message_text_for_mention(message)
    if combined and _text_has_mention(combined, bot_username):
        return Engagement(call_llm=True, force_reply=True)
    aliases = chat_aliases or []
    if combined and _text_matches_chat_alias(combined, aliases):
        return Engagement(call_llm=True, force_reply=True)

    reply_to_bot = bool(
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.id == bot_id
    )
    if reply_to_bot:
        p = float(settings.get("reply_to_bot_probability", reply_to_bot_probability))
        return Engagement(call_llm=random.random() < p, force_reply=False)

    p = float(settings.get("reply_probability", reply_probability))
    return Engagement(call_llm=random.random() < p, force_reply=False)
