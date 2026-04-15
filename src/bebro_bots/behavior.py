from __future__ import annotations

from dataclasses import dataclass
from random import Random


@dataclass
class Engagement:
    """Решение: вызывать ли LLM и обязательно ли ответить текстом."""

    call_llm: bool
    force_reply: bool  # явное обращение — ожидаем ответ (не только реакция)


def decide_stochastic_engagement(reply_probability: float, rng: Random) -> Engagement:
    """Ответ без явного обращения: одна вероятность ``reply_probability`` из конфига (не из БД)."""
    p = float(reply_probability)
    return Engagement(call_llm=rng.random() < p, force_reply=False)
