"""Оси тона и лимиты контекста в зависимости от восприимчивости к обратной связи."""

from __future__ import annotations

from typing import Any

# Значения: -1 (низко / вторая полюса), 0 нейтрально, 1 (высоко / первая полюса)
TONE_KEYS: tuple[str, ...] = (
    "tone_warmth",  # нежность/тепло vs холод/отстранённость
    "tone_sharpness",  # раздражение/острота vs спокойствие
    "tone_anxiety",  # тревога vs расслабленность
    "tone_joy",  # воодушевление vs грусть/меланхолия
    "tone_irony",  # ирония
    "tone_pathos",  # пафос/торжественность vs бытовая простота
    "tone_skepticism",
    "tone_cynicism",
    "tone_confidence",  # уверенность vs неуверенность
    "tone_apathy",  # апатия vs вовлечённость
    "tone_playfulness",  # игривость/лёгкий юмор
)

TONE_LABELS_RU: dict[str, str] = {
    "tone_warmth": "теплота/нежность ↔ холод",
    "tone_sharpness": "острота/раздражение ↔ спокойствие",
    "tone_anxiety": "тревога ↔ расслабленность",
    "tone_joy": "подъём/энтузиазм ↔ грусть/меланхолия",
    "tone_irony": "ирония",
    "tone_pathos": "пафос/торжественность ↔ простота",
    "tone_skepticism": "скептицизм",
    "tone_cynicism": "цинизм",
    "tone_confidence": "уверенность ↔ неуверенность",
    "tone_apathy": "апатия ↔ вовлечённость",
    "tone_playfulness": "игривость/юмор",
}

DEFAULT_TONE_SETTINGS: dict[str, int] = {k: 0 for k in TONE_KEYS}

MOOD_RANDOM_POOL: tuple[str, ...] = (
    "нейтральное",
    "игривое",
    "задумчивое",
    "бодрое",
    "сдержанно ироничное",
    "мягкое",
    "саркастичное (без оскорблений)",
    "спокойное любопытство",
)


def default_tone_settings_dict() -> dict[str, int]:
    return dict(DEFAULT_TONE_SETTINGS)


def merged_tone_values(settings: dict[str, Any], drift: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for k in TONE_KEYS:
        base = int(settings.get(k, 0))
        if base < -1:
            base = -1
        elif base > 1:
            base = 1
        d = int(drift.get(k, 0))
        v = base + d
        if v < -1:
            v = -1
        elif v > 1:
            v = 1
        out[k] = v
    return out


def format_tone_lines(merged: dict[str, int]) -> str:
    parts: list[str] = []
    for k in TONE_KEYS:
        v = merged.get(k, 0)
        label = TONE_LABELS_RU.get(k, k)
        if v == -1:
            parts.append(f"- {label}: низко")
        elif v == 1:
            parts.append(f"- {label}: высоко")
        else:
            parts.append(f"- {label}: умеренно")
    return "\n".join(parts)


def situation_rules_limit(sensitivity: int) -> int:
    s = max(1, min(5, sensitivity))
    return {1: 0, 2: 2, 3: 4, 4: 6, 5: 8}[s]


def long_memory_limit(sensitivity: int) -> int:
    s = max(1, min(5, sensitivity))
    return 4 + s * 2


def format_situation_rules_block(rules: list[tuple[str, str]]) -> str:
    if not rules:
        return ""
    lines: list[str] = []
    for trig, ins in rules:
        lines.append(f"- Когда: {trig} → {ins}")
    return "Ситуационные договорённости (кратко):\n" + "\n".join(lines)


def drift_random_axes() -> dict[str, int]:
    """Небольшой случайный сдвиг по части осей (-1, 0, 1)."""
    import random

    drift: dict[str, int] = {}
    for k in TONE_KEYS:
        if random.random() < 0.15:
            drift[k] = random.choice((-1, 0, 1))
    if not drift:
        drift[random.choice(TONE_KEYS)] = random.choice((-1, 1))
    return drift
