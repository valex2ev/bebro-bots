from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from bebro_bots.db import Database
from bebro_bots.tone_profile import TONE_KEYS, long_memory_limit, situation_rules_limit

logger = logging.getLogger(__name__)

PREFERENCE_KEYS = {
    "window_messages",
    "recent_for_llm",
    "reply_probability",
    "reaction_probability",
    "reply_to_bot_probability",
    "initiative_probability",
    "initiative_interval_minutes",
    "proactive_reaction_probability",
    "verbosity",
    "frequency",
    "short_reply_bias",
    "response_depth",
    *TONE_KEYS,
}

MUTATING_TOOL_NAMES = frozenset(
    {
        "set_preference",
        "add_long_instruction",
        "set_mood",
        "update_conversation_summary",
        "add_situation_rule",
        "set_bot_persona",
    }
)


@dataclass
class CompleteTurnResult:
    action: str  # message | reaction | sticker | none
    text: str | None
    reaction_emoji: str | None
    sticker_file_id: str | None


def sanitize_user_visible_text(text: str | None) -> str | None:
    """Убирает из ответа хвосты с именами инструментов (модель иногда их дублирует)."""
    if not text:
        return None
    t = text.strip()
    cut = re.search(r"(?is)\bcomplete_turn\b", t)
    if cut:
        t = t[: cut.start()].rstrip(" \t.;:,-—")
    lines_out: list[str] = []
    for line in t.splitlines():
        s = line.strip()
        if not s:
            continue
        sl = s.lower()
        if "complete_turn" in sl:
            continue
        if re.match(r"^action\s*=\s*\w+", s, re.I):
            continue
        lines_out.append(s)
    out = "\n".join(lines_out).strip()
    out = re.sub(
        r"(?is)[\s.;,]*\baction\s*=\s*(message|reaction|sticker|none)\b[\s.;,]*$",
        "",
        out,
    )
    out = out.strip(" \t.;:,-—")
    return out if out else None


def _finalize_turn_result(r: CompleteTurnResult) -> CompleteTurnResult:
    if r.action != "message":
        return r
    if not r.text:
        return CompleteTurnResult(
            action="none",
            text=None,
            reaction_emoji=None,
            sticker_file_id=None,
        )
    clean = sanitize_user_visible_text(r.text)
    if not clean:
        return CompleteTurnResult(
            action="none",
            text=None,
            reaction_emoji=None,
            sticker_file_id=None,
        )
    return CompleteTurnResult(
        action=r.action,
        text=clean,
        reaction_emoji=r.reaction_emoji,
        sticker_file_id=r.sticker_file_id,
    )


def _tools_schema(*, include_web_search: bool = False) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "get_group_state",
                "description": "Прочитать настроения, предпочтения, краткую память и саммари беседы по текущему чату.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_preference",
                "description": "Сохранить долговременное предпочтение для этого чата (частота, длина ответов и т.д.).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "enum": sorted(PREFERENCE_KEYS),
                        },
                        "value": {
                            "type": "string",
                            "description": "Число как строка, или short|medium|long для verbosity, rare|medium|often для frequency",
                        },
                    },
                    "required": ["key", "value"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "add_long_instruction",
                "description": "Добавить долгую инструкцию, которая должна влиять на поведение в этом чате.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_mood",
                "description": "Обновить текущее настроение бота (короткая строка).",
                "parameters": {
                    "type": "object",
                    "properties": {"mood": {"type": "string"}},
                    "required": ["mood"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_conversation_summary",
                "description": "Обновить сжатый контекст беседы (саммари) для экономии токенов.",
                "parameters": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "add_situation_rule",
                "description": (
                    "Сохранить правило: в какой ситуации (trigger) как себя вести (instruction). "
                    "Только если просит человек (не другой бот)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "trigger_text": {
                            "type": "string",
                            "description": "Кратко: когда применять (тема, настроение чата, тип сообщения)",
                        },
                        "instruction_text": {
                            "type": "string",
                            "description": "Что делать в таких случаях (1–2 предложения)",
                        },
                    },
                    "required": ["trigger_text", "instruction_text"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_bot_persona",
                "description": (
                    "Обновить базовое текстовое описание личности бота (глобально для всех чатов). "
                    "Только по просьбе человека."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    if include_web_search:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": (
                        "Поиск в интернете (Tavily). Доступен только в этом диалоге: пользователь явно написал ключевое слово. "
                        "Передай сжатый поисковый запрос в query."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Запрос для поиска (суть вопроса без служебных слов)",
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "complete_turn",
                "description": "Завершить ход одним вызовом. Не дублируй это описание в поле text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["message", "reaction", "sticker", "none"],
                        },
                        "text": {
                            "type": "string",
                            "description": "Только текст/emoji для людей в чате. Без имён инструментов, без JSON и без фраз вроде complete_turn.",
                        },
                        "reaction_emoji": {
                            "type": "string",
                            "description": "Одна emoji для реакции на сообщение пользователя",
                        },
                        "sticker_file_id": {
                            "type": "string",
                            "description": "file_id стикера или пусто для случайного из пула чата",
                        },
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
        }
    )
    return tools


def _parse_pref_value(key: str, raw: str) -> Any:
    s = raw.strip()
    if key in {
        "window_messages",
        "recent_for_llm",
        "initiative_interval_minutes",
    }:
        return int(s)
    if key in {
        "reply_probability",
        "reaction_probability",
        "reply_to_bot_probability",
        "initiative_probability",
        "short_reply_bias",
        "proactive_reaction_probability",
    }:
        return float(s)
    if key in TONE_KEYS:
        v = int(s)
        if v not in (-1, 0, 1):
            raise ValueError("ось тона: -1, 0 или 1")
        return v
    if key == "verbosity":
        if s not in ("short", "medium", "long"):
            raise ValueError("verbosity: short|medium|long")
        return s
    if key == "frequency":
        if s not in ("rare", "medium", "often"):
            raise ValueError("frequency: rare|medium|often")
        return s
    if key == "response_depth":
        if s not in ("shallow", "normal", "deep"):
            raise ValueError("response_depth: shallow|normal|deep")
        return s
    raise ValueError(f"неизвестный ключ {key}")


def _apply_frequency_to_probs(patch: dict[str, Any], frequency: str) -> None:
    if frequency == "rare":
        patch.setdefault("reply_probability", 0.08)
        patch.setdefault("initiative_probability", 0.02)
    elif frequency == "often":
        patch.setdefault("reply_probability", 0.28)
        patch.setdefault("initiative_probability", 0.12)
    else:
        patch.setdefault("reply_probability", 0.15)
        patch.setdefault("initiative_probability", 0.05)


def _apply_verbosity(patch: dict[str, Any], verbosity: str) -> None:
    if verbosity == "short":
        patch["short_reply_bias"] = min(1.0, float(patch.get("short_reply_bias", 0.0)) + 0.3)
    elif verbosity == "long":
        patch["short_reply_bias"] = max(-1.0, float(patch.get("short_reply_bias", 0.0)) - 0.3)


async def run_agent_turn(
    *,
    client: AsyncOpenAI,
    model: str,
    db: Database,
    bot_id: str,
    chat_id: int,
    system_prompt: str,
    user_lines: list[str],
    force_substantive_reply: bool,
    allow_web_search: bool = False,
    allow_mutations: bool = True,
    feedback_sensitivity: int = 4,
) -> CompleteTurnResult:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_lines)},
    ]
    if force_substantive_reply:
        messages.append(
            {
                "role": "user",
                "content": "Системное напоминание: пользователь явно упомянул тебя через @username — дай содержательный ответ (не только одну реакцию), если нет просьбы именно о реакции.",
            }
        )

    tools = _tools_schema(include_web_search=allow_web_search)

    for _ in range(8):
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.85,
        )
        choice = resp.choices[0].message
        if choice.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": choice.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                        for tc in choice.tool_calls
                    ],
                }
            )
            complete: CompleteTurnResult | None = None
            for tc in choice.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                if name == "complete_turn":
                    complete = _parse_complete_turn(args)
                    tool_out = json.dumps({"ok": True}, ensure_ascii=False)
                else:
                    tool_out = await _dispatch_tool(
                        name=name,
                        args=args,
                        db=db,
                        bot_id=bot_id,
                        chat_id=chat_id,
                        allow_web_search=allow_web_search,
                        allow_mutations=allow_mutations,
                        feedback_sensitivity=feedback_sensitivity,
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_out,
                    }
                )
            if complete:
                return _finalize_turn_result(complete)
            continue

        text = (choice.content or "").strip()
        if text:
            return _finalize_turn_result(
                CompleteTurnResult(action="message", text=text, reaction_emoji=None, sticker_file_id=None)
            )
        break

    return CompleteTurnResult(action="none", text=None, reaction_emoji=None, sticker_file_id=None)


def _parse_complete_turn(args: dict[str, Any]) -> CompleteTurnResult:
    raw_text = args.get("text")
    if isinstance(raw_text, str):
        raw_text = sanitize_user_visible_text(raw_text)
    return CompleteTurnResult(
        action=str(args.get("action") or "none"),
        text=raw_text,
        reaction_emoji=args.get("reaction_emoji"),
        sticker_file_id=args.get("sticker_file_id") or None,
    )


async def _dispatch_tool(
    *,
    name: str,
    args: dict[str, Any],
    db: Database,
    bot_id: str,
    chat_id: int,
    allow_web_search: bool = False,
    allow_mutations: bool = True,
    feedback_sensitivity: int = 4,
) -> str:
    try:
        if name in MUTATING_TOOL_NAMES and not allow_mutations:
            return json.dumps(
                {"ok": False, "error": "настройки по сообщениям от ботов не меняются"},
                ensure_ascii=False,
            )
        if name == "web_search":
            if not allow_web_search:
                return json.dumps(
                    {"ok": False, "error": "web_search недоступен в этом запросе"},
                    ensure_ascii=False,
                )
            from bebro_bots.tavily_search import (
                format_tavily_for_tool,
                run_tavily_search,
                tavily_api_key_from_env,
            )

            res = await run_tavily_search(
                query=str(args.get("query", "")),
                api_key=tavily_api_key_from_env(),
            )
            return format_tavily_for_tool(res)
        if name == "get_group_state":
            prefs = db.get_settings(bot_id, chat_id)
            mood = db.get_mood(bot_id)
            summary = db.get_summary(bot_id, chat_id)
            lim = long_memory_limit(feedback_sensitivity)
            mem = db.list_long_memory(bot_id, chat_id, limit=lim)
            sr_lim = situation_rules_limit(feedback_sensitivity)
            rules = db.list_situation_rules(bot_id, chat_id, limit=sr_lim)
            persona_text = db.get_bot_persona(bot_id)
            tone_drift = db.get_tone_drift(bot_id)
            rules_payload = [{"trigger": t, "instruction": i} for t, i in rules]
            return json.dumps(
                {
                    "persona_text": persona_text,
                    "tone_drift": tone_drift,
                    "preferences": prefs,
                    "mood": mood,
                    "conversation_summary": summary,
                    "long_instructions": mem,
                    "situation_rules": rules_payload,
                    "feedback_sensitivity": feedback_sensitivity,
                },
                ensure_ascii=False,
            )
        if name == "set_preference":
            key = str(args["key"])
            raw_val = str(args["value"])
            if key not in PREFERENCE_KEYS:
                return json.dumps({"ok": False, "error": "bad key"})
            val = _parse_pref_value(key, raw_val)
            patch: dict[str, Any] = {key: val}
            if key == "frequency":
                _apply_frequency_to_probs(patch, str(val))
            if key == "verbosity":
                _apply_verbosity(patch, str(val))
            merged = db.merge_settings(bot_id, chat_id, patch)
            return json.dumps({"ok": True, "merged_preferences": merged}, ensure_ascii=False)
        if name == "add_long_instruction":
            db.add_long_memory(bot_id, chat_id, str(args["text"]))
            return json.dumps({"ok": True}, ensure_ascii=False)
        if name == "set_mood":
            db.set_mood(bot_id, str(args["mood"]))
            return json.dumps({"ok": True}, ensure_ascii=False)
        if name == "update_conversation_summary":
            db.set_summary(bot_id, chat_id, str(args["summary"]))
            return json.dumps({"ok": True}, ensure_ascii=False)
        if name == "add_situation_rule":
            db.add_situation_rule(
                bot_id,
                chat_id,
                trigger_text=str(args.get("trigger_text", "")),
                instruction_text=str(args.get("instruction_text", "")),
            )
            return json.dumps({"ok": True}, ensure_ascii=False)
        if name == "set_bot_persona":
            db.set_bot_persona(bot_id, str(args.get("text", "")))
            return json.dumps({"ok": True}, ensure_ascii=False)
    except Exception as e:
        logger.exception("tool error")
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    return json.dumps({"ok": False, "error": "unknown tool"}, ensure_ascii=False)


def build_system_prompt(
    *,
    personality: str,
    mood: str | None,
    bot_username: str,
    allow_web_search: bool = False,
    tone_instructions: str = "",
    situation_block: str = "",
    feedback_sensitivity: int = 4,
) -> str:
    m = mood or "нейтральное"
    tools_enum = (
        "get_group_state, set_preference, add_long_instruction, set_mood, update_conversation_summary, "
        "add_situation_rule, set_bot_persona"
    )
    if allow_web_search:
        tools_enum += ", web_search (только в этом ответе — пользователь явно запросил поиск ключевым словом)"
    web_extra = ""
    if allow_web_search:
        web_extra = (
            "В этом сообщении разрешён поиск в сети: пользователь ввёл ключевое слово-триггер. "
            "При необходимости один раз вызови web_search с полем query по сути вопроса, затем ответь пользователю "
            "по найденному. Не вызывай web_search без пользы.\n"
        )
    tone_part = ""
    if tone_instructions.strip():
        tone_part = f"Тон и стиль (оси −1…1 с учётом базы чата и кратковременного сдвига):\n{tone_instructions}\n\n"
    sit_part = ""
    if situation_block.strip():
        sit_part = f"{situation_block}\n\n"
    learn = (
        "Учитывай пожелания только от людей (не от ботов). Для сохранения правок используй инструменты; "
        f"уровень учёта деталей из состояния чата (sensitivity) сейчас {feedback_sensitivity} из 5.\n"
    )
    return (
        f"Ты Telegram-бот @{bot_username} в групповом чате.\n"
        f"Базовая личность (можно уточнять через set_bot_persona):\n{personality}\n\n"
        f"Текущее настроение: {m}.\n"
        f"{tone_part}"
        f"{sit_part}"
        "Отвечай в основном по-русски; короткий английский допустим, если уместно.\n"
        f"{learn}"
        f"{web_extra}"
        "Служебные вызовы только через инструменты: "
        f"{tools_enum}. Финальный шаг — complete_turn (action и при необходимости text/reaction_emoji/sticker_file_id).\n"
        "Важно: в поле text у complete_turn и в любом тексте для пользователя — только живая реплика для чата. "
        "Никогда не пиши имена инструментов, JSON, «complete_turn», «action=…» и подобные пометки — их увидят люди.\n"
        "Просьбы о длине/частоте/тоне фиксируй через set_preference (ключи tone_*, response_depth, verbosity, frequency, "
        "числовые вероятности и т.д.). Ситуационные правила — add_situation_rule.\n"
    )


def format_recent_for_prompt(rows: list, limit: int) -> list[str]:
    lines: list[str] = []
    for r in rows[-limit:]:
        who = r.username or (str(r.user_id) if r.user_id else "?")
        if r.content_type == "sticker" and r.sticker_file_id:
            lines.append(f"[{who}] [sticker file_id={r.sticker_file_id}]")
        elif r.content:
            role = r.role
            lines.append(f"[{role} {who}] {r.content}")
    return lines


async def maybe_summarize(
    *,
    client: AsyncOpenAI,
    model: str,
    db: Database,
    bot_id: str,
    chat_id: int,
    recent_lines: list[str],
) -> None:
    """Периодическое обновление саммари по последним репликам."""
    prompt = (
        "Сожми следующий фрагмент переписки в краткое саммари на русском (5–10 предложений максимум), "
        "сохрани имена и договорённости:\n\n" + "\n".join(recent_lines[-40:])
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    text = (resp.choices[0].message.content or "").strip()
    if text:
        db.set_summary(bot_id, chat_id, text)
