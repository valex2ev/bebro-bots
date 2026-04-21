from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from openai import AsyncOpenAI

from bebro_bots.config import AppConfig
from bebro_bots.db import Database, usage_day_utc_string
from bebro_bots.telegram_vision import data_url_for_openai
from bebro_bots.tool_definitions import _tools_schema  # noqa: F401  (re-exported for anthropic_llm)
from bebro_bots.tool_dispatch import _dispatch_tool  # noqa: F401  (re-exported for anthropic_llm)

logger = logging.getLogger(__name__)

_LLM_RETRY_DELAYS = (1.0, 2.0, 4.0)  # секунды между попытками (3 retry = 4 попытки итого)


async def _llm_call_with_retry(coro_fn, delays: tuple = _LLM_RETRY_DELAYS):
    """Выполняет coroutine-функцию coro_fn() с exponential backoff при временных сбоях.

    coro_fn — callable без аргументов, возвращающий coroutine (вызывается заново на каждую попытку).
    """
    last_exc: BaseException | None = None
    for attempt, delay in enumerate((-1.0, *delays)):
        if attempt > 0:
            logger.warning("LLM call retry %d/%d after %.0fs (prev error: %s)", attempt, len(delays), delay, last_exc)
            await asyncio.sleep(delay)
        try:
            return await coro_fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise last_exc  # type: ignore[misc]


@dataclass
class CompleteTurnResult:
    action: str  # message | reaction | sticker | none
    text: str | None
    reaction_emoji: str | None
    sticker_file_id: str | None


@dataclass
class AgentTurnResult:
    """Результат run_agent_turn: ответ и флаг блокировки по глобальному лимиту input-токенов."""

    result: CompleteTurnResult
    blocked_by_input_token_limit: bool = False


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


def _user_message_content(
    user_lines: list[str],
    vision_images: list[tuple[bytes, str]] | None,
) -> str | list[dict[str, Any]]:
    text = "\n".join(user_lines)
    imgs = vision_images or []
    if not imgs:
        return text
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for data, mime in imgs:
        url = data_url_for_openai(data, mime)
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


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
    allow_chat_alias_tools: bool = True,
    explicitly_addressed: bool = True,
    feedback_sensitivity: int = 4,
    vision_images: list[tuple[bytes, str]] | None = None,
    app_cfg: AppConfig | None = None,
    web_search_mode: Literal["on_trigger", "always"] = "on_trigger",
    record_usage: bool = True,
    usage_day: str | None = None,
    effective_daily_limit: int | None = None,
) -> AgentTurnResult:
    day = usage_day or usage_day_utc_string()
    if app_cfg and record_usage:
        limit = effective_daily_limit if effective_daily_limit is not None else app_cfg.openai.daily_input_tokens
        if not db.input_tokens_below_daily_limit(day, limit, provider="openai"):
            return AgentTurnResult(
                CompleteTurnResult(action="none", text=None, reaction_emoji=None, sticker_file_id=None),
                blocked_by_input_token_limit=True,
            )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _user_message_content(user_lines, vision_images)},
    ]
    if force_substantive_reply:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Системное напоминание: к тебе явно обратились (@username или сохранённый текстовый алиас чата) — "
                    "дай содержательный ответ (не только одну реакцию), если нет просьбы именно о реакции."
                ),
            }
        )
    if not explicitly_addressed:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Системное напоминание: ты включился в этот разговор самостоятельно (явного @username или алиаса не было). "
                    "Сохраняй правила и инструкции только если они явно адресованы тебе или всем ботам сразу, "
                    "но не конкретному другому боту."
                ),
            }
        )

    async def _consolidate() -> None:
        try:
            await consolidate_rules(
                client=client, model=model, db=db, bot_id=bot_id, chat_id=chat_id,
                app_cfg=app_cfg, usage_day=day, effective_daily_limit=effective_daily_limit,
            )
        except Exception:
            logger.exception("consolidate_rules failed bot=%s chat=%s", bot_id, chat_id)

    tools = _tools_schema(
        include_web_search=allow_web_search,
        web_search_mode=web_search_mode if allow_web_search else "on_trigger",
    )

    # Проверку лимита между итерациями не делаем: после первого вызова токены уже учтены,
    # повторная проверка обрывала цепочку tool_calls до complete_turn.
    for _ in range(8):
        resp = await _llm_call_with_retry(
            lambda: client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.85,
            )
        )
        if app_cfg and record_usage:
            u = getattr(resp, "usage", None)
            pt = getattr(u, "prompt_tokens", None) if u is not None else None
            if pt is not None:
                db.add_input_tokens(day, int(pt), provider="openai")

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
                        allow_chat_alias_tools=allow_chat_alias_tools,
                        feedback_sensitivity=feedback_sensitivity,
                        app_cfg=app_cfg,
                        usage_day=day,
                        consolidate_callback=_consolidate,
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_out,
                    }
                )
            if complete:
                return AgentTurnResult(_finalize_turn_result(complete), blocked_by_input_token_limit=False)
            continue

        text = (choice.content or "").strip()
        if text:
            return AgentTurnResult(
                _finalize_turn_result(
                    CompleteTurnResult(action="message", text=text, reaction_emoji=None, sticker_file_id=None)
                ),
                blocked_by_input_token_limit=False,
            )
        break

    return AgentTurnResult(
        CompleteTurnResult(action="none", text=None, reaction_emoji=None, sticker_file_id=None),
        blocked_by_input_token_limit=False,
    )


def build_system_prompt(
    *,
    personality: str,
    mood: str | None,
    bot_username: str,
    allow_web_search: bool = False,
    web_search_mode: Literal["on_trigger", "always"] = "on_trigger",
    tone_instructions: str = "",
    situation_block: str = "",
    feedback_sensitivity: int = 4,
) -> str:
    m = mood or "нейтральное"
    tools_enum = (
        "get_group_state, set_preference, add_long_instruction, remove_long_instruction, set_mood, "
        "update_conversation_summary, add_situation_rule, remove_situation_rule, set_bot_persona, "
        "add_chat_alias, remove_chat_alias"
    )
    if allow_web_search:
        if web_search_mode == "always":
            tools_enum += ", web_search (при необходимости — актуальные сведения из сети)"
        else:
            tools_enum += ", web_search (только в этом ответе — пользователь явно запросил поиск ключевым словом)"
    web_extra = ""
    if allow_web_search:
        if web_search_mode == "always":
            web_extra = (
                "Поиск в сети (Tavily) в этом ходе доступен при необходимости: для свежих фактов, цен, новостей, "
                "если вопрос требует данных из интернета. Вызови web_search с полем query по сути; не злоупотребляй "
                "и не вызывай без пользы для ответа пользователю. Опирайся на результаты поиска для проверяемых фактов; "
                "не требуется оформлять в ответе явные цитаты или ссылки, если об этом не просили.\n"
            )
        else:
            web_extra = (
                "В этом сообщении разрешён поиск в сети: пользователь ввёл ключевое слово-триггер. "
                "При необходимости один раз вызови web_search с полем query по сути вопроса, затем ответь пользователю "
                "по найденному. Не вызывай web_search без пользы. Явные цитаты источников в тексте для людей не обязательны.\n"
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
        "числовые вероятности и т.д.; reply_probability только в YAML). Долгие инструкции — add_long_instruction / remove_long_instruction; "
        "ситуационные правила — add_situation_rule / remove_situation_rule. "
        "Различай: «отзывайтесь на фразу …» — add_chat_alias; «перестань откликаться на …» / «убери алиас …» — remove_chat_alias "
        "с text=фраза (алиас только у этого бота; если в сообщении @ другого бота без обращения к тебе — инструменты недоступны). "
        "Без @username людей и без логинов Telegram; «называйте кого-то так-то» — add_long_instruction, не add_chat_alias.\n"
    )


def format_recent_for_prompt(rows: list, limit: int) -> list[str]:
    lines: list[str] = []
    for r in rows[-limit:]:
        who = r.username or (str(r.user_id) if r.user_id else "?")
        is_bot = getattr(r, "role", "user") == "assistant"
        prefix = f"бот {who}" if is_bot else f"user {who}"
        if r.content_type == "sticker" and r.sticker_file_id:
            lines.append(f"[{prefix}] [sticker file_id={r.sticker_file_id}]")
        elif r.content_type == "photo":
            cap = (r.content or "").strip()
            cap_part = f" подпись: {cap}" if cap else " (без подписи)"
            lines.append(f"[{prefix}] [фото]{cap_part}")
        elif r.content:
            lines.append(f"[{prefix}] {r.content}")
    return lines


async def maybe_summarize(
    *,
    client: AsyncOpenAI,
    model: str,
    db: Database,
    bot_id: str,
    chat_id: int,
    recent_lines: list[str],
    app_cfg: AppConfig | None = None,
    effective_daily_limit: int | None = None,
) -> None:
    """Периодическое обновление саммари по последним репликам."""
    day = usage_day_utc_string()
    limit = effective_daily_limit if effective_daily_limit is not None else (app_cfg.openai.daily_input_tokens if app_cfg else 0)
    if app_cfg and not db.input_tokens_below_daily_limit(day, limit, provider="openai"):
        return
    prompt = (
        "Сожми следующий фрагмент переписки в краткое саммари на русском (5–10 предложений максимум), "
        "сохрани имена и договорённости:\n\n" + "\n".join(recent_lines[-40:])
    )
    resp = await _llm_call_with_retry(
        lambda: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
    )
    if app_cfg:
        u = getattr(resp, "usage", None)
        pt = getattr(u, "prompt_tokens", None) if u is not None else None
        if pt is not None:
            db.add_input_tokens(day, int(pt), provider="openai")
    text = (resp.choices[0].message.content or "").strip()
    if text:
        db.set_summary(bot_id, chat_id, text)


async def consolidate_rules(
    *,
    client: AsyncOpenAI,
    model: str,
    db: Database,
    bot_id: str,
    chat_id: int,
    app_cfg: AppConfig | None = None,
    usage_day: str | None = None,
    effective_daily_limit: int | None = None,
) -> None:
    """Консолидировать situation_rules и long_memory через LLM: убрать дубли, объединить похожие."""
    day = usage_day or usage_day_utc_string()
    if app_cfg:
        limit = effective_daily_limit if effective_daily_limit is not None else app_cfg.openai.daily_input_tokens
        if not db.input_tokens_below_daily_limit(day, limit, provider="openai"):
            return

    rules = db.list_situation_rules_with_ids(bot_id, chat_id, limit=200, min_created_at=None)
    if len(rules) > 1:
        rules_text = "\n".join(
            f"{i + 1}. trigger: {t}\n   instruction: {ins}"
            for i, (_, t, ins) in enumerate(rules)
        )
        prompt = (
            "Это список ситуационных правил Telegram-бота (trigger → instruction). "
            "Убери точные дубли, объедини правила с одинаковым или перекрывающимся смыслом в одно, "
            "разреши противоречия (предпочти более конкретное или новое), "
            "сократи формулировки без потери смысла.\n\n"
            f"{rules_text}\n\n"
            'Верни ТОЛЬКО JSON-массив: [{"trigger": "...", "instruction": "..."}]. Без пояснений.'
        )
        try:
            resp = await _llm_call_with_retry(
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
            )
            if app_cfg:
                u = getattr(resp, "usage", None)
                pt = getattr(u, "prompt_tokens", None) if u else None
                if pt:
                    db.add_input_tokens(day, int(pt), provider="openai")
            raw = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                consolidated = json.loads(m.group())
                new_rules = [
                    (r["trigger"], r["instruction"])
                    for r in consolidated
                    if isinstance(r, dict) and r.get("trigger") and r.get("instruction")
                ]
                if new_rules:
                    db.replace_situation_rules(bot_id, chat_id, new_rules)
                    logger.info(
                        "consolidate situation_rules bot=%s chat=%s: %d → %d",
                        bot_id, chat_id, len(rules), len(new_rules),
                    )
        except Exception:
            logger.exception("consolidate situation_rules failed bot=%s chat=%s", bot_id, chat_id)

    memory = db.list_long_memory_with_ids(bot_id, chat_id, limit=200, min_created_at=None)
    if len(memory) > 1:
        mem_text = "\n".join(f"{i + 1}. {content}" for i, (_, content) in enumerate(memory))
        prompt = (
            "Это список долговременных инструкций Telegram-бота. "
            "Убери точные дубли, объедини похожие инструкции, разреши противоречия, "
            "сократи без потери смысла.\n\n"
            f"{mem_text}\n\n"
            'Верни ТОЛЬКО JSON-массив строк: ["...", "..."]. Без пояснений.'
        )
        try:
            resp = await _llm_call_with_retry(
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
            )
            if app_cfg:
                u = getattr(resp, "usage", None)
                pt = getattr(u, "prompt_tokens", None) if u else None
                if pt:
                    db.add_input_tokens(day, int(pt), provider="openai")
            raw = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                consolidated = json.loads(m.group())
                new_entries = [e for e in consolidated if isinstance(e, str) and e.strip()]
                if new_entries:
                    db.replace_long_memory(bot_id, chat_id, new_entries)
                    logger.info(
                        "consolidate long_memory bot=%s chat=%s: %d → %d",
                        bot_id, chat_id, len(memory), len(new_entries),
                    )
        except Exception:
            logger.exception("consolidate long_memory failed bot=%s chat=%s", bot_id, chat_id)
