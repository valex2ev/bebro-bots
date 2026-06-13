"""Реализация run_agent_turn и maybe_summarize для Anthropic API (AsyncAnthropic)."""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Literal

from bebro_bots.config import AppConfig
from bebro_bots.db import Database, usage_day_utc_string
from bebro_bots.llm import (
    AgentTurnResult,
    CompleteTurnResult,
    _dispatch_tool,
    _finalize_turn_result,
    _llm_call_with_retry,
    _parse_complete_turn,
    _tools_schema,
)

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

# Запасной лимит токенов ответа, если AppConfig недоступен.
_DEFAULT_MAX_OUTPUT_TOKENS = 2048


def _count_input_tokens(usage: Any) -> int:
    """Суммирует все входные токены из usage: обычные + создание кэша + чтение кэша."""
    if usage is None:
        return 0
    return (
        (getattr(usage, "input_tokens", None) or 0)
        + (getattr(usage, "cache_creation_input_tokens", None) or 0)
        + (getattr(usage, "cache_read_input_tokens", None) or 0)
    )


def _openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Конвертирует список инструментов из формата OpenAI в формат Anthropic.

    OpenAI:  [{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}]
    Anthropic: [{"name": ..., "description": ..., "input_schema": {...}}]
    """
    out: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function", {})
        out.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return out


def _image_block_for_anthropic(data: bytes, mime: str) -> dict[str, Any]:
    """Блок изображения в формате Anthropic (base64 source)."""
    b64 = base64.standard_b64encode(data).decode("ascii")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime,
            "data": b64,
        },
    }


def _user_message_content_anthropic(
    user_lines: list[str],
    vision_images: list[tuple[bytes, str]] | None,
) -> list[dict[str, Any]]:
    """Формирует content первого пользовательского сообщения (текст + изображения)."""
    text = "\n".join(user_lines)
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for data, mime in (vision_images or []):
        parts.append(_image_block_for_anthropic(data, mime))
    return parts


async def run_agent_turn(
    *,
    client: AsyncAnthropic,
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
        limit = effective_daily_limit if effective_daily_limit is not None else app_cfg.anthropic.daily_input_tokens
        if not db.input_tokens_below_daily_limit(day, limit, provider="anthropic"):
            return AgentTurnResult(
                CompleteTurnResult(action="none", text=None, reaction_emoji=None, sticker_file_id=None),
                blocked_by_input_token_limit=True,
            )

    # Первое пользовательское сообщение (текст + опциональные изображения)
    first_user_content = _user_message_content_anthropic(user_lines, vision_images)

    if force_substantive_reply:
        first_user_content.append(
            {
                "type": "text",
                "text": (
                    "Системное напоминание: к тебе явно обратились (@username или сохранённый текстовый алиас чата) — "
                    "дай содержательный ответ (не только одну реакцию), если нет просьбы именно о реакции."
                ),
            }
        )
    if not explicitly_addressed:
        first_user_content.append(
            {
                "type": "text",
                "text": (
                    "Системное напоминание: ты включился в этот разговор самостоятельно (явного @username или алиаса не было). "
                    "Сохраняй правила и инструкции только если они явно адресованы тебе или всем ботам сразу, "
                    "но не конкретному другому боту."
                ),
            }
        )

    # Anthropic: messages без system; system передаётся отдельно.
    # content каждого сообщения — список блоков или строка.
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": first_user_content},
    ]

    openai_tools = _tools_schema(
        include_web_search=allow_web_search,
        web_search_mode=web_search_mode if allow_web_search else "on_trigger",
    )
    tools = _openai_tools_to_anthropic(openai_tools)

    # Кэшируем список инструментов: добавляем cache_control к последнему инструменту.
    if tools:
        tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}

    # Кэшируем system prompt: передаём как список блоков с cache_control.
    system_blocks: list[dict[str, Any]] = [
        {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
    ]

    max_output_tokens = (
        app_cfg.anthropic.max_output_tokens if app_cfg else _DEFAULT_MAX_OUTPUT_TOKENS
    )
    for _ in range(8):
        resp = await _llm_call_with_retry(
            lambda: client.messages.create(
                model=model,
                max_tokens=max_output_tokens,
                system=system_blocks,
                messages=messages,
                tools=tools,
                tool_choice={"type": "auto"},
                temperature=0.85,
            )
        )
        if app_cfg and record_usage:
            pt = _count_input_tokens(getattr(resp, "usage", None))
            if pt:
                db.add_input_tokens(day, pt, provider="anthropic")

        # resp.content — список блоков: TextBlock | ToolUseBlock
        tool_use_blocks = [b for b in resp.content if b.type == "tool_use"]
        text_blocks = [b for b in resp.content if b.type == "text"]

        if tool_use_blocks:
            # Добавляем ответ ассистента с вызовами инструментов
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": b.id,
                            "name": b.name,
                            "input": b.input,
                        }
                        for b in tool_use_blocks
                    ],
                }
            )

            # Результаты инструментов собираем в одно user-сообщение (Anthropic требует)
            tool_results: list[dict[str, Any]] = []
            complete: CompleteTurnResult | None = None

            async def _consolidate() -> None:
                try:
                    await consolidate_rules(
                        client=client, model=model, db=db, bot_id=bot_id, chat_id=chat_id,
                        app_cfg=app_cfg, usage_day=day, effective_daily_limit=effective_daily_limit,
                    )
                except Exception:
                    logger.exception("consolidate_rules failed bot=%s chat=%s", bot_id, chat_id)

            for b in tool_use_blocks:
                name = b.name
                args = b.input if isinstance(b.input, dict) else {}
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
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": tool_out,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

            if complete:
                return AgentTurnResult(_finalize_turn_result(complete), blocked_by_input_token_limit=False)
            continue

        # Нет вызовов инструментов — обычный текстовый ответ
        text = " ".join(b.text for b in text_blocks if b.text).strip()
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


async def maybe_summarize(
    *,
    client: AsyncAnthropic,
    model: str,
    db: Database,
    bot_id: str,
    chat_id: int,
    recent_lines: list[str],
    app_cfg: AppConfig | None = None,
    effective_daily_limit: int | None = None,
) -> None:
    """Периодическое обновление саммари (Anthropic)."""
    day = usage_day_utc_string()
    if app_cfg:
        limit = effective_daily_limit if effective_daily_limit is not None else app_cfg.anthropic.daily_input_tokens
        if not db.input_tokens_below_daily_limit(day, limit, provider="anthropic"):
            return
    prompt = (
        "Сожми следующий фрагмент переписки в краткое саммари на русском (5–10 предложений максимум), "
        "сохрани имена и договорённости:\n\n" + "\n".join(recent_lines[-40:])
    )
    resp = await _llm_call_with_retry(
        lambda: client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
    )
    if app_cfg:
        pt = _count_input_tokens(getattr(resp, "usage", None))
        if pt:
            db.add_input_tokens(day, pt, provider="anthropic")
    text_blocks = [b for b in resp.content if b.type == "text"]
    text = " ".join(b.text for b in text_blocks if b.text).strip()
    if text:
        db.set_summary(bot_id, chat_id, text)


async def consolidate_rules(
    *,
    client: AsyncAnthropic,
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
        limit = effective_daily_limit if effective_daily_limit is not None else app_cfg.anthropic.daily_input_tokens
        if not db.input_tokens_below_daily_limit(day, limit, provider="anthropic"):
            return

    max_tokens = app_cfg.anthropic.max_output_tokens if app_cfg else 2048

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
                lambda: client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
            )
            if app_cfg:
                pt = _count_input_tokens(getattr(resp, "usage", None))
                if pt:
                    db.add_input_tokens(day, pt, provider="anthropic")
            text_blocks = [b for b in resp.content if b.type == "text"]
            raw = " ".join(b.text for b in text_blocks if b.text).strip()
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
                lambda: client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
            )
            if app_cfg:
                pt = _count_input_tokens(getattr(resp, "usage", None))
                if pt:
                    db.add_input_tokens(day, pt, provider="anthropic")
            text_blocks = [b for b in resp.content if b.type == "text"]
            raw = " ".join(b.text for b in text_blocks if b.text).strip()
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
