"""Реализация run_agent_turn и maybe_summarize для Anthropic API (AsyncAnthropic)."""

from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING, Any, Literal

from bebro_bots.config import AppConfig
from bebro_bots.db import Database, usage_day_utc_string
from bebro_bots.llm import (
    AgentTurnResult,
    CompleteTurnResult,
    _dispatch_tool,
    _finalize_turn_result,
    _parse_complete_turn,
    _tools_schema,
)

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

# Максимум токенов в ответе модели (Anthropic требует явного max_tokens)
_MAX_OUTPUT_TOKENS = 1024


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
    feedback_sensitivity: int = 4,
    vision_images: list[tuple[bytes, str]] | None = None,
    app_cfg: AppConfig | None = None,
    web_search_mode: Literal["on_trigger", "always"] = "on_trigger",
    record_usage: bool = True,
    usage_day: str | None = None,
) -> AgentTurnResult:
    day = usage_day or usage_day_utc_string()
    if app_cfg and record_usage:
        limit = app_cfg.anthropic.daily_input_tokens
        if not db.input_tokens_below_daily_limit(day, limit):
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

    for _ in range(8):
        resp = await client.messages.create(
            model=model,
            max_tokens=_MAX_OUTPUT_TOKENS,
            system=system_prompt,
            messages=messages,
            tools=tools,
            tool_choice={"type": "auto"},
            temperature=0.85,
        )
        if app_cfg and record_usage:
            pt = getattr(getattr(resp, "usage", None), "input_tokens", None)
            if pt is not None:
                db.add_input_tokens(day, int(pt))

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
) -> None:
    """Периодическое обновление саммари (Anthropic)."""
    day = usage_day_utc_string()
    if app_cfg:
        limit = app_cfg.anthropic.daily_input_tokens
        if not db.input_tokens_below_daily_limit(day, limit):
            return
    prompt = (
        "Сожми следующий фрагмент переписки в краткое саммари на русском (5–10 предложений максимум), "
        "сохрани имена и договорённости:\n\n" + "\n".join(recent_lines[-40:])
    )
    resp = await client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    if app_cfg:
        pt = getattr(getattr(resp, "usage", None), "input_tokens", None)
        if pt is not None:
            db.add_input_tokens(day, int(pt))
    text_blocks = [b for b in resp.content if b.type == "text"]
    text = " ".join(b.text for b in text_blocks if b.text).strip()
    if text:
        db.set_summary(bot_id, chat_id, text)
