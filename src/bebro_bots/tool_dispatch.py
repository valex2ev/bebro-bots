"""Диспетчер LLM-инструментов и вспомогательные функции парсинга аргументов."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any, TYPE_CHECKING

from bebro_bots.chat_aliases_policy import reject_reason_for_chat_alias
from bebro_bots.config import LongMemoryPolicy, SituationRulesPolicy, min_created_at_cutoff
from bebro_bots.db import Database, usage_day_utc_string
from bebro_bots.tool_definitions import MUTATING_TOOL_NAMES, PREFERENCE_KEYS
from bebro_bots.tone_profile import TONE_KEYS

if TYPE_CHECKING:
    from bebro_bots.config import AppConfig

logger = logging.getLogger(__name__)


def _parse_pref_value(key: str, raw: str) -> Any:
    s = raw.strip()
    if key in {
        "window_messages",
        "recent_for_llm",
        "initiative_interval_minutes",
    }:
        return int(s)
    if key in {
        "reaction_probability",
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
        patch.setdefault("initiative_probability", 0.02)
    elif frequency == "often":
        patch.setdefault("initiative_probability", 0.12)
    else:
        patch.setdefault("initiative_probability", 0.05)


def _apply_verbosity(patch: dict[str, Any], verbosity: str) -> None:
    if verbosity == "short":
        patch["short_reply_bias"] = min(1.0, float(patch.get("short_reply_bias", 0.0)) + 0.3)
    elif verbosity == "long":
        patch["short_reply_bias"] = max(-1.0, float(patch.get("short_reply_bias", 0.0)) - 0.3)


async def _dispatch_tool(
    *,
    name: str,
    args: dict[str, Any],
    db: Database,
    bot_id: str,
    chat_id: int,
    allow_web_search: bool = False,
    allow_mutations: bool = True,
    allow_chat_alias_tools: bool = True,
    feedback_sensitivity: int = 4,
    app_cfg: AppConfig | None = None,
    usage_day: str | None = None,
    consolidate_callback: Callable[[], Coroutine[Any, Any, None]] | None = None,
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
            uday = usage_day or usage_day_utc_string()
            if app_cfg and not db.web_search_below_daily_limit(uday, app_cfg.limits.daily_web_searches):
                return json.dumps(
                    {"ok": False, "error": "daily_web_search_limit_exhausted"},
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
            if app_cfg and res.get("ok"):
                db.increment_web_search(uday)
            return format_tavily_for_tool(res)
        if name == "get_group_state":
            lm_pol = app_cfg.long_memory if app_cfg else LongMemoryPolicy()
            sr_pol = app_cfg.situation_rules if app_cfg else SituationRulesPolicy()
            prefs = db.get_settings(bot_id, chat_id)
            mood = db.get_mood(bot_id)
            summary = db.get_summary(bot_id, chat_id)
            mem_rows = db.list_long_memory_with_ids(
                bot_id,
                chat_id,
                limit=lm_pol.context_limit,
                min_created_at=min_created_at_cutoff(lm_pol.retention_days),
            )
            mem = [t for _, t in mem_rows]
            long_with_ids = [{"id": mid, "text": txt} for mid, txt in mem_rows]
            rules_rows = db.list_situation_rules_with_ids(
                bot_id,
                chat_id,
                limit=sr_pol.context_limit,
                min_created_at=min_created_at_cutoff(sr_pol.retention_days),
            )
            rules = [(t, i) for _, t, i in rules_rows]
            rules_payload = [{"trigger": t, "instruction": i} for t, i in rules]
            rules_with_ids = [
                {"id": rid, "trigger": t, "instruction": i} for rid, t, i in rules_rows
            ]
            persona_text = db.get_bot_persona(bot_id)
            tone_drift = db.get_tone_drift(bot_id)
            chat_aliases = db.list_chat_aliases(bot_id, chat_id)
            return json.dumps(
                {
                    "persona_text": persona_text,
                    "tone_drift": tone_drift,
                    "preferences": prefs,
                    "mood": mood,
                    "conversation_summary": summary,
                    "long_instructions": mem,
                    "long_instructions_with_ids": long_with_ids,
                    "situation_rules": rules_payload,
                    "situation_rules_with_ids": rules_with_ids,
                    "chat_aliases": chat_aliases,
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
            lm_pol = app_cfg.long_memory if app_cfg else LongMemoryPolicy()
            db.add_long_memory(
                bot_id,
                chat_id,
                str(args["text"]),
                retention_days=lm_pol.retention_days,
            )
            if consolidate_callback:
                asyncio.create_task(consolidate_callback())
            return json.dumps({"ok": True}, ensure_ascii=False)
        if name == "remove_long_instruction":
            ok = db.delete_long_memory_by_id(bot_id, chat_id, int(args.get("memory_id", 0)))
            return json.dumps({"ok": ok}, ensure_ascii=False)
        if name == "set_mood":
            db.set_mood(bot_id, str(args["mood"]))
            return json.dumps({"ok": True}, ensure_ascii=False)
        if name == "update_conversation_summary":
            db.set_summary(bot_id, chat_id, str(args["summary"]))
            return json.dumps({"ok": True}, ensure_ascii=False)
        if name == "add_situation_rule":
            sr_pol = app_cfg.situation_rules if app_cfg else SituationRulesPolicy()
            db.add_situation_rule(
                bot_id,
                chat_id,
                trigger_text=str(args.get("trigger_text", "")),
                instruction_text=str(args.get("instruction_text", "")),
                max_rows=sr_pol.max_stored,
                retention_days=sr_pol.retention_days,
            )
            if consolidate_callback:
                asyncio.create_task(consolidate_callback())
            return json.dumps({"ok": True}, ensure_ascii=False)
        if name == "remove_situation_rule":
            ok = db.delete_situation_rule_by_id(bot_id, chat_id, int(args.get("rule_id", 0)))
            return json.dumps({"ok": ok}, ensure_ascii=False)
        if name == "set_bot_persona":
            db.set_bot_persona(bot_id, str(args.get("text", "")))
            return json.dumps({"ok": True}, ensure_ascii=False)
        if name == "add_chat_alias":
            if not allow_chat_alias_tools:
                return json.dumps(
                    {
                        "ok": False,
                        "error": "instruction_addressed_to_other_bot",
                        "hint": (
                            "В сообщении указан другой бот через @ — add_chat_alias доступен только у того бота, "
                            "к которому обратились (или без @ ботов в тексте)."
                        ),
                    },
                    ensure_ascii=False,
                )
            raw_alias = str(args.get("text", ""))
            reason = reject_reason_for_chat_alias(raw_alias)
            if reason:
                return json.dumps(
                    {
                        "ok": False,
                        "error": reason,
                        "hint": (
                            "Алиасы — только фразы, на которые боты должны откликаться. "
                            "Правила «называйте участника X так-то» — используй add_long_instruction, не add_chat_alias."
                        ),
                    },
                    ensure_ascii=False,
                )
            added = db.add_chat_alias(bot_id, chat_id, raw_alias)
            return json.dumps(
                {"ok": True, "added": added, "aliases": db.list_chat_aliases(bot_id, chat_id)},
                ensure_ascii=False,
            )
        if name == "remove_chat_alias":
            if not allow_chat_alias_tools:
                return json.dumps(
                    {
                        "ok": False,
                        "error": "instruction_addressed_to_other_bot",
                        "hint": (
                            "В сообщении указан другой бот через @ — remove_chat_alias только у адресата."
                        ),
                    },
                    ensure_ascii=False,
                )
            removed = db.remove_chat_alias(bot_id, chat_id, str(args.get("text", "")))
            return json.dumps(
                {"ok": True, "removed": removed, "aliases": db.list_chat_aliases(bot_id, chat_id)},
                ensure_ascii=False,
            )
    except Exception as e:
        logger.exception("tool error")
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
    return json.dumps({"ok": False, "error": "unknown tool"}, ensure_ascii=False)
