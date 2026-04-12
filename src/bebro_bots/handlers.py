from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from typing import TYPE_CHECKING

from telegram import Message, ReactionTypeEmoji, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes

from bebro_bots.addressing import (
    collect_explicitly_addressed_bot_ids,
    command_is_for_this_bot_username,
)
from bebro_bots.behavior import Engagement, decide_stochastic_engagement
from bebro_bots.config import AppConfig, BotPersonaConfig, min_created_at_cutoff
from bebro_bots.db import Database, usage_day_utc_string
from bebro_bots.llm import (
    CompleteTurnResult,
    build_system_prompt,
    format_recent_for_prompt,
    maybe_summarize,
    run_agent_turn,
)
from bebro_bots.tone_profile import (
    MOOD_RANDOM_POOL,
    drift_random_axes,
    format_tone_lines,
    merged_tone_values,
)
from bebro_bots.telegram_vision import collect_vision_images_for_turn
from bebro_bots.chat_aliases_policy import chat_alias_mutations_allowed_for_message
from bebro_bots.triggers import message_allows_web_search

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def _allowed_chat(cfg: AppConfig, chat_id: int) -> bool:
    if not cfg.allowed_chat_ids:
        return True
    return chat_id in cfg.allowed_chat_ids


def _is_admin(cfg: AppConfig, user_id: int | None, username: str | None = None) -> bool:
    if user_id is None:
        return False
    un = (username or "").strip().lower() or None
    for entry in cfg.admin_user_ids:
        if isinstance(entry, int) and entry == user_id:
            return True
        if isinstance(entry, str) and un is not None and entry == un:
            return True
    return False


def _allow_learning(message: Message) -> bool:
    u = message.from_user
    if not u:
        return False
    return not u.is_bot


def _compose_prompt_extras(
    db: Database,
    bot_id: str,
    chat_id: int,
    settings: dict,
    app_cfg: AppConfig,
) -> tuple[str, str]:
    drift = db.get_tone_drift(bot_id)
    merged = merged_tone_values(settings, drift)
    tone_instructions = format_tone_lines(merged)
    return tone_instructions, ""


def _memory_and_rules_prompt_lines(
    db: Database,
    bot_id: str,
    chat_id: int,
    app_cfg: AppConfig,
) -> list[str]:
    """Явные блоки long_memory и situation_rules в пользовательском промпте (лимиты из YAML)."""
    lm = app_cfg.long_memory
    sr = app_cfg.situation_rules
    mem = db.list_long_memory(
        bot_id,
        chat_id,
        limit=lm.context_limit,
        min_created_at=min_created_at_cutoff(lm.retention_days),
    )
    rules = db.list_situation_rules(
        bot_id,
        chat_id,
        limit=sr.context_limit,
        min_created_at=min_created_at_cutoff(sr.retention_days),
    )
    out: list[str] = []
    if mem:
        out.append("— Долгие инструкции (long_memory) —")
        for i, t in enumerate(mem, 1):
            out.append(f"{i}. {t}")
    if rules:
        out.append("— Ситуационные правила —")
        for t, ins in rules:
            out.append(f"• когда: {t}")
            out.append(f"  → {ins}")
    return out


async def _maybe_notify_limit(
    *,
    application,
    chat_id: int,
    db: Database,
    app_cfg: AppConfig,
    kind: str,
    text: str,
) -> None:
    """kind: input_tokens | web_search. Антиспам по limit_notify_cooldown_minutes."""
    cooldown_sec = float(app_cfg.limit_notify_cooldown_minutes) * 60.0
    if not db.can_send_limit_notify(chat_id, kind, cooldown_sec):
        return
    bot = application.bot
    try:
        await bot.send_message(chat_id=chat_id, text=text)
        db.record_limit_notify(chat_id, kind)
    except Exception:
        logger.exception("limit notify failed")


def _format_params_message(
    *,
    app_cfg: AppConfig,
    persona: BotPersonaConfig,
    db: Database,
    bot_id: str,
    chat_id: int,
) -> str:
    settings = db.get_settings(bot_id, chat_id)
    mood = db.get_mood(bot_id) or persona.mood_default
    persona_text = db.get_bot_persona(bot_id)
    drift = db.get_tone_drift(bot_id)
    merged = merged_tone_values(settings, drift)
    tone_lines = format_tone_lines(merged)
    sr = app_cfg.situation_rules
    rules = db.list_situation_rules(
        bot_id,
        chat_id,
        limit=min(50, sr.max_stored),
        min_created_at=min_created_at_cutoff(sr.retention_days),
    )
    day = usage_day_utc_string()
    used_tok, used_ws = db.get_global_usage(day)
    lim = app_cfg.limits
    lines: list[str] = [
        f"Бот «{bot_id}», чат {chat_id}",
        f"Модель OpenAI: {app_cfg.openai.model}",
        f"Web search (режим): {app_cfg.web_search.mode}",
        f"Сутки учёта лимитов (UTC): {day}",
        f"Input-токены (потрачено / лимит): {used_tok} / {lim.daily_input_tokens}",
        f"Web search за сутки (потрачено / лимит): {used_ws} / {lim.daily_web_searches}",
        f"Восприимчивость к пожеланиям (конфиг): {app_cfg.feedback_sensitivity} из 5",
        f"Вероятность ответа без @ (конфиг): {app_cfg.reply_probability}",
        "",
        "— Персона (БД) —",
        persona_text,
        "",
        "— Настроение —",
        mood,
        "",
        "— Тон (база чата + кратковременный сдвиг) —",
        tone_lines,
        "",
        "— Прочие настройки чата —",
        json.dumps(settings, ensure_ascii=False, indent=2),
        "",
        "— Сдвиг тона (временный) —",
        json.dumps(drift, ensure_ascii=False) if drift else "{}",
        "",
        "— Ситуационные правила —",
    ]
    if rules:
        for t, i in rules:
            lines.append(f"• когда: {t}")
            lines.append(f"  → {i}")
    else:
        lines.append("(нет)")
    aliases = db.list_chat_aliases(bot_id, chat_id)
    lines.extend(["", "— Текстовые алиасы (отклик этого бота по подстроке в сообщении) —"])
    if aliases:
        for a in aliases:
            lines.append(f"• {a}")
    else:
        lines.append("(нет)")
    summ = db.get_summary(bot_id, chat_id)
    lines.extend(["", "— Саммари —", summ or "(нет)"])
    return "\n".join(lines)


async def _record_incoming(
    db: Database,
    bot_id: str,
    message: Message,
) -> None:
    chat_id = message.chat_id
    u = message.from_user
    uid = u.id if u else None
    uname = u.username if u else None
    role = "assistant" if (u and u.is_bot) else "user"
    if message.sticker:
        db.remember_sticker(
            bot_id,
            chat_id,
            message.sticker.file_id,
            message.sticker.set_name,
            source="group",
        )
        db.append_message(
            bot_id,
            chat_id,
            message_id=message.message_id,
            user_id=uid,
            username=uname,
            role=role,
            content=message.caption,
            content_type="sticker",
            sticker_file_id=message.sticker.file_id,
            photo_file_id=None,
        )
        return
    if message.photo:
        ph = message.photo[-1]
        cap = message.caption or ""
        db.append_message(
            bot_id,
            chat_id,
            message_id=message.message_id,
            user_id=uid,
            username=uname,
            role=role,
            content=cap,
            content_type="photo",
            sticker_file_id=None,
            photo_file_id=ph.file_id,
        )
        return
    text = message.text or message.caption or ""
    db.append_message(
        bot_id,
        chat_id,
        message_id=message.message_id,
        user_id=uid,
        username=uname,
        role=role,
        content=text,
        content_type="text",
        sticker_file_id=None,
        photo_file_id=None,
    )


def _looks_emoji_only(s: str) -> bool:
    return bool(s) and all(ord(ch) > 127 or ch.isspace() for ch in s)


async def _execute_turn_result(
    *,
    result: CompleteTurnResult,
    application: Application,
    chat_id: int,
    reply_to_message_id: int | None,
    db: Database,
    bot_id: str,
) -> None:
    """Отправить ответ. Если reply_to_message_id is None — сообщение в чат без reply (инициатива)."""
    bot = application.bot
    if result.action == "none":
        return
    if result.action == "reaction" and result.reaction_emoji:
        emoji = result.reaction_emoji.strip()
        if not emoji:
            return
        if reply_to_message_id is None:
            await bot.send_message(chat_id=chat_id, text=emoji)
            db.append_message(
                bot_id,
                chat_id,
                message_id=0,
                user_id=None,
                username=None,
                role="assistant",
                content=emoji,
                content_type="text",
                sticker_file_id=None,
                photo_file_id=None,
            )
            return
        try:
            await bot.set_message_reaction(
                chat_id=chat_id,
                message_id=reply_to_message_id,
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )
        except Exception:
            logger.exception("set_message_reaction failed")
        return
    if result.action == "sticker":
        fid = result.sticker_file_id or db.random_sticker_file_id(bot_id, chat_id)
        if fid:
            try:
                await bot.send_sticker(chat_id=chat_id, sticker=fid)
                db.append_message(
                    bot_id,
                    chat_id,
                    message_id=0,
                    user_id=None,
                    username=None,
                    role="assistant",
                    content="[sticker]",
                    content_type="sticker",
                    sticker_file_id=fid,
                    photo_file_id=None,
                )
            except Exception:
                logger.exception("send_sticker failed")
        return
    if result.action == "message" and result.text:
        text = result.text.strip()
        if not text:
            return
        if (
            reply_to_message_id is not None
            and len(text) <= 8
            and not re.search(r"\s", text)
            and _looks_emoji_only(text)
        ):
            try:
                await bot.set_message_reaction(
                    chat_id=chat_id,
                    message_id=reply_to_message_id,
                    reaction=[ReactionTypeEmoji(emoji=text)],
                )
            except Exception:
                await bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_to_message_id=reply_to_message_id,
                )
            db.append_message(
                bot_id,
                chat_id,
                message_id=0,
                user_id=None,
                username=None,
                role="assistant",
                content=text,
                content_type="text",
                sticker_file_id=None,
                photo_file_id=None,
            )
            return
        kwargs: dict = {"chat_id": chat_id, "text": text}
        if reply_to_message_id is not None:
            kwargs["reply_to_message_id"] = reply_to_message_id
        await bot.send_message(**kwargs)
        db.append_message(
            bot_id,
            chat_id,
            message_id=0,
            user_id=None,
            username=None,
            role="assistant",
            content=text,
            content_type="text",
            sticker_file_id=None,
            photo_file_id=None,
        )


async def on_group_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message or not message.chat:
        return
    app_cfg: AppConfig = context.application.bot_data["app_cfg"]
    persona: BotPersonaConfig = context.application.bot_data["persona"]
    db: Database = context.application.bot_data["db"]
    client: AsyncOpenAI = context.application.bot_data["openai_client"]
    model: str = context.application.bot_data["model"]

    chat_id = message.chat_id
    if not _allowed_chat(app_cfg, chat_id):
        return

    logger.info(
        "Сообщение получено: bot=%s chat_id=%s type=%s mid=%s",
        persona.id,
        chat_id,
        getattr(message.chat, "type", None),
        message.message_id,
    )

    await _record_incoming(db, persona.id, message)

    settings = db.get_settings(persona.id, chat_id)
    addressed = collect_explicitly_addressed_bot_ids(message, app_cfg, db, chat_id)
    if addressed:
        if persona.id not in addressed:
            return
        eng = Engagement(call_llm=True, force_reply=True)
    else:
        eng = decide_stochastic_engagement(float(app_cfg.reply_probability))
    if not eng.call_llm:
        return

    jmin = int(app_cfg.response_jitter_ms_min)
    jmax = int(app_cfg.response_jitter_ms_max)
    if jmax > jmin:
        await asyncio.sleep(random.uniform(jmin, jmax) / 1000.0)
    elif jmax == jmin and jmin > 0:
        await asyncio.sleep(jmin / 1000.0)

    recent_n = int(settings.get("recent_for_llm", 16))
    rows = db.recent_messages(persona.id, chat_id, limit=max(recent_n, 5))

    day = usage_day_utc_string()
    if not db.input_tokens_below_daily_limit(day, app_cfg.limits.daily_input_tokens):
        await _maybe_notify_limit(
            application=context.application,
            chat_id=chat_id,
            db=db,
            app_cfg=app_cfg,
            kind="input_tokens",
            text="Дневной лимит запросов к модели исчерпан. Попробуйте позже (лимит обновляется по UTC).",
        )
        return

    msg_text = message.text or message.caption or ""
    ws_mode = app_cfg.web_search.mode
    if ws_mode == "always":
        allow_web_search = True
    else:
        allow_web_search = message_allows_web_search(msg_text)
    if allow_web_search and not db.web_search_below_daily_limit(day, app_cfg.limits.daily_web_searches):
        await _maybe_notify_limit(
            application=context.application,
            chat_id=chat_id,
            db=db,
            app_cfg=app_cfg,
            kind="web_search",
            text="Дневной лимит поиска в интернете исчерпан. Повторите завтра (сутки по UTC).",
        )
        allow_web_search = False

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    allow_mutations = _allow_learning(message)
    allow_chat_alias_tools = chat_alias_mutations_allowed_for_message(
        msg_text,
        my_username=persona.username,
        all_bot_usernames=[b.username for b in app_cfg.bots],
    )
    sens = int(app_cfg.feedback_sensitivity)

    mood = db.get_mood(persona.id) or persona.mood_default
    persona_text = db.get_bot_persona(persona.id)
    tone_instructions, _sit_unused = _compose_prompt_extras(
        db, persona.id, chat_id, settings, app_cfg,
    )
    system = build_system_prompt(
        personality=persona_text,
        mood=mood,
        bot_username=persona.username,
        allow_web_search=allow_web_search,
        web_search_mode=ws_mode,
        tone_instructions=tone_instructions,
        situation_block="",
        feedback_sensitivity=sens,
    )
    user_lines: list[str] = [
        "Контекст последних сообщений (ты — ассистент в этом чате):",
    ]
    summary = db.get_summary(persona.id, chat_id)
    if summary:
        user_lines.append(f"Сжатый контекст (саммари): {summary}")
    mem_rules = _memory_and_rules_prompt_lines(db, persona.id, chat_id, app_cfg)
    if mem_rules:
        user_lines.extend(mem_rules)
    user_lines.extend(format_recent_for_prompt(rows, recent_n))

    vc = app_cfg.vision
    vision_images = await collect_vision_images_for_turn(
        context.bot,
        message,
        rows,
        enabled=vc.enabled,
        max_image_bytes=vc.max_image_bytes,
        max_images=vc.max_images_per_turn,
        history_photo_policy="if_text_asks",
    )
    if vision_images:
        user_lines.append(
            "К запросу приложены изображения в хронологическом порядке (по сообщениям в чате); "
            "они соответствуют пометкам [фото] в контексте выше."
        )

    try:
        turn = await run_agent_turn(
            client=client,
            model=model,
            db=db,
            bot_id=persona.id,
            chat_id=chat_id,
            system_prompt=system,
            user_lines=user_lines,
            force_substantive_reply=eng.force_reply,
            allow_web_search=allow_web_search,
            allow_mutations=allow_mutations,
            allow_chat_alias_tools=allow_chat_alias_tools,
            feedback_sensitivity=sens,
            vision_images=vision_images if vision_images else None,
            app_cfg=app_cfg,
            web_search_mode=ws_mode,
        )
        if turn.blocked_by_input_token_limit:
            await _maybe_notify_limit(
                application=context.application,
                chat_id=chat_id,
                db=db,
                app_cfg=app_cfg,
                kind="input_tokens",
                text="Дневной лимит запросов к модели исчерпан. Попробуйте позже (лимит обновляется по UTC).",
            )
            return
        await _execute_turn_result(
            result=turn.result,
            application=context.application,
            chat_id=chat_id,
            reply_to_message_id=message.message_id,
            db=db,
            bot_id=persona.id,
        )
    except Exception:
        logger.exception("LLM turn failed")

    try:
        csum = app_cfg.conversation_summary
        if len(rows) >= csum.min_messages_in_window and random.random() < csum.random_summarize_probability:
            lines = format_recent_for_prompt(rows, len(rows))
            await maybe_summarize(
                client=client,
                model=model,
                db=db,
                bot_id=persona.id,
                chat_id=chat_id,
                recent_lines=lines,
                app_cfg=app_cfg,
            )
    except Exception:
        logger.exception("summarize failed")


async def on_ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Проверка, что бот видит чат. Команды доставляются даже при включённой group privacy."""
    message = update.effective_message
    persona: BotPersonaConfig = context.application.bot_data["persona"]
    if not message:
        return
    if not command_is_for_this_bot_username(message, persona.username):
        return
    uname = persona.username.lstrip("@")
    await message.reply_text(
        f"ok, бот «{persona.id}», chat_id={message.chat_id}.\n\n"
        "Если на обычный текст бот не реагирует: @BotFather → ваш бот → "
        "Bot Settings → Group Privacy → Turn off (Disable). "
        "Без этого Telegram не присылает боту сообщения группы.\n\n"
        "Явное обращение: @username (сущности Telegram), текстовый алиас вне кавычек или ответ на сообщение бота — "
        "откликается только адресат; несколько обращений — все указанные боты. "
        f"Иначе ответ со случайной вероятностью reply_probability из config."
    )


async def on_admin_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    app_cfg: AppConfig = context.application.bot_data["app_cfg"]
    persona: BotPersonaConfig = context.application.bot_data["persona"]
    db: Database = context.application.bot_data["db"]
    if not message or not user:
        return
    if not command_is_for_this_bot_username(message, persona.username):
        return
    chat_id = message.chat_id
    if not _allowed_chat(app_cfg, chat_id):
        await message.reply_text("Access denied")
        return
    if not _is_admin(app_cfg, user.id, user.username):
        await message.reply_text("Команда только для администраторов из конфигурации.")
        return
    db.clear_chat_memory(persona.id, chat_id)
    await message.reply_text("Память для этого чата очищена.")


async def on_bebro_params(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Все параметры бота для текущего чата (читаемый вид)."""
    message = update.effective_message
    app_cfg: AppConfig = context.application.bot_data["app_cfg"]
    persona: BotPersonaConfig = context.application.bot_data["persona"]
    db: Database = context.application.bot_data["db"]
    if not message:
        return
    if not command_is_for_this_bot_username(message, persona.username):
        return
    chat_id = message.chat_id
    text = _format_params_message(
        app_cfg=app_cfg,
        persona=persona,
        db=db,
        bot_id=persona.id,
        chat_id=chat_id,
    )
    if len(text) > 4090:
        text = text[:4080] + "\n…(обрезано)"
    await message.reply_text(text)


async def on_admin_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    app_cfg: AppConfig = context.application.bot_data["app_cfg"]
    persona: BotPersonaConfig = context.application.bot_data["persona"]
    db: Database = context.application.bot_data["db"]
    if not message or not user:
        return
    if not command_is_for_this_bot_username(message, persona.username):
        return
    chat_id = message.chat_id
    if not _allowed_chat(app_cfg, chat_id):
        await message.reply_text("Access denied")
        return
    if not _is_admin(app_cfg, user.id, user.username):
        await message.reply_text("Команда только для администраторов из конфигурации.")
        return
    s = db.get_settings(persona.id, chat_id)
    await message.reply_text(
        f"Бот {persona.id}\nНастройки:\n{json.dumps(s, ensure_ascii=False, indent=2)}",
    )


async def initiative_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Редкая инициатива в случайном чате с историей."""
    app: Application = context.application
    persona: BotPersonaConfig = app.bot_data["persona"]
    db: Database = app.bot_data["db"]
    app_cfg: AppConfig = app.bot_data["app_cfg"]
    client: AsyncOpenAI = app.bot_data["openai_client"]
    model: str = app.bot_data["model"]

    chats = db.distinct_chat_ids(persona.id)
    if not chats:
        return
    random.shuffle(chats)
    chat_id = chats[0]
    if not _allowed_chat(app_cfg, chat_id):
        return

    settings = db.get_settings(persona.id, chat_id)
    if random.random() > float(settings.get("initiative_probability", 0.05)):
        return

    interval = float(settings.get("initiative_interval_minutes", 90)) * 60.0
    last = db.last_initiative_ts(persona.id, chat_id)
    now = time.time()
    if last and (now - last) < interval:
        return

    day = usage_day_utc_string()
    if not db.input_tokens_below_daily_limit(day, app_cfg.limits.daily_input_tokens):
        return

    mood = db.get_mood(persona.id) or persona.mood_default
    sens = int(app_cfg.feedback_sensitivity)
    persona_text = db.get_bot_persona(persona.id)
    tone_instructions, _ = _compose_prompt_extras(
        db, persona.id, chat_id, settings, app_cfg,
    )
    system = build_system_prompt(
        personality=persona_text,
        mood=mood,
        bot_username=persona.username,
        allow_web_search=False,
        tone_instructions=tone_instructions,
        situation_block="",
        feedback_sensitivity=sens,
    )
    rows = db.recent_messages(persona.id, chat_id, limit=int(settings.get("recent_for_llm", 16)))
    user_lines = [
        "[инициатива бота] Напиши одно короткое сообщение в группу без прямого обращения: вопрос, шутка или заметка по контексту. Вызови complete_turn с action=message или sticker.",
    ]
    mem_rules = _memory_and_rules_prompt_lines(db, persona.id, chat_id, app_cfg)
    if mem_rules:
        user_lines.extend(mem_rules)
    user_lines.extend(format_recent_for_prompt(rows, len(rows)))
    vc = app_cfg.vision
    vision_images = await collect_vision_images_for_turn(
        app.bot,
        None,
        rows,
        enabled=vc.enabled,
        max_image_bytes=vc.max_image_bytes,
        max_images=vc.max_images_per_turn,
        history_photo_policy="never",
    )
    if vision_images:
        user_lines.append(
            "К запросу приложены изображения в хронологическом порядке; они соответствуют пометкам [фото] в контексте выше."
        )
    try:
        turn = await run_agent_turn(
            client=client,
            model=model,
            db=db,
            bot_id=persona.id,
            chat_id=chat_id,
            system_prompt=system,
            user_lines=user_lines,
            force_substantive_reply=False,
            allow_web_search=False,
            allow_mutations=False,
            feedback_sensitivity=sens,
            vision_images=vision_images if vision_images else None,
            app_cfg=app_cfg,
        )
        if turn.blocked_by_input_token_limit:
            return
        result = turn.result
        if result.action == "reaction":
            result = CompleteTurnResult(
                action="message",
                text="😊",
                reaction_emoji=None,
                sticker_file_id=None,
            )
        await _execute_turn_result(
            result=result,
            application=app,
            chat_id=chat_id,
            reply_to_message_id=None,
            db=db,
            bot_id=persona.id,
        )
        db.set_last_initiative_ts(persona.id, chat_id, now)
    except Exception:
        logger.exception("initiative failed")


async def proactive_reaction_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Иногда ставит emoji-реакцию на недавнее сообщение человека без текстового ответа."""
    app: Application = context.application
    persona: BotPersonaConfig = app.bot_data["persona"]
    db: Database = app.bot_data["db"]
    app_cfg: AppConfig = app.bot_data["app_cfg"]
    client: AsyncOpenAI = app.bot_data["openai_client"]
    model: str = app.bot_data["model"]

    chats = db.distinct_chat_ids(persona.id)
    if not chats:
        return
    random.shuffle(chats)
    chat_id = chats[0]
    if not _allowed_chat(app_cfg, chat_id):
        return

    day = usage_day_utc_string()
    if not db.input_tokens_below_daily_limit(day, app_cfg.limits.daily_input_tokens):
        return

    settings = db.get_settings(persona.id, chat_id)
    sens = max(1, min(5, int(app_cfg.feedback_sensitivity)))
    base_p = float(settings.get("proactive_reaction_probability", 0.04))
    if random.random() > base_p * (sens / 5.0):
        return

    me = await app.bot.get_me()
    bot_uid = me.id
    rows = db.recent_messages(persona.id, chat_id, limit=30)
    candidates = [
        r
        for r in rows
        if r.role == "user"
        and r.user_id is not None
        and r.user_id != bot_uid
        and (r.content or r.sticker_file_id or r.content_type == "photo")
    ]
    if not candidates:
        return
    target = random.choice(candidates)
    who = target.username or (str(target.user_id) if target.user_id else "?")
    if target.content_type == "sticker" and target.sticker_file_id:
        line = f"[{who}] [sticker file_id={target.sticker_file_id}]"
    elif target.content_type == "photo":
        cap = (target.content or "").strip()
        line = f"[{who}] [фото] {cap or '(без подписи)'}"
    else:
        line = f"[user {who}] {target.content or ''}"

    mood = db.get_mood(persona.id) or persona.mood_default
    persona_text = db.get_bot_persona(persona.id)
    tone_instructions, _ = _compose_prompt_extras(
        db, persona.id, chat_id, settings, app_cfg,
    )
    system = build_system_prompt(
        personality=persona_text,
        mood=mood,
        bot_username=persona.username,
        allow_web_search=False,
        tone_instructions=tone_instructions,
        situation_block="",
        feedback_sensitivity=sens,
    )
    user_lines = [
        "[проактивная реакция] Поставь ровно одну emoji-реакцию на указанное сообщение. "
        "Только complete_turn: action=reaction и reaction_emoji. Без текста в чат.",
        f"message_id={target.message_id}",
        line,
    ]
    vc = app_cfg.vision
    vision_images = await collect_vision_images_for_turn(
        app.bot,
        None,
        [target],
        enabled=vc.enabled,
        max_image_bytes=vc.max_image_bytes,
        max_images=1,
        history_photo_policy="explicit",
    )
    if vision_images:
        user_lines.append(
            "К указанному сообщению приложено изображение — учти содержимое при выборе emoji."
        )
    try:
        turn = await run_agent_turn(
            client=client,
            model=model,
            db=db,
            bot_id=persona.id,
            chat_id=chat_id,
            system_prompt=system,
            user_lines=user_lines,
            force_substantive_reply=False,
            allow_web_search=False,
            allow_mutations=False,
            feedback_sensitivity=sens,
            vision_images=vision_images if vision_images else None,
            app_cfg=app_cfg,
        )
        if turn.blocked_by_input_token_limit:
            return
        result = turn.result
        if result.action == "reaction" and result.reaction_emoji:
            await _execute_turn_result(
                result=result,
                application=app,
                chat_id=chat_id,
                reply_to_message_id=target.message_id,
                db=db,
                bot_id=persona.id,
            )
    except Exception:
        logger.exception("proactive_reaction failed")


async def random_tone_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Случайный кратковременный сдвиг тона и иногда настроения (без участия LLM)."""
    app: Application = context.application
    persona: BotPersonaConfig = app.bot_data["persona"]
    db: Database = app.bot_data["db"]
    app_cfg: AppConfig = app.bot_data["app_cfg"]

    sens = max(1, min(5, int(app_cfg.feedback_sensitivity)))
    if random.random() > 0.18 * (sens / 5.0):
        return

    drift = drift_random_axes()
    db.set_tone_drift(persona.id, drift)
    if random.random() < 0.45:
        db.set_mood(persona.id, random.choice(MOOD_RANDOM_POOL))
    logger.info("Случайный сдвиг тона для бота %s: %s", persona.id, drift)
