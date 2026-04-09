from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import TYPE_CHECKING

from telegram import Message, ReactionTypeEmoji, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes

from bebro_bots.behavior import decide_engagement
from bebro_bots.config import AppConfig, BotPersonaConfig, min_created_at_cutoff
from bebro_bots.db import Database
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
    format_situation_rules_block,
    format_tone_lines,
    merged_tone_values,
)
from bebro_bots.telegram_vision import download_photo_file_bytes, sniff_image_mime
from bebro_bots.chat_aliases_policy import chat_alias_mutations_allowed_for_message
from bebro_bots.triggers import message_allows_web_search

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def _allowed_chat(cfg: AppConfig, chat_id: int) -> bool:
    if not cfg.allowed_chat_ids:
        return True
    return chat_id in cfg.allowed_chat_ids


def _is_admin(cfg: AppConfig, user_id: int | None) -> bool:
    if user_id is None:
        return False
    return user_id in cfg.admin_user_ids


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
    sr = app_cfg.situation_rules
    rules = db.list_situation_rules(
        bot_id,
        chat_id,
        limit=sr.context_limit,
        min_created_at=min_created_at_cutoff(sr.retention_days),
    )
    situation_block = format_situation_rules_block(rules)
    return tone_instructions, situation_block


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
    lines: list[str] = [
        f"Бот «{bot_id}», чат {chat_id}",
        f"Восприимчивость к пожеланиям (конфиг): {app_cfg.feedback_sensitivity} из 5",
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
    uid = message.from_user.id if message.from_user else None
    uname = message.from_user.username if message.from_user else None
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
            role="user",
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
            role="user",
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
        role="user",
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

    bot = context.bot
    me = await bot.get_me()
    settings = db.get_settings(persona.id, chat_id)
    chat_aliases = db.list_chat_aliases(persona.id, chat_id)
    eng = decide_engagement(
        message,
        bot_id=me.id,
        bot_username=persona.username,
        settings=settings,
        chat_aliases=chat_aliases,
        reply_probability=float(app_cfg.reply_probability),
        reply_to_bot_probability=float(app_cfg.reply_to_bot_probability),
    )
    if not eng.call_llm:
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    msg_text = message.text or message.caption or ""
    ws_mode = app_cfg.web_search.mode
    if ws_mode == "always":
        allow_web_search = True
    else:
        allow_web_search = message_allows_web_search(msg_text)
    allow_mutations = _allow_learning(message)
    allow_chat_alias_tools = chat_alias_mutations_allowed_for_message(
        msg_text,
        my_username=persona.username,
        all_bot_usernames=[b.username for b in app_cfg.bots],
    )
    sens = int(app_cfg.feedback_sensitivity)

    mood = db.get_mood(persona.id) or persona.mood_default
    persona_text = db.get_bot_persona(persona.id)
    tone_instructions, situation_block = _compose_prompt_extras(
        db, persona.id, chat_id, settings, app_cfg,
    )
    system = build_system_prompt(
        personality=persona_text,
        mood=mood,
        bot_username=persona.username,
        allow_web_search=allow_web_search,
        web_search_mode=ws_mode,
        tone_instructions=tone_instructions,
        situation_block=situation_block,
        feedback_sensitivity=sens,
    )
    recent_n = int(settings.get("recent_for_llm", 16))
    rows = db.recent_messages(persona.id, chat_id, limit=max(recent_n, 5))
    user_lines = [
        "Контекст последних сообщений (ты — ассистент в этом чате):",
        *format_recent_for_prompt(rows, recent_n),
    ]
    summary = db.get_summary(persona.id, chat_id)
    if summary:
        user_lines.insert(1, f"Сжатый контекст (саммари): {summary}")

    vision_image: tuple[bytes, str] | None = None
    if message.photo and app_cfg.vision.enabled:
        try:
            raw = await download_photo_file_bytes(context.bot, message.photo[-1].file_id)
            if len(raw) > app_cfg.vision.max_image_bytes:
                logger.warning(
                    "Фото слишком большое (%s байт), vision пропущен",
                    len(raw),
                )
            else:
                mime = sniff_image_mime(raw)
                vision_image = (raw, mime)
                user_lines.append(
                    "К последнему сообщению пользователя приложено изображение — опиши или ответь по содержимому."
                )
        except Exception:
            logger.exception("Не удалось загрузить фото для vision")

    try:
        result = await run_agent_turn(
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
            vision_image=vision_image,
            app_cfg=app_cfg,
            web_search_mode=ws_mode,
        )
        await _execute_turn_result(
            result=result,
            application=context.application,
            chat_id=chat_id,
            reply_to_message_id=message.message_id,
            db=db,
            bot_id=persona.id,
        )
    except Exception:
        logger.exception("LLM turn failed")

    try:
        if len(rows) >= 15 and random.random() < 0.06:
            lines = format_recent_for_prompt(rows, len(rows))
            await maybe_summarize(
                client=client,
                model=model,
                db=db,
                bot_id=persona.id,
                chat_id=chat_id,
                recent_lines=lines,
            )
    except Exception:
        logger.exception("summarize failed")


async def on_ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Проверка, что бот видит чат. Команды доставляются даже при включённой group privacy."""
    message = update.effective_message
    persona: BotPersonaConfig = context.application.bot_data["persona"]
    if not message:
        return
    uname = persona.username.lstrip("@")
    await message.reply_text(
        f"ok, бот «{persona.id}», chat_id={message.chat_id}.\n\n"
        "Если на обычный текст бот не реагирует: @BotFather → ваш бот → "
        "Bot Settings → Group Privacy → Turn off (Disable). "
        "Без этого Telegram не присылает боту сообщения группы.\n\n"
        f"Без @упоминания ответ случайный (~15% по умолчанию); с @{uname} — ответ обязателен."
    )


async def on_admin_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    app_cfg: AppConfig = context.application.bot_data["app_cfg"]
    persona: BotPersonaConfig = context.application.bot_data["persona"]
    db: Database = context.application.bot_data["db"]
    if not message or not user:
        return
    if not _is_admin(app_cfg, user.id):
        await message.reply_text("Команда только для администраторов из конфигурации.")
        return
    chat_id = message.chat_id
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
    if not _is_admin(app_cfg, user.id):
        await message.reply_text("Команда только для администраторов из конфигурации.")
        return
    chat_id = message.chat_id
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

    mood = db.get_mood(persona.id) or persona.mood_default
    sens = int(app_cfg.feedback_sensitivity)
    persona_text = db.get_bot_persona(persona.id)
    tone_instructions, situation_block = _compose_prompt_extras(
        db, persona.id, chat_id, settings, app_cfg,
    )
    system = build_system_prompt(
        personality=persona_text,
        mood=mood,
        bot_username=persona.username,
        allow_web_search=False,
        tone_instructions=tone_instructions,
        situation_block=situation_block,
        feedback_sensitivity=sens,
    )
    rows = db.recent_messages(persona.id, chat_id, limit=int(settings.get("recent_for_llm", 16)))
    user_lines = [
        "[инициатива бота] Напиши одно короткое сообщение в группу без прямого обращения: вопрос, шутка или заметка по контексту. Вызови complete_turn с action=message или sticker.",
        *format_recent_for_prompt(rows, len(rows)),
    ]
    try:
        result = await run_agent_turn(
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
            app_cfg=app_cfg,
        )
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
    tone_instructions, situation_block = _compose_prompt_extras(
        db, persona.id, chat_id, settings, app_cfg,
    )
    system = build_system_prompt(
        personality=persona_text,
        mood=mood,
        bot_username=persona.username,
        allow_web_search=False,
        tone_instructions=tone_instructions,
        situation_block=situation_block,
        feedback_sensitivity=sens,
    )
    user_lines = [
        "[проактивная реакция] Поставь ровно одну emoji-реакцию на указанное сообщение. "
        "Только complete_turn: action=reaction и reaction_emoji. Без текста в чат.",
        f"message_id={target.message_id}",
        line,
    ]
    try:
        result = await run_agent_turn(
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
            app_cfg=app_cfg,
        )
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
