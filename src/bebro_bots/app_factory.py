from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from openai import AsyncOpenAI

from bebro_bots.config import AppConfig, BotPersonaConfig, env_token
from bebro_bots.db import Database
from bebro_bots.handlers import (
    initiative_job,
    on_admin_reset,
    on_admin_status,
    on_bebro_params,
    on_group_message,
    on_ping,
    proactive_reaction_job,
    random_tone_job,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


async def _preload_sticker_sets(application: Application, persona: BotPersonaConfig, db: Database) -> None:
    bot = application.bot
    for pack_name in persona.sticker_set_names:
        try:
            st = await bot.get_sticker_set(pack_name)
            for sticker in st.stickers:
                db.remember_sticker(persona.id, 0, sticker.file_id, st.name, source="pack")
            logger.info("Загружен стикерпак %s для бота %s", pack_name, persona.id)
        except Exception:
            logger.exception("Не удалось загрузить стикерпак %s", pack_name)


def build_application(
    *,
    persona: BotPersonaConfig,
    app_cfg: AppConfig,
    db: Database,
    client: AsyncOpenAI,
) -> Application:
    token = env_token(persona.token_env)

    async def post_init(app: Application) -> None:
        await _preload_sticker_sets(app, persona, db)
        jq = app.job_queue
        if jq:
            jq.run_repeating(
                initiative_job,
                interval=180,
                first=90,
                name=f"initiative_{persona.id}",
            )
            jq.run_repeating(
                proactive_reaction_job,
                interval=120,
                first=55,
                name=f"proactive_reaction_{persona.id}",
            )
            jq.run_repeating(
                random_tone_job,
                interval=600,
                first=240,
                name=f"random_tone_{persona.id}",
            )
        else:
            logger.warning("JobQueue недоступна — инициатива отключена (установите python-telegram-bot[job-queue]).")

    application = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .build()
    )

    application.bot_data["app_cfg"] = app_cfg
    application.bot_data["persona"] = persona
    application.bot_data["db"] = db
    application.bot_data["openai_client"] = client
    application.bot_data["model"] = app_cfg.openai.model
    application.bot_data["feedback_sensitivity"] = app_cfg.feedback_sensitivity

    group = filters.ChatType.GROUPS
    msg_filter = group & ~filters.COMMAND & (filters.TEXT | filters.Sticker.ALL | filters.Caption)

    application.add_handler(MessageHandler(msg_filter, on_group_message))
    application.add_handler(CommandHandler("bebro_ping", on_ping, filters=group))
    application.add_handler(CommandHandler("bebro_params", on_bebro_params, filters=group))
    application.add_handler(CommandHandler("bebroadmin_reset", on_admin_reset, filters=group))
    application.add_handler(CommandHandler("bebroadmin_status", on_admin_status, filters=group))

    async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error("Ошибка при обработке update", exc_info=context.error)

    application.add_error_handler(_on_error)

    return application
