"""Периодическая проверка канала Telegram + OpenAI и перезапуск VPN (служба Windows)."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time
from typing import Any

from openai import AsyncOpenAI
from telegram.ext import ContextTypes

from bebro_bots.config import AppConfig

logger = logging.getLogger(__name__)

PROBE_TIMEOUT = 20.0
RESTART_TIMEOUT = 120


async def probe_connectivity(bot: Any, client: AsyncOpenAI) -> bool:
    """Успех только если доступны и Telegram Bot API, и OpenAI API."""
    try:
        await asyncio.wait_for(bot.get_me(), timeout=PROBE_TIMEOUT)
    except Exception:
        logger.warning("VPN watch: проверка Telegram не удалась", exc_info=True)
        return False
    try:
        await asyncio.wait_for(client.models.list(), timeout=PROBE_TIMEOUT)
    except Exception:
        logger.warning("VPN watch: проверка OpenAI не удалась", exc_info=True)
        return False
    return True


def _restart_windows_service_sync(service_name: str) -> None:
    """Синхронный перезапуск службы через PowerShell (вызывать из asyncio.to_thread)."""
    # Экранирование одинарной кавычки в имени службы для PowerShell
    safe = service_name.replace("'", "''")
    cmd = (
        f"$ErrorActionPreference='Stop'; "
        f"Restart-Service -Name '{safe}' -Force"
    )
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            cmd,
        ],
        check=True,
        timeout=RESTART_TIMEOUT,
        capture_output=True,
        text=True,
    )


async def restart_windows_service(service_name: str) -> None:
    await asyncio.to_thread(_restart_windows_service_sync, service_name)


def _fmt_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


async def vpn_watch_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    if sys.platform != "win32":
        return

    app = context.application
    cfg: AppConfig = app.bot_data["app_cfg"]
    vw = cfg.vpn_watch
    if not vw.enabled:
        return

    persona = app.bot_data["persona"]
    if persona.id != vw.notify_bot_id:
        return

    client: AsyncOpenAI = app.bot_data["openai_client"]
    bot = app.bot

    state: dict[str, Any] = app.bot_data.setdefault(
        "vpn_watch_state",
        {"failure_detected_at": None, "failure_notified": False},
    )

    ok = await probe_connectivity(bot, client)
    if ok:
        state["failure_detected_at"] = None
        state["failure_notified"] = False
        return

    now = time.time()
    if state.get("failure_detected_at") is None:
        state["failure_detected_at"] = now
    detected_at: float = state["failure_detected_at"]

    for attempt in range(1, vw.reconnect_max_attempts + 1):
        try:
            await restart_windows_service(vw.windows_service_name)
            logger.info(
                "VPN watch: Restart-Service «%s», попытка %s/%s",
                vw.windows_service_name,
                attempt,
                vw.reconnect_max_attempts,
            )
        except Exception:
            logger.exception(
                "VPN watch: ошибка перезапуска службы «%s», попытка %s",
                vw.windows_service_name,
                attempt,
            )

        await asyncio.sleep(vw.post_restart_wait_seconds)

        if await probe_connectivity(bot, client):
            recovery_ts = time.time()
            text = (
                "VPN: соединение восстановлено.\n"
                f"Обнаружение сбоя: {_fmt_ts(detected_at)}\n"
                f"Восстановление: {_fmt_ts(recovery_ts)}\n"
                f"Попыток перезапуска службы: {attempt}"
            )
            try:
                await bot.send_message(chat_id=vw.notify_chat_id, text=text)
            except Exception:
                logger.exception("VPN watch: не удалось отправить сообщение о восстановлении")
            state["failure_detected_at"] = None
            state["failure_notified"] = False
            return

        if attempt < vw.reconnect_max_attempts:
            await asyncio.sleep(vw.reconnect_backoff_seconds)

    logger.error(
        "VPN watch: не удалось восстановить канал после %s попыток",
        vw.reconnect_max_attempts,
    )
    if not state.get("failure_notified"):
        fail_text = (
            "VPN: не удалось восстановить соединение после "
            f"{vw.reconnect_max_attempts} попыток перезапуска службы.\n"
            f"Первое обнаружение сбоя: {_fmt_ts(detected_at)}"
        )
        try:
            await bot.send_message(chat_id=vw.notify_chat_id, text=fail_text)
        except Exception:
            logger.exception("VPN watch: не удалось отправить сообщение об ошибке")
        state["failure_notified"] = True
