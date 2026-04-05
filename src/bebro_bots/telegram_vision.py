"""Загрузка фото из Telegram и подготовка к OpenAI vision (data URL)."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telegram import Bot

logger = logging.getLogger(__name__)


def sniff_image_mime(data: bytes) -> str:
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def data_url_for_openai(data: bytes, mime: str) -> str:
    b64 = base64.standard_b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


async def download_photo_file_bytes(bot: Bot, file_id: str) -> bytes:
    tg_file = await bot.get_file(file_id)
    buf = await tg_file.download_as_bytearray()
    return bytes(buf)
