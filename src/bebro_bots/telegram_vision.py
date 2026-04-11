"""Загрузка фото из Telegram и подготовка к OpenAI vision (data URL)."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from telegram import Bot, Message

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


async def collect_vision_images_for_turn(
    bot: Bot,
    message: Message | None,
    history_rows: list[Any],
    *,
    enabled: bool,
    max_image_bytes: int,
    max_images: int,
) -> list[tuple[bytes, str]]:
    """Собирает изображения для multimodal-запроса: текущее фото, reply на фото, фото из окна истории.

    Порядок — хронологический по ``message_id`` Telegram; дубликаты ``file_id`` отбрасываются.
    ``history_rows`` — строки из БД (есть ``message_id``, ``content_type``, ``photo_file_id``).
    """
    if not enabled or max_images < 1:
        return []

    candidates: list[tuple[int, str]] = []

    if message is not None:
        if message.photo:
            candidates.append((message.message_id, message.photo[-1].file_id))
        rm = message.reply_to_message
        if rm is not None and rm.photo:
            candidates.append((rm.message_id, rm.photo[-1].file_id))

    for row in history_rows:
        ct = getattr(row, "content_type", None) or ""
        fid = getattr(row, "photo_file_id", None)
        mid = getattr(row, "message_id", None)
        if ct == "photo" and fid and mid is not None:
            candidates.append((int(mid), str(fid)))

    candidates.sort(key=lambda x: x[0])
    ordered_fids: list[str] = []
    seen: set[str] = set()
    for _mid, fid in candidates:
        if fid in seen:
            continue
        seen.add(fid)
        ordered_fids.append(fid)
        if len(ordered_fids) >= max_images:
            break

    out: list[tuple[bytes, str]] = []
    for fid in ordered_fids:
        try:
            raw = await download_photo_file_bytes(bot, fid)
            if len(raw) > max_image_bytes:
                logger.warning(
                    "Фото %s… пропущено (%s байт > max_image_bytes)",
                    fid[:16],
                    len(raw),
                )
                continue
            out.append((raw, sniff_image_mime(raw)))
        except Exception:
            logger.exception("Не удалось загрузить фото file_id для vision")
    return out
