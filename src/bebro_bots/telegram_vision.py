"""Загрузка фото из Telegram и подготовка к OpenAI vision (data URL)."""

from __future__ import annotations

import base64
import logging
import re
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from telegram import Bot, Message

logger = logging.getLogger(__name__)

HistoryPhotoPolicy = Literal["never", "if_text_asks", "explicit"]


def text_asks_prior_chat_images(text: str | None) -> bool:
    """True, если по тексту/подписи запрашивают действие с картинкой из более ранних сообщений (не только текущее/reply).

    Эвристика по рус. (и частично англ.) формулировкам: «фото выше», «прокомментируй картинку в чате», «что изображено на том фото» и т.п.
    """
    if not text:
        return False
    t = text.lower().strip()
    if len(t) < 4:
        return False

    prior = (
        "выше",
        "раньше",
        "в чате",
        "в переписке",
        "то фото",
        "эту картинку",
        "это фото",
        "прошл",
        "предыдущ",
        "сверху",
        "над этим",
        "above",  # e.g. "photo above"
        "earlier",
    )
    img = ("фото", "фотк", "картин", "изображен", "снимк", "рисунк", "picture", "image", "photo")
    action = (
        "прокоммент",
        "комментир",
        "мнение",
        "оцени",
        "что изображено",
        "что нарисовано",
        "что на ",
        "что это за",
        "опиши",
        "describe",
        "comment on",
    )

    has_prior = any(p in t for p in prior)
    has_img = any(p in t for p in img)
    has_action = any(p in t for p in action)

    if has_prior and has_img:
        return True
    if has_prior and has_action:
        return True

    if re.search(
        r"(?:фото|картинк|изображен|снимок|picture|image)[^.!?\n]{0,50}(?:выше|раньше|в\s+чате|above|earlier)",
        t,
    ):
        return True
    if re.search(
        r"(?:выше|раньше|в\s+чате|above|earlier)[^.!?\n]{0,50}(?:фото|картинк|изображен|снимок|picture|image)",
        t,
    ):
        return True
    return False


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


def image_block_for_anthropic(data: bytes, mime: str) -> dict:
    """Блок изображения в формате Anthropic Messages API (base64 source)."""
    b64 = base64.standard_b64encode(data).decode("ascii")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime,
            "data": b64,
        },
    }


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
    history_photo_policy: HistoryPhotoPolicy = "if_text_asks",
) -> list[tuple[bytes, str]]:
    """Собирает изображения для multimodal-запроса.

    Всегда (если есть ``message``): фото в текущем сообщении; фото у ``reply_to_message``.

    Фото из ``history_rows`` (SQLite) добавляются только если ``history_photo_policy``:
    ``never`` — не добавлять; ``if_text_asks`` — когда :func:`text_asks_prior_chat_images` по тексту/подписи
    текущего сообщения; ``explicit`` — все фото из переданных строк (например одно целевое сообщение в проактиве).

    Порядок — хронологический по ``message_id``; дубликаты ``file_id`` отбрасываются.
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

    include_from_rows = False
    if history_photo_policy == "explicit":
        include_from_rows = True
    elif history_photo_policy == "if_text_asks":
        combined = ""
        if message is not None:
            combined = (message.text or message.caption or "").strip()
        include_from_rows = text_asks_prior_chat_images(combined or None)
    # "never" → include_from_rows остаётся False

    if include_from_rows:
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
