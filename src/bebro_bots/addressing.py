"""Явные обращения к ботам: @username (entities), алиасы вне кавычек, reply на сообщение бота."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from telegram.constants import MessageEntityType

if TYPE_CHECKING:
    from telegram import Message

    from bebro_bots.config import AppConfig
    from bebro_bots.db import Database

logger = logging.getLogger(__name__)

# persona_id -> telegram user id (заполняется в post_init каждого Application)
_bot_user_id_to_persona_id: dict[int, str] = {}


def register_bot_user_id(persona_id: str, telegram_user_id: int) -> None:
    _bot_user_id_to_persona_id[telegram_user_id] = persona_id


def _merge_intervals(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Объединение полуинтервалов [start, end) по возрастанию start."""
    if not spans:
        return []
    spans = sorted(spans)
    out: list[tuple[int, int]] = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def quoted_spans_excluding_content(text: str) -> list[tuple[int, int]]:
    """Индексы символов (Unicode), попадающих внутрь кавычек — полуинтервалы [start, end)."""
    if not text:
        return _merge_intervals([])

    raw: list[tuple[int, int]] = []
    raw.extend(_pair_chars(text, '"', '"'))
    raw.extend(_pair_chars(text, "\u201c", "\u201d"))
    raw.extend(_pair_chars(text, "'", "'"))
    raw.extend(_pair_chars(text, "\u2018", "\u2019"))
    raw.extend(_pair_guillemets(text))
    raw.extend(_pair_chars(text, "`", "`"))
    return _merge_intervals(raw)


def _pair_chars(s: str, open_ch: str, close_ch: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == open_ch:
            j = s.find(close_ch, i + 1)
            if j == -1:
                break
            # внутри кавычек, без самих символов кавычек — как в постановке «строка алиаса внутри»
            out.append((i + 1, j))
            i = j + 1
            continue
        i += 1
    return out


def _pair_guillemets(s: str) -> list[tuple[int, int]]:
    O, C = "\u00ab", "\u00bb"
    out: list[tuple[int, int]] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == O:
            j = s.find(C, i + 1)
            if j == -1:
                break
            out.append((i + 1, j))
            i = j + 1
            continue
        i += 1
    return out


def _range_overlaps_excluded(start: int, end: int, excluded: list[tuple[int, int]]) -> bool:
    for s, e in excluded:
        if start < e and end > s:
            return True
    return False


def _alias_matches_outside_quotes(haystack: str, alias: str, excluded: list[tuple[int, int]]) -> bool:
    if not haystack or not alias:
        return False
    h = haystack.lower()
    a = alias.lower()
    la = len(a)
    if la == 0:
        return False
    start = 0
    while True:
        idx = h.find(a, start)
        if idx == -1:
            return False
        if not _range_overlaps_excluded(idx, idx + la, excluded):
            return True
        start = idx + 1


def _username_to_persona_id(app_cfg: Any) -> dict[str, str]:
    return {b.username.lstrip("@").lower(): b.id for b in app_cfg.bots}


def _entities_to_bot_ids(message: Message, entities: Any, app_cfg: Any) -> set[str]:
    out: set[str] = set()
    if not entities:
        return out
    uname_map = _username_to_persona_id(app_cfg)
    for entity in entities:
        try:
            if entity.type == MessageEntityType.MENTION:
                frag = message.parse_entity(entity).strip()
                u = frag.lstrip("@").lower()
                pid = uname_map.get(u)
                if pid:
                    out.add(pid)
            elif entity.type == MessageEntityType.TEXT_MENTION and entity.user:
                uid = entity.user.id
                if uid in _bot_user_id_to_persona_id:
                    out.add(_bot_user_id_to_persona_id[uid])
                else:
                    uu = (entity.user.username or "").lower()
                    if uu:
                        pid = uname_map.get(uu)
                        if pid:
                            out.add(pid)
        except Exception:
            logger.debug("parse_entity failed for entity", exc_info=True)
    return out


def _bot_ids_from_entities(message: Message, app_cfg: Any) -> set[str]:
    out: set[str] = set()
    out |= _entities_to_bot_ids(message, message.entities, app_cfg)
    out |= _entities_to_bot_ids(message, message.caption_entities, app_cfg)
    return out


def _bot_ids_from_aliases(
    message: Message,
    app_cfg: Any,
    db: Database,
    chat_id: int,
) -> set[str]:
    out: set[str] = set()
    if message.text:
        excl = quoted_spans_excluding_content(message.text)
        for b in app_cfg.bots:
            for alias in db.list_chat_aliases(b.id, chat_id):
                if _alias_matches_outside_quotes(message.text, alias, excl):
                    out.add(b.id)
                    break
    if message.caption:
        excl = quoted_spans_excluding_content(message.caption)
        for b in app_cfg.bots:
            for alias in db.list_chat_aliases(b.id, chat_id):
                if _alias_matches_outside_quotes(message.caption, alias, excl):
                    out.add(b.id)
                    break
    return out


def _bot_id_from_reply_to_bot(message: Message, app_cfg: Any) -> str | None:
    rm = message.reply_to_message
    if not rm or not rm.from_user or not rm.from_user.is_bot:
        return None
    u = (rm.from_user.username or "").lower()
    if not u:
        uid = rm.from_user.id
        return _bot_user_id_to_persona_id.get(uid)
    for b in app_cfg.bots:
        if b.username.lstrip("@").lower() == u:
            return b.id
    uid = rm.from_user.id
    return _bot_user_id_to_persona_id.get(uid)


def collect_explicitly_addressed_bot_ids(
    message: Message,
    app_cfg: Any,
    db: Database,
    chat_id: int,
) -> set[str]:
    """Множество ``bots[].id``, к которым в сообщении есть явное обращение. Пустое — обращения не было."""
    addressed: set[str] = set()

    rid = _bot_id_from_reply_to_bot(message, app_cfg)
    if rid:
        addressed.add(rid)

    addressed |= _bot_ids_from_entities(message, app_cfg)
    addressed |= _bot_ids_from_aliases(message, app_cfg, db, chat_id)
    return addressed


def command_target_bot_username(message: Message) -> str | None:
    """Username из ``/cmd@bot`` или ``/cmd@@bot`` (берётся последний непустой фрагмент после ``@``).

    ``None`` — в команде не указан конкретный бот (в группе с несколькими ботами ответят все).
    """
    text = (message.text or "").strip()
    if not text:
        return None
    first = text.split()[0]
    if "@" not in first:
        return None
    parts = first.split("@")
    tail = [p for p in parts[1:] if p]
    if not tail:
        return None
    return tail[-1].lower()


def command_is_for_this_bot_username(message: Message, bot_username: str) -> bool:
    """``False``, если в тексте команды указан ``@`` не этот бот; иначе ``True``."""
    target = command_target_bot_username(message)
    if target is None:
        return True
    return target == bot_username.lstrip("@").lower()
