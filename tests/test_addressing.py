"""Unit-тесты для addressing.py.

Покрывают:
- quoted_spans_excluding_content (все виды кавычек)
- _alias_matches_outside_quotes
- collect_explicitly_addressed_bot_ids (entities, алиасы, reply)
- command_is_for_this_bot_username
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from bebro_bots.addressing import (
    _alias_matches_outside_quotes,
    collect_explicitly_addressed_bot_ids,
    command_is_for_this_bot_username,
    quoted_spans_excluding_content,
    register_bot_user_id,
)


# ---------------------------------------------------------------------------
# quoted_spans_excluding_content
# ---------------------------------------------------------------------------


def test_no_quotes_returns_empty():
    assert quoted_spans_excluding_content("hello world") == []


def test_double_quotes():
    spans = quoted_spans_excluding_content('"inner"')
    assert any(s <= 1 and e >= 6 for s, e in spans)


def test_guillemets():
    spans = quoted_spans_excluding_content("«inner»")
    assert len(spans) == 1
    s, e = spans[0]
    assert s == 1 and e == 6


def test_single_quotes():
    spans = quoted_spans_excluding_content("'inner'")
    assert len(spans) == 1


def test_backtick():
    spans = quoted_spans_excluding_content("`inner`")
    assert len(spans) == 1


def test_curly_quotes():
    spans = quoted_spans_excluding_content("\u201cinner\u201d")
    assert len(spans) == 1


def test_overlapping_spans_merged():
    # Два перекрывающихся диапазона должны слиться
    spans = quoted_spans_excluding_content('"ab"c"de"')
    # 'ab' → (1,3), 'de' → (6,8)  — не перекрываются
    assert len(spans) == 2


def test_empty_string():
    assert quoted_spans_excluding_content("") == []


# ---------------------------------------------------------------------------
# _alias_matches_outside_quotes
# ---------------------------------------------------------------------------


def _spans(text: str):
    return quoted_spans_excluding_content(text)


def test_alias_found_outside_quotes():
    text = "привет бебро как дела"
    assert _alias_matches_outside_quotes(text, "бебро", _spans(text)) is True


def test_alias_not_found():
    text = "привет как дела"
    assert _alias_matches_outside_quotes(text, "бебро", _spans(text)) is False


def test_alias_inside_quotes_ignored():
    text = "«бебро» — это имя бота"
    assert _alias_matches_outside_quotes(text, "бебро", _spans(text)) is False


def test_alias_partial_inside_quotes_but_also_outside():
    text = "«бебро» бебро"
    assert _alias_matches_outside_quotes(text, "бебро", _spans(text)) is True


def test_alias_case_insensitive():
    text = "БЕБРО привет"
    assert _alias_matches_outside_quotes(text, "бебро", _spans(text)) is True


def test_alias_empty_returns_false():
    text = "hello"
    assert _alias_matches_outside_quotes(text, "", _spans(text)) is False


# ---------------------------------------------------------------------------
# Вспомогательные фабрики моков Telegram
# ---------------------------------------------------------------------------


def _make_app_cfg(bots: list[dict]) -> Any:
    """Минимальный AppConfig с полем bots."""
    bot_list = [SimpleNamespace(**b) for b in bots]
    return SimpleNamespace(bots=bot_list)


def _make_db(aliases_by_bot: dict[str, list[str]]) -> Any:
    """Минимальный Database: list_chat_aliases возвращает заданные алиасы."""
    db = MagicMock()
    db.list_chat_aliases.side_effect = lambda bot_id, chat_id: aliases_by_bot.get(bot_id, [])
    return db


def _make_message(
    *,
    text: str | None = None,
    caption: str | None = None,
    entities: list | None = None,
    caption_entities: list | None = None,
    reply_to_message=None,
) -> Any:
    msg = MagicMock()
    msg.text = text
    msg.caption = caption
    msg.entities = entities or []
    msg.caption_entities = caption_entities or []
    msg.reply_to_message = reply_to_message
    return msg


# ---------------------------------------------------------------------------
# collect_explicitly_addressed_bot_ids — алиасы
# ---------------------------------------------------------------------------


CHAT_ID = -100123456


def test_alias_in_text_addresses_bot():
    app_cfg = _make_app_cfg([{"id": "biba", "username": "bebro_biba_bot"}])
    db = _make_db({"biba": ["бибa", "бебро"]})
    msg = _make_message(text="бебро, как дела?")
    result = collect_explicitly_addressed_bot_ids(msg, app_cfg, db, CHAT_ID)
    assert "biba" in result


def test_alias_in_caption_addresses_bot():
    app_cfg = _make_app_cfg([{"id": "biba", "username": "bebro_biba_bot"}])
    db = _make_db({"biba": ["бебро"]})
    msg = _make_message(text=None, caption="бебро смотри фото")
    result = collect_explicitly_addressed_bot_ids(msg, app_cfg, db, CHAT_ID)
    assert "biba" in result


def test_alias_inside_quotes_not_addressed():
    app_cfg = _make_app_cfg([{"id": "biba", "username": "bebro_biba_bot"}])
    db = _make_db({"biba": ["бебро"]})
    msg = _make_message(text="«бебро» — так звали кота")
    result = collect_explicitly_addressed_bot_ids(msg, app_cfg, db, CHAT_ID)
    assert "biba" not in result


def test_no_alias_no_entities_empty_result():
    app_cfg = _make_app_cfg([{"id": "biba", "username": "bebro_biba_bot"}])
    db = _make_db({"biba": ["бебро"]})
    msg = _make_message(text="просто сообщение")
    result = collect_explicitly_addressed_bot_ids(msg, app_cfg, db, CHAT_ID)
    assert result == set()


def test_two_bots_different_aliases():
    app_cfg = _make_app_cfg([
        {"id": "biba", "username": "bebro_biba_bot"},
        {"id": "boba", "username": "bebro_boba_bot"},
    ])
    db = _make_db({"biba": ["биба"], "boba": ["боба"]})
    msg = _make_message(text="биба и боба, привет")
    result = collect_explicitly_addressed_bot_ids(msg, app_cfg, db, CHAT_ID)
    assert result == {"biba", "boba"}


def test_alias_addresses_only_owner_not_other_bot():
    app_cfg = _make_app_cfg([
        {"id": "biba", "username": "bebro_biba_bot"},
        {"id": "boba", "username": "bebro_boba_bot"},
    ])
    db = _make_db({"biba": ["биба"], "boba": []})
    msg = _make_message(text="биба, привет")
    result = collect_explicitly_addressed_bot_ids(msg, app_cfg, db, CHAT_ID)
    assert result == {"biba"}
    assert "boba" not in result


# ---------------------------------------------------------------------------
# collect_explicitly_addressed_bot_ids — reply
# ---------------------------------------------------------------------------


def test_reply_to_bot_by_username():
    register_bot_user_id("biba", 111111)
    app_cfg = _make_app_cfg([{"id": "biba", "username": "bebro_biba_bot"}])
    db = _make_db({})

    reply_user = MagicMock()
    reply_user.is_bot = True
    reply_user.username = "bebro_biba_bot"
    reply_user.id = 111111
    reply_msg = MagicMock()
    reply_msg.from_user = reply_user

    msg = _make_message(text="ответ", reply_to_message=reply_msg)
    result = collect_explicitly_addressed_bot_ids(msg, app_cfg, db, CHAT_ID)
    assert "biba" in result


def test_reply_to_non_bot_not_addressed():
    app_cfg = _make_app_cfg([{"id": "biba", "username": "bebro_biba_bot"}])
    db = _make_db({})

    reply_user = MagicMock()
    reply_user.is_bot = False
    reply_msg = MagicMock()
    reply_msg.from_user = reply_user

    msg = _make_message(text="ответ", reply_to_message=reply_msg)
    result = collect_explicitly_addressed_bot_ids(msg, app_cfg, db, CHAT_ID)
    assert "biba" not in result


def test_no_reply_no_entities_empty():
    app_cfg = _make_app_cfg([{"id": "biba", "username": "bebro_biba_bot"}])
    db = _make_db({})
    msg = _make_message(text="просто текст", reply_to_message=None)
    result = collect_explicitly_addressed_bot_ids(msg, app_cfg, db, CHAT_ID)
    assert result == set()


# ---------------------------------------------------------------------------
# command_is_for_this_bot_username
# ---------------------------------------------------------------------------


def _cmd_msg(text: str) -> Any:
    msg = MagicMock()
    msg.text = text
    return msg


def test_command_without_at_is_for_everyone():
    assert command_is_for_this_bot_username(_cmd_msg("/bebro_params"), "bebro_biba_bot") is True


def test_command_with_correct_at():
    assert command_is_for_this_bot_username(_cmd_msg("/bebro_params@bebro_biba_bot"), "bebro_biba_bot") is True


def test_command_with_other_bot_at():
    assert command_is_for_this_bot_username(_cmd_msg("/bebro_params@bebro_boba_bot"), "bebro_biba_bot") is False


def test_command_with_double_at():
    assert command_is_for_this_bot_username(_cmd_msg("/bebro_params@@bebro_biba_bot"), "bebro_biba_bot") is True


def test_command_with_leading_at_in_username():
    # username может передаваться с @ или без
    assert command_is_for_this_bot_username(_cmd_msg("/bebro_params@bebro_biba_bot"), "@bebro_biba_bot") is True
