from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bebro_bots.chat_aliases_policy import purge_username_like_aliases, reject_reason_for_chat_alias
from bebro_bots.tone_profile import DEFAULT_TONE_SETTINGS

logger = logging.getLogger(__name__)

DEFAULT_CHAT_SETTINGS: dict[str, Any] = {
    "window_messages": 40,
    "recent_for_llm": 16,
    "reaction_probability": 0.12,
    "initiative_probability": 0.05,
    "initiative_interval_minutes": 90,
    "proactive_reaction_probability": 0.04,
    "verbosity": "medium",  # short | medium | long
    "frequency": "medium",  # rare | medium | often
    "short_reply_bias": 0.0,
    "response_depth": "normal",  # shallow | normal | deep
    **DEFAULT_TONE_SETTINGS,
}

DEFAULT_BOT_PERSONA = (
    "Ты участник группового чата. Стиль, тон и подробность ответов настраиваются участниками группы."
)


@dataclass
class ChatMessageRow:
    message_id: int
    user_id: int | None
    username: str | None
    role: str
    content: str | None
    content_type: str
    sticker_file_id: str | None
    photo_file_id: str | None
    created_at: float


class Database:
    def __init__(self, sqlite_path: str) -> None:
        self.path = Path(sqlite_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_settings (
                  bot_id TEXT NOT NULL,
                  chat_id INTEGER NOT NULL,
                  settings_json TEXT NOT NULL,
                  PRIMARY KEY (bot_id, chat_id)
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  bot_id TEXT NOT NULL,
                  chat_id INTEGER NOT NULL,
                  message_id INTEGER NOT NULL,
                  user_id INTEGER,
                  username TEXT,
                  role TEXT NOT NULL,
                  content TEXT,
                  content_type TEXT DEFAULT 'text',
                  sticker_file_id TEXT,
                  photo_file_id TEXT,
                  created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cm_bot_chat
                  ON chat_messages(bot_id, chat_id, id);

                CREATE TABLE IF NOT EXISTS conversation_summaries (
                  bot_id TEXT NOT NULL,
                  chat_id INTEGER NOT NULL,
                  summary TEXT NOT NULL,
                  updated_at REAL NOT NULL,
                  PRIMARY KEY (bot_id, chat_id)
                );

                CREATE TABLE IF NOT EXISTS long_memory (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  bot_id TEXT NOT NULL,
                  chat_id INTEGER NOT NULL,
                  content TEXT NOT NULL,
                  created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS stickers (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  bot_id TEXT NOT NULL,
                  chat_id INTEGER NOT NULL,
                  file_id TEXT NOT NULL,
                  set_name TEXT,
                  source TEXT DEFAULT 'group',
                  last_seen REAL NOT NULL,
                  UNIQUE (bot_id, chat_id, file_id)
                );

                CREATE TABLE IF NOT EXISTS bot_state (
                  bot_id TEXT PRIMARY KEY,
                  mood TEXT,
                  updated_at REAL
                );

                CREATE TABLE IF NOT EXISTS chat_last_initiative (
                  bot_id TEXT NOT NULL,
                  chat_id INTEGER NOT NULL,
                  last_ts REAL NOT NULL,
                  PRIMARY KEY (bot_id, chat_id)
                );

                CREATE TABLE IF NOT EXISTS situation_rules (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  bot_id TEXT NOT NULL,
                  chat_id INTEGER NOT NULL,
                  trigger_text TEXT NOT NULL,
                  instruction_text TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sr_bot_chat
                  ON situation_rules(bot_id, chat_id, updated_at DESC);
                """
            )
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(bot_state)").fetchall()}
        if "persona_text" not in cols:
            conn.execute("ALTER TABLE bot_state ADD COLUMN persona_text TEXT")
        if "tone_drift_json" not in cols:
            conn.execute("ALTER TABLE bot_state ADD COLUMN tone_drift_json TEXT")
        cm = {r[1] for r in conn.execute("PRAGMA table_info(chat_messages)").fetchall()}
        if "photo_file_id" not in cm:
            conn.execute("ALTER TABLE chat_messages ADD COLUMN photo_file_id TEXT")
        self._migrate_chat_aliases_bot_id(conn)
        n = purge_username_like_aliases(conn)
        if n:
            logger.info(
                "Миграция chat_aliases: удалено строк с @username/похожих на логин: %s",
                n,
            )

    def _migrate_chat_aliases_bot_id(self, conn: sqlite3.Connection) -> None:
        """Создание chat_aliases при первом запуске; миграция старого формата без bot_id."""
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chat_aliases'"
        ).fetchone()
        if not exists:
            conn.execute(
                """
                CREATE TABLE chat_aliases (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  bot_id TEXT NOT NULL,
                  chat_id INTEGER NOT NULL,
                  alias_text TEXT NOT NULL,
                  created_at REAL NOT NULL,
                  UNIQUE (bot_id, chat_id, alias_text)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_aliases_bot_chat ON chat_aliases(bot_id, chat_id)"
            )
            return
        ca_cols = {r[1] for r in conn.execute("PRAGMA table_info(chat_aliases)").fetchall()}
        if "bot_id" in ca_cols:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_aliases_bot_chat ON chat_aliases(bot_id, chat_id)"
            )
            return
        logger.info("Миграция chat_aliases: добавление bot_id (раздельные алиасы на бота)")
        conn.execute("ALTER TABLE chat_aliases RENAME TO chat_aliases_old")
        conn.execute(
            """
            CREATE TABLE chat_aliases (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              bot_id TEXT NOT NULL,
              chat_id INTEGER NOT NULL,
              alias_text TEXT NOT NULL,
              created_at REAL NOT NULL,
              UNIQUE (bot_id, chat_id, alias_text)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_aliases_bot_chat ON chat_aliases(bot_id, chat_id)"
        )
        old_rows = conn.execute("SELECT chat_id, alias_text, created_at FROM chat_aliases_old").fetchall()
        for r in old_rows:
            cid = int(r["chat_id"])
            bots = conn.execute(
                "SELECT DISTINCT bot_id FROM chat_messages WHERE chat_id=?",
                (cid,),
            ).fetchall()
            for b in bots:
                bid = str(b["bot_id"])
                conn.execute(
                    """
                    INSERT OR IGNORE INTO chat_aliases (bot_id, chat_id, alias_text, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (bid, cid, str(r["alias_text"]), float(r["created_at"])),
                )
        conn.execute("DROP TABLE chat_aliases_old")

    @staticmethod
    def normalize_alias_text(raw: str) -> str:
        return raw.strip().lower()

    def list_chat_aliases(self, bot_id: str, chat_id: int) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT alias_text FROM chat_aliases
                WHERE bot_id=? AND chat_id=?
                ORDER BY id ASC
                """,
                (bot_id, chat_id),
            ).fetchall()
        return [str(r["alias_text"]) for r in rows]

    def add_chat_alias(self, bot_id: str, chat_id: int, raw: str) -> bool:
        if reject_reason_for_chat_alias(raw) is not None:
            return False
        key = self.normalize_alias_text(raw)
        if not key:
            return False
        now = time.time()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO chat_aliases (bot_id, chat_id, alias_text, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (bot_id, chat_id, key, now),
            )
            return cur.rowcount > 0

    def remove_chat_alias(self, bot_id: str, chat_id: int, raw: str) -> bool:
        key = self.normalize_alias_text(raw)
        if not key:
            return False
        with self.connect() as conn:
            cur = conn.execute(
                "DELETE FROM chat_aliases WHERE bot_id=? AND chat_id=? AND alias_text=?",
                (bot_id, chat_id, key),
            )
            return cur.rowcount > 0

    def get_settings(self, bot_id: str, chat_id: int) -> dict[str, Any]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT settings_json FROM chat_settings WHERE bot_id=? AND chat_id=?",
                (bot_id, chat_id),
            ).fetchone()
        base = dict(DEFAULT_CHAT_SETTINGS)
        if row:
            data = json.loads(row["settings_json"])
            base.update(data)
        return base

    def merge_settings(self, bot_id: str, chat_id: int, patch: dict[str, Any]) -> dict[str, Any]:
        cur = self.get_settings(bot_id, chat_id)
        cur.update(patch)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_settings (bot_id, chat_id, settings_json)
                VALUES (?, ?, ?)
                ON CONFLICT(bot_id, chat_id) DO UPDATE SET settings_json=excluded.settings_json
                """,
                (bot_id, chat_id, json.dumps(cur, ensure_ascii=False)),
            )
        return cur

    def append_message(
        self,
        bot_id: str,
        chat_id: int,
        *,
        message_id: int,
        user_id: int | None,
        username: str | None,
        role: str,
        content: str | None,
        content_type: str = "text",
        sticker_file_id: str | None = None,
        photo_file_id: str | None = None,
    ) -> None:
        now = time.time()
        settings = self.get_settings(bot_id, chat_id)
        window = int(settings.get("window_messages", 40))
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages
                (bot_id, chat_id, message_id, user_id, username, role, content, content_type, sticker_file_id, photo_file_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bot_id,
                    chat_id,
                    message_id,
                    user_id,
                    username,
                    role,
                    content,
                    content_type,
                    sticker_file_id,
                    photo_file_id,
                    now,
                ),
            )
            self._trim_messages(conn, bot_id, chat_id, window)

    def _trim_messages(self, conn: sqlite3.Connection, bot_id: str, chat_id: int, keep: int) -> None:
        conn.execute(
            """
            DELETE FROM chat_messages
            WHERE bot_id=? AND chat_id=?
              AND id NOT IN (
                SELECT id FROM chat_messages
                WHERE bot_id=? AND chat_id=?
                ORDER BY id DESC
                LIMIT ?
              )
            """,
            (bot_id, chat_id, bot_id, chat_id, keep),
        )

    def recent_messages(self, bot_id: str, chat_id: int, limit: int) -> list[ChatMessageRow]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT message_id, user_id, username, role, content, content_type, sticker_file_id, photo_file_id, created_at
                FROM chat_messages
                WHERE bot_id=? AND chat_id=?
                ORDER BY id DESC
                LIMIT ?
                """,
                (bot_id, chat_id, limit),
            ).fetchall()
        out: list[ChatMessageRow] = []
        for r in reversed(rows):
            out.append(
                ChatMessageRow(
                    message_id=int(r["message_id"]),
                    user_id=r["user_id"],
                    username=r["username"],
                    role=str(r["role"]),
                    content=r["content"],
                    content_type=str(r["content_type"] or "text"),
                    sticker_file_id=r["sticker_file_id"],
                    photo_file_id=r["photo_file_id"],
                    created_at=float(r["created_at"]),
                )
            )
        return out

    def get_summary(self, bot_id: str, chat_id: int) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT summary FROM conversation_summaries WHERE bot_id=? AND chat_id=?",
                (bot_id, chat_id),
            ).fetchone()
        return str(row["summary"]) if row else None

    def set_summary(self, bot_id: str, chat_id: int, summary: str) -> None:
        now = time.time()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_summaries (bot_id, chat_id, summary, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(bot_id, chat_id) DO UPDATE SET summary=excluded.summary, updated_at=excluded.updated_at
                """,
                (bot_id, chat_id, summary, now),
            )

    def add_long_memory(
        self,
        bot_id: str,
        chat_id: int,
        content: str,
        *,
        retention_days: float | None = None,
    ) -> None:
        now = time.time()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO long_memory (bot_id, chat_id, content, created_at) VALUES (?, ?, ?, ?)",
                (bot_id, chat_id, content, now),
            )
            if retention_days is not None:
                cutoff = now - float(retention_days) * 86400.0
                conn.execute(
                    "DELETE FROM long_memory WHERE bot_id=? AND chat_id=? AND created_at < ?",
                    (bot_id, chat_id, cutoff),
                )

    def list_long_memory(
        self,
        bot_id: str,
        chat_id: int,
        limit: int = 50,
        *,
        min_created_at: float | None = None,
    ) -> list[str]:
        with self.connect() as conn:
            if min_created_at is None:
                rows = conn.execute(
                    """
                    SELECT content FROM long_memory
                    WHERE bot_id=? AND chat_id=?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (bot_id, chat_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT content FROM long_memory
                    WHERE bot_id=? AND chat_id=? AND created_at >= ?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (bot_id, chat_id, min_created_at, limit),
                ).fetchall()
        return [str(r["content"]) for r in rows]

    def remember_sticker(
        self,
        bot_id: str,
        chat_id: int,
        file_id: str,
        set_name: str | None,
        source: str = "group",
    ) -> None:
        now = time.time()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO stickers (bot_id, chat_id, file_id, set_name, source, last_seen)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (bot_id, chat_id, file_id) DO UPDATE SET
                  set_name=COALESCE(excluded.set_name, stickers.set_name),
                  last_seen=excluded.last_seen
                """,
                (bot_id, chat_id, file_id, set_name, source, now),
            )

    def random_sticker_file_id(self, bot_id: str, chat_id: int) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT file_id FROM stickers
                WHERE bot_id=? AND (chat_id=? OR chat_id=0)
                ORDER BY RANDOM() LIMIT 1
                """,
                (bot_id, chat_id),
            ).fetchone()
        return str(row["file_id"]) if row else None

    def get_mood(self, bot_id: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT mood FROM bot_state WHERE bot_id=?",
                (bot_id,),
            ).fetchone()
        return str(row["mood"]) if row and row["mood"] else None

    def get_bot_persona(self, bot_id: str) -> str:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT persona_text FROM bot_state WHERE bot_id=?",
                (bot_id,),
            ).fetchone()
        if row and row["persona_text"]:
            return str(row["persona_text"])
        return DEFAULT_BOT_PERSONA

    def set_bot_persona(self, bot_id: str, text: str) -> None:
        now = time.time()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO bot_state (bot_id, persona_text, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(bot_id) DO UPDATE SET
                  persona_text=excluded.persona_text,
                  updated_at=excluded.updated_at
                """,
                (bot_id, text.strip(), now),
            )

    def get_tone_drift(self, bot_id: str) -> dict[str, int]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT tone_drift_json FROM bot_state WHERE bot_id=?",
                (bot_id,),
            ).fetchone()
        if not row or not row["tone_drift_json"]:
            return {}
        try:
            raw = json.loads(row["tone_drift_json"])
        except json.JSONDecodeError:
            return {}
        out: dict[str, int] = {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    iv = int(v)
                    if iv in (-1, 0, 1):
                        out[str(k)] = iv
                except (TypeError, ValueError):
                    continue
        return out

    def set_tone_drift(self, bot_id: str, drift: dict[str, int]) -> None:
        now = time.time()
        payload = json.dumps(drift, ensure_ascii=False)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO bot_state (bot_id, tone_drift_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(bot_id) DO UPDATE SET
                  tone_drift_json=excluded.tone_drift_json,
                  updated_at=excluded.updated_at
                """,
                (bot_id, payload, now),
            )

    def set_mood(self, bot_id: str, mood: str) -> None:
        now = time.time()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO bot_state (bot_id, mood, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(bot_id) DO UPDATE SET mood=excluded.mood, updated_at=excluded.updated_at
                """,
                (bot_id, mood, now),
            )

    def last_initiative_ts(self, bot_id: str, chat_id: int) -> float | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT last_ts FROM chat_last_initiative WHERE bot_id=? AND chat_id=?",
                (bot_id, chat_id),
            ).fetchone()
        return float(row["last_ts"]) if row else None

    def set_last_initiative_ts(self, bot_id: str, chat_id: int, ts: float) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_last_initiative (bot_id, chat_id, last_ts)
                VALUES (?, ?, ?)
                ON CONFLICT(bot_id, chat_id) DO UPDATE SET last_ts=excluded.last_ts
                """,
                (bot_id, chat_id, ts),
            )

    def distinct_chat_ids(self, bot_id: str) -> list[int]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT chat_id FROM chat_messages WHERE bot_id=?",
                (bot_id,),
            ).fetchall()
        return [int(r["chat_id"]) for r in rows]

    def list_situation_rules(
        self,
        bot_id: str,
        chat_id: int,
        limit: int,
        *,
        min_created_at: float | None = None,
    ) -> list[tuple[str, str]]:
        with self.connect() as conn:
            if min_created_at is None:
                rows = conn.execute(
                    """
                    SELECT trigger_text, instruction_text FROM situation_rules
                    WHERE bot_id=? AND chat_id=?
                    ORDER BY updated_at DESC, id DESC
                    LIMIT ?
                    """,
                    (bot_id, chat_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT trigger_text, instruction_text FROM situation_rules
                    WHERE bot_id=? AND chat_id=? AND created_at >= ?
                    ORDER BY updated_at DESC, id DESC
                    LIMIT ?
                    """,
                    (bot_id, chat_id, min_created_at, limit),
                ).fetchall()
        return [(str(r["trigger_text"]), str(r["instruction_text"])) for r in rows]

    def add_situation_rule(
        self,
        bot_id: str,
        chat_id: int,
        *,
        trigger_text: str,
        instruction_text: str,
        max_rows: int = 50,
        retention_days: float | None = None,
    ) -> None:
        now = time.time()
        trig = trigger_text.strip()
        ins = instruction_text.strip()
        if not trig or not ins:
            return
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO situation_rules
                (bot_id, chat_id, trigger_text, instruction_text, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (bot_id, chat_id, trig, ins, now, now),
            )
            if retention_days is not None:
                cutoff = now - float(retention_days) * 86400.0
                conn.execute(
                    "DELETE FROM situation_rules WHERE bot_id=? AND chat_id=? AND created_at < ?",
                    (bot_id, chat_id, cutoff),
                )
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM situation_rules WHERE bot_id=? AND chat_id=?",
                (bot_id, chat_id),
            ).fetchone()
            cnt = int(row["c"]) if row else 0
            if cnt > max_rows:
                excess = cnt - max_rows
                conn.execute(
                    """
                    DELETE FROM situation_rules WHERE id IN (
                      SELECT id FROM situation_rules
                      WHERE bot_id=? AND chat_id=?
                      ORDER BY updated_at ASC, id ASC
                      LIMIT ?
                    )
                    """,
                    (bot_id, chat_id, excess),
                )

    def clear_chat_memory(self, bot_id: str, chat_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM chat_messages WHERE bot_id=? AND chat_id=?",
                (bot_id, chat_id),
            )
            conn.execute(
                "DELETE FROM conversation_summaries WHERE bot_id=? AND chat_id=?",
                (bot_id, chat_id),
            )
            conn.execute(
                "DELETE FROM long_memory WHERE bot_id=? AND chat_id=?",
                (bot_id, chat_id),
            )
            conn.execute(
                "DELETE FROM situation_rules WHERE bot_id=? AND chat_id=?",
                (bot_id, chat_id),
            )
            conn.execute(
                "DELETE FROM chat_aliases WHERE bot_id=? AND chat_id=?",
                (bot_id, chat_id),
            )
