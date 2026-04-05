from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from telegram import Update

from bebro_bots.app_factory import build_application
from bebro_bots.config import create_openai_client, load_app_config
from bebro_bots.db import Database

logger = logging.getLogger(__name__)

_apps: dict[str, Any] = {}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    cfg = load_app_config()
    db = Database(cfg.sqlite_path)
    client = create_openai_client(cfg)
    if not cfg.bots:
        raise RuntimeError("Нет ботов в конфиге")
    for persona in cfg.bots:
        tg = build_application(persona=persona, app_cfg=cfg, db=db, client=client)
        await tg.initialize()
        await tg.start()
        _apps[persona.id] = tg
        logger.info("Telegram app %s initialized for webhook", persona.id)
    yield
    for pid, tg in list(_apps.items()):
        try:
            await tg.stop()
            await tg.shutdown()
        except Exception:
            logger.exception("shutdown %s", pid)
        _apps.pop(pid, None)


def build_fastapi_app() -> FastAPI:
    web = FastAPI(lifespan=_lifespan)

    @web.post("/hook/{bot_key}")
    async def telegram_webhook(bot_key: str, request: Request) -> dict[str, bool]:
        tg = _apps.get(bot_key)
        if not tg:
            raise HTTPException(status_code=404, detail="unknown bot")
        body = await request.json()
        bot = tg.bot
        update = Update.de_json(body, bot)
        await tg.process_update(update)
        return {"ok": True}

    @web.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return web


def run_webhook_server() -> None:
    import uvicorn

    cfg = load_app_config()
    host = cfg.server.webhook_host
    port = int(cfg.server.webhook_port)
    app = build_fastapi_app()
    uvicorn.run(app, host=host, port=port, log_level="info")
