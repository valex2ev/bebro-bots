from __future__ import annotations

import argparse
import logging
import threading
import sys

from telegram import Update

from bebro_bots.app_factory import build_application
from bebro_bots.config import create_llm_client, load_app_config
from bebro_bots.db import Database


class _ContextDefaults(logging.Filter):
    """Добавляет bot_id/chat_id='-' записям от логгеров без LoggerAdapter.

    Filter срабатывает после makeRecord, поэтому не конфликтует с extra
    из LoggerAdapter (в отличие от setLogRecordFactory в Python 3.12+).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "bot_id"):
            record.bot_id = "-"
        if not hasattr(record, "chat_id"):
            record.chat_id = "-"
        return True


def _run_polling() -> None:
    cfg = load_app_config()
    db = Database(cfg.sqlite_path)
    if not cfg.bots:
        raise SystemExit("В config.yaml нет ботов (ключ bots).")

    threads: list[threading.Thread] = []
    for persona in cfg.bots:
        client = create_llm_client(cfg, persona)
        app = build_application(persona=persona, app_cfg=cfg, db=db, client=client)

        def runner(application=app) -> None:
            application.run_polling(allowed_updates=Update.ALL_TYPES)

        t = threading.Thread(target=runner, name=f"bot-{persona.id}", daemon=True)
        t.start()
        threads.append(t)
        logging.info("Запущен long polling для бота %s", persona.id)

    for t in threads:
        t.join()


def main() -> None:
    parser = argparse.ArgumentParser(description="bebro-bots")
    parser.add_argument(
        "mode",
        nargs="?",
        default="polling",
        choices=("polling", "webhook"),
        help="polling (по умолчанию) или webhook",
    )
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        default=None,
        help="Путь к лог-файлу; если не задан — запись в файл отключена.",
    )
    args = parser.parse_args()

    fmt = "%(asctime)s %(levelname)s %(name)s [%(bot_id)s/%(chat_id)s]: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(format=fmt, level=logging.INFO, handlers=handlers)
    logging.getLogger().addFilter(_ContextDefaults())

    if args.mode == "polling":
        _run_polling()
        return

    from bebro_bots.webhook import run_webhook_server

    run_webhook_server()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
