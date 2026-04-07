from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


class OpenAIConfig(BaseModel):
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4o-mini"
    base_url: str | None = None


class ServerConfig(BaseModel):
    webhook_host: str = "0.0.0.0"
    webhook_port: int = 8080
    public_base_url: str | None = None


class BotPersonaConfig(BaseModel):
    id: str
    token_env: str
    username: str
    mood_default: str = "нейтральное"
    sticker_set_names: list[str] = Field(default_factory=list)


class VisionConfig(BaseModel):
    """Передача изображений из Telegram в multimodal-модель OpenAI."""

    enabled: bool = True
    max_image_bytes: int = Field(default=20 * 1024 * 1024, ge=1024 * 100)


class VpnWatchConfig(BaseModel):
    """Контроль VPN: проверка Telegram + OpenAI, перезапуск службы Windows (только при enabled)."""

    enabled: bool = False
    check_interval_seconds: int = Field(default=120, ge=30, le=86400)
    reconnect_max_attempts: int = Field(default=5, ge=1, le=50)
    post_restart_wait_seconds: float = Field(default=15.0, ge=0.0, le=600.0)
    windows_service_name: str = Field(
        default="Adguard VPN Service",
        description="Имя службы Windows (Restart-Service).",
    )
    service_exe_path: str | None = Field(
        default=None,
        description="Путь к exe службы (справочно; перезапуск только по имени службы).",
    )
    notify_bot_id: str | None = Field(
        default=None,
        description="id бота из bots, который шлёт уведомления о VPN во все чаты из chat_messages.",
    )

    @model_validator(mode="after")
    def _notify_when_enabled(self) -> VpnWatchConfig:
        if self.enabled and not self.notify_bot_id:
            raise ValueError(
                "vpn_watch: при enabled=true задайте notify_bot_id (строка id бота из bots)"
            )
        return self


class AppConfig(BaseModel):
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    allowed_chat_ids: list[int] = Field(default_factory=list)
    admin_user_ids: list[int] = Field(default_factory=list)
    bots: list[BotPersonaConfig] = Field(default_factory=list)
    sqlite_path: str = "data/bebro.sqlite"
    """1 — минимум контекста из предпочтений; 5 — максимум. По умолчанию 4."""
    feedback_sensitivity: int = Field(default=4, ge=1, le=5)
    vpn_watch: VpnWatchConfig = Field(default_factory=VpnWatchConfig)

    @model_validator(mode="after")
    def _vpn_notify_bot_in_list(self) -> AppConfig:
        if not self.vpn_watch.enabled:
            return self
        ids = {b.id for b in self.bots}
        nid = self.vpn_watch.notify_bot_id
        if nid and nid not in ids:
            raise ValueError(
                f"vpn_watch.notify_bot_id={nid!r} отсутствует в списке bots[].id"
            )
        return self


def _resolve_config_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    for name in ("config.yaml", "config.yml"):
        p = root / name
        if p.is_file():
            return p
    ex = root / "config.example.yaml"
    if ex.is_file():
        return ex
    raise FileNotFoundError(
        "Не найден config.yaml. Скопируйте config.example.yaml в config.yaml."
    )


def load_app_config() -> AppConfig:
    load_dotenv()
    path = _resolve_config_path()
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = AppConfig.model_validate(raw)
    extra = os.environ.get("ADMIN_USER_IDS", "").strip()
    if extra:
        ids: list[int] = []
        for part in extra.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                ids.append(int(part))
            except ValueError:
                continue
        cfg.admin_user_ids = list(dict.fromkeys([*cfg.admin_user_ids, *ids]))
    return cfg


def create_openai_client(cfg: AppConfig):
    from openai import AsyncOpenAI

    kwargs: dict[str, Any] = {"api_key": openai_api_key(cfg)}
    if cfg.openai.base_url:
        kwargs["base_url"] = cfg.openai.base_url
    return AsyncOpenAI(**kwargs)


def env_token(token_env: str) -> str:
    v = os.environ.get(token_env, "").strip()
    if not v:
        raise RuntimeError(f"Переменная окружения {token_env} не задана или пуста")
    return v


def openai_api_key(cfg: AppConfig) -> str:
    return env_token(cfg.openai.api_key_env)
