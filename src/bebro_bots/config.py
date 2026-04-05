from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


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
