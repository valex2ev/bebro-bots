from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OpenAIConfig(BaseModel):
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    daily_input_tokens: int = Field(default=650_000, ge=0, description="Prompt-токены OpenAI за сутки (UTC).")


class AnthropicConfig(BaseModel):
    api_key_env: str = "ANTHROPIC_API_KEY"
    model: str = "claude-opus-4-6"
    daily_input_tokens: int = Field(default=300_000, ge=0, description="Input-токены Anthropic за сутки (UTC).")
    max_output_tokens: int = Field(default=2048, ge=64, description="Максимум токенов в ответе модели (Anthropic требует явного max_tokens).")


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


class LimitsConfig(BaseModel):
    """Глобальные суточные лимиты (все боты и чаты, один процесс). День — календарный UTC."""

    model_config = ConfigDict(extra="ignore")

    daily_web_searches: int = Field(default=30, ge=0, description="Успешные вызовы Tavily за сутки (UTC).")
    use_utc_calendar_day: bool = Field(
        default=True,
        description="Считать сутки по UTC (зафиксировано; локальный пояс не используется).",
    )


class ConversationSummaryPolicy(BaseModel):
    """Параметры периодического обновления саммари."""

    min_messages_in_window: int = Field(
        default=15,
        ge=1,
        description="Минимум сообщений в окне, чтобы запускать случайное саммари.",
    )
    random_summarize_probability: float = Field(
        default=0.06,
        ge=0.0,
        le=1.0,
        description="Вероятность вызова саммари после ответа бота при достаточном окне.",
    )


class VisionConfig(BaseModel):
    """Передача изображений из Telegram в multimodal-модель OpenAI."""

    enabled: bool = True
    max_image_bytes: int = Field(default=20 * 1024 * 1024, ge=1024 * 100)
    max_images_per_turn: int = Field(
        default=6,
        ge=1,
        le=32,
        description="Максимум изображений в одном запросе к модели (текущее сообщение, reply на фото, фото из окна истории).",
    )


class LongMemoryPolicy(BaseModel):
    """Таблица `long_memory`: инструмент `add_long_instruction`."""

    retention_days: float | None = Field(
        default=None,
        ge=0.0,
        description="Удалять записи старше N дней; null — не ограничивать по возрасту.",
    )
    context_limit: int = Field(
        default=12,
        ge=1,
        le=500,
        description="Сколько последних записей учитывать в промпте и в get_group_state.",
    )


class WebSearchPolicy(BaseModel):
    """Поиск в интернете (Tavily), инструмент `web_search`."""

    mode: Literal["on_trigger", "always"] = Field(
        default="on_trigger",
        description=(
            "on_trigger — инструмент только если в сообщении есть подстрока «найди» (см. triggers). "
            "always — инструмент доступен в каждом ответе на сообщение группы (не для инициативы/проактивных задач)."
        ),
    )


class SituationRulesPolicy(BaseModel):
    """Таблица `situation_rules`: инструмент `add_situation_rule`."""

    retention_days: float | None = Field(
        default=None,
        ge=0.0,
        description="Удалять правила старше N дней; null — не ограничивать по возрасту.",
    )
    context_limit: int = Field(
        default=6,
        ge=0,
        le=200,
        description="Сколько последних правил подмешивать в промпт и get_group_state.",
    )
    max_stored: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Максимум строк на чат; при превышении удаляются самые старые.",
    )


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
    model_config = ConfigDict(extra="ignore")

    provider: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="Активный LLM-провайдер: openai или anthropic.",
    )
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    """Пауза перед ответом на обычное сообщение группы (мс); снижает одновременные ответы нескольких ботов."""
    response_jitter_ms_min: int = Field(default=0, ge=0)
    response_jitter_ms_max: int = Field(default=8000, ge=0)
    """Не чаще одного уведомления об исчерпании лимита на чат и тип (минуты)."""
    limit_notify_cooldown_minutes: float = Field(default=45.0, ge=0.5, le=24 * 60.0)
    conversation_summary: ConversationSummaryPolicy = Field(default_factory=ConversationSummaryPolicy)
    allowed_chat_ids: list[int] = Field(default_factory=list)
    admin_user_ids: list[int | str] = Field(
        default_factory=list,
        description="Целые Telegram user id и/или username без @ (строка из YAML в кавычках, напр. \"@valex2ev\").",
    )
    bots: list[BotPersonaConfig] = Field(default_factory=list)
    sqlite_path: str = "data/bebro.sqlite"
    """1 — минимум контекста из предпочтений; 5 — максимум. По умолчанию 4."""
    feedback_sensitivity: int = Field(default=4, ge=1, le=5)
    reply_probability: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Вероятность вызова LLM без @ и без текстового алиаса; только из YAML, не из БД.",
    )
    long_memory: LongMemoryPolicy = Field(default_factory=LongMemoryPolicy)
    situation_rules: SituationRulesPolicy = Field(default_factory=SituationRulesPolicy)
    web_search: WebSearchPolicy = Field(default_factory=WebSearchPolicy)
    vpn_watch: VpnWatchConfig = Field(default_factory=VpnWatchConfig)

    @field_validator("admin_user_ids", mode="before")
    @classmethod
    def _normalize_admin_user_ids(cls, v: Any) -> list[int | str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise TypeError("admin_user_ids must be a list")
        out: list[int | str] = []
        for item in v:
            if isinstance(item, bool):
                raise ValueError("admin_user_ids: boolean is not allowed")
            if isinstance(item, int):
                out.append(item)
                continue
            if isinstance(item, str):
                s = item.strip()
                if not s:
                    continue
                core = s.lstrip("-")
                if core.isdigit() and core != "":
                    out.append(int(s))
                else:
                    out.append(s.lstrip("@").lower())
                continue
            raise ValueError(f"admin_user_ids: unsupported entry {item!r}")
        return out

    @model_validator(mode="after")
    def _jitter_order(self) -> AppConfig:
        if self.response_jitter_ms_max < self.response_jitter_ms_min:
            raise ValueError("response_jitter_ms_max must be >= response_jitter_ms_min")
        return self

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


def min_created_at_cutoff(retention_days: float | None) -> float | None:
    """Нижняя граница created_at для выборки «не старше N дней»; None — без фильтра по возрасту."""
    if retention_days is None:
        return None
    return time.time() - float(retention_days) * 86400.0


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


def create_anthropic_client(cfg: AppConfig):
    from anthropic import AsyncAnthropic

    return AsyncAnthropic(api_key=env_token(cfg.anthropic.api_key_env))


def create_llm_client(cfg: AppConfig):
    """Создать клиент активного провайдера (AsyncOpenAI или AsyncAnthropic)."""
    if cfg.provider == "anthropic":
        return create_anthropic_client(cfg)
    return create_openai_client(cfg)


def active_model(cfg: AppConfig) -> str:
    """Модель активного провайдера."""
    if cfg.provider == "anthropic":
        return cfg.anthropic.model
    return cfg.openai.model


def active_daily_input_tokens(cfg: AppConfig) -> int:
    """Суточный лимит input-токенов активного провайдера."""
    if cfg.provider == "anthropic":
        return cfg.anthropic.daily_input_tokens
    return cfg.openai.daily_input_tokens


def env_token(token_env: str) -> str:
    v = os.environ.get(token_env, "").strip()
    if not v:
        raise RuntimeError(f"Переменная окружения {token_env} не задана или пуста")
    return v


def openai_api_key(cfg: AppConfig) -> str:
    return env_token(cfg.openai.api_key_env)
