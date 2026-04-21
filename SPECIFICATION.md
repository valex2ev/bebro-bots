# bebro-bots: спецификация

Актуальное описание системы.

---

## 1. Назначение

Несколько независимых Telegram-ботов в одних и тех же групповых чатах. Каждый бот общается через LLM, ведёт персистентную память, поддерживает настройку естественным языком и техкоманды для администрирования.

**Окружение:** Windows-сервер; публикация через Cloudflare Tunnel.  
**Стек:** Python 3.11+, `python-telegram-bot` v21+ (async), `openai` (AsyncOpenAI), `anthropic` (AsyncAnthropic), FastAPI + uvicorn (webhook).  
**Хранилище:** SQLite (WAL-режим, один файл, путь — `sqlite_path` в конфиге).  
**Репозиторий:** GitHub.

---

## 2. LLM-провайдеры

- Провайдер задаётся **на каждого бота** полем `bots[].provider` (`openai` / `anthropic`, по умолчанию `openai`).
- Параметры подключения — в секциях `openai` / `anthropic` конфига и `.env`; переключение провайдера — только смена `provider` в YAML.
- Блок `openai`: `api_key_env`, `model`, `base_url`, `daily_input_tokens`.
- Блок `anthropic`: `api_key_env`, `model`, `daily_input_tokens`, `max_output_tokens` (по умолчанию 2048).
- **Function calling / tools** — обязательны для полноценной работы памяти и настроек.

### Лимиты (UTC-сутки)

Таблица `global_usage_daily` накапливает input-токены и успешные Tavily-запросы **по всем ботам и чатам суммарно**.  
- `openai.daily_input_tokens` / `anthropic.daily_input_tokens` — лимит на провайдер.  
- `limits.daily_web_searches` — лимит поиска.  
- При исчерпании — разовое уведомление в чат (антиспам по `limit_notify_cooldown_minutes` на пару чат + тип лимита).  
- Оверрайд суточного лимита токенов через `/bebroadmin_set_daily_limit`.

---

## 3. Поиск в интернете

Провайдер: **Tavily** (`TAVILY_API_KEY` в `.env`).  
Режим `web_search.mode`:
- `on_trigger` — инструмент `web_search` доступен только если в тексте сообщения есть подстрока-триггер (`WEB_SEARCH_TRIGGER` в `triggers.py`, по умолчанию `найди`).
- `always` — `web_search` доступен в каждом ответе на сообщение группы; инициатива и проактивные реакции — без поиска.

Явное цитирование источников не требуется.

---

## 4. Кто и когда отвечает

### Явное обращение

В одном сообщении может быть несколько явных обращений — откликаются **только** адресованные боты, остальные **игнорируют** сообщение:

- **`@username` бота** — по сущностям Telegram (`MessageEntity`: mention / text_mention), из текста и подписи к фото.
- **Текстовый алиас** — подстрока из `chat_aliases` (пара бот+чат), без учёта регистра, **не считается** обращением если вся совпадающая подстрока лежит внутри кавычек (`"`, `'`, `«»`, `""`, `''`, `` ` ``).
- **Reply на сообщение бота** — явное обращение к автору.

### Стохастика

Если в сообщении **нет** ни одного явного обращения — каждый бот независимо решает отвечать или нет по вероятности `reply_probability` из корневого YAML (только YAML, не БД, не `set_preference`).

### Сохранение правил при стохастике

Когда бот включился **самостоятельно** (без явного `@username` или алиаса), модель получает напоминание: сохранять правила и инструкции только если они явно адресованы ему или всем ботам, но не конкретному другому боту. Это предотвращает ситуацию, когда бот сохраняет чужую инструкцию, случайно включившись в разговор.

---

## 5. Форматы ответа

- **Текст** — основной формат.
- **Один или несколько emoji** — допустимо вместо текста.
- **Стикер** — отправка одного стикера.
- **Реакция** — emoji-реакция на сообщение без текстового ответа.
- **none** — молчание (`complete_turn` с `action=none`).

---

## 6. Стикеры

- Накопление `file_id` из входящих сообщений группы (`source=group`).
- Стикерпаки из BotFather: поле `sticker_set_names` у бота в YAML — при старте вызывается `getStickerSet`, стикеры кладутся в БД с `chat_id=0`.

---

## 7. Персона и тон

- **Персона** — текстовое описание личности в `bot_state.persona_text` (глобально на бота), меняется инструментом `set_bot_persona`. В YAML не задаётся; стартовый текст при пустой БД — нейтральный шаблон в коде.
- **Настроение** — `bot_state.mood`, глобально на бота; меняется `set_mood` и случайной фоновой задачей.
- **Тон** — числовые оси по чату в `chat_settings` (теплота, острота, тревога, радость/грусть, ирония, пафос, скептицизм, цинизм, уверенность, апатия, игривость; значения −1/0/1). Поверх — кратковременный сдвиг `tone_drift_json` (случайные изменения фоновой задачей).
- **`/bebro_params`** / **`/bebro_params@username`** — просмотр текущих параметров бота.

---

## 8. Память и контекст

### История сообщений

Скользящее окно в `chat_messages` (размер `window_messages`, настраивается). В промпт подаётся последние `recent_for_llm` сообщений. После джиттера дополнительно подмешивается **хвост** — сообщения с `message_id` строго больше триггерного (лимит `min(64, max(recent_for_llm*2, 16))`).

### Саммари

Периодическое сжатие через отдельный вызов LLM. Параметры: `conversation_summary.min_messages_in_window`, `conversation_summary.random_summarize_probability`.

### Долговременная память

- **`long_memory`** — текстовые инструкции (как называть участников, правила поведения и т.п.). Лимиты: `long_memory.retention_days`, `long_memory.context_limit`.
- **`situation_rules`** — правила вида «триггер → инструкция» (когда / как вести себя). Лимиты: `situation_rules.retention_days`, `situation_rules.context_limit`, `situation_rules.max_stored`.

Оба типа загружаются в пользовательский блок промпта. Удаление — по id (см. `get_group_state`: `long_instructions_with_ids`, `situation_rules_with_ids`).

### Консолидация

После каждого вызова `add_situation_rule` или `add_long_instruction` в **фоне** (asyncio.create_task) запускается `consolidate_rules`. Функция отправляет полный список правил/инструкций LLM с задачей убрать дубли, объединить похожие, разрешить противоречия. Результат атомарно заменяет данные в БД (`replace_situation_rules` / `replace_long_memory` — delete+insert в одной транзакции). При ошибке LLM данные остаются без изменений.

Реализована для обоих провайдеров: `llm.consolidate_rules` (OpenAI) и `anthropic_llm.consolidate_rules` (Anthropic).

### Восприимчивость

`feedback_sensitivity` (1–5, по умолчанию 4) — уровень учёта пожеланий, влияет на формулировки в системном промпте.

---

## 9. Инструменты LLM

| Инструмент | Назначение |
|---|---|
| `get_group_state` | Прочитать текущее состояние: настройки, память, правила, алиасы |
| `set_preference` | Сохранить предпочтение (тон, частота, длина и т.п.) |
| `add_long_instruction` | Добавить долговременную инструкцию; предварительно проверяет похожие через `get_group_state` |
| `remove_long_instruction` | Удалить инструкцию по id |
| `add_situation_rule` | Добавить ситуационное правило; предварительно проверяет похожие через `get_group_state` |
| `remove_situation_rule` | Удалить правило по id |
| `set_bot_persona` | Обновить базовое описание личности бота |
| `set_mood` | Обновить текущее настроение |
| `update_conversation_summary` | Обновить саммари беседы |
| `add_chat_alias` | Добавить текстовый триггер-алиас для этого бота в чате |
| `remove_chat_alias` | Удалить алиас |
| `web_search` | Поиск в интернете через Tavily (опционально, по режиму и лимиту) |
| `complete_turn` | Завершить ход (action + text/reaction_emoji/sticker_file_id) |

**Ограничения мутирующих инструментов:**
- Не применяются к сообщениям от ботов (`from_user.is_bot`).
- `add_chat_alias` / `remove_chat_alias` — заблокированы если в сообщении есть `@username` другого бота без обращения к текущему.
- `add_situation_rule` / `add_long_instruction` — перед вызовом модель проверяет `get_group_state`, удаляет старую похожую запись и только потом добавляет новую.
- При стохастическом включении — модель не сохраняет правила, адресованные другому боту (инструкция в промпте).

---

## 10. Настройки

- Изменение **естественным языком** в чате — основной путь.
- Может изменять **любой участник** (не только админ).
- `reply_probability` — только из корневого YAML.

---

## 11. Изображения (vision)

Блок `vision` в конфиге: `enabled`, `max_image_bytes`, `max_images_per_turn`.

В multimodal-запрос передаются:
- Фото из **текущего сообщения**.
- Фото из **сообщения, на которое сделан reply**.
- Фото из **окна истории** — только если `text_asks_prior_chat_images` (эвристика в `telegram_vision.py`) обнаруживает явный запрос на картинку из переписки (`history_photo_policy=if_text_asks`).

Инициатива и случайная реакция — без истории фото.

---

## 12. Транспорт

- `python -m bebro_bots polling` — long polling (все боты в отдельных потоках).
- `python -m bebro_bots webhook` — один HTTP-порт, пути `/hook/<bot_id>`.

---

## 13. Конфигурация

Файл `config.yaml` (локально, не в git; при отсутствии — `config.example.yaml`). Секреты — `.env`.

| Блок / поле | Назначение |
|---|---|
| `bots[]` | `id`, `token_env`, `username`, `provider`, `mood_default`, `sticker_set_names` |
| `openai` | `api_key_env`, `model`, `base_url`, `daily_input_tokens` |
| `anthropic` | `api_key_env`, `model`, `daily_input_tokens`, `max_output_tokens` |
| `limits` | `daily_web_searches`, `use_utc_calendar_day` |
| `long_memory` | `retention_days`, `context_limit` |
| `situation_rules` | `retention_days`, `context_limit`, `max_stored` |
| `conversation_summary` | `min_messages_in_window`, `random_summarize_probability` |
| `web_search` | `mode` (`on_trigger` / `always`) |
| `vision` | `enabled`, `max_image_bytes`, `max_images_per_turn` |
| `reply_probability` | Вероятность стохастического ответа (только YAML) |
| `feedback_sensitivity` | 1–5 |
| `response_jitter_ms_min/max` | Задержка перед запросом к LLM (равномерное распределение) |
| `limit_notify_cooldown_minutes` | Антиспам уведомлений о лимите |
| `allowed_chat_ids` | Whitelist чатов (опционально) |
| `admin_user_ids` | Целые user id и/или `"@username"` |
| `vpn_watch` | Мониторинг доступности и перезапуск Windows-службы (опционально) |

---

## 14. Команды

### Групповые (все участники)
- `/bebro_ping` — проверка видимости.
- `/bebro_params` / `/bebro_params@username` — параметры бота.

### Администраторские (`admin_user_ids`)
- `/bebroadmin_reset` — сброс памяти чата.
- `/bebroadmin_status` — статус процесса.
- `/bebroadmin_export` — экспорт памяти.
- `/bebroadmin_set_daily_limit <provider> <limit>` — оверрайд суточного лимита токенов на текущий день (применяет только бот с совпадающим `provider`).

---

## 15. Фоновые задачи (JobQueue)

| Задача | Интервал | Описание |
|---|---|---|
| Инициатива | 180 с | С вероятностью `initiative_probability` при соблюдении `initiative_interval_minutes` |
| Проактивная реакция | 120 с | С вероятностью `proactive_reaction_probability` — emoji-реакция на недавнее сообщение человека |
| Тон/настроение | 600 с | Случайное обновление `tone_drift_json` и иногда `mood` |
| VPN-мониторинг | `vpn_watch.check_interval_seconds` | Только Windows, только у `notify_bot_id`; проверяет `get_me` + LLM; при сбое — `Restart-Service`; уведомления во все чаты из `chat_messages` |

---

## 16. Структура модулей

| Модуль | Назначение |
|---|---|
| `db.py` | Схема и все операции с SQLite |
| `handlers.py` | Обработчики Telegram-апдейтов, сборка промпта |
| `llm.py` | OpenAI: `run_agent_turn`, `maybe_summarize`, `consolidate_rules`, `build_system_prompt` |
| `anthropic_llm.py` | Anthropic: `run_agent_turn`, `maybe_summarize`, `consolidate_rules` |
| `tool_dispatch.py` | Диспетчер LLM-инструментов; запуск консолидации после add-вызовов |
| `tool_definitions.py` | JSON-схемы инструментов |
| `addressing.py` | Определение явного обращения к боту |
| `config.py` | Pydantic-модели конфигурации |
| `behavior.py` | Решение о стохастическом участии |
| `tone_profile.py` | Оси тона, сдвиги, форматирование |
| `chat_aliases_policy.py` | Политика валидации алиасов |
| `telegram_vision.py` | Скачивание и подготовка изображений |
| `triggers.py` | Триггеры (`WEB_SEARCH_TRIGGER` и др.) |
| `tavily_search.py` | Запросы к Tavily API |
| `vpn_watch.py` | Мониторинг VPN / Windows-служба |
| `app_factory.py` | Сборка `Application` для каждого бота |
| `webhook.py` | FastAPI-приложение для webhook-режима |
| `main.py` / `__main__.py` | Точка входа |
