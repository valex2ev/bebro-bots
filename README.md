# bebro-bots

Telegram-боты для групп с OpenAI, локальной SQLite-памятью и настраиваемым поведением. Подробности — в [SPECIFICATION.md](SPECIFICATION.md).

## Быстрый старт (Windows)

1. Python 3.11+.
2. Скопируйте `config.example.yaml` в `config.yaml` (файл в `.gitignore`).
3. Создайте `.env` с переменными:
   - `OPENAI_API_KEY`
   - токены ботов (`BEBRO_BIBA_BOT_TOKEN`, `BEBRO_BOBA_BOT_TOKEN` и т.д. — как в `config.yaml`).
   - опционально `TAVILY_API_KEY` — поиск в интернете (Tavily); бот вызывает поиск **только** если в сообщении есть подстрока **`найди`** (см. [SPECIFICATION.md](SPECIFICATION.md) §2).
4. Установка и запуск long polling:

```powershell
cd C:\Work\AI\Cursor\bebro-bots
pip install -e .
python -m bebro_bots
```

Или явно: `python -m bebro_bots polling`.

## Группа: бот «молчит» / не видит сообщения

1. **Режим приватности (частая причина).** По умолчанию у бота в BotFather включён *Group Privacy*: тогда Telegram **не присылает** боту обычные сообщения в группах (только команды, ответы боту и @упоминания). Нужно для **каждого** бота: [@BotFather](https://t.me/BotFather) → ваш бот → **Bot Settings** → **Group Privacy** → **Turn off** (Disable). После смены настройки может понадобиться **удалить бота из группы и добавить снова** (рекомендация Telegram).

2. **Проверка связи.** В группе отправьте `/bebro_ping` — бот должен ответить и показать `chat_id`. Если пинг есть, а на обычный текст тишина — смотри пункт 1.

3. **Логи.** При успешной доставке сообщений в логе появляется строка `Сообщение получено: bot=... chat_id=...`. Если пишете в группу, а этой строки нет — снова пункт 1.

4. **Поведение по постановке.** Без `@username` бота ответ **не гарантирован** (случайность ~15% по умолчанию). С явным `@упоминанием` ответ обязателен.

## Webhook (Cloudflare и т.д.)

```powershell
python -m bebro_bots webhook
```

Сервер слушает `server.webhook_host` / `server.webhook_port` из конфига. Маршрут для Telegram: `POST /hook/<id_бота_из_config>` (поле `id` у записи в `bots`).

После деплоя выполните `setWebhook` на URL вида `https://<ваш_домен>/hook/biba` для каждого бота.

## Админ-команды в группе

- `/bebro_ping` — проверка, что бот в группе отвечает; подсказка по `chat_id` и приватности.
- `/bebro_params` — вывести в чат основные параметры **этого** бота для текущей группы (персона из БД, тон, настройки, правила).
  - Один бот в группе: отправьте в группу **`/bebro_params`** (достаточно слэша и команды, `@` не обязателен).
  - Несколько ботов: укажите получателя в форме Telegram **`/bebro_params@username_бота`**, иначе команда может уйти всем ботам с таким же именем команды. Пример: `/bebro_params@bebro_biba_bot` (username без пробела, как в BotFather / в профиле бота).
- `/bebroadmin_reset` — сброс памяти чата (только user id из `admin_user_ids` в конфиге или `ADMIN_USER_IDS` в `.env`).
- `/bebroadmin_status` — показать настройки чата (JSON, для админов).

## Разработка без установки пакета

```powershell
$env:PYTHONPATH="C:\Work\AI\Cursor\bebro-bots\src"
python -m bebro_bots
```
