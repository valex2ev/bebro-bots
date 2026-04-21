"""Схемы инструментов LLM (OpenAI function-calling формат)."""

from __future__ import annotations

from typing import Any, Literal

from bebro_bots.tone_profile import TONE_KEYS

PREFERENCE_KEYS = {
    "window_messages",
    "recent_for_llm",
    "reaction_probability",
    "initiative_probability",
    "initiative_interval_minutes",
    "proactive_reaction_probability",
    "verbosity",
    "frequency",
    "short_reply_bias",
    "response_depth",
    *TONE_KEYS,
}

MUTATING_TOOL_NAMES = frozenset(
    {
        "set_preference",
        "add_long_instruction",
        "remove_long_instruction",
        "set_mood",
        "update_conversation_summary",
        "add_situation_rule",
        "remove_situation_rule",
        "set_bot_persona",
        "add_chat_alias",
        "remove_chat_alias",
    }
)


def _tools_schema(
    *,
    include_web_search: bool = False,
    web_search_mode: Literal["on_trigger", "always"] = "on_trigger",
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "get_group_state",
                "description": "Прочитать настроения, предпочтения, краткую память и саммари беседы по текущему чату.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_preference",
                "description": (
                    "Сохранить долговременное предпочтение для этого чата (частота, длина ответов и т.д.). "
                    "Ключи reply_probability и reply_to_bot_probability недоступны — частота ответов задаётся только в config.yaml."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "enum": sorted(PREFERENCE_KEYS),
                        },
                        "value": {
                            "type": "string",
                            "description": "Число как строка, или short|medium|long для verbosity, rare|medium|often для frequency",
                        },
                    },
                    "required": ["key", "value"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "add_long_instruction",
                "description": (
                    "Добавить долгую инструкцию для поведения в чате: в т.ч. как называть участников, "
                    "прозвища, кто как к кому обращается. Не использовать для триггеров «откликайтесь на слово» — "
                    "под это add_chat_alias. "
                    "Перед добавлением вызови get_group_state и проверь long_instructions_with_ids: "
                    "если похожая инструкция уже есть — сначала удали через remove_long_instruction, затем добавь обновлённую. "
                    "Если ты включился самостоятельно (без явного @username или алиаса) и инструкция явно адресована другому боту — не сохраняй."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "remove_long_instruction",
                "description": (
                    "Удалить одну долгую инструкцию по id (см. get_group_state: long_instructions)."
                    "Только по просьбе человека."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "integer",
                            "description": "Числовой id записи long_memory",
                        }
                    },
                    "required": ["memory_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_mood",
                "description": "Обновить текущее настроение бота (короткая строка).",
                "parameters": {
                    "type": "object",
                    "properties": {"mood": {"type": "string"}},
                    "required": ["mood"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_conversation_summary",
                "description": "Обновить сжатый контекст беседы (саммари) для экономии токенов.",
                "parameters": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "add_situation_rule",
                "description": (
                    "Сохранить правило: в какой ситуации (trigger) как себя вести (instruction). "
                    "Только если просит человек (не другой бот). "
                    "Перед добавлением вызови get_group_state и проверь situation_rules_with_ids: "
                    "если похожее правило уже есть — сначала удали через remove_situation_rule, затем добавь обновлённое. "
                    "Если ты включился самостоятельно (без явного @username или алиаса) и правило явно адресовано другому боту — не сохраняй."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "trigger_text": {
                            "type": "string",
                            "description": "Кратко: когда применять (тема, настроение чата, тип сообщения)",
                        },
                        "instruction_text": {
                            "type": "string",
                            "description": "Что делать в таких случаях (1–2 предложения)",
                        },
                    },
                    "required": ["trigger_text", "instruction_text"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "remove_situation_rule",
                "description": (
                    "Удалить ситуационное правило по id (см. get_group_state: situation_rules). "
                    "Только по просьбе человека."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rule_id": {
                            "type": "integer",
                            "description": "Числовой id правила в situation_rules",
                        }
                    },
                    "required": ["rule_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_bot_persona",
                "description": (
                    "Обновить базовое текстовое описание личности бота (глобально для всех чатов). "
                    "Только по просьбе человека."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "add_chat_alias",
                "description": (
                    "Только если люди просят, чтобы этот бот откликался на фразу (алиас хранится на этого бота в чате). "
                    "Если в сообщении указан другой бот через @ — инструмент недоступен (кроме случая, когда обращаются к тебе). "
                    "Не для «называйте пользователя X так-то» — add_long_instruction. "
                    "Нельзя: @username людей, строки-логины Telegram (ASCII). "
                    "Подстрока в тексте — отклик как при @username только у ботов, у которых сохранён этот алиас."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Подстрока-триггер (без @ и без логинов Telegram)",
                        }
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "remove_chat_alias",
                "description": (
                    "Удалить сохранённый у этого бота текстовый триггер (алиас) в чате. "
                    "Используй, когда человек просит перестать откликаться на фразу: например «перестань откликаться на …», "
                    "«убери алиас …», «забудь триггер …». В поле text передай саму подстроку-триггер (как в списке алиасов, "
                    "без лишних слов). Те же ограничения, что у add_chat_alias (не @ другого бота без обращения к тебе)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Подстрока-триггер для удаления (как сохраняли)",
                        }
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    if include_web_search:
        ws_desc = (
            "Поиск в интернете (Tavily). Доступен только в этом диалоге: пользователь явно написал ключевое слово. "
            "Передай сжатый поисковый запрос в query."
            if web_search_mode == "on_trigger"
            else (
                "Поиск в интернете (Tavily). Используй при необходимости свежих или проверяемых сведений из сети; "
                "не вызывай без пользы. Сжатый запрос в query."
            )
        )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": ws_desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Запрос для поиска (суть вопроса без служебных слов)",
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "complete_turn",
                "description": "Завершить ход одним вызовом. Не дублируй это описание в поле text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["message", "reaction", "sticker", "none"],
                        },
                        "text": {
                            "type": "string",
                            "description": "Только текст/emoji для людей в чате. Без имён инструментов, без JSON и без фраз вроде complete_turn.",
                        },
                        "reaction_emoji": {
                            "type": "string",
                            "description": "Одна emoji для реакции на сообщение пользователя",
                        },
                        "sticker_file_id": {
                            "type": "string",
                            "description": "file_id стикера или пусто для случайного из пула чата",
                        },
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
        }
    )
    return tools
