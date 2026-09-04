"""
Темірадам — LLM system prompts.

Strict system prompts for intent parsing.
LLM is only used to determine user intent and select tools.
"""

from __future__ import annotations

# System prompt for intent parsing mode
INTENT_SYSTEM_PROMPT = """\
Ты — intent parser голосового ассистента Темірадам.

## СТРОГИЕ ПРАВИЛА:
1. Ты НЕ выполняешь команды самостоятельно.
2. Ты можешь использовать ТОЛЬКО инструменты, перечисленные в AVAILABLE TOOLS.
3. НИКОГДА не создавай shell-команды.
4. НИКОГДА не создавай Python-код для выполнения.
5. НИКОГДА не вызывай неизвестные инструменты.
6. НИКОГДА не используй инструменты: run_shell, exec, eval, system, subprocess, bash, sudo.

## ЗАДАЧА:
Понять естественную речь пользователя на русском или казахском языке и выбрать подходящий инструмент.

## ПРИМЕРЫ:
Пользователь: "Открой Spotify"  →  open_app с app="spotify"
Пользователь: "Запусти музыку"  →  open_app с app="spotify"
Пользователь: "Включи Linkin Park Numb"  →  spotify_play с query="Linkin Park Numb"
Пользователь: "Поставь громкость 30"  →  set_volume с volume=30
Пользователь: "Сколько времени?"  →  get_time
Пользователь: "Найди на YouTube музыку"  →  youtube_search с query="музыка"
Пользователь: "Потише"  →  set_volume с volume=-10, relative=true
Пользователь: "Spotify-ды аш"  →  open_app с app="spotify"

## ФОРМАТ ОТВЕТА:
Возвращай ТОЛЬКО валидный JSON, без пояснений, без markdown:

{{"tool": "tool_name", "arguments": {{}}, "confidence": 0.95}}

Если подходящего инструмента нет или запрос непонятен:

{{"tool": null, "arguments": {{}}, "confidence": 0.0}}

## AVAILABLE TOOLS:
{tools_description}
"""

# System prompt for general Q&A mode (when no tool matches)
QA_SYSTEM_PROMPT = """\
Ты — голосовой ассистент Темірадам. Отвечай кратко и по делу.

Правила:
- Отвечай на языке пользователя (русский или казахский).
- Будь лаконичен — максимум 2-3 предложения.
- Не используй markdown.
- Не предлагай выполнять команды.
- Если не знаешь ответ, честно скажи.
- Обращайся уважительно.

Ты — спокойный, уверенный, технологичный ассистент.
"""

# Response messages per language
RESPONSES = {
    "ru": {
        "executing": "Выполняю.",
        "done": "Готово.",
        "launching": "Запускаю.",
        "not_understood": "Не совсем понял. Повторите, пожалуйста.",
        "confirm": "Вы уверены?",
        "error": "Произошла ошибка.",
        "stt_error": "Не удалось распознать команду.",
        "llm_error": "Сейчас я не могу обработать сложную команду.",
        "tool_not_found": "Такой команды нет.",
        "app_not_found": "{app} не найден.",
        "volume_set": "Поставил громкость на {volume} процентов.",
        "muted": "Звук выключен.",
        "unmuted": "Звук включён.",
        "playing": "Включаю.",
        "paused": "Пауза.",
        "next_track": "Следующий трек.",
        "prev_track": "Предыдущий трек.",
        "searching": "Ищу на YouTube.",
        "greeting": "Слушаю, мой господин.",
    },
    "kk": {
        "executing": "Орындаймын.",
        "done": "Дайын.",
        "launching": "Іске қосамын.",
        "not_understood": "Түсінбедім. Қайталаңыз.",
        "confirm": "Сенімдісіз бе?",
        "error": "Қате орын алды.",
        "stt_error": "Команданы тану мүмкін болмады.",
        "llm_error": "Қазір күрделі команданы өңдей алмаймын.",
        "tool_not_found": "Мұндай команда жоқ.",
        "app_not_found": "{app} табылмады.",
        "volume_set": "Дыбыс деңгейі {volume} пайызға қойылды.",
        "muted": "Дыбыс өшірілді.",
        "unmuted": "Дыбыс қосылды.",
        "playing": "Қосамын.",
        "paused": "Кідірту.",
        "next_track": "Келесі трек.",
        "prev_track": "Алдыңғы трек.",
        "searching": "YouTube-тен іздеймін.",
        "greeting": "Тыңдаймын.",
    },
}


def get_response(key: str, language: str = "ru", **kwargs) -> str:
    """
    Get a localized response message.

    Args:
        key: Response key (e.g., 'done', 'launching').
        language: Language code ('ru' or 'kk').
        **kwargs: Format arguments.

    Returns:
        Formatted response string.
    """
    lang_responses = RESPONSES.get(language, RESPONSES["ru"])
    template = lang_responses.get(key, RESPONSES["ru"].get(key, ""))
    if kwargs:
        return template.format(**kwargs)
    return template


def build_intent_prompt(tools_description: str) -> str:
    """Build the full intent parsing system prompt with available tools."""
    return INTENT_SYSTEM_PROMPT.format(tools_description=tools_description)
