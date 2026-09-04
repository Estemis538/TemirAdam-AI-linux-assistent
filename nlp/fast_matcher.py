"""
Темірадам — Fast command matcher.

Lightweight regex/keyword-based matcher for common commands.
Bypasses LLM for obvious, simple commands to save resources.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("temiradam.nlp.fast_matcher")


@dataclass
class FastMatchResult:
    """Result from the fast command matcher."""

    tool: str
    arguments: dict
    confidence: float
    matched_pattern: str


KNOWN_WEBSITES = {
    'ютуб', 'youtube', 'гугл', 'google', 'гитхаб', 'github',
    'яндекс', 'yandex', 'вк', 'вконтакте', 'vk', 'википедия',
    'wikipedia', 'чатгпт', 'chatgpt', 'оллама', 'ollama'
}

DOMAIN_EXTENSIONS = ('.com', '.ru', '.kz', '.org', '.net', '.io', '.dev', '.app', '.me')

FILLER_WORDS = [
    'пожалуйста', 'эй', 'слушай', 'тако', 'така', 'такам', 'компьютер', 'ассистент'
]


def _is_website(target: str) -> bool:
    clean = target.lower().strip()
    if clean.startswith(('сайт ', 'вебсайт ', 'страница ', 'урл ', 'http://', 'https://')):
        return True
    if any(site in clean for site in KNOWN_WEBSITES):
        return True
    if any(ext in clean for ext in DOMAIN_EXTENSIONS):
        return True
    return False


# Each pattern: (compiled_regex, tool_name, argument_extractor, confidence)
_PATTERNS: list[tuple[re.Pattern, str, callable, float]] = []


def _p(
    pattern: str,
    tool: str,
    arg_fn: callable = lambda m: {},
    confidence: float = 0.95,
) -> None:
    """Helper to register a pattern."""
    _PATTERNS.append((re.compile(pattern, re.IGNORECASE), tool, arg_fn, confidence))


# ─── Time / Date ──────────────────────────────────────────────

_p(r"^(время|сколько времени|который час|қазір сағат қанша|сағат қанша|уақыт)$", "get_time")
_p(r"^(дата|какое сегодня число|какой сегодня день|бүгін қандай күн|бүгінгі күн)$", "get_date")

# ─── Spotify basic controls ──────────────────────────────────

_p(r"^(пауза|стоп|останови|тоқта|pause)$", "spotify_pause")
_p(r"^(продолжи|играй|play|ойнат|жалғастыр)$", "spotify_play")
_p(r"^(следующий|следующий трек|next|дальше|келесі|келесі трек)$", "spotify_next")
_p(r"^(предыдущий|предыдущий трек|назад|previous|алдыңғы|алдыңғы трек)$", "spotify_previous")

# ─── Volume with number ──────────────────────────────────────

_p(
    r"^(?:громкость|volume|дыбыс)\s+(?:на\s+)?(\d+)\s*(?:процент(?:ов)?|%)?$",
    "set_volume",
    lambda m: {"volume": int(m.group(1))},
)
_p(
    r"^(?:поставь|установи|set)\s+(?:громкость|volume|дыбыс)\s+(?:на\s+)?(\d+)\s*(?:процент(?:ов)?|%)?$",
    "set_volume",
    lambda m: {"volume": int(m.group(1))},
)

# ─── Volume simple commands ──────────────────────────────────

_p(r"^(выключи звук|без звука|mute|дыбысты өшір)$", "mute")
_p(r"^(включи звук|unmute|верни звук|дыбысты қос)$", "unmute")
_p(r"^(какая громкость|текущая громкость|громкость\??|volume\??)$", "get_volume")

# ─── Volume relative ─────────────────────────────────────────

_p(
    r"^(потише|тише|убавь|убавь звук|тихо|кішірек|азайт)$",
    "set_volume",
    lambda m: {"volume": -10, "relative": True},
    0.90,
)
_p(
    r"^(погромче|громче|прибавь|прибавь звук|громко|қаттырақ|көбейт)$",
    "set_volume",
    lambda m: {"volume": 10, "relative": True},
    0.90,
)

# ─── Close apps ──────────────────────────────────────────────

_p(
    r"^(?:закрой|выключи|close|kill|жап|өшір)\s+(.+)$",
    "close_app",
    lambda m: {"app": m.group(1).strip()},
)


def normalize_command_text(text: str) -> str:
    """Normalize text to fix common Whisper STT typos and split words."""
    clean = text.lower().strip()

    # 1. Fix space-split open verbs & typos
    clean = re.sub(r'\b(?:от\s*крой|от\s*рвей|от\s*край|ат\s*крой|акгрой|отбрай|аккорит)\b', 'открой', clean)
    clean = re.sub(r'\b(?:от\s*крою|от\s*крывай)\b', 'открою', clean)
    clean = re.sub(r'\b(?:от\s*кройся|от\s*крывайся)\b', 'открой', clean)

    # 2. Fix typos in app names
    clean = re.sub(r'\b(?:файфонс|файрфокс|фаерфокс|файерфокс|бронсия)\b', 'firefox', clean)
    clean = re.sub(r'\b(?:спотифай|спотифи|смотифай|спатифай|спутифай)\b', 'spotify', clean)

    # 3. Strip prepositions between open verb and app name ('открой с spotify' -> 'открой spotify')
    clean = re.sub(r'^(откр\w*|запуст\w*|включ\w*)\s+(?:с|в|на|ди|из)\s+(.+)$', r'\1 \2', clean)

    return clean


class FastMatcher:
    """
    Fast command matcher for common/simple commands.

    Uses regex patterns to match commands without invoking LLM.
    Returns None if no pattern matches (command should go to LLM).
    """

    def match(self, text: str) -> FastMatchResult | None:
        text = normalize_command_text(text)

        # Clean filler words
        for filler in FILLER_WORDS:
            if text.startswith(filler + " "):
                text = text[len(filler):].strip()
            if text.endswith(" " + filler):
                text = text[:-len(filler)].strip()

        # Remove trailing punctuation
        text = text.strip(" .!?,;:")

        # 1. Smart Open Matcher (highest priority for opening commands)
        open_match = re.match(
            r"^(?:откр\w*|запуст\w*|включ\w*|перейди\s+на|покаж\w*|показат\w*|open|launch|аш\w*|іске\s+қос|ашып\s+бер)\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if open_match:
            target = open_match.group(1).strip()
            if _is_website(target):
                logger.info("Fast match open_website: '%s' -> target='%s'", text, target)
                return FastMatchResult(
                    tool="open_website",
                    arguments={"url": target},
                    confidence=0.98,
                    matched_pattern="open_website_heuristic",
                )
            else:
                logger.info("Fast match open_app: '%s' -> query='%s'", text, target)
                return FastMatchResult(
                    tool="open_app",
                    arguments={"query": target},
                    confidence=0.98,
                    matched_pattern="open_app_heuristic",
                )

        # 2. Other patterns
        for pattern, tool, arg_fn, confidence in _PATTERNS:
            m = pattern.match(text)
            if m:
                args = arg_fn(m)
                result = FastMatchResult(
                    tool=tool,
                    arguments=args,
                    confidence=confidence,
                    matched_pattern=pattern.pattern,
                )
                logger.info(
                    "Fast match: '%s' -> %s(%s) [%.2f]",
                    text,
                    tool,
                    args,
                    confidence,
                )
                return result

        logger.debug("No fast match for: '%s'", text)
        return None

