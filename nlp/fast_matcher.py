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


# Each pattern: (compiled_regex, tool_name, argument_extractor, confidence)
# argument_extractor is a callable: (match) -> dict
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


class FastMatcher:
    """
    Fast command matcher for common/simple commands.

    Uses regex patterns to match commands without invoking LLM.
    Returns None if no pattern matches (command should go to LLM).
    """

    def match(self, text: str) -> FastMatchResult | None:
        """
        Try to match text against known command patterns.

        Args:
            text: Normalized text from STT (lowercase, stripped).

        Returns:
            FastMatchResult if matched, None otherwise.
        """
        text = text.strip().lower()

        # Remove trailing punctuation
        text = text.rstrip(".!?,;:")

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
