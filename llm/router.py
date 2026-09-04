"""
Темірадам — Task Router.

Lightweight router to determine the complexity of a user request
(EASY, MEDIUM, HARD) based on keywords and rules, without loading a heavy LLM.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger("temiradam.llm.router")

Tier = Literal["easy", "medium", "hard"]


@dataclass
class RouteResult:
    tier: Tier
    reason: str


class TaskRouter:
    """
    Determines the appropriate LLM tier for a given task.
    """

    def __init__(self) -> None:
        # Keywords that indicate a complex question/analysis (HARD)
        self.hard_keywords = re.compile(
            r"\b(объясни|расскажи|как работает|почему|разберись|проанализируй|код|ошибка|"
            r"напиши|составь|план|разница|түсіндір|қалай жұмыс істейді|неге|неліктен)\b",
            re.IGNORECASE,
        )

        # Keywords that indicate contextual commands (MEDIUM)
        self.medium_keywords = re.compile(
            r"\b(найди|поищи|информацию|что-нибудь|какую-нибудь|для программирования|"
            r"спокойную|из|табыңыз|ізде|ақпарат)\b",
            re.IGNORECASE,
        )

        # Basic action verbs (EASY)
        self.easy_keywords = re.compile(
            r"\b(открой|запусти|поставь|включи|пауза|сделай|аш|қос|ойнат)\b",
            re.IGNORECASE,
        )

    def route(self, text: str) -> RouteResult:
        """
        Route the request to the appropriate tier.
        
        Rules:
        1. Explicit "HARD" keywords -> hard
        2. Long queries (> 15 words) -> hard
        3. Contextual/search keywords -> medium
        4. Queries (8-15 words) -> medium
        5. Basic action verbs -> easy
        6. Default -> medium (safe fallback)
        """
        text = text.strip()
        word_count = len(text.split())

        # 1. Hard keywords
        if self.hard_keywords.search(text):
            return RouteResult(tier="hard", reason="Found 'hard' keywords (question/analysis)")

        # 2. Length check (very long queries are usually complex)
        if word_count >= 15:
            return RouteResult(tier="hard", reason=f"Query is long ({word_count} words)")

        # 3. Medium keywords
        if self.medium_keywords.search(text):
            return RouteResult(tier="medium", reason="Found 'medium' contextual keywords")

        # 4. Length check (medium length)
        if word_count >= 8:
            return RouteResult(tier="medium", reason=f"Query is medium length ({word_count} words)")

        # 5. Easy keywords
        if self.easy_keywords.search(text):
            return RouteResult(tier="easy", reason="Found 'easy' action keywords")

        # 6. Default
        return RouteResult(tier="medium", reason="Default fallback")
