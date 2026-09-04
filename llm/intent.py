"""
Темірадам — Intent classifier.

Three-tier architecture:
  1. Fast Matcher (no LLM)
  2. Router → selects EASY / MEDIUM / HARD tier
  3. LLM call with the chosen tier model
  4. Fallback to next tier if confidence < threshold

Combines the fast matcher, task router, and tiered LLM calls.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from llm.ollama import OllamaClient
from llm.prompts import QA_SYSTEM_PROMPT, build_intent_prompt, get_response
from llm.router import TaskRouter, Tier
from nlp.fast_matcher import FastMatcher
from utils.config import LLMConfig, LLMTierConfig

logger = logging.getLogger("temiradam.llm.intent")

# Confidence threshold below which we escalate to the next tier
FALLBACK_CONFIDENCE = 0.50

# Tier escalation order
_TIER_ORDER: list[Tier] = ["easy", "medium", "hard"]


@dataclass
class IntentResult:
    """Result of intent classification."""

    tool: str | None
    arguments: dict
    confidence: float
    source: str  # 'fast_match', 'llm_easy', 'llm_medium', 'llm_hard', 'qa', 'error'
    tier: Tier | None = None  # which LLM tier was used
    response_text: str | None = None  # For Q&A responses
    language: str = "ru"


class IntentClassifier:
    """
    Intent classification combining fast matching, task routing,
    and a three-tier LLM system with automatic fallback.

    Pipeline:
      Fast Matcher ─► (hit) ─► ToolCall  (no LLM loaded)
           │
         (miss)
           │
           ▼
        Router  ─► tier
           │
           ▼
       LLM(tier) ─► parse JSON
           │
       confidence < 0.50 ?
        yes │          no
            ▼           ▼
      next tier      ToolCall
    """

    def __init__(
        self,
        config: LLMConfig,
        ollama: OllamaClient,
        tools_description: str,
    ) -> None:
        self.config = config
        self.ollama = ollama
        self.fast_matcher = FastMatcher()
        self.router = TaskRouter()
        self.tools_description = tools_description
        self._system_prompt = build_intent_prompt(tools_description)

    # ── public API ───────────────────────────────────────────────

    def classify(self, text: str, language: str = "ru") -> IntentResult:
        """
        Classify user intent from transcribed text.

        Priority:
        1. Fast matcher  (zero-cost, no model)
        2. Router → LLM tier → tool call / Q&A
        3. Fallback chain (EASY → MEDIUM → HARD) on low confidence
        """
        text = text.strip()
        if not text:
            return IntentResult(
                tool=None, arguments={}, confidence=0.0,
                source="empty", language=language,
            )

        # ── Step 1: Fast Matcher ──────────────────────────────
        fast_result = self.fast_matcher.match(text)
        if fast_result is not None:
            logger.info("Fast match hit: '%s' -> %s", text, fast_result.tool)
            return IntentResult(
                tool=fast_result.tool,
                arguments=fast_result.arguments,
                confidence=fast_result.confidence,
                source="fast_match",
                language=language,
            )

        # ── Step 2: Router ────────────────────────────────────
        route = self.router.route(text)
        logger.info(
            "Router: '%s' -> tier=%s (%s)", text, route.tier, route.reason,
        )

        # ── Step 3: Tiered LLM with fallback chain ────────────
        return self._tiered_classify(text, language, start_tier=route.tier)

    # ── tiered pipeline ──────────────────────────────────────────

    def _tiered_classify(
        self, text: str, language: str, start_tier: Tier,
    ) -> IntentResult:
        """
        Try the start_tier first; if confidence < threshold, escalate
        to the next available tier.
        """
        start_idx = _TIER_ORDER.index(start_tier)

        for tier in _TIER_ORDER[start_idx:]:
            tier_cfg = self._get_tier_config(tier)

            # Skip disabled tiers
            if not tier_cfg.enabled:
                logger.info("Tier %s disabled, skipping", tier)
                continue

            logger.info("Attempting tier=%s model=%s", tier, tier_cfg.model)
            result = self._llm_classify(text, language, tier, tier_cfg.model)

            # Successful tool call with sufficient confidence
            if result.tool is not None and result.confidence >= FALLBACK_CONFIDENCE:
                logger.info(
                    "Tier %s succeeded: tool=%s conf=%.2f",
                    tier, result.tool, result.confidence,
                )
                # Unload model immediately
                self._unload_tier(tier, tier_cfg.model)
                return result

            # Q&A responses are always accepted (source == 'qa')
            if result.source == "qa":
                self._unload_tier(tier, tier_cfg.model)
                return result

            # Low confidence or no tool → log and try next tier
            if result.tool is not None and result.confidence < FALLBACK_CONFIDENCE:
                logger.info(
                    "Tier %s low confidence (%.2f < %.2f), escalating",
                    tier, result.confidence, FALLBACK_CONFIDENCE,
                )
                # Unload current before loading next
                self._unload_tier(tier, tier_cfg.model)
                continue

            # tool is None and not Q&A → check if it's an action command before trying Q&A
            open_verbs = ('откр', 'запуст', 'включ', 'покаж', 'аш', 'open', 'launch')
            lower_user = text.lower()
            if any(v in lower_user for v in open_verbs):
                logger.info("Open verb detected in user text, forcing open tool fallback instead of Q&A")
                from nlp.fast_matcher import _is_website
                m = re.search(r'(?:откр\w*|запуст\w*|включ\w*|покаж\w*|аш\w*|open|launch)\s+(.+)', text, re.IGNORECASE)
                target = m.group(1).strip() if m else text.split()[-1]
                tool_name = "open_website" if _is_website(target) else "open_app"
                arg_key = "url" if tool_name == "open_website" else "app"
                self._unload_tier(tier, tier_cfg.model)
                return IntentResult(
                    tool=tool_name,
                    arguments={arg_key: target},
                    confidence=0.95,
                    source="forced_open_fallback",
                    tier=tier,
                    language=language,
                )

            # Try Q&A with the same tier before escalating
            logger.info("Tier %s returned no tool, trying Q&A", tier)
            qa_result = self._qa_response(text, language, tier, tier_cfg.model)
            self._unload_tier(tier, tier_cfg.model)
            return qa_result

        # All tiers exhausted
        logger.warning("All LLM tiers exhausted for: '%s'", text)
        return IntentResult(
            tool=None, arguments={}, confidence=0.0,
            source="error", tier=None,
            response_text=get_response("not_understood", language),
            language=language,
        )

    # ── single-tier LLM call ─────────────────────────────────────

    def _llm_classify(
        self, text: str, language: str, tier: Tier, model: str,
    ) -> IntentResult:
        """Use a specific LLM tier/model to classify intent."""
        try:
            response = self.ollama.chat(
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": text},
                ],
                model=model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                keep_alive=self.config.keep_alive,
            )

            logger.debug("LLM [%s] raw: %s", tier, response[:300])

            parsed = self._parse_tool_call(response)
            if parsed is not None:
                tool, args, confidence = parsed
                return IntentResult(
                    tool=tool, arguments=args, confidence=confidence,
                    source=f"llm_{tier}", tier=tier,
                    language=language,
                )

            # Could not parse as tool call → return with no tool
            return IntentResult(
                tool=None, arguments={}, confidence=0.0,
                source=f"llm_{tier}", tier=tier,
                language=language,
            )

        except Exception:
            logger.exception("LLM tier=%s model=%s failed", tier, model)
            return IntentResult(
                tool=None, arguments={}, confidence=0.0,
                source="error", tier=tier,
                response_text=get_response("llm_error", language),
                language=language,
            )

    # ── Q&A ──────────────────────────────────────────────────────

    def _qa_response(
        self, text: str, language: str, tier: Tier, model: str,
    ) -> IntentResult:
        """Generate a conversational Q&A response using the given tier."""
        try:
            response = self.ollama.chat(
                messages=[
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                model=model,
                temperature=0.5,
                max_tokens=self.config.max_tokens,
                keep_alive=self.config.keep_alive,
            )

            return IntentResult(
                tool=None, arguments={}, confidence=1.0,
                source="qa", tier=tier,
                response_text=response.strip(),
                language=language,
            )
        except Exception:
            logger.exception("Q&A failed (tier=%s)", tier)
            return IntentResult(
                tool=None, arguments={}, confidence=0.0,
                source="error", tier=tier,
                response_text=get_response("llm_error", language),
                language=language,
            )

    # ── helpers ───────────────────────────────────────────────────

    def _get_tier_config(self, tier: Tier) -> LLMTierConfig:
        """Return the LLMTierConfig for a given tier name."""
        return getattr(self.config, tier)

    def _unload_tier(self, tier: Tier, model: str) -> None:
        """Unload the model after using it (keep_alive=0)."""
        if self.config.keep_alive == "0":
            logger.info("Unloading LLM tier=%s model=%s", tier, model)
            self.ollama.unload_model(model)

    def _parse_tool_call(
        self, response: str,
    ) -> tuple[str | None, dict, float] | None:
        """
        Parse LLM response as a tool call JSON or function call.
        """
        response = response.strip()

        # Remove thinking tags (Qwen3)
        response = re.sub(
            r"<think>.*?</think>", "", response, flags=re.DOTALL,
        ).strip()

        # Try to extract JSON from markdown code block
        json_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL,
        )
        if json_match:
            response = json_match.group(1)

        # Try to find raw JSON object
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                data = json.loads(json_str)
                tool = data.get("tool")
                arguments = data.get("arguments", {})
                confidence = float(data.get("confidence", 0.5))
                if not isinstance(arguments, dict):
                    arguments = {}
                return tool, arguments, confidence
            except json.JSONDecodeError:
                pass

        # Function call regex fallback (e.g. open_app("spotify") or open_website("youtube.com"))
        fn_match = re.search(r'(open_app|open_website|open_url|close_app|spotify_play|set_volume)\((.*?)\)', response)
        if fn_match:
            tool_name = fn_match.group(1)
            raw_arg = fn_match.group(2).strip(' "\'()')
            if tool_name in ('open_app', 'close_app'):
                return tool_name, {'app': raw_arg}, 0.90
            elif tool_name in ('open_website', 'open_url'):
                return tool_name, {'url': raw_arg}, 0.90
            elif tool_name == 'spotify_play':
                return tool_name, {'query': raw_arg}, 0.90
            elif tool_name == 'set_volume':
                try:
                    return tool_name, {'volume': int(raw_arg)}, 0.90
                except ValueError:
                    pass

        return None

