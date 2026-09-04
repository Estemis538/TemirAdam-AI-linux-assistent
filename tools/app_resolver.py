from __future__ import annotations

import difflib
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from tools.app_catalog import AppCatalog, AppEntry

logger = logging.getLogger("temiradam.tools.app_resolver")

# Category mapping to generic queries
CATEGORY_MAP = {
    "terminal": ["terminalemulator", "consoleonly", "system"],
    "терминал": ["terminalemulator", "consoleonly", "system"],
    "консоль": ["terminalemulator", "consoleonly", "system"],
    "browser": ["webbrowser", "network"],
    "браузер": ["webbrowser", "network"],
    "шолғыш": ["webbrowser", "network"],
    "editor": ["development", "texteditor", "ide"],
    "редактор": ["development", "texteditor", "ide"],
    "код": ["development", "texteditor", "ide"],
    "music": ["audio", "audiovideo", "player"],
    "музыка": ["audio", "audiovideo", "player"],
    "files": ["filemanager", "filesystem"],
    "файлы": ["filemanager", "filesystem"],
}

DEFAULT_ALIASES = {
    "terminal": "kitty",
    "терминал": "kitty",
    "консоль": "kitty",
    "browser": "firefox",
    "браузер": "firefox",
    "шолғыш": "firefox",
    "music": "spotify",
    "музыка": "spotify",
    "музыку": "spotify",
    "спотифай": "spotify",
    "дискорд": "discord",
    "стим": "steam",
    "нвидиа": "nvidia-settings",
    "телеграм": "telegram-desktop",
}


@dataclass
class ResolveResult:
    """Result of app resolution."""

    status: str  # 'single', 'ambiguous', 'not_found'
    selected: AppEntry | None = None
    candidates: list[tuple[AppEntry, float]] = field(default_factory=list)  # (AppEntry, score)
    suggestions: list[str] = field(default_factory=list)


class AppResolver:
    """
    Multi-level App Resolver.
    Resolves natural language user queries (e.g. 'Firefox', 'спотифай', 'терминал', 'NVIDIA')
    to an installed Linux desktop application.
    """

    def __init__(
        self,
        catalog: AppCatalog,
        defaults: dict[str, str] | None = None,
        aliases: dict[str, str] | None = None,
        threshold: float = 0.55,
    ) -> None:
        self.catalog = catalog
        self.defaults = defaults or {}
        self.aliases = {**DEFAULT_ALIASES, **(aliases or {})}
        self.threshold = threshold

    def resolve_app(self, query: str) -> ResolveResult:
        """
        Resolve a user query string to an AppEntry using multi-level search.

        Search Levels:
        1. Exact Match (Score = 1.0)
        2. Case-Insensitive Match (Score = 0.95)
        3. Config Defaults & Aliases (Score = 0.92)
        4. Localized Name / Category / GenericName (Score = 0.85)
        5. Fuzzy Search / Ranking (Score = 0.50 - 0.84)
        """
        clean_query = self._normalize_query(query)
        if not clean_query:
            return ResolveResult(status="not_found")

        logger.info("[APP RESOLVER] Query: '%s' (normalized: '%s')", query, clean_query)

        apps = self.catalog.get_all()
        if not apps:
            logger.warning("[APP RESOLVER] Catalog is empty!")
            return ResolveResult(status="not_found")

        scored_candidates: dict[str, tuple[AppEntry, float]] = {}  # app_id -> (entry, score)

        for app in apps:
            score = self._score_app(clean_query, app)
            if score > 0.0:
                scored_candidates[app.id] = (app, score)

        # Convert to list and sort by score descending
        sorted_candidates = sorted(
            scored_candidates.values(), key=lambda x: x[1], reverse=True
        )

        # Log candidates
        for entry, score in sorted_candidates[:5]:
            logger.info("  candidate='%s' (id=%s) score=%.2f", entry.name, entry.id, score)

        if not sorted_candidates or sorted_candidates[0][1] < self.threshold:
            # Not found: provide top 3 suggestions if available
            suggestions = [entry.name for entry, _ in sorted_candidates[:3]]
            logger.info("[APP RESOLVER] No candidate met threshold %.2f", self.threshold)
            return ResolveResult(status="not_found", suggestions=suggestions)

        top_entry, top_score = sorted_candidates[0]

        # Check if single clear winner
        if len(sorted_candidates) == 1:
            logger.info("[APP RESOLVER] Selected single match: '%s'", top_entry.name)
            return ResolveResult(status="single", selected=top_entry, candidates=sorted_candidates)

        second_entry, second_score = sorted_candidates[1]

        # If top candidate is significantly better than second candidate (margin >= 0.15)
        # OR top candidate score is very high (>= 0.90), pick it automatically!
        if (top_score - second_score >= 0.15) or top_score >= 0.90:
            logger.info(
                "[APP RESOLVER] Selected clear top match: '%s' (score=%.2f vs %.2f)",
                top_entry.name, top_score, second_score
            )
            return ResolveResult(status="single", selected=top_entry, candidates=sorted_candidates)

        # Multiple candidates with close high scores -> Ambiguous
        suggestions = [entry.name for entry, _ in sorted_candidates[:3]]
        logger.info("[APP RESOLVER] Ambiguous matches between: %s", suggestions)
        return ResolveResult(
            status="ambiguous",
            selected=top_entry,  # Fallback best
            candidates=sorted_candidates,
            suggestions=suggestions,
        )

    def _normalize_query(self, query: str) -> str:
        """Clean user query string."""
        clean = query.lower().strip()
        # Remove verbs if left over
        clean = re.sub(
            r"^(?:откр\w*|запуст\w*|включ\w*|покаж\w*|open|launch|аш\w*)\s+", "", clean
        ).strip()
        # Remove leading prepositions
        for prep in ["с ", "в ", "на ", "ди ", "из "]:
            if clean.startswith(prep):
                clean = clean[len(prep):].strip()
        clean = clean.strip(" .!?,;:")
        return clean

    def _score_app(self, query: str, app: AppEntry) -> float:
        """Calculate relevance score (0.0 to 1.0) of an AppEntry for a query."""
        app_name_clean = app.name.lower()
        app_id_clean = app.id.lower()
        app_id_no_ext = app_id_clean.replace(".desktop", "")
        exec_name = os.path.basename(app.exec_cmd.split()[0]).lower() if app.exec_cmd else ""

        # Level 1 — Exact Match (1.0)
        if query == app.name or query == exec_name or query == app_id_no_ext:
            return 1.0

        # Level 2 — Case-Insensitive Match (0.95)
        if query == app_name_clean:
            return 0.95

        # Level 3 — Config Defaults & Aliases (0.92)
        # Check alias override
        if query in self.aliases:
            alias_target = self.aliases[query].lower()
            if alias_target == app_id_no_ext or alias_target == exec_name or alias_target in app_name_clean:
                return 0.92

        # Check default category override
        if query in self.defaults:
            def_target = self.defaults[query].lower()
            if def_target == app_id_no_ext or def_target == exec_name or def_target in app_name_clean:
                return 0.92

        # Level 4 — Localized Names / GenericName / Category Match (0.85)
        for lang_code, loc_name in app.localized_names.items():
            if query == loc_name.lower():
                return 0.88
            elif query in loc_name.lower():
                return 0.84

        if app.generic_name and query == app.generic_name.lower():
            return 0.86

        for lang_code, loc_gen in app.localized_generics.items():
            if query == loc_gen.lower():
                return 0.85

        # Check category keywords map (e.g. query='terminal' -> Categories contains 'TerminalEmulator')
        if query in CATEGORY_MAP:
            target_cats = CATEGORY_MAP[query]
            app_cats = [c.lower() for c in app.categories]
            if any(cat in app_cats for cat in target_cats):
                return 0.85

        for kw in app.keywords:
            if query == kw.lower():
                return 0.85

        # Substring matches (0.70 - 0.80)
        if query in app_name_clean:
            # Return higher score if query matches start of name (e.g. "nvidia" in "NVIDIA Settings")
            if app_name_clean.startswith(query):
                return 0.82
            return 0.75

        if query in app_id_no_ext or (exec_name and query in exec_name):
            return 0.75

        if app.generic_name and query in app.generic_name.lower():
            return 0.70

        # Level 5 — Fuzzy Search (0.50 - 0.70)
        matcher = difflib.SequenceMatcher(None, query, app_name_clean)
        ratio = matcher.ratio()
        if ratio >= 0.60:
            return round(ratio * 0.85, 2)

        # Check fuzzy on exec or ID
        id_ratio = difflib.SequenceMatcher(None, query, app_id_no_ext).ratio()
        if id_ratio >= 0.65:
            return round(id_ratio * 0.80, 2)

        return 0.0
