from __future__ import annotations

import configparser
import glob
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("temiradam.tools.app_catalog")

DEFAULT_DESKTOP_DIRS = [
    "/usr/share/applications",
    "/usr/local/share/applications",
    os.path.expanduser("~/.local/share/applications"),
    "/var/lib/flatpak/exports/share/applications",
    os.path.expanduser("~/.local/share/flatpak/exports/share/applications"),
]

CACHE_FILE_PATH = os.path.expanduser("data/app_catalog.json")


@dataclass
class AppEntry:
    """Represents a Linux application desktop entry."""

    id: str  # e.g., 'firefox.desktop' or 'org.kde.dolphin.desktop'
    name: str  # e.g., 'Firefox Web Browser'
    generic_name: str = ""
    comment: str = ""
    exec_cmd: str = ""
    icon: str = ""
    categories: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    desktop_file: str = ""
    localized_names: dict[str, str] = field(default_factory=dict)  # e.g., {'ru': '...', 'kk': '...'}
    localized_generics: dict[str, str] = field(default_factory=dict)
    terminal: bool = False
    no_display: bool = False
    is_flatpak: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppEntry:
        return cls(**data)


class AppCatalog:
    """
    Scans and manages Linux application desktop entries.
    Provides cached access to all installed GUI applications on the system.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.desktop_dirs = self.config.get("desktop_dirs", DEFAULT_DESKTOP_DIRS)
        self.cache_file = os.path.expanduser(
            self.config.get("cache_file", CACHE_FILE_PATH)
        )
        self.auto_refresh = self.config.get("auto_refresh", True)

        self._apps: dict[str, AppEntry] = {}  # id -> AppEntry
        self._dir_mtimes: dict[str, float] = {}

    def get_all(self) -> list[AppEntry]:
        """Return all discovered valid applications."""
        if not self._apps:
            self.load_or_scan()
        return list(self._apps.values())

    def get_by_id(self, desktop_id: str) -> AppEntry | None:
        """Get an application by its desktop ID (e.g. 'firefox.desktop')."""
        if not self._apps:
            self.load_or_scan()
        return self._apps.get(desktop_id)

    def load_or_scan(self) -> None:
        """Load from cache if fresh, otherwise perform full scan."""
        if self._try_load_cache():
            if self.auto_refresh and self._is_cache_stale():
                logger.info("[APP CATALOG] Desktop directories changed, refreshing catalog...")
                self.scan_applications()
        else:
            self.scan_applications()

    def refresh(self) -> int:
        """Force a full re-scan of all application directories."""
        return self.scan_applications()

    def scan_applications(self) -> int:
        """Scan all configured desktop directories for .desktop files."""
        logger.info("[APP CATALOG] Scanning applications in Linux desktop directories...")
        discovered: dict[str, AppEntry] = {}
        new_mtimes: dict[str, float] = {}

        for raw_dir in self.desktop_dirs:
            dir_path = Path(os.path.expanduser(raw_dir))
            if not dir_path.exists() or not dir_path.is_dir():
                continue

            try:
                new_mtimes[str(dir_path)] = dir_path.stat().st_mtime
            except OSError:
                pass

            desktop_files = glob.glob(os.path.join(str(dir_path), "*.desktop"))
            is_flatpak_dir = "flatpak" in str(dir_path).lower()

            for filepath in desktop_files:
                entry = self._parse_desktop_file(filepath, is_flatpak=is_flatpak_dir)
                if entry and not entry.no_display:
                    # Prefer user local entries over system entries if duplicate ID
                    if entry.id not in discovered or "~/.local" in filepath:
                        discovered[entry.id] = entry

        self._apps = discovered
        self._dir_mtimes = new_mtimes
        logger.info("[APP CATALOG] Found %d valid applications.", len(discovered))

        self.save_cache()
        return len(discovered)

    def _parse_desktop_file(self, filepath: str, is_flatpak: bool = False) -> AppEntry | None:
        """Parse a single .desktop file using configparser."""
        parser = configparser.ConfigParser(interpolation=None, strict=False)
        try:
            parser.read(filepath, encoding="utf-8")
        except Exception:
            try:
                parser.read(filepath, encoding="latin-1")
            except Exception as e:
                logger.debug("Failed to read %s: %e", filepath, e)
                return None

        if "Desktop Entry" not in parser:
            return None

        sec = parser["Desktop Entry"]

        # Only process Type=Application
        app_type = sec.get("Type", "").strip()
        if app_type and app_type != "Application":
            return None

        # Check NoDisplay or Hidden
        no_display_str = sec.get("NoDisplay", "false").lower()
        hidden_str = sec.get("Hidden", "false").lower()
        if no_display_str in ("true", "1") or hidden_str in ("true", "1"):
            return None

        name = sec.get("Name", "").strip()
        exec_cmd = sec.get("Exec", "").strip()
        if not name or not exec_cmd:
            return None

        desktop_id = os.path.basename(filepath)
        generic_name = sec.get("GenericName", "").strip()
        comment = sec.get("Comment", "").strip()
        icon = sec.get("Icon", "").strip()
        terminal = sec.get("Terminal", "false").lower() in ("true", "1")

        # Categories & Keywords
        raw_cats = sec.get("Categories", "")
        categories = [c.strip() for c in raw_cats.split(";") if c.strip()]

        raw_kw = sec.get("Keywords", "")
        keywords = [k.strip() for k in raw_kw.split(";") if k.strip()]

        # Localized fields (e.g. Name[ru], GenericName[ru], Name[kk], etc.)
        localized_names: dict[str, str] = {}
        localized_generics: dict[str, str] = {}
        for key, val in sec.items():
            if key.startswith("name[") and key.endswith("]"):
                lang = key[5:-1].lower()
                localized_names[lang] = val.strip()
            elif key.startswith("genericname[") and key.endswith("]"):
                lang = key[12:-1].lower()
                localized_generics[lang] = val.strip()

        return AppEntry(
            id=desktop_id,
            name=name,
            generic_name=generic_name,
            comment=comment,
            exec_cmd=exec_cmd,
            icon=icon,
            categories=categories,
            keywords=keywords,
            desktop_file=filepath,
            localized_names=localized_names,
            localized_generics=localized_generics,
            terminal=terminal,
            no_display=False,
            is_flatpak=is_flatpak,
        )

    def _try_load_cache(self) -> bool:
        """Load catalog from cache file if it exists."""
        if not os.path.exists(self.cache_file):
            return False

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            apps_data = data.get("apps", {})
            self._apps = {
                app_id: AppEntry.from_dict(info)
                for app_id, info in apps_data.items()
            }
            self._dir_mtimes = data.get("mtimes", {})
            logger.info("[APP CATALOG] Loaded %d applications from cache.", len(self._apps))
            return True
        except Exception as e:
            logger.warning("[APP CATALOG] Failed to load cache: %s", e)
            return False

    def save_cache(self) -> None:
        """Save application catalog to cache file."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            data = {
                "mtimes": self._dir_mtimes,
                "apps": {app_id: entry.to_dict() for app_id, entry in self._apps.items()},
            }
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("[APP CATALOG] Cache saved to %s", self.cache_file)
        except Exception as e:
            logger.error("[APP CATALOG] Failed to save cache: %s", e)

    def _is_cache_stale(self) -> bool:
        """Check if any application directory mtime changed."""
        for raw_dir in self.desktop_dirs:
            dir_path = Path(os.path.expanduser(raw_dir))
            if dir_path.exists() and dir_path.is_dir():
                try:
                    current_mtime = dir_path.stat().st_mtime
                    cached_mtime = self._dir_mtimes.get(str(dir_path), 0.0)
                    if current_mtime != cached_mtime:
                        return True
                except OSError:
                    pass
        return False
