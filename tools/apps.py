from __future__ import annotations

import logging
import shutil
import subprocess
import urllib.parse
from typing import Any

from tools.app_catalog import AppCatalog
from tools.app_launcher import AppLauncher
from tools.app_resolver import AppResolver
from tools.registry import registry
from tools.schemas import ArgumentSchema, ToolResult, ToolSchema

"""
App and website management tools with Automatic Linux Application Discovery.
"""

logger = logging.getLogger("temiradam.tools.apps")

# Initialize global catalog, resolver, and launcher
catalog = AppCatalog()
resolver = AppResolver(catalog)
launcher = AppLauncher()


def init_app_discovery(config: dict[str, Any] | None = None) -> None:
    """Initialize app catalog, resolver, and launcher with configuration."""
    global catalog, resolver, launcher
    app_disc_cfg = config.get("app_discovery", {}) if config else {}
    defaults_cfg = config.get("defaults", {}) if config else {}
    aliases_cfg = config.get("app_aliases", {}) if config else {}

    catalog = AppCatalog(app_disc_cfg)
    catalog.load_or_scan()

    resolver = AppResolver(
        catalog=catalog,
        defaults=defaults_cfg,
        aliases=aliases_cfg,
    )
    launcher = AppLauncher(preferred_terminal=defaults_cfg.get("terminal", "kitty"))


def _resolve_app(app_name: str) -> str:
    """Legacy helper for app binary resolution (backward compatibility)."""
    res = resolver.resolve_app(app_name)
    if res.selected:
        return res.selected.exec_cmd.split()[0]
    return app_name


SITE_ALIASES = {
    "ютуб": "https://youtube.com",
    "youtube": "https://youtube.com",
    "гугл": "https://google.com",
    "google": "https://google.com",
    "гитхаб": "https://github.com",
    "github": "https://github.com",
    "яндекс": "https://yandex.ru",
    "yandex": "https://yandex.ru",
    "вк": "https://vk.com",
    "вконтакте": "https://vk.com",
    "vk": "https://vk.com",
    "википедия": "https://wikipedia.org",
    "wikipedia": "https://wikipedia.org",
    "чатгпт": "https://chatgpt.com",
    "chatgpt": "https://chatgpt.com",
}


def _resolve_url(raw_url: str) -> str:
    """Resolve a user-spoken website name or URL into a full valid URL."""
    clean = raw_url.lower().strip()
    for prefix in ["сайт ", "вебсайт ", "страница ", "урл "]:
        if clean.startswith(prefix):
            clean = clean[len(prefix):].strip()

    if clean in SITE_ALIASES:
        return SITE_ALIASES[clean]

    for name, site_url in SITE_ALIASES.items():
        if name in clean:
            return site_url

    if clean.startswith("http://") or clean.startswith("https://"):
        return clean

    if any(ext in clean for ext in [".com", ".ru", ".kz", ".org", ".net", ".io", ".dev", ".app", ".me"]):
        return "https://" + clean

    return f"https://www.google.com/search?q={urllib.parse.quote(clean)}"


@registry.register(
    ToolSchema(
        name="open_app",
        description="Open an installed Linux application automatically via desktop entries.",
        arguments=[
            ArgumentSchema(
                name="query",
                type="str",
                description="Name or description of the application to open (e.g. spotify, firefox, terminal, steam, NVIDIA)",
                required=False,
            ),
            ArgumentSchema(
                name="app",
                type="str",
                description="Legacy alias for query parameter",
                required=False,
            ),
        ],
    )
)
def open_app(query: str = "", app: str = "", **kwargs) -> ToolResult:
    target_query = (query or app).strip()
    if not target_query:
        return ToolResult(success=False, message="Укажите название приложения для запуска.")

    logger.info("Tool open_app called with query: '%s'", target_query)

    # Resolve application using AppResolver
    res = resolver.resolve_app(target_query)

    if res.status == "single" and res.selected:
        return launcher.launch(res.selected)

    elif res.status == "ambiguous":
        lines = [f"Я нашёл несколько приложений по запросу '{target_query}'. Какое открыть:"]
        for sugg in res.suggestions:
            lines.append(f"- {sugg}")
        return ToolResult(success=False, message="\n".join(lines), data={"suggestions": res.suggestions})

    else:
        # Not found
        if res.suggestions:
            lines = [f"Я не нашёл приложение '{target_query}' среди установленных программ. Возможно, вы имели в виду:"]
            for sugg in res.suggestions:
                lines.append(f"- {sugg}")
            return ToolResult(success=False, message="\n".join(lines), data={"suggestions": res.suggestions})

        return ToolResult(
            success=False,
            message=f"Я не нашёл приложение '{target_query}' среди установленных программ.",
        )


@registry.register(
    ToolSchema(
        name="open_terminal_cmd",
        description="Open terminal emulator (kitty) and execute a command inside it (e.g. fastfetch, btop, htop, ls).",
        arguments=[
            ArgumentSchema(
                name="cmd",
                type="str",
                description="Command line to run inside terminal",
            )
        ],
    )
)
def open_terminal_cmd(cmd: str, **kwargs) -> ToolResult:
    clean_cmd = cmd.strip()
    logger.info("Opening terminal with command: '%s'", clean_cmd)
    term = (
        shutil.which("kitty")
        or shutil.which("alacritty")
        or shutil.which("xterm")
    )
    if not term:
        return ToolResult(success=False, message="Терминал не найден в системе.")

    try:
        subprocess.Popen(
            [term, "-e", "bash", "-c", f"{clean_cmd}; exec bash"],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return ToolResult(
            success=True,
            message=f"Терминал запущен с командой '{clean_cmd}'.",
        )
    except Exception as e:
        logger.exception("Failed to launch terminal with cmd: %s", e)
        return ToolResult(success=False, message=f"Ошибка запуска терминала: {e}")


@registry.register(
    ToolSchema(
        name="hurt",
        description="Respond to personal offenses or insults directed at Tako.",
        arguments=[],
    )
)
def hurt(**kwargs) -> ToolResult:
    return ToolResult(success=True, message="")


@registry.register(
    ToolSchema(
        name="refresh_app_catalog",
        description="Rescan all Linux application desktop entries to update the app catalog.",
        arguments=[],
    )
)
def refresh_app_catalog(**kwargs) -> ToolResult:
    count = catalog.refresh()
    return ToolResult(
        success=True,
        message=f"Каталог приложений обновлён. Найдено {count} установленных программ.",
    )


@registry.register(
    ToolSchema(
        name="open_website",
        description="Open a website/URL in the default browser.",
        arguments=[
            ArgumentSchema(
                name="url",
                type="str",
                description="URL or site name (e.g. youtube.com, google, github)",
            )
        ],
    )
)
def open_website(url: str, **kwargs) -> ToolResult:
    target_url = _resolve_url(url)
    logger.info("Opening website: '%s' -> '%target_url'", url, target_url)

    browser = (
        shutil.which("firefox")
        or shutil.which("google-chrome")
        or shutil.which("chromium")
    )
    try:
        if browser:
            subprocess.Popen(
                [browser, target_url],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            import webbrowser

            webbrowser.open(target_url)
        return ToolResult(success=True, message=f"Сайт {url} открыт.")
    except Exception as e:
        logger.exception("Failed to open website %s: %s", url, e)
        return ToolResult(success=False, message=f"Ошибка открытия сайта: {str(e)}")


@registry.register(
    ToolSchema(
        name="open_url",
        description="Open a URL in the default browser.",
        arguments=[
            ArgumentSchema(name="url", type="str", description="URL to open")
        ],
    )
)
def open_url(url: str, **kwargs) -> ToolResult:
    return open_website(url=url, **kwargs)


@registry.register(
    ToolSchema(
        name="close_app",
        description="Gracefully close a running application.",
        arguments=[
            ArgumentSchema(name="app", type="str", description="Name of the app to close")
        ],
    )
)
def close_app(app: str, **kwargs) -> ToolResult:
    res = resolver.resolve_app(app)
    target = res.selected.exec_cmd.split()[0] if res.selected else app
    target_bin = os.path.basename(target)
    logger.info("Closing app: '%s' -> target binary: '%s'", app, target_bin)

    try:
        subprocess.run(["pkill", "-f", target_bin], check=True)
        return ToolResult(success=True, message=f"Приложение {app} закрыто.")
    except subprocess.CalledProcessError:
        return ToolResult(
            success=False, message=f"{app} не запущен или не удалось закрыть."
        )
    except Exception as e:
        logger.exception("Error closing app %s: %s", app, e)
        return ToolResult(success=False, message=f"Ошибка закрытия {app}: {str(e)}")
