from __future__ import annotations

import logging
import re
import shutil
import subprocess
from typing import Any

from tools.app_catalog import AppEntry
from tools.schemas import ToolResult

logger = logging.getLogger("temiradam.tools.app_launcher")


class AppLauncher:
    """
    Safely launches Linux application desktop entries.
    Uses official Linux desktop entry launch mechanisms (gio launch / gtk-launch)
    or sanitized Exec execution.
    """

    def __init__(self, preferred_terminal: str = "kitty") -> None:
        self.preferred_terminal = preferred_terminal

    def launch(self, entry: AppEntry) -> ToolResult:
        """
        Launch a Linux desktop entry.

        Args:
            entry: The AppEntry to launch.

        Returns:
            ToolResult indicating launch status.
        """
        logger.info(
            "[APP LAUNCHER] Launching app '%s' (id=%s, desktop_file=%s)",
            entry.name,
            entry.id,
            entry.desktop_file,
        )

        # 1. Try Linux Desktop Entry launch mechanism (gio launch or gtk-launch)
        if entry.desktop_file and shutil.which("gio"):
            try:
                subprocess.Popen(
                    ["gio", "launch", entry.desktop_file],
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info("[APP LAUNCHER] Launched via 'gio launch %s'", entry.desktop_file)
                return ToolResult(
                    success=True,
                    message=f"Приложение {entry.name} запущено.",
                    data={"app_id": entry.id, "method": "gio"},
                )
            except Exception as e:
                logger.debug("gio launch failed: %s", e)

        if entry.id and shutil.which("gtk-launch"):
            try:
                subprocess.Popen(
                    ["gtk-launch", entry.id],
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info("[APP LAUNCHER] Launched via 'gtk-launch %s'", entry.id)
                return ToolResult(
                    success=True,
                    message=f"Приложение {entry.name} запущено.",
                    data={"app_id": entry.id, "method": "gtk-launch"},
                )
            except Exception as e:
                logger.debug("gtk-launch failed: %s", e)

        # 2. Fallback: Parse Exec line
        if not entry.exec_cmd:
            return ToolResult(
                success=False,
                message=f"У приложения {entry.name} отсутствует команда для запуска.",
            )

        cmd_args = self._clean_exec_command(entry.exec_cmd)
        if not cmd_args:
            return ToolResult(
                success=False,
                message=f"Не удалось распарсить команду для {entry.name}.",
            )

        binary = cmd_args[0]
        binary_path = shutil.which(binary)
        if not binary_path:
            logger.error("[APP LAUNCHER] Binary not found in PATH: %s", binary)
            return ToolResult(
                success=False,
                message=f"Исполняемый файл '{binary}' для {entry.name} не найден.",
            )

        # Update binary to full path
        cmd_args[0] = binary_path

        # Handle Terminal=true applications
        if entry.terminal:
            term_binary = shutil.which(self.preferred_terminal) or shutil.which("alacritty") or shutil.which("xterm")
            if term_binary:
                cmd_args = [term_binary, "-e"] + cmd_args
                logger.info("[APP LAUNCHER] Terminal app, wrapping in: %s", term_binary)

        try:
            subprocess.Popen(
                cmd_args,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("[APP LAUNCHER] Launched via Popen: %s", cmd_args)
            return ToolResult(
                success=True,
                message=f"Приложение {entry.name} запущено.",
                data={"app_id": entry.id, "method": "popen", "cmd": cmd_args},
            )
        except Exception as e:
            logger.exception("[APP LAUNCHER] Failed to launch %s: %s", entry.name, e)
            return ToolResult(
                success=False,
                message=f"Ошибка при запуске {entry.name}: {str(e)}",
            )

    def _clean_exec_command(self, raw_exec: str) -> list[str]:
        """
        Clean desktop entry Exec line by removing field codes (%f, %u, etc.).

        Field codes in Desktop Entry Specification:
        %f, %F, %u, %U, %d, %D, %n, %N, %i, %c, %k, %v, %m
        """
        # Remove field codes (%u, %f, etc.)
        cleaned = re.sub(r"%[fFuUdDnNiIckvm]", "", raw_exec).strip()
        # Split into tokens taking quotes into account
        import shlex
        try:
            tokens = shlex.split(cleaned)
        except Exception:
            tokens = cleaned.split()

        return [t for t in tokens if t.strip()]
