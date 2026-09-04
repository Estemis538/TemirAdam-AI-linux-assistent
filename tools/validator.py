from __future__ import annotations
import logging
from typing import Any

from tools.schemas import ToolCall
from tools.registry import ToolRegistry

"""
Tool validator for whitelist enforcement.
"""

logger = logging.getLogger("temiradam.tools.validator")

BLACKLISTED_TOOLS = {
    'run_shell', 'exec', 'eval', 'system', 'subprocess',
    'bash', 'fish', 'sudo', 'rm', 'delete', 'format'
}

class ToolValidator:
    def __init__(self, config: dict[str, Any], registry: ToolRegistry):
        self.config = config
        self.registry = registry

    def validate(self, tool_call: ToolCall) -> tuple[bool, str]:
        """Returns (is_valid, error_message). Empty string if valid."""
        if not tool_call.tool:
            return False, "Tool name is missing"

        if tool_call.tool in BLACKLISTED_TOOLS:
            logger.warning(f"Blocked attempt to use blacklisted tool: {tool_call.tool}")
            return False, f"Tool '{tool_call.tool}' is permanently blacklisted."

        tool_info = self.registry.get(tool_call.tool)
        if not tool_info:
            return False, f"Tool '{tool_call.tool}' not found in registry."

        schema, _ = tool_info
        tool_cfg = self.config.get(tool_call.tool)

        if tool_cfg and not getattr(tool_cfg, 'enabled', True):
            return False, f"Tool '{tool_call.tool}' is disabled."

        for arg in schema.arguments:
            if arg.required and arg.name not in tool_call.arguments:
                return False, f"Missing required argument '{arg.name}' for tool '{tool_call.tool}'."

        if tool_call.tool == 'open_app':
            app_arg = tool_call.arguments.get('app')
            allowed_apps = getattr(self.config.get('open_app'), 'allowed_apps', []) if self.config.get('open_app') else []
            if isinstance(allowed_apps, dict):
                allowed_apps = list(allowed_apps.keys())
            if app_arg not in allowed_apps and app_arg not in getattr(self.config.get('open_app'), 'aliases', {}):
                # Simple check for now, can be improved based on exact config structure
                return False, f"App '{app_arg}' is not in the allowed applications list."

        return True, ""
