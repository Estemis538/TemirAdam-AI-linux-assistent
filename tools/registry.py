from __future__ import annotations
import logging
from typing import Callable, Any

from tools.schemas import ToolSchema

"""
Tool registry with decorator-based registration.
"""

logger = logging.getLogger("temiradam.tools.registry")

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, tuple[ToolSchema, Callable]] = {}

    def register(self, schema: ToolSchema) -> Callable:
        def decorator(func: Callable) -> Callable:
            self._tools[schema.name] = (schema, func)
            logger.debug(f"Registered tool: {schema.name}")
            return func
        return decorator

    def get(self, name: str) -> tuple[ToolSchema, Callable] | None:
        return self._tools.get(name)

    def list_enabled(self, config: dict[str, Any]) -> list[ToolSchema]:
        enabled_tools = []
        for name, (schema, _) in self._tools.items():
            tool_cfg = config.get(name)
            if tool_cfg is None or getattr(tool_cfg, 'enabled', True):
                enabled_tools.append(schema)
        return enabled_tools

    def get_schemas_for_llm(self, config: dict[str, Any]) -> str:
        schemas = self.list_enabled(config)
        output = []
        for s in schemas:
            args = ", ".join(f"{arg.name}: {arg.type}" for arg in s.arguments)
            output.append(f"- {s.name}({args}): {s.description}")
        return "\n".join(output)

# Global registry instance
registry = ToolRegistry()
