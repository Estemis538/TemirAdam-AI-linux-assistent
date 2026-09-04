from __future__ import annotations
import logging
from typing import Any

from tools.schemas import ToolCall, ToolResult
from tools.validator import ToolValidator
from tools.registry import ToolRegistry

"""
Safe tool executor.
"""

logger = logging.getLogger("temiradam.tools.executor")

class ToolExecutor:
    def __init__(self, validator: ToolValidator, registry: ToolRegistry):
        self.validator = validator
        self.registry = registry

    async def execute(self, tool_call: ToolCall, config: dict[str, Any]) -> ToolResult:
        logger.info(f"Executing tool: {tool_call.tool} with args: {tool_call.arguments}")
        
        is_valid, error_msg = self.validator.validate(tool_call)
        if not is_valid:
            logger.error(f"Validation failed for {tool_call.tool}: {error_msg}")
            return ToolResult(success=False, message=error_msg)

        tool_info = self.registry.get(tool_call.tool) # type: ignore
        schema, func = tool_info # type: ignore

        if schema.confirmation:
            logger.info(f"Tool {tool_call.tool} requires confirmation.")
            # In a real scenario, this would block or request user input.
            # Assuming it's handled or we just return a message indicating confirmation is needed.
            return ToolResult(success=False, message="User confirmation required.")

        try:
            # Checking if the function expects the config parameter
            import inspect
            sig = inspect.signature(func)
            kwargs = dict(tool_call.arguments)
            if 'config' in sig.parameters:
                kwargs['config'] = config

            if inspect.iscoroutinefunction(func):
                result = await func(**kwargs)
            else:
                result = func(**kwargs)
                
            logger.info(f"Tool {tool_call.tool} executed successfully.")
            return result
        except Exception as e:
            logger.exception(f"Error executing tool {tool_call.tool}: {e}")
            return ToolResult(success=False, message=f"Execution error: {str(e)}")
