from __future__ import annotations
import logging
import subprocess
import urllib.parse

from tools.schemas import ToolSchema, ArgumentSchema, ToolResult
from tools.registry import registry

"""
YouTube search tools.
"""

logger = logging.getLogger("temiradam.tools.youtube")

@registry.register(ToolSchema(
    name='youtube_search',
    description='Search YouTube for a query.',
    arguments=[
        ArgumentSchema(name='query', type='str', description='Search query for YouTube')
    ]
))
def youtube_search(query: str) -> ToolResult:
    logger.info(f"Searching YouTube for: {query}")
    try:
        url = f'https://www.youtube.com/results?search_query={urllib.parse.quote(query)}'
        subprocess.Popen(['firefox', url], start_new_session=True)
        return ToolResult(success=True, message=f"Opened YouTube search for '{query}'.")
    except FileNotFoundError:
        return ToolResult(success=False, message="Firefox browser not found.")
    except Exception as e:
        logger.exception(f"Error launching YouTube search: {e}")
        return ToolResult(success=False, message=f"Error searching YouTube: {str(e)}")
