from __future__ import annotations
import logging
import subprocess
from typing import Any

from tools.schemas import ToolSchema, ArgumentSchema, ToolResult
from tools.registry import registry

"""
App management tools.
"""

logger = logging.getLogger("temiradam.tools.apps")

@registry.register(ToolSchema(
    name='open_app',
    description='Open an allowed application.',
    arguments=[
        ArgumentSchema(name='app', type='str', description='Name of the app to open (e.g., browser, music)')
    ]
))
def open_app(app: str, config: dict[str, Any]) -> ToolResult:
    logger.info(f"Opening app: {app}")
    app_config = config.get('open_app')
    allowed_apps = getattr(app_config, 'allowed_apps', {}) if app_config else {}
    
    if isinstance(allowed_apps, list):
        # Fallback if allowed_apps is a list of commands
        command = app
    else:
        # Assuming allowed_apps is a dict mapping app name/alias to command
        command = allowed_apps.get(app, app)
    
    # Simple alias mapping for common terms
    aliases = {
        'browser': 'firefox',
        'music': 'spotify',
        'terminal': 'gnome-terminal'
    }
    command = aliases.get(app.lower(), command)
        
    try:
        subprocess.Popen([command], start_new_session=True)
        return ToolResult(success=True, message=f"Application {app} opened successfully.")
    except FileNotFoundError:
        return ToolResult(success=False, message=f"Command for app {app} not found.")
    except Exception as e:
        logger.exception(f"Failed to open app {app}: {e}")
        return ToolResult(success=False, message=f"Failed to open {app}: {str(e)}")


@registry.register(ToolSchema(
    name='close_app',
    description='Gracefully close a running application.',
    arguments=[
        ArgumentSchema(name='app', type='str', description='Name of the app to close')
    ]
))
def close_app(app: str) -> ToolResult:
    logger.info(f"Closing app: {app}")
    
    aliases = {
        'browser': 'firefox',
        'music': 'spotify'
    }
    target = aliases.get(app.lower(), app)
    
    try:
        subprocess.run(['pkill', target], check=True)
        return ToolResult(success=True, message=f"Application {app} closed.")
    except subprocess.CalledProcessError:
        return ToolResult(success=False, message=f"Application {app} might not be running or failed to close.")
    except Exception as e:
        logger.exception(f"Error closing app {app}: {e}")
        return ToolResult(success=False, message=f"Error closing {app}: {str(e)}")
