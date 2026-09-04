from __future__ import annotations
import logging
import subprocess

from tools.schemas import ToolSchema, ArgumentSchema, ToolResult
from tools.registry import registry

"""
Volume control via wpctl.
"""

logger = logging.getLogger("temiradam.tools.volume")

@registry.register(ToolSchema(
    name='set_volume',
    description='Set the system volume.',
    arguments=[
        ArgumentSchema(name='volume', type='int', description='Volume level from 0 to 100')
    ]
))
def set_volume(volume: int) -> ToolResult:
    logger.info(f"Setting volume to {volume}")
    volume = max(0, min(100, volume))
    try:
        subprocess.run(['wpctl', 'set-volume', '@DEFAULT_AUDIO_SINK@', f'{volume/100:.2f}'], check=True)
        return ToolResult(success=True, message=f"Volume set to {volume}%.")
    except Exception as e:
        logger.exception(f"Error setting volume: {e}")
        return ToolResult(success=False, message=f"Failed to set volume: {str(e)}")

@registry.register(ToolSchema(
    name='get_volume',
    description='Get the current system volume.',
    arguments=[]
))
def get_volume() -> ToolResult:
    logger.info("Getting volume")
    try:
        result = subprocess.run(['wpctl', 'get-volume', '@DEFAULT_AUDIO_SINK@'], capture_output=True, text=True, check=True)
        return ToolResult(success=True, message=f"Current volume: {result.stdout.strip()}")
    except Exception as e:
        logger.exception(f"Error getting volume: {e}")
        return ToolResult(success=False, message=f"Failed to get volume: {str(e)}")

@registry.register(ToolSchema(
    name='mute',
    description='Mute the system volume.',
    arguments=[]
))
def mute() -> ToolResult:
    logger.info("Muting volume")
    try:
        subprocess.run(['wpctl', 'set-mute', '@DEFAULT_AUDIO_SINK@', '1'], check=True)
        return ToolResult(success=True, message="System muted.")
    except Exception as e:
        logger.exception(f"Error muting volume: {e}")
        return ToolResult(success=False, message=f"Failed to mute: {str(e)}")

@registry.register(ToolSchema(
    name='unmute',
    description='Unmute the system volume.',
    arguments=[]
))
def unmute() -> ToolResult:
    logger.info("Unmuting volume")
    try:
        subprocess.run(['wpctl', 'set-mute', '@DEFAULT_AUDIO_SINK@', '0'], check=True)
        return ToolResult(success=True, message="System unmuted.")
    except Exception as e:
        logger.exception(f"Error unmuting volume: {e}")
        return ToolResult(success=False, message=f"Failed to unmute: {str(e)}")
