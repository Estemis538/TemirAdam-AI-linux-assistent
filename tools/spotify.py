from __future__ import annotations
import logging
import subprocess
import time

from tools.schemas import ToolSchema, ArgumentSchema, ToolResult
from tools.registry import registry

"""
Spotify control tools via playerctl/MPRIS.
"""

logger = logging.getLogger("temiradam.tools.spotify")

def _run_playerctl(*args: str) -> tuple[bool, str]:
    try:
        cmd = ['playerctl', '-p', 'spotify'] + list(args)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"playerctl failed: {e.stderr}")
        return False, e.stderr.strip()
    except FileNotFoundError:
        return False, "playerctl not installed"

@registry.register(ToolSchema(
    name='spotify_play',
    description='Play music on Spotify or resume playback. Provide a query to search and play.',
    arguments=[
        ArgumentSchema(name='query', type='str', required=False, description='What to search for and play')
    ]
))
def spotify_play(query: str = "") -> ToolResult:
    logger.info(f"Spotify play called with query: '{query}'")
    
    # Check if Spotify is running, if not start it
    check = subprocess.run(['pgrep', '-x', 'spotify'], capture_output=True)
    if check.returncode != 0:
        logger.info("Spotify not running, starting it...")
        subprocess.Popen(['spotify'], start_new_session=True)
        time.sleep(2)  # Give it time to start
        
    if query:
        # Use playerctl open or dbus
        success, msg = _run_playerctl('open', f'spotify:search:{query}')
        if not success:
            logger.warning("playerctl open failed, trying dbus-send...")
            try:
                subprocess.run([
                    'dbus-send', '--print-reply', '--dest=org.mpris.MediaPlayer2.spotify',
                    '/org/mpris/MediaPlayer2', 'org.mpris.MediaPlayer2.Player.OpenUri',
                    f'string:spotify:search:{query}'
                ], check=True)
                success = True
            except Exception as e:
                msg = str(e)
                
        if success:
            return ToolResult(success=True, message=f"Searching and playing '{query}' on Spotify.")
        else:
            return ToolResult(success=False, message=f"Failed to play query: {msg}")
    else:
        success, msg = _run_playerctl('play')
        if success:
            return ToolResult(success=True, message="Resumed Spotify playback.")
        return ToolResult(success=False, message=f"Failed to resume: {msg}")

@registry.register(ToolSchema(
    name='spotify_pause',
    description='Pause Spotify playback.',
    arguments=[]
))
def spotify_pause() -> ToolResult:
    logger.info("Pausing Spotify")
    success, msg = _run_playerctl('pause')
    if success:
        return ToolResult(success=True, message="Paused Spotify.")
    return ToolResult(success=False, message=f"Failed to pause: {msg}")

@registry.register(ToolSchema(
    name='spotify_next',
    description='Skip to the next track on Spotify.',
    arguments=[]
))
def spotify_next() -> ToolResult:
    logger.info("Skipping to next track on Spotify")
    success, msg = _run_playerctl('next')
    if success:
        return ToolResult(success=True, message="Skipped to next track.")
    return ToolResult(success=False, message=f"Failed to skip: {msg}")

@registry.register(ToolSchema(
    name='spotify_previous',
    description='Go back to the previous track on Spotify.',
    arguments=[]
))
def spotify_previous() -> ToolResult:
    logger.info("Going to previous track on Spotify")
    success, msg = _run_playerctl('previous')
    if success:
        return ToolResult(success=True, message="Going to previous track.")
    return ToolResult(success=False, message=f"Failed to go to previous track: {msg}")
