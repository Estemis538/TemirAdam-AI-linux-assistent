from __future__ import annotations
import logging
from datetime import datetime

from tools.schemas import ToolSchema, ToolResult
from tools.registry import registry

"""
Time and date tools.
"""

logger = logging.getLogger("temiradam.tools.time_tools")

@registry.register(ToolSchema(
    name='get_time',
    description='Get the current time.',
    arguments=[]
))
def get_time() -> ToolResult:
    now = datetime.now()
    logger.info("Getting current time")
    return ToolResult(success=True, message=f"Сейчас {now.strftime('%H:%M')}")

@registry.register(ToolSchema(
    name='get_date',
    description='Get the current date.',
    arguments=[]
))
def get_date() -> ToolResult:
    now = datetime.now()
    logger.info("Getting current date")
    
    months = [
        "января", "февраля", "марта", "апреля", "мая", "июня",
        "июля", "августа", "сентября", "октября", "ноября", "декабря"
    ]
    weekdays = [
        "понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"
    ]
    
    day = now.day
    month_name = months[now.month - 1]
    year = now.year
    weekday_name = weekdays[now.weekday()]
    
    date_str = f"{day} {month_name} {year} года, {weekday_name}"
    return ToolResult(success=True, message=f"Сегодня {date_str}")
