from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

"""
Tool and argument schema definitions.
"""

@dataclass
class ArgumentSchema:
    name: str
    type: str  # 'str', 'int', 'float', 'bool'
    required: bool = True
    description: str = ''
    default: Any = None

@dataclass
class ToolSchema:
    name: str
    description: str
    arguments: list[ArgumentSchema]
    permission: str = 'safe'  # 'safe' or 'dangerous'
    confirmation: bool = False

@dataclass
class ToolCall:
    tool: str | None
    arguments: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

@dataclass
class ToolResult:
    success: bool
    message: str
    data: Any = None
