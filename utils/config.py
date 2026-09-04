"""
Темірадам — Configuration loader.

Loads config.toml into typed dataclasses for safe access throughout the app.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("temiradam.utils.config")


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 30
    silence_threshold_ms: int = 800
    max_listen_seconds: int = 10
    device: str = ""


@dataclass
class WakeWordConfig:
    enabled: bool = True
    strategy: str = "whisper_kws"
    whisper_model: str = "tiny"
    vad_aggressiveness: int = 2
    min_speech_ms: int = 300


@dataclass
class SpeakerVerificationConfig:
    enabled: bool = True
    threshold: float = 0.70
    profile_dir: str = "data/speaker_profile"
    min_samples: int = 3
    max_samples: int = 5


@dataclass
class STTConfig:
    enabled: bool = True
    engine: str = "faster_whisper"
    model: str = "small"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5


@dataclass
class LLMTierConfig:
    enabled: bool = True
    model: str = ""
    max_memory_gb: float = 0.0


@dataclass
class LLMConfig:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 512
    timeout_seconds: int = 30
    keep_alive: str = "0"
    
    easy: LLMTierConfig = field(default_factory=lambda: LLMTierConfig(model="qwen2.5:0.5b"))
    medium: LLMTierConfig = field(default_factory=lambda: LLMTierConfig(model="qwen3:4b"))
    hard: LLMTierConfig = field(default_factory=lambda: LLMTierConfig(model="qwen3:8b"))


@dataclass
class TTSConfig:
    enabled: bool = True
    engine: str = "piper"
    voice_ru: str = "ru_RU-denis-medium"
    voice_kk: str = "kk_KZ-issai-high"
    output_device: str = ""
    speed: float = 1.0


@dataclass
class QuickshellConfig:
    enabled: bool = True
    socket_path: str = "/run/user/1000/quickshell"
    fallback_notify: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "logs"
    max_bytes: int = 10485760
    backup_count: int = 3


@dataclass
class ToolAppEntry:
    name: str
    command: str


@dataclass
class ToolConfig:
    enabled: bool = False
    description: str = ""
    permission: str = "safe"
    confirmation: bool = False
    allowed_apps: list[ToolAppEntry] = field(default_factory=list)


@dataclass
class AssistantConfig:
    name: str = "Темірадам"
    wake_word: str = "темірадам"
    wake_word_variants: list[str] = field(
        default_factory=lambda: ["темір адам", "темирадам", "темир адам"]
    )
    language: str = "ru"


@dataclass
class LanguagesConfig:
    default: str = "ru"
    supported: list[str] = field(default_factory=lambda: ["ru", "kk"])


@dataclass
class TemirAdamConfig:
    """Root configuration object."""

    assistant: AssistantConfig = field(default_factory=AssistantConfig)
    languages: LanguagesConfig = field(default_factory=LanguagesConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    wakeword: WakeWordConfig = field(default_factory=WakeWordConfig)
    speaker_verification: SpeakerVerificationConfig = field(
        default_factory=SpeakerVerificationConfig
    )
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    quickshell: QuickshellConfig = field(default_factory=QuickshellConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tools: dict[str, ToolConfig] = field(default_factory=dict)

    # Base directory of the project (set at load time)
    base_dir: Path = field(default_factory=lambda: Path.cwd())


def _parse_tool_config(raw: dict[str, Any]) -> ToolConfig:
    """Parse a single tool configuration from raw TOML dict."""
    allowed_apps_raw = raw.get("allowed_apps", [])
    allowed_apps = []
    for entry in allowed_apps_raw:
        if isinstance(entry, dict):
            allowed_apps.append(
                ToolAppEntry(
                    name=entry.get("name", ""),
                    command=entry.get("command", ""),
                )
            )
    return ToolConfig(
        enabled=raw.get("enabled", False),
        description=raw.get("description", ""),
        permission=raw.get("permission", "safe"),
        confirmation=raw.get("confirmation", False),
        allowed_apps=allowed_apps,
    )


def _dataclass_from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Create a dataclass instance from a dict, ignoring unknown keys."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def load_config(config_path: str | Path) -> TemirAdamConfig:
    """
    Load configuration from a TOML file.

    Args:
        config_path: Path to config.toml

    Returns:
        Fully populated TemirAdamConfig
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("Config file not found: %s, using defaults", config_path)
        return TemirAdamConfig()

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    config = TemirAdamConfig()
    config.base_dir = config_path.parent

    # Parse each section
    if "assistant" in raw:
        config.assistant = _dataclass_from_dict(AssistantConfig, raw["assistant"])

    if "languages" in raw:
        config.languages = _dataclass_from_dict(LanguagesConfig, raw["languages"])

    if "audio" in raw:
        config.audio = _dataclass_from_dict(AudioConfig, raw["audio"])

    if "wakeword" in raw:
        config.wakeword = _dataclass_from_dict(WakeWordConfig, raw["wakeword"])

    if "speaker_verification" in raw:
        config.speaker_verification = _dataclass_from_dict(
            SpeakerVerificationConfig, raw["speaker_verification"]
        )

    if "stt" in raw:
        config.stt = _dataclass_from_dict(STTConfig, raw["stt"])

    if "llm" in raw:
        base_llm_data = {k: v for k, v in raw["llm"].items() if not isinstance(v, dict)}
        config.llm = _dataclass_from_dict(LLMConfig, base_llm_data)
        
        if "easy" in raw["llm"]:
            config.llm.easy = _dataclass_from_dict(LLMTierConfig, raw["llm"]["easy"])
        if "medium" in raw["llm"]:
            config.llm.medium = _dataclass_from_dict(LLMTierConfig, raw["llm"]["medium"])
        if "hard" in raw["llm"]:
            config.llm.hard = _dataclass_from_dict(LLMTierConfig, raw["llm"]["hard"])

    if "tts" in raw:
        config.tts = _dataclass_from_dict(TTSConfig, raw["tts"])

    if "quickshell" in raw:
        config.quickshell = _dataclass_from_dict(QuickshellConfig, raw["quickshell"])

    if "logging" in raw:
        config.logging = _dataclass_from_dict(LoggingConfig, raw["logging"])

    # Parse tools
    if "tools" in raw:
        for tool_name, tool_raw in raw["tools"].items():
            if isinstance(tool_raw, dict):
                config.tools[tool_name] = _parse_tool_config(tool_raw)

    logger.info("Configuration loaded from %s", config_path)
    logger.info(
        "Enabled tools: %s",
        [name for name, tc in config.tools.items() if tc.enabled],
    )

    return config
