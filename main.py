#!/usr/bin/env python3
"""
Темірадам — Point of Entry.

Main CLI to run the voice assistant, or enroll a new voice profile.
"""

from __future__ import annotations

import argparse
import logging
import sys

from core.assistant import AssistantOrchestrator
from speaker.enrollment import SpeakerEnrollment
from utils.config import load_config
from utils.logger import setup_logging

logger = logging.getLogger("temiradam.main")


def run_assistant() -> None:
    """Run the main assistant loop."""
    config = load_config("config.toml")
    
    setup_logging(
        log_dir=config.logging.log_dir,
        level=config.logging.level,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
    )
    
    logger.info("Initializing Темірадам Orchestrator...")
    orchestrator = AssistantOrchestrator(config)
    
    try:
        orchestrator.run()
    except Exception:
        logger.exception("Fatal error in main loop")
        sys.exit(1)


def run_enrollment() -> None:
    """Run the interactive voice enrollment CLI."""
    config = load_config("config.toml")
    
    setup_logging(
        log_dir=config.logging.log_dir,
        level="INFO",
    )
    
    logger.info("Starting Speaker Enrollment...")
    enrollment = SpeakerEnrollment(config.speaker_verification, config.audio)
    
    try:
        embedding_path = enrollment.enroll()
        if embedding_path:
            logger.info("Enrollment completed successfully.")
            print(f"\n✅ Голосовой профиль успешно сохранён: {embedding_path}")
        else:
            logger.warning("Enrollment failed or was cancelled.")
            print("\n❌ Не удалось создать голосовой профиль.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nОтменено пользователем.")
    except Exception as e:
        logger.exception("Enrollment error")
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Темірадам Voice Assistant")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    subparsers.add_parser("run", help="Run the assistant (default)")
    
    # Enroll command
    subparsers.add_parser("enroll", help="Record voice profile for speaker verification")
    
    args = parser.parse_args()
    
    command = args.command or "run"
    
    if command == "run":
        run_assistant()
    elif command == "enroll":
        run_enrollment()


if __name__ == "__main__":
    main()
