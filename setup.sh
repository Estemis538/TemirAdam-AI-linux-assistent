#!/usr/bin/env bash
# Темірадам Installation Script

set -e

echo "===================================================="
echo "    Темірадам — Установка и настройка системы"
echo "===================================================="
echo ""

# 1. Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не найден!"
    exit 1
fi
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅ Python $PYTHON_VERSION"

# 2. Check dependencies (Ollama, PipeWire, Quickshell)
if command -v ollama &> /dev/null; then
    echo "✅ Ollama установлен"
else
    echo "⚠️ Ollama не найден. Пожалуйста, установите Ollama."
fi

if command -v qs &> /dev/null; then
    echo "✅ Quickshell установлен"
else
    echo "⚠️ Quickshell не найден. UI будет работать в режиме fallback (notify-send)."
fi

if command -v wpctl &> /dev/null; then
    echo "✅ WirePlumber (wpctl) установлен"
else
    echo "⚠️ wpctl не найден. Управление громкостью может работать некорректно."
fi

echo ""
echo "📦 Установка Python зависимостей..."
# Install using pacman for CachyOS/Arch or pip as fallback
echo "Рекомендуется устанавливать системные пакеты через pacman:"
echo "sudo pacman -S python-numpy python-sounddevice python-httpx"
echo ""
echo "Устанавливаем pip-зависимости:"
pip install --user --break-system-packages -r requirements.txt || pip install --user -r requirements.txt

echo ""
echo "⚙️ Установка systemd-сервиса..."
mkdir -p ~/.config/systemd/user/
cp temiradam.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable temiradam.service

echo ""
echo "🤖 Скачивание LLM моделей (3 уровня)..."
if command -v ollama &> /dev/null; then
    echo ""
    echo "EASY  — qwen2.5:0.5b  (~0.5 GB)"
    ollama pull qwen2.5:0.5b || echo "⚠️ Не удалось скачать qwen2.5:0.5b"
    echo ""
    echo "MEDIUM — qwen3:4b  (~2.5 GB)"
    ollama pull qwen3:4b || echo "⚠️ Не удалось скачать qwen3:4b"
    echo ""
    echo "HARD  — qwen3:8b  (~5 GB)"
    ollama pull qwen3:8b || echo "⚠️ Не удалось скачать qwen3:8b"
else
    echo "⚠️ Ollama не найден. Установите Ollama и скачайте модели вручную:"
    echo "   ollama pull qwen2.5:0.5b"
    echo "   ollama pull qwen3:4b"
    echo "   ollama pull qwen3:8b"
fi

echo ""
echo "===================================================="
echo "✅ Установка завершена!"
echo ""
echo "Следующие шаги:"
echo "1. Запишите свой голосовой профиль:"
echo "   python3 main.py enroll"
echo ""
echo "2. Запустите ассистента:"
echo "   python3 main.py run"
echo "   (или через systemd: systemctl --user start temiradam)"
echo ""
echo "Модели LLM (3 уровня):"
echo "  EASY   = qwen2.5:0.5b   (простые команды, ~0.5 GB)"
echo "  MEDIUM = qwen3:4b       (контекстные запросы, ~2.5 GB)"
echo "  HARD   = qwen3:8b       (сложные вопросы, ~5 GB)"
echo "===================================================="
