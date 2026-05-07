#!/usr/bin/env bash
# Creates a virtual environment and installs dependencies.
# Usage: bash setup.sh
set -e

PYTHON="${PYTHON:-python}"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv venv
fi

# Pick the right activate path for the current OS
if [ -f "venv/Scripts/activate" ]; then
    # Windows (Git Bash / WSL with Windows venv)
    ACTIVATE="venv/Scripts/activate"
    PIP="venv/Scripts/pip"
else
    # macOS / Linux
    ACTIVATE="venv/bin/activate"
    PIP="venv/bin/pip"
fi

echo "Upgrading pip..."
"$PIP" install --upgrade pip

echo "Installing dependencies..."
"$PIP" install -r requirements.txt

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source $ACTIVATE"
