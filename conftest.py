"""Pytest root conftest — adds project root to sys.path for `src.*` imports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
