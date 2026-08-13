"""Test path setup for the training and fine-tuning lesson."""

from __future__ import annotations

import sys
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))
