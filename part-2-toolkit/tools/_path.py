"""Add the part-2-toolkit root to sys.path so `from helpers...` resolves
when scripts in this directory are run as `python tools/<script>.py`.

Imported for side effects only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
