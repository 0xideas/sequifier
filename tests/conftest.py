"""Use the working tree, independently of any installed Sequifier version."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
