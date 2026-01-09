#!/usr/bin/env python3
"""
Compatibility wrapper.

The implementation lives in `src/rules_checker.py`, but we keep this script to
avoid breaking existing workflows that call `rules_check.py` directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rules_checker import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

