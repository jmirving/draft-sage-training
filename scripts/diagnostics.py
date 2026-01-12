"""CLI entry point for DraftSage training diagnostics."""

from __future__ import annotations

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from draft_sage_training.diagnostics import main


if __name__ == "__main__":
    raise SystemExit(main())
