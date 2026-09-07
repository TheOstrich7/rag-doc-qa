"""把项目根目录加到 sys.path，让 tests/*.py 可以直接 import ragdoc.*。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
