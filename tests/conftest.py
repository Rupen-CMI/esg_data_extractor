"""
conftest.py — pytest shared fixtures/setup for agentic_estimation's
deterministic unit-test suite (Tier-0 validators, Confidence Gate, critic
panel logic, verifier orchestration, v5 saturation, v8 reconcile).

All tests in this directory are pure/synthetic -- no live network calls, no
LLM calls (critic_panel/pillar_extractors internals are exercised via
monkeypatched functions, never real zen_client calls). A few tests import
modules that construct a DB URL at import time (e.g. peer_anchor.py via
formula_estimator.py) -- .env is loaded here so ASYNC_DB_URL is available,
matching the project's existing env-loading convention.
"""
import os
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env", override=True)
except Exception:
    pass
