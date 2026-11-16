"""Fork-only history audit, excluded from the production PR."""

import importlib.util
import json
from pathlib import Path

import pytest


def test_000_tdmpc_complete_public_history(capsys: pytest.CaptureFixture[str]) -> None:
    path = Path(__file__).parents[3] / "contribution-validation/tdmpc-warm-start/audit_history.py"
    spec = importlib.util.spec_from_file_location("tdmpc_history_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.audit()
    with capsys.disabled():
        print("TDMPC_HISTORY_AUDIT_BEGIN", flush=True)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        print("TDMPC_HISTORY_AUDIT_END", flush=True)
    assert result["complete"], result["parse_failures"]
