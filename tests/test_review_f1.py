#!/usr/bin/env python3
"""Regression tests for review findings F1 (rank fallback / has consistency).

Run:  HERMES_HOME=<temp> python3 tests/test_review_f1.py
No network, no API key.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

FAILS: list[str] = []
TMP = Path(tempfile.mkdtemp(prefix="gw-f1-"))
os.environ["HERMES_HOME"] = str(TMP)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugin"))

import planning  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def ep(provider, quant="fp8", up=100.0, tools=True, status=0, ctx=1_000_000, pin=1e-7, pout=4e-7):
    tags = ["structured-output"]
    if tools:
        tags.insert(0, "tool-use")
    return {
        "provider_name": provider, "quantization": quant, "status": status,
        "pricing": {"prompt": pin, "completion": pout, "input_cache_read": pin / 10},
        "uptime_last_15m": up, "uptime_last_1h": up, "uptime_last_1d": up,
        "throughput_last_1h": {"p50": 100}, "latency_last_1h": {"p50": 800},
        "tags": tags, "context_length": ctx,
    }


def rule(**kw):
    base = {
        "workload": {"input_tokens": 1000, "output_tokens": 200, "cached_ratio": 0.0},
        "max_candidates": 4, "session_affinity": False, "time_value_cents_per_s": 0.15,
        "require_tools": True, "min_uptime_15m": 95,
        "precision": {"mode": "auto"},
    }
    base.update(kw)
    return base


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


capture = Capture()
planning.log.addHandler(capture)
planning.log.setLevel(logging.DEBUG)
plugin = None


# --- F1a: precision is never relaxed by the fallback ---
eps = [ep("fp4-prov", quant="fp4", up=10.0, tools=True)]   # fails precision (and floor)
r = rule(min_uptime_15m=95, precision={"mode": "strict", "require": "fp8"})
ranked = planning.rank("m/f1", r, eps, {})
check("F1a: fallback does not revive a provider that fails precision", ranked == [], json.dumps(ranked))

planning.rule_for = lambda model: r
planning.stats = lambda model, ttl=600.0: (eps, False)
gw = planning.build_gateway("m/f1", None)
check("F1a: build_gateway returns {} when every candidate is filtered", gw == {}, json.dumps(gw))
check("F1a: a WARNING is logged once instead of silent routing-off",
      any(rec.levelno == logging.WARNING and "m/f1" in rec.getMessage() for rec in capture.records),
      str([rec.getMessage() for rec in capture.records]))

# --- F1b: require_tools is never relaxed ---
eps_b = [ep("untooled", quant="fp8", up=100.0, tools=False)]
planning.stats = lambda model, ttl=600.0: (eps_b, False)
ranked_b = planning.rank("m/f1b", rule(require_tools=True, min_uptime_15m=0), eps_b, {})
check("F1b: no tool-use tag → not selected even as last resort", ranked_b == [], json.dumps(ranked_b))

# --- F1c: strict + allow_unknown keeps 'only' consistent with 'has' ---
eps_c = [ep("declared", quant="fp8", up=100.0), ep("undeclared", quant=None, up=100.0)]
r_c = rule(require_tools=False, min_uptime_15m=0,
           precision={"mode": "strict", "require": "fp8", "allow_unknown": True})
planning.rule_for = lambda model: r_c
planning.stats = lambda model, ttl=600.0: (eps_c, False)
gw_c = planning.build_gateway("m/f1c", None)
h_c = gw_c.get("has") or []
contradiction = bool(h_c) and any(p == "undeclared" for p in gw_c.get("only", []))
check("F1c: 'has' never rejects a provider that 'only' allows",
      not contradiction, json.dumps(gw_c))
check("F1c: undeclared provider still routable (only keeps it)",
      "undeclared" in gw_c.get("only", []), json.dumps(gw_c))

# --- F1e: HTTP 400 no_capable_providers_available is reported, not swallowed ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugin"))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("gw_plugin_f1",
                                             Path(__file__).resolve().parents[1] / "plugin" / "__init__.py")
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)
plugin.log.addHandler(capture)
plugin.log.setLevel(logging.DEBUG)
capture.records.clear()
plugin._profile.classify_api_error(None, status_code=400,
                                   message="no_capable_providers_available",
                                   body={"error": {"message": "no_capable_providers_available"}},
                                   model="m/f1c")
msgs = [rec.getMessage() for rec in capture.records]
check("F1e: routing-config 400 is logged as a warning",
      any("no_capable_providers_available" in m or "routing" in m.lower() for m in msgs),
      str(msgs))

print(f"\n{len(FAILS)} hata" if FAILS else "\ntüm testler geçti")
sys.exit(1 if FAILS else 0)
