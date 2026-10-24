#!/usr/bin/env python3
"""Regression tests for review findings F2 (scoring / filter input).

Run:  HERMES_HOME=<temp> python3 tests/test_review_f2.py
No network, no API key.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path

FAILS: list[str] = []
TMP = Path(tempfile.mkdtemp(prefix="gw-f2-"))
os.environ["HERMES_HOME"] = str(TMP)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugin"))

import planning  # noqa: E402

WL = {"input_tokens": 1000, "output_tokens": 200, "cached_ratio": 0.0}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def ep(provider, quant="fp8", up=100.0, tps=100, ttft=800, status=0, tools=True, pin=1e-7, pout=4e-7):
    e = {"provider_name": provider, "quantization": quant, "status": status,
         "pricing": {"prompt": pin, "completion": pout, "input_cache_read": pin / 10},
         "tags": (["tool-use", "structured-output"] if tools else ["structured-output"]),
         "context_length": 1_000_000}
    if up is not None:
        e.update({"uptime_last_15m": up, "uptime_last_1h": up, "uptime_last_1d": up})
    e["throughput_last_1h"] = {"p50": tps} if tps is not None else {}
    e["latency_last_1h"] = {"p50": ttft} if ttft is not None else {}
    return e


RULE = {"workload": WL, "max_candidates": 4, "session_affinity": False,
        "time_value_cents_per_s": 0.5, "require_tools": True, "min_uptime_15m": 95,
        "precision": {"mode": "strict", "require": "fp8"}}


# --- F2a: missing tps/ttft must not look like "zero wait" ---
eps_a = [ep("measured", tps=200, ttft=500), ep("unmeasured", tps=None, ttft=None)]
ranked = planning.rank("m/f2a", RULE, eps_a, {})
order = [r["provider"] for r in ranked]
check("F2a: provider without metrics does not win on price alone",
      order[:1] == ["measured"], json.dumps(ranked))

# --- F2b: uptime 0 / NaN never beats a healthy provider and never passes the floor ---
eps_b = [ep("zero-up", up=0.0), ep("half-up", up=50.0)]
ranked_b = planning.rank("m/f2b", {**RULE, "min_uptime_15m": None}, eps_b, {})
check("F2b: 0% uptime ranks behind 50% uptime",
      [r["provider"] for r in ranked_b][:1] == ["half-up"], json.dumps(ranked_b))

eps_nan = [ep("nan-up", up=float("nan")), ep("healthy", up=100.0)]
ranked_nan = planning.rank("m/f2b2", RULE, eps_nan, {})
check("F2b: NaN uptime fails the uptime floor",
      [r["provider"] for r in ranked_nan] == ["healthy"], json.dumps(ranked_nan))

# --- F2c: one malformed endpoint must not turn routing off for the model ---
eps_c = [ep("good"), ep("garbage", status="high")]
planning.rule_for = lambda model: RULE
planning.stats = lambda model, ttl=600.0: (eps_c, False)
gw_c = planning.build_gateway("m/f2c", None)
check("F2c: malformed endpoint is skipped, routing keeps working",
      gw_c.get("order") and "good" in gw_c["order"] and "garbage" not in gw_c["order"],
      json.dumps(gw_c))

# --- F2d: quantization is matched case-insensitively ---
eps_d = [ep("upper", quant="FP8"), ep("lower", quant="fp8")]
ranked_d = planning.rank("m/f2d", RULE, eps_d, {})
check("F2d: 'FP8' is treated as fp8 in the precision filter",
      sorted(r["provider"] for r in ranked_d) == ["lower", "upper"], json.dumps(ranked_d))

# --- F2: require_metrics excludes providers without metrics when enabled ---
eps_e = [ep("measured", tps=200, ttft=500), ep("unmeasured", tps=None, ttft=None)]
ranked_e = planning.rank("m/f2e", {**RULE, "require_metrics": True}, eps_e, {})
check("F2: require_metrics=true drops providers without metrics",
      [r["provider"] for r in ranked_e] == ["measured"], json.dumps(ranked_e))

print(f"\n{len(FAILS)} hata" if FAILS else "\ntüm testler geçti")
sys.exit(1 if FAILS else 0)
