#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Regression tests for review findings F5 (strike context).

Run:  HERMES_HOME=<temp> python3 tests/test_review_f5.py
No network, no API key (stats/rule lookups are patched).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

FAILS: list[str] = []
TMP = Path(tempfile.mkdtemp(prefix="gw-f5-"))
os.environ["HERMES_HOME"] = str(TMP)
PLUGIN = Path(__file__).resolve().parents[1] / "plugin"
sys.path.insert(0, str(PLUGIN))

# The plugin is loaded first and its own planning module is used, exactly like a
# package-style load in Hermes: patching a different copy would prove nothing.
spec = importlib.util.spec_from_file_location(
    "gwplug", PLUGIN / "__init__.py", submodule_search_locations=[str(PLUGIN)])
plugin = importlib.util.module_from_spec(spec)
sys.modules["gwplug"] = plugin
spec.loader.exec_module(plugin)
planning = plugin.planning  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


A, B = "vendor/model-a", "vendor/model-b"


def ep(name, pin=1e-7, pout=4e-7, up=100.0):
    return {"provider_name": name, "quantization": "fp8", "status": 0,
            "pricing": {"prompt": pin, "completion": pout, "input_cache_read": pin / 10},
            "uptime_last_15m": up, "uptime_last_1h": up, "uptime_last_1d": up,
            "throughput_last_1h": {"p50": 100}, "latency_last_1h": {"p50": 800},
            "tags": ["tool-use", "structured-output"], "context_length": 10**6}


RULE = {"workload": {"input_tokens": 1000, "output_tokens": 200, "cached_ratio": 0.0},
        "max_candidates": 4, "session_affinity": False, "time_value_cents_per_s": 0.15,
        "require_tools": True, "min_uptime_15m": 95, "precision": {"mode": "strict", "require": "fp8"}}

POOL = [ep("p1"), ep("p2"), ep("p3")]
planning.rule_for = lambda model: RULE
planning.stats = lambda model, ttl=600.0: (POOL, False)

prof = plugin._profile


def reset():
    for name in ("strikes.json", "demoted.json", "strike_pause.json", "sessions.json"):
        planning._save_state(name, {})
    planning.LAST_GATEWAY.clear()


# --- F5b: strikes are keyed by (model, provider) -----------------------------
reset()
for _ in range(2):
    planning.record_strike(A, None, "p1", "HTTP 503")
    st = planning._json_state("strikes.json")
    for rec in st.values():
        rec["ts"] = time.time() - planning.STRIKE_COOLDOWN_S - 1
    planning._save_state("strikes.json", st)

_keys = list(planning._strikes())
check("F5b: legacy-format state still reads back with composite keys",
      all(planning.KEY_SEP in k for k in _keys) and planning.state_key(A, "p1") not in _keys,
      f"keys={_keys} (p1 is quarantined, so it leaves strikes.json)")
dem = planning.demoted(A)
check("F5b: model A quarantine recorded for p1", planning.state_key(A, "p1") in dem, str(dem))
check("F5b: model B is NOT quarantined for p1", planning.demoted(B) == {}, str(planning.demoted(B)))

gw_b = planning.build_gateway(B, None)
check("F5b: model B still routes to p1", "p1" in gw_b.get("order", []), json.dumps(gw_b))

# --- F5a: context for another model is ignored -------------------------------
reset()
prof.build_extra_body(session_id="sess-A", model=A)      # ctx: model A, locked p1
prof.classify_api_error(None, status_code=503, message="HTTP 503", body=None, model=B)
check("F5a: a context from model A never strikes for model B",
      planning._json_state("strikes.json") == {}, str(planning._json_state("strikes.json")))

# --- F5c: routing off records no strikes ------------------------------------
reset()
os.environ["HERMES_AI_GATEWAY_ROUTING"] = "off"
try:
    prof.build_extra_body(session_id="sess-A", model=A)
    prof.classify_api_error(None, status_code=503, message="HTTP 503", body=None, model=A)
finally:
    os.environ.pop("HERMES_AI_GATEWAY_ROUTING", None)
check("F5c: routing off → no strike at all", planning._json_state("strikes.json") == {},
      str(planning._json_state("strikes.json")))

# --- F5d: candidate count survives a restart (LAST_GATEWAY empty) ------------
reset()
planning.stats = lambda model, ttl=600.0: (POOL[:2], False)   # only 2 candidates
planning.LAST_GATEWAY.clear()          # simulates a fresh process
planning.record_strike(A, None, "p1", "HTTP 503")
st = planning._json_state("strikes.json")
st[planning.state_key(A, "p1")]["ts"] = time.time() - planning.STRIKE_COOLDOWN_S - 1
planning._save_state("strikes.json", st)
hit = planning.record_strike(A, None, "p1", "HTTP 503")
dem = planning.demoted(A)
check("F5d: B4 guard works after a restart (pool recomputed from stats)",
      hit is None and dem == {}, f"hit={hit} demoted={dem}")

# 3 candidates → quarantine IS applied after the same restart
reset()
planning.stats = lambda model, ttl=600.0: (POOL, False)
planning.LAST_GATEWAY.clear()
planning.record_strike(A, None, "p1", "HTTP 503")
st = planning._json_state("strikes.json")
st[planning.state_key(A, "p1")]["ts"] = time.time() - planning.STRIKE_COOLDOWN_S - 1
planning._save_state("strikes.json", st)
hit2 = planning.record_strike(A, None, "p1", "HTTP 503")
check("F5d: with 3 candidates the quarantine is applied",
      hit2 == "p1" and planning.state_key(A, "p1") in planning.demoted(A), str(planning.demoted(A)))

# --- F5: legacy provider-keyed state is migrated ------------------------------
reset()
planning._save_state("strikes.json", {"p2": {"count": 1, "ts": time.time(), "model": A}})
migrated = planning._strikes()
check("F5: legacy provider-only strike keys migrate to (model, provider)",
      planning.state_key(A, "p2") in migrated, str(migrated))

# --- F5: timeouts and every retryable status produce a strike ----------------
for status in (500, 502, 504, 529):
    reset()
    prof.build_extra_body(session_id=f"sess-{status}", model=A)
    prof.classify_api_error(None, status_code=status, message=f"HTTP {status}", body=None, model=A)
    check(f"F5: HTTP {status} records a strike",
          planning.state_key(A, "p1") in planning._json_state("strikes.json"),
          str(planning._json_state("strikes.json")))

reset()


class TimeoutError_(TimeoutError):
    pass


prof.build_extra_body(session_id="sess-timeout", model=A)
prof.classify_api_error(TimeoutError_("read timeout"), status_code=None, message="timeout",
                        body=None, model=A)
check("F5: a timeout (no status code) records a strike",
      planning.state_key(A, "p1") in planning._json_state("strikes.json"),
      str(planning._json_state("strikes.json")))

# --- session_ttl_s: the pin's idle TTL is a rule key, not a constant ---
def _seed_pin(provider: str, age_s: float, session: str = "sess-ttl") -> None:
    planning._save_state("sessions.json", {
        session: {"model": A, "provider": provider, "ts": time.time() - age_s}})


def _order_for(ttl: float | None) -> list[str]:
    rule = {"session_affinity": True, "require_tools": False, "min_uptime_15m": 0,
            "precision": {"mode": "any"}, "max_candidates": 4,
            "time_value_cents_per_s": 0.15}
    if ttl is not None:
        rule["session_ttl_s"] = ttl
    planning.rule_for = lambda model: rule
    planning.stats = lambda model, ttl_=600.0: ([
        ep("beta", pin=3e-7), ep("alpha2", pin=1e-7)], False)   # beta is pinned but pricier
    gw = planning.build_gateway(A, "sess-ttl")
    return gw.get("only") or []


_seed_pin("beta", age_s=120.0)
order_long = _order_for(3600.0)     # build_gateway re-locks, so seed again below
_seed_pin("beta", age_s=120.0)
order_short = _order_for(60.0)
check("session_ttl_s: taze pin (TTL 3600) önde tutulur",
      bool(order_long) and order_long[0] == "beta", str(order_long))
check("session_ttl_s: TTL 60 geçmiş pini düşürür",
      bool(order_short) and order_short[0] != "beta", str(order_short))

_seed_pin("beta", age_s=0.0)
check("session_ttl_s: yeni pin okunur",
      planning.locked_provider(A, "sess-ttl", 60.0) == "beta")
check("session_ttl_s: TTL dışındaki pin yok sayılır",
      planning.locked_provider(A, "sess-ttl", 0.0) is None)

print(f"\n{len(FAILS)} hata" if FAILS else "\ntüm testler geçti")
sys.exit(1 if FAILS else 0)
