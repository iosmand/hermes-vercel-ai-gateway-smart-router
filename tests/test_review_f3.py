#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Regression tests for review findings F3 (stats cold cache).

Run:  HERMES_HOME=<temp> python3 tests/test_review_f3.py
No network, no API key.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

FAILS: list[str] = []
TMP = Path(tempfile.mkdtemp(prefix="gw-f3-"))
os.environ["HERMES_HOME"] = str(TMP)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugin"))

import planning  # noqa: E402

MODEL = "m/f3"


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def ep(name="p"):
    return {"provider_name": name, "quantization": "fp8", "status": 0,
            "pricing": {"prompt": 1e-7, "completion": 4e-7, "input_cache_read": 1e-8},
            "uptime_last_15m": 100, "uptime_last_1h": 100, "uptime_last_1d": 100,
            "throughput_last_1h": {"p50": 100}, "latency_last_1h": {"p50": 800},
            "tags": ["tool-use", "structured-output"], "context_length": 10**6}


# --- F3: atomic stats write (a failed write must not truncate the cache) ---
path = planning._stats_path(MODEL)
planning._write_stats(path, [ep("old")], fetched_at=time.time())
real_dump = json.dump


def boom(*a, **kw):
    raise OSError("disk full")


json.dump = boom
try:
    planning._write_stats(path, [ep("new")], fetched_at=time.time())
    raised = False
except Exception:
    raised = True
finally:
    json.dump = real_dump

data = json.loads(path.read_text(encoding="utf-8"))
check("F3: a failed stats write leaves the old cache intact",
      raised and data["endpoints"][0]["provider_name"] == "old", json.dumps(data)[:200])
check("F3: no temporary file is left behind",
      not [p for p in path.parent.iterdir() if p.name.startswith(".stats-")],
      str([p.name for p in path.parent.iterdir()]))

# --- F3: negative cache (60 s) after a failed fetch ---
calls = {"n": 0}


def failing_fetch(model, timeout=None):
    calls["n"] += 1
    raise OSError("network down")


planning._fetch_endpoints = failing_fetch
planning._NEG_CACHE_TS.clear()
path.unlink(missing_ok=True)
planning.stats(MODEL)
planning.stats(MODEL)
check("F3: failed fetch is negatively cached (1 call for 2 stats())",
      calls["n"] == 1, f"fetch çağrısı={calls['n']}")

# --- F3: cold-start timeout is 3 s (background refresh keeps 20 s) ---
check("F3: cold-start fetch timeout is 3 s",
      getattr(planning, "STATS_FETCH_TIMEOUT_S", None) == 3.0,
      str(getattr(planning, "STATS_FETCH_TIMEOUT_S", "missing")))

seen = {}


def slow_fetch(model, timeout=None):
    seen["timeout"] = timeout
    calls["n"] += 1
    time.sleep(0.2)
    return {"endpoints": [ep("fresh")]}


planning._fetch_endpoints = slow_fetch
planning._NEG_CACHE_TS.clear()
path.unlink(missing_ok=True)
calls["n"] = 0
threads = [threading.Thread(target=planning.stats, args=(MODEL,)) for _ in range(5)]
t0 = time.time()
for t in threads:
    t.start()
for t in threads:
    t.join()
elapsed = time.time() - t0
check("F3: cold start uses the 3 s timeout", seen.get("timeout") == 3.0, str(seen))
check("F3: only one sync fetch per model per process", calls["n"] == 1, f"fetch çağrısı={calls['n']}")
check("F3: other threads return immediately instead of waiting for the fetch",
      elapsed < 0.5, f"geçen süre={elapsed:.3f}s")

# --- F3: an empty endpoints list is not a valid cache ---
path.write_text(json.dumps({"fetched_at": time.time(), "endpoints": []}), encoding="utf-8")
calls["n"] = 0
planning._NEG_CACHE_TS.clear()
eps, _stale = planning.stats(MODEL)
check("F3: cached endpoints: [] triggers a real fetch",
      calls["n"] == 1 and [e["provider_name"] for e in eps] == ["fresh"],
      f"fetch çağrısı={calls['n']}, eps={len(eps)}")

print(f"\n{len(FAILS)} hata" if FAILS else "\ntüm testler geçti")
sys.exit(1 if FAILS else 0)
