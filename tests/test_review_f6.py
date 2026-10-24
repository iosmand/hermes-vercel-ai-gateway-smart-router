#!/usr/bin/env python3
"""Regression tests for review findings F6 (collect_desktop robustness).

Run:  HERMES_HOME=<temp> python3 tests/test_review_f6.py
No network: collect_desktop.fetch_generation is patched.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

FAILS: list[str] = []
TMP = Path(tempfile.mkdtemp(prefix="gw-f6-"))
os.environ["HERMES_HOME"] = str(TMP)
# Placeholder only: the fetch function is patched, nothing leaves the process.
os.environ["AI_GATEWAY_API_KEY"] = "test-key-not-real"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import collect_desktop as cd  # noqa: E402

LOGS = TMP / "logs"
STATE = TMP / "ai-gateway-router"
LOGS.mkdir(parents=True, exist_ok=True)
STATE.mkdir(parents=True, exist_ok=True)
LOG = LOGS / "agent.log"
OFFSET = STATE / "collect_desktop.offset"
PENDING = STATE / "collect_desktop.pending.jsonl"
MISSING = STATE / "collect_desktop.missing.jsonl"
LEDGER = STATE / "ledger_desktop.jsonl"


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def line(ts: str, sid: str, gen: str, model: str = "m/x") -> str:
    return (f"{ts},123 INFO [{sid}] agent.conversation_loop: API call #1: "
            f"model={model} provider=particle id={gen}\n")


def make_db(desktop_sessions: list[str]) -> None:
    db = sqlite3.connect(TMP / "state.db")
    db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT, source TEXT)")
    db.execute("DELETE FROM sessions")
    for sid in desktop_sessions:
        db.execute("INSERT INTO sessions VALUES (?, 'desktop')", (sid,))
    db.commit()
    db.close()


def reset_all() -> None:
    for p in (OFFSET, PENDING, MISSING, LEDGER):
        p.unlink(missing_ok=True)
    (TMP / ".env").unlink(missing_ok=True)
    LOG.write_text("", encoding="utf-8")


def ok_fetch(data_by_id: dict, calls: list | None = None):
    def _fake(gen_id, key, timeout=60.0):
        if calls is not None:
            calls.append((gen_id, key))
        if gen_id in data_by_id:
            return data_by_id[gen_id], "ok", ""
        return None, "not_ready", "HTTP 404"
    return _fake


def data_row(provider="particle"):
    return {"provider_name": provider, "latency": 100, "generation_time": 500,
            "tokens_prompt": 10, "tokens_completion": 5, "native_tokens_reasoning": 1,
            "native_tokens_cached": 2, "total_cost": 0.001}


OLD = "2026-10-24 10:00:00"     # older than the 60-minute window
# two minutes old: past the 20 s "too young" guard, well inside the 60 min window
FRESH = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 120))


# --- F6a: unreadable state.db must not advance the offset -------------------
reset_all()
LOG.write_text(line(FRESH, "sess-1", "gen_AAA000000000000000001"), encoding="utf-8")
cd.fetch_generation = ok_fetch({})
rc = cd.main([])
check("F6a: unreadable state.db stops the run", rc == 2, f"rc={rc}")
check("F6a: offset is not advanced when sessions cannot be classified",
      not OFFSET.exists() or json.loads(OFFSET.read_text()).get("offset") == 0,
      OFFSET.read_text() if OFFSET.exists() else "no offset")
check("F6a: no ledger row written", not LEDGER.exists(), "ledger exists")

# --- F6d: .env parsing (quotes + export prefix, missing file) ---------------
_saved_key = os.environ.pop("AI_GATEWAY_API_KEY", None)
try:
    check("F6d: missing .env returns None instead of crashing", cd.read_api_key() is None)
    (TMP / ".env").write_text('export AI_GATEWAY_API_KEY="sk-test-value"\n', encoding="utf-8")
    check("F6d: 'export ' and quotes are stripped",
          cd.read_api_key() == "sk-test-value", str(cd.read_api_key())[:6])
    (TMP / ".env").unlink()
finally:
    if _saved_key is not None:
        os.environ["AI_GATEWAY_API_KEY"] = _saved_key

# --- F6f: a partial last line is not consumed -------------------------------
reset_all()
make_db(["sess-1"])
cd.fetch_generation = ok_fetch({"gen_BBB000000000000000001": data_row()})
full = line(FRESH, "sess-1", "gen_BBB000000000000000001")
LOG.write_text(full[:-1], encoding="utf-8")          # no trailing newline
rc = cd.main([])
check("F6f: partial line → nothing consumed", rc == 0 and not LEDGER.exists(),
      f"rc={rc} ledger={LEDGER.exists()}")
with LOG.open("a", encoding="utf-8") as fh:
    fh.write("\n")
rc = cd.main([])
rows = [json.loads(l) for l in LEDGER.open()] if LEDGER.exists() else []
check("F6f: completed line is processed exactly once",
      rc == 0 and len(rows) == 1 and rows[0]["generation_id"] == "gen_BBB000000000000000001",
      f"rows={len(rows)}")

# --- F6c: only HTTP 404 ages out; 5xx does not -----------------------------
reset_all()
make_db(["sess-1"])
LOG.write_text(line(OLD, "sess-1", "gen_CCC000000000000000001"), encoding="utf-8")
cd.fetch_generation = ok_fetch({})
cd.main([])
pend = [json.loads(l) for l in PENDING.open()] if PENDING.exists() else []
check("F6c: 404 keeps the id pending with the log timestamp",
      len(pend) == 1 and pend[0]["ts"] == OLD and pend[0].get("last_error_kind") == "not_ready",
      str(pend))
check("F6c: a fresh 404 does not go straight to missing", not MISSING.exists(), "missing exists")

# second run: now the window is over → final attempt, then missing
rc = cd.main([])
miss = [json.loads(l) for l in MISSING.open()] if MISSING.exists() else []
check("F6c: after the window the id is written to missing once",
      rc == 0 and len(miss) == 1, f"missing={len(miss)}")
cd.main([])
miss2 = [json.loads(l) for l in MISSING.open()] if MISSING.exists() else []
check("F6c: missing writes stay idempotent", len(miss2) == 1, f"missing={len(miss2)}")


def transient_fetch(gen_id, key, timeout=60.0):
    return None, "transient", "HTTP 503"


reset_all()
make_db(["sess-1"])
LOG.write_text(line(OLD, "sess-1", "gen_DDD000000000000000001"), encoding="utf-8")
cd.fetch_generation = transient_fetch
cd.main([])
cd.main([])
pend = [json.loads(l) for l in PENDING.open()] if PENDING.exists() else []
check("F6c: 5xx never counts towards the 60-minute limit",
      len(pend) == 1 and not MISSING.exists(), f"pending={len(pend)} missing={MISSING.exists()}")


def auth_fetch(gen_id, key, timeout=60.0):
    return None, "auth", "HTTP 401"


reset_all()
make_db(["sess-1"])
LOG.write_text(line(OLD, "sess-1", "gen_EEE000000000000000001"), encoding="utf-8")
cd.fetch_generation = auth_fetch
rc = cd.main([])
check("F6c: 401/403 stops the run with a clear error", rc == 3, f"rc={rc}")

# --- F6b: rotation (inode / first line) resets the offset -------------------
reset_all()
make_db(["sess-1"])
cd.fetch_generation = ok_fetch({"gen_FFF000000000000000001": data_row()})
LOG.write_text(line(FRESH, "sess-1", "gen_FFF000000000000000001"), encoding="utf-8")
cd.main([])
first_off = json.loads(OFFSET.read_text())["offset"]
check("F6b: offset advanced after the first run", first_off > 0, str(first_off))
rotated = LOGS / "agent.log.1"
LOG.rename(rotated)
LOG.write_text(line(FRESH, "sess-1", "gen_GGG000000000000000001"), encoding="utf-8")
cd.fetch_generation = ok_fetch({"gen_GGG000000000000000001": data_row()})
rc = cd.main([])
rows = [json.loads(l) for l in LEDGER.open()]
check("F6b: a rotated log is read from the top again",
      rc == 0 and [r["generation_id"] for r in rows] ==
      ["gen_FFF000000000000000001", "gen_GGG000000000000000001"],
      str([r["generation_id"] for r in rows]))

# --- F6e: overlapping run exits 0 ------------------------------------------
reset_all()
make_db(["sess-1"])
with cd.RunLock(STATE / "collect_desktop.lock") as held:
    rc = cd.main([])
check("F6e: a second concurrent run exits 0 without doing work",
      held is not None and rc == 0 and not OFFSET.exists(), f"rc={rc}")

# --- F6e: pending is written before the offset -----------------------------
reset_all()
make_db(["sess-1"])
LOG.write_text(line(FRESH, "sess-1", "gen_HHH000000000000000001"), encoding="utf-8")
cd.fetch_generation = ok_fetch({})          # 404 → pending

real_save_offset = cd.save_offset


def boom(*a, **kw):
    raise RuntimeError("crash between the two writes")


cd.save_offset = boom
try:
    cd.main([])
    crashed = False
except RuntimeError:
    crashed = True
finally:
    cd.save_offset = real_save_offset

pend = [json.loads(l) for l in PENDING.open()] if PENDING.exists() else []
check("F6e: pending survives a crash before the offset write",
      crashed and len(pend) == 1, f"crashed={crashed} pending={len(pend)}")

# --- F6g: log timestamps are local time unless LOG_UTC=1 -------------------
import time as _time  # noqa: E402

local = cd.log_epoch("2026-10-24 10:00:00")
expected = _time.mktime(_time.strptime("2026-10-24 10:00:00", "%Y-%m-%d %H:%M:%S"))
check("F6g: log_epoch uses local time by default", local == expected, f"{local} != {expected}")
os.environ[cd.LOG_UTC_ENV] = "1"
try:
    utc = cd.log_epoch("2026-10-24 10:00:00")
finally:
    os.environ.pop(cd.LOG_UTC_ENV, None)
import calendar  # noqa: E402

check("F6g: HERMES_AI_GATEWAY_LOG_UTC=1 switches to UTC",
      utc == calendar.timegm(_time.strptime("2026-10-24 10:00:00", "%Y-%m-%d %H:%M:%S")),
      str(utc))

print(f"\n{len(FAILS)} hata" if FAILS else "\ntüm testler geçti")
sys.exit(1 if FAILS else 0)
