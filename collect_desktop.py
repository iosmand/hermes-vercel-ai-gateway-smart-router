#!/usr/bin/env python3
"""collect_desktop.py — collect desktop turns from agent.log and write a ledger.

Why: desktop turns bypass the transport wrapper, so they produce no ledger rows.
The "API call #N ... id=gen_..." log lines exist for every turn, and
GET /v1/generation?id=... returns the real provider/latency/token/cost data.

Usage (NOT a daemon — run every 10 minutes from cron/systemd):
    python3 collect_desktop.py            # desktop sessions only (state.db source='desktop')
    python3 collect_desktop.py --all      # every session
    python3 collect_desktop.py --limit 50 # at most 50 new ids per run
    python3 collect_desktop.py --coverage # coverage report (collects nothing)
    python3 collect_desktop.py --retry-missing  # move missing ids back to pending

Behaviour:
  - The byte offset lives in ~/.hermes/ai-gateway-router/collect_desktop.offset as
    JSON {"offset": N, "inode": I, "head_sha1": H}. A rotated/truncated log
    (different inode or first line) resets the offset to 0.
  - A trailing partial line (no newline yet) is never consumed: the offset stops
    before it so the next run sees the complete line.
  - If state.db cannot be read while desktop-only mode is on, the run stops
    WITHOUT advancing the offset (otherwise those turns are lost forever).
  - Ids younger than 20 s go to the pending file and are retried every run.
  - Only HTTP 404 counts as "not ready". 401/403 stop the run with a clear error.
    5xx, timeouts and DNS errors are transient: they are retried and do NOT count
    towards the 60-minute give-up window (age is measured from the log timestamp).
  - A single non-blocking flock prevents overlapping runs; a busy lock exits 0.
  - pending is written before the offset, and every state file is written
    atomically (temp file + os.replace). Missing ids are written idempotently.
  - Prompt/response text is never read or written.

Row schema: ts, model, agent_session_id, generation_id, provider_actual, latency_ms,
generation_time_ms, input_tokens, output_tokens, output_total_tokens,
cache_read_tokens, reasoning_tokens, cost_usd.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

API_CALL_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ INFO \[([^\]]+)\] agent\.conversation_loop: "
    r"API call #(\d+): model=(\S+) provider=(\S+).*?id=(gen_[0-9A-Za-z]+)")
MIN_AGE_S = 20.0
PENDING_MAX_AGE_S = 3600.0
TIMER_START_DEFAULT = "1970-01-01 00:00:00"  # --coverage default; narrow with --since
LEDGER_FIELDS = ("ts", "model", "agent_session_id", "generation_id", "provider_actual",
                 "latency_ms", "generation_time_ms", "input_tokens", "output_tokens",
                 "output_total_tokens", "cache_read_tokens", "reasoning_tokens", "cost_usd")

# Log timestamps are written in the host's local time. Set this to 1 only when the
# Hermes log is known to be in UTC.
LOG_UTC_ENV = "HERMES_AI_GATEWAY_LOG_UTC"


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def state_dir(home: Path | None = None) -> Path:
    d = (home or hermes_home()) / "ai-gateway-router"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------- helpers
def log_epoch(ts: str) -> float:
    """Naive log timestamp → epoch seconds (local time unless LOG_UTC_ENV=1)."""
    parsed = time.strptime(ts, "%Y-%m-%d %H:%M:%S")
    if (os.environ.get(LOG_UTC_ENV) or "").strip() in {"1", "true", "yes"}:
        import calendar
        return calendar.timegm(parsed)
    return time.mktime(parsed)


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + "-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -------------------------------------------------------------------- API key
def read_api_key(home: Path | None = None) -> str | None:
    """AI_GATEWAY_API_KEY from the environment or $HERMES_HOME/.env (never printed)."""
    key = (os.environ.get("AI_GATEWAY_API_KEY") or "").strip()
    if key:
        return key
    env = (home or hermes_home()) / ".env"
    try:
        lines = env.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if line.split("=", 1)[0].strip() != "AI_GATEWAY_API_KEY":
            continue
        value = line.split("=", 1)[1].strip() if "=" in line else ""
        return value.strip('"').strip("'") or None
    return None


# ------------------------------------------------------------------ state I/O
def log_fingerprint(log_path: Path) -> dict:
    """Identify the log file: inode + hash of its first line (rotation detector)."""
    try:
        st = log_path.stat()
        head = b""
        with log_path.open("rb") as fh:
            head = fh.readline(4096)
        return {"inode": st.st_ino, "head_sha1": hashlib.sha1(head).hexdigest()}
    except OSError:
        return {}


def load_offset(state_file: Path, log_size: int, fingerprint: dict) -> int:
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    if fingerprint:
        if data.get("inode") not in (None, fingerprint.get("inode")):
            print("log rotasyonu: inode değişti — ofset sıfırlandı")
            return 0
        if data.get("head_sha1") not in (None, fingerprint.get("head_sha1")):
            print("log rotasyonu: ilk satır değişti — ofset sıfırlandı")
            return 0
    try:
        off = int(data.get("offset") or 0)
    except (TypeError, ValueError):
        return 0
    return 0 if off > log_size else off


def save_offset(state_file: Path, offset: int, fingerprint: dict) -> None:
    payload = {"offset": offset, **({"inode": fingerprint.get("inode"),
                                     "head_sha1": fingerprint.get("head_sha1")} if fingerprint else {})}
    _atomic_write(state_file, json.dumps(payload))


def save_pending(path: Path, pending: list[dict]) -> None:
    _atomic_write(path, "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in pending))


class RunLock:
    """Non-blocking exclusive lock so two timer runs never overlap."""

    def __init__(self, path: Path):
        self.path = path
        self.fh = None

    def __enter__(self):
        self.fh = self.path.open("a+")
        try:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.close()
            self.fh = None
            return None
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(str(os.getpid()))
        self.fh.flush()
        return self

    def __exit__(self, *exc):
        if self.fh is not None:
            try:
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            finally:
                self.fh.close()
        return False


# ------------------------------------------------------------------- sessions
def desktop_sessions(home: Path) -> set[str] | None:
    """Desktop session ids, or None when state.db cannot be read."""
    db_path = home / "state.db"
    if not db_path.exists():
        return None
    try:
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            return {r[0] for r in db.execute("SELECT id FROM sessions WHERE source='desktop'")}
        finally:
            db.close()
    except sqlite3.Error as exc:
        print(f"state.db okunamadı: {exc}")
        return None


# ----------------------------------------------------------------- generation
def fetch_generation(gen_id: str, key: str, timeout: float = 60.0) -> tuple[dict | None, str, str]:
    """(data, kind, detail); kind ∈ ok | not_ready | auth | transient | fatal."""
    req = urllib.request.Request(
        f"https://ai-gateway.vercel.sh/v1/generation?id={gen_id}",
        headers={"Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None, "not_ready", "HTTP 404"
        if exc.code in (401, 403):
            return None, "auth", f"HTTP {exc.code}"
        if 500 <= exc.code < 600:
            return None, "transient", f"HTTP {exc.code}"
        return None, "fatal", f"HTTP {exc.code}"
    except Exception as exc:               # timeout, DNS, connection reset, TLS…
        return None, "transient", f"{type(exc).__name__}: {exc}"
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, dict):
        return data, "ok", ""
    return None, "transient", "beklenmeyen yanıt biçimi"


def _sum_or_none(a, b):
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


def build_row(ts: str, model: str, session_id: str, gen_id: str, data: dict) -> dict:
    return {
        "ts": ts,
        "model": model,
        "agent_session_id": session_id,
        "generation_id": gen_id,
        "provider_actual": data.get("provider_name"),
        "latency_ms": data.get("latency"),
        "generation_time_ms": data.get("generation_time"),
        "input_tokens": data.get("tokens_prompt"),
        "output_tokens": data.get("tokens_completion"),
        # Faturalanan toplam çıktı = tokens_completion + native_tokens_reasoning
        "output_total_tokens": _sum_or_none(data.get("tokens_completion"),
                                            data.get("native_tokens_reasoning")),
        "cache_read_tokens": data.get("native_tokens_cached"),
        "reasoning_tokens": data.get("native_tokens_reasoning"),
        "cost_usd": data.get("total_cost"),
    }


def write_row(ledger: Path, ts: str, model: str, sid: str, gen_id: str, data: dict) -> None:
    row = build_row(ts, model, sid, gen_id, data)
    with ledger.open("a", encoding="utf-8") as out:
        out.write(json.dumps({k: row.get(k) for k in LEDGER_FIELDS}, ensure_ascii=False) + "\n")


def seen_ids(ledger: Path) -> set[str]:
    out: set[str] = set()
    if ledger.exists():
        for line in ledger.open(encoding="utf-8"):
            try:
                out.add(json.loads(line).get("generation_id"))
            except ValueError:
                continue
    out.discard(None)
    return out


# -------------------------------------------------------------------- pending
def retry_pending(pending: list[dict], ledger: Path, key: str, now: float,
                  done: set[str], missing_file: Path) -> tuple[list[dict], int, int]:
    """Retry pending ids; give up only after a final attempt once the window closes."""
    still: list[dict] = []
    written = 0
    gave_up = 0
    for e in sorted(pending, key=lambda x: x.get("ts") or ""):
        gid = e.get("generation_id")
        if not gid or gid in done:
            continue
        data, kind, detail = fetch_generation(gid, key)
        e["attempts"] = int(e.get("attempts") or 0) + 1
        if data is not None:
            write_row(ledger, e.get("ts") or "", e.get("model") or "",
                      e.get("agent_session_id") or "", gid, data)
            done.add(gid)
            written += 1
            continue
        e["last_error"] = detail
        e["last_error_kind"] = kind
        if kind == "auth":
            raise PermissionError(f"/v1/generation yetkisiz ({detail})")
        # Age is measured from the log timestamp, and only after a real attempt.
        age = now - log_epoch(e["ts"]) if e.get("ts") else 0.0
        if kind == "not_ready" and age > PENDING_MAX_AGE_S:
            append_jsonl(missing_file, {
                "generation_id": gid,
                "first_seen_ts": e.get("first_seen_ts"),
                "ts": _iso(now),
                "attempts": e["attempts"],
                "reason": f"log damgasından {age / 60:.0f} dk sonra hâlâ 404; "
                          f"son hata: {detail}",
            })
            gave_up += 1
            continue
        still.append(e)
    return still, written, gave_up


# ------------------------------------------------------------------- coverage
def coverage(home: Path, since_str: str, only_desktop: bool) -> int:
    since = log_epoch(since_str)
    desk = desktop_sessions(home) if only_desktop else set()
    if desk is None:
        print("state.db okunamadı — kapsam raporu üretilemiyor")
        return 2
    state = state_dir(home)
    log_ids: set[str] = set()
    log_path = home / "logs" / "agent.log"
    if log_path.exists():
        for raw in log_path.open("rb"):
            m = API_CALL_RE.match(raw.decode("utf-8", "replace"))
            if not m:
                continue
            ts, sid, _n, _model, _prov, gen_id = m.groups()
            if log_epoch(ts) >= since and (not only_desktop or sid in desk):
                log_ids.add(gen_id)
    rows = load_jsonl(state / "ledger_desktop.jsonl")
    row_ids = {r.get("generation_id") for r in rows}
    rows_since = [r for r in rows if (r.get("ts") or "") >= since_str]
    pending = load_jsonl(state / "collect_desktop.pending.jsonl")
    missing = load_jsonl(state / "collect_desktop.missing.jsonl")
    pending_ids = {e.get("generation_id") for e in pending}
    missing_ids = {e.get("generation_id") for e in missing}
    uncovered = log_ids - row_ids
    print(f"pencere başlangıcı        : {since_str} (--since ile değiştirilir)")
    print(f"log'daki masaüstü gen_ id : {len(log_ids)}")
    print(f"ledger_desktop satırı     : {len(rows)} (pencerede {len(rows_since)})")
    print(f"pending id                : {len(pending_ids)}")
    print(f"missing id                : {len(missing_ids)}")
    print(f"pencere id'si kapsandı    : {len(log_ids & row_ids)}")
    print(f"kapsanmayan pencere id    : {len(uncovered)} "
          f"(pending {len(uncovered & pending_ids)}, missing {len(uncovered & missing_ids)}, "
          f"hiçbir yerde {len(uncovered - pending_ids - missing_ids)})")
    return 0


# ----------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true", help="tüm oturumlar (yalnız masaüstü değil)")
    ap.add_argument("--limit", type=int, default=0, help="bu koşulda en fazla N yeni id (0=sınırsız)")
    ap.add_argument("--coverage", action="store_true", help="kapsam raporu göster ve çık")
    ap.add_argument("--since", default=TIMER_START_DEFAULT, help="--coverage pencere başlangıcı")
    ap.add_argument("--retry-missing", action="store_true",
                    help="missing id'leri yeni bir 60 dk penceresiyle pending'e geri al")
    args = ap.parse_args(argv)

    home = hermes_home()
    if args.coverage:
        return coverage(home, args.since, not args.all)

    state = state_dir(home)
    state_file = state / "collect_desktop.offset"
    ledger = state / "ledger_desktop.jsonl"
    pending_file = state / "collect_desktop.pending.jsonl"
    missing_file = state / "collect_desktop.missing.jsonl"

    with RunLock(state / "collect_desktop.lock") as lock:
        if lock is None:
            print("başka bir collect_desktop koşusu sürüyor — çıkıldı")
            return 0
        return _run(home, state, state_file, ledger, pending_file, missing_file, args)


def _run(home: Path, state: Path, state_file: Path, ledger: Path, pending_file: Path,
         missing_file: Path, args) -> int:
    log_path = home / "logs" / "agent.log"
    if not log_path.exists():
        print(f"log yok: {log_path}")
        return 0

    pending = load_jsonl(pending_file)
    if args.retry_missing:
        revived = []
        for e in load_jsonl(missing_file):
            gid = e.get("generation_id")
            if not gid:
                continue
            revived.append({"generation_id": gid, "first_seen_ts": _iso(time.time()),
                            "attempts": 0, "ts": e.get("ts") or _iso(time.time()),
                            "model": e.get("model"), "agent_session_id": e.get("agent_session_id"),
                            "last_error": e.get("reason")})
        if revived:
            pending.extend(revived)
            save_pending(pending_file, pending)
            _atomic_write(missing_file, "")
            print(f"--retry-missing: {len(revived)} id pending'e alındı")

    desk: set[str] = set()
    if not args.all:
        desk_ids = desktop_sessions(home)
        if desk_ids is None:
            print("state.db okunamadı — ofset İLERLETİLMEDİ (masaüstü oturumları sınıflandırılamadı)")
            return 2
        desk = desk_ids

    done = seen_ids(ledger)
    missing_ids = {e.get("generation_id") for e in load_jsonl(missing_file)}
    skip = done | {e.get("generation_id") for e in pending} | missing_ids
    now = time.time()

    fingerprint = log_fingerprint(log_path)
    offset = load_offset(state_file, log_path.stat().st_size, fingerprint)

    # 1) pending retries (needs the key)
    retried_ok = 0
    gave_up = 0
    if pending and any(e.get("generation_id") not in done for e in pending):
        key = read_api_key(home)
        if not key:
            print("AI_GATEWAY_API_KEY yok — pending yeniden denenemedi")
            return 2
        try:
            pending, retried_ok, gave_up = retry_pending(pending, ledger, key, now, done, missing_file)
        except PermissionError as exc:
            print(f"DURDURULDU: {exc} — anahtarı kontrol edin; ofset değişmedi")
            return 3
    skip = done | {e.get("generation_id") for e in pending} | missing_ids

    # 2) new ids from the log
    written = 0
    stop_offset = offset
    key = None
    with log_path.open("rb") as fh:
        fh.seek(offset)
        consumed = offset
        for raw in fh:
            if not raw.endswith(b"\n"):
                break          # partial last line: never consume it
            consumed += len(raw)
            m = API_CALL_RE.match(raw.decode("utf-8", "replace"))
            if not m:
                stop_offset = consumed
                continue
            ts, sid, _n, model, _prov, gen_id = m.groups()
            if gen_id in skip or (not args.all and sid not in desk):
                stop_offset = consumed
                continue
            if args.limit and written + retried_ok >= args.limit:
                break          # limit: bu satır sonraki koşuma (ofset ilerlemez)
            if now - log_epoch(ts) < MIN_AGE_S:
                pending.append({"generation_id": gen_id, "first_seen_ts": _iso(now),
                                "attempts": 0, "ts": ts, "model": model,
                                "agent_session_id": sid, "last_error": None})
                skip.add(gen_id)
                stop_offset = consumed
                continue
            if key is None:
                key = read_api_key(home)
                if not key:
                    print("AI_GATEWAY_API_KEY yok — yeni id'ler sorgulanamadı; "
                          "ofset İLERLETİLMEDİ")
                    return 2
            data, kind, detail = fetch_generation(gen_id, key)
            if data is not None:
                write_row(ledger, ts, model, sid, gen_id, data)
                done.add(gen_id)
                written += 1
            elif kind == "auth":
                print(f"DURDURULDU: /v1/generation yetkisiz ({detail}); ofset değişmedi")
                return 3
            else:
                pending.append({"generation_id": gen_id, "first_seen_ts": _iso(now),
                                "attempts": 1, "ts": ts, "model": model,
                                "agent_session_id": sid, "last_error": detail,
                                "last_error_kind": kind})
                skip.add(gen_id)
            stop_offset = consumed

    # pending BEFORE offset: a crash in between re-reads a few lines instead of
    # losing them (seen_ids skips anything already written).
    save_pending(pending_file, pending)
    save_offset(state_file, stop_offset, fingerprint)
    print(f"işlendi: {written} yeni + {retried_ok} pending | pending: {len(pending)} | "
          f"missing'e bırakılan: {gave_up} | ofset {offset} -> {stop_offset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
