#!/usr/bin/env python3
"""collect_desktop.py — masaüstü turlarını agent.log'dan toplayıp /v1/generation ile ledger yazar.

Neden: masaüstü turları bizim transport sarmalayıcımızdan GEÇMEDİĞİ için ledger.jsonl'e
satır yazmıyor; ama "API call #N ... id=gen_..." log satırları her tur için var ve
GET /v1/generation?id=... gerçek sağlayıcı/gecikme/token/maliyet verisini veriyor.

Kullanım (daemon DEĞİL — 10 dakikada bir cron/timer ile):
    python3 collect_desktop.py            # yalnız masaüstü oturumları (state.db source='desktop')
    python3 collect_desktop.py --all      # tüm oturumlar
    python3 collect_desktop.py --limit 50 # tur başına en fazla 50 yeni id
    python3 collect_desktop.py --coverage # kapsam raporu (toplama yapmaz)

Davranış:
  - agent.log bayt ofseti ~/.hermes/ai-gateway-router/collect_desktop.offset içinde tutulur
    (JSON {"offset": N}); dosya dönerse (boyut < ofset) başa sarılır.
  - Yeni bir gen_ id'si için log satırının damgasından en az 20 saniye geçmeden
    sorgulanmaz; 20 sn'den genç id'ler pending dosyasına alınır.
  - /v1/generation 404 (veya başka hata) dönerse id PENDING dosyasına yazılır
    (collect_desktop.pending.jsonl: generation_id, first_seen_ts, attempts + satır
    üstverisi) ve HER KOŞUMDA yeniden denenir. 60 dakikada bulunamazsa id
    collect_desktop.missing.jsonl'e nedeniyle yazılır ve bırakılır.
  - ledger_desktop.jsonl'de satırı olan id TEKRAR sorgulanmaz; pending/missing id'ler de.
  - Prompt/yanıt metni ASLA okunmaz/yazılmaz.

Satır şeması: ts, model, agent_session_id, generation_id, provider_actual, latency_ms,
generation_time_ms, input_tokens, output_tokens, output_total_tokens, cache_read_tokens,
reasoning_tokens, cost_usd.
"""
from __future__ import annotations

import argparse
import calendar
import json
import re
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

API_CALL_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ INFO \[([^\]]+)\] agent\.conversation_loop: "
    r"API call #(\d+): model=(\S+) provider=(\S+).*?id=(gen_[0-9A-Za-z]+)")
MIN_AGE_S = 20.0
PENDING_MAX_AGE_S = 3600.0
TIMER_START_DEFAULT = "1970-01-01 00:00:00"  # --coverage varsayılanı; operatör --since ile daraltabilir
LEDGER_FIELDS = ("ts", "model", "agent_session_id", "generation_id", "provider_actual",
                 "latency_ms", "generation_time_ms", "input_tokens", "output_tokens",
                 "output_total_tokens", "cache_read_tokens", "reasoning_tokens", "cost_usd")


def hermes_home() -> Path:
    import os
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def api_key() -> str:
    import os
    key = (os.environ.get("AI_GATEWAY_API_KEY") or "").strip()
    if key:
        return key
    env = hermes_home() / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.split("=", 1)[0].strip() == "AI_GATEWAY_API_KEY":
            return line.split("=", 1)[1].strip()
    raise SystemExit("AI_GATEWAY_API_KEY bulunamadı (env veya %s)" % env)


def desktop_sessions(home: Path) -> set[str]:
    try:
        db = sqlite3.connect(f"file:{home / 'state.db'}?mode=ro", uri=True)
        return {r[0] for r in db.execute("SELECT id FROM sessions WHERE source='desktop'")}
    except sqlite3.Error:
        return set()


def load_offset(state_file: Path, log_size: int) -> int:
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        off = int(data.get("offset") or 0)
        return 0 if off > log_size else off
    except (OSError, ValueError, KeyError):
        return 0


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


def fetch_generation(gen_id: str, key: str, timeout: float = 60.0) -> tuple[dict | None, str | None]:
    """(data, hata_nedeni) döner; data None ise neden doludur."""
    req = urllib.request.Request(
        f"https://ai-gateway.vercel.sh/v1/generation?id={gen_id}",
        headers={"Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except Exception as exc:
        return None, str(exc)  # örn. "HTTP Error 404: Not Found"
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, dict):
        return data, None
    return None, "beklenmeyen yanıt biçimi"


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
        # Kanıt: faturalanan toplam çıktı = tokens_completion + native_tokens_reasoning
        # (5 modelde liste fiyatıyla %0.00 sapma; Hermes log out da bu toplama eşit)
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


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _epoch(iso_ts: str) -> float:
    return calendar.timegm(time.strptime(iso_ts, "%Y-%m-%dT%H:%M:%SZ"))


def retry_pending(pending: list[dict], ledger: Path, key: str, now: float,
                  done: set[str], missing_file: Path) -> tuple[list[dict], int, int]:
    """Pending id'leri yeniden dener; 60 dk'yı geçenleri missing'e yazar."""
    still: list[dict] = []
    written = 0
    gave_up = 0
    for e in sorted(pending, key=lambda x: x.get("first_seen_ts") or ""):
        gid = e.get("generation_id")
        if not gid or gid in done:
            continue  # bu arada satırı yazıldı → pending'den düş
        if now - _epoch(e["first_seen_ts"]) > PENDING_MAX_AGE_S:
            append_jsonl(missing_file, {
                "generation_id": gid,
                "first_seen_ts": e["first_seen_ts"],
                "ts": _iso(now),
                "attempts": e.get("attempts", 0),
                "reason": f"60 dk içinde /v1/generation bulunamadı; son hata: {e.get('last_error')}",
            })
            gave_up += 1
            continue
        data, err = fetch_generation(gid, key)
        e["attempts"] = int(e.get("attempts") or 0) + 1
        if data is not None:
            write_row(ledger, e.get("ts") or "", e.get("model") or "",
                      e.get("agent_session_id") or "", gid, data)
            done.add(gid)
            written += 1
        else:
            e["last_error"] = err
            still.append(e)
    return still, written, gave_up


def coverage(home: Path, since_str: str, only_desktop: bool) -> int:
    since = calendar.timegm(time.strptime(since_str, "%Y-%m-%d %H:%M:%S"))
    desk = set() if not only_desktop else desktop_sessions(home)
    state_dir = home / "ai-gateway-router"
    log_ids: set[str] = set()
    log_path = home / "logs" / "agent.log"
    if log_path.exists():
        for raw in log_path.open("rb"):
            m = API_CALL_RE.match(raw.decode("utf-8", "replace"))
            if not m:
                continue
            ts, sid, _n, _model, _prov, gen_id = m.groups()
            if calendar.timegm(time.strptime(ts, "%Y-%m-%d %H:%M:%S")) >= since and \
                    (not only_desktop or sid in desk):
                log_ids.add(gen_id)
    rows = load_jsonl(state_dir / "ledger_desktop.jsonl")
    row_ids = {r.get("generation_id") for r in rows}
    rows_since = [r for r in rows if (r.get("ts") or "") >= since_str]
    pending = load_jsonl(state_dir / "collect_desktop.pending.jsonl")
    missing = load_jsonl(state_dir / "collect_desktop.missing.jsonl")
    pending_ids = {e.get("generation_id") for e in pending}
    missing_ids = {e.get("generation_id") for e in missing}
    uncovered = log_ids - row_ids
    print(f"pencere başlangıcı        : {since_str} (timer başlangıcı; --since ile ezilir)")
    print(f"log'daki masaüstü gen_ id : {len(log_ids)}")
    print(f"ledger_desktop satırı     : {len(rows)} (pencerede {len(rows_since)})")
    print(f"pending id                : {len(pending_ids)}")
    print(f"missing id                : {len(missing_ids)}")
    print(f"pencere id'si kapsandı    : {len(log_ids & row_ids)}")
    print(f"kapsanmayan pencere id    : {len(uncovered)} "
          f"(pending {len(uncovered & pending_ids)}, missing {len(uncovered & missing_ids)}, "
          f"hiçbir yerde {len(uncovered - pending_ids - missing_ids)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true", help="yalnız masaüstü değil, tüm oturumlar")
    ap.add_argument("--limit", type=int, default=0, help="bu koşulda en fazla N yeni id (0=sınırsız)")
    ap.add_argument("--coverage", action="store_true", help="kapsam raporu göster ve çık")
    ap.add_argument("--since", default=TIMER_START_DEFAULT, help="--coverage pencere başlangıcı")
    args = ap.parse_args()

    home = hermes_home()
    if args.coverage:
        return coverage(home, args.since, not args.all)

    log_path = home / "logs" / "agent.log"
    state_dir = home / "ai-gateway-router"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "collect_desktop.offset"
    ledger = state_dir / "ledger_desktop.jsonl"
    pending_file = state_dir / "collect_desktop.pending.jsonl"
    missing_file = state_dir / "collect_desktop.missing.jsonl"

    log_size = log_path.stat().st_size
    offset = load_offset(state_file, log_size)
    done = seen_ids(ledger)
    pending = load_jsonl(pending_file)
    missing_ids = {e.get("generation_id") for e in load_jsonl(missing_file)}
    skip = done | {e.get("generation_id") for e in pending} | missing_ids
    desk = set() if args.all else desktop_sessions(home)
    key = api_key()
    now = time.time()

    # 1) pending yeniden deneme (60 dk sınırı ile)
    pending, retried_ok, gave_up = retry_pending(pending, ledger, key, now, done, missing_file)
    skip = done | {e.get("generation_id") for e in pending} | missing_ids

    # 2) log'dan yeni id'ler
    written = 0
    stop_offset = offset
    with log_path.open("rb") as fh:
        fh.seek(offset)
        consumed = offset
        for raw in fh:
            consumed += len(raw)
            m = API_CALL_RE.match(raw.decode("utf-8", "replace"))
            if not m:
                continue
            ts, sid, _n, model, _prov, gen_id = m.groups()
            if gen_id in skip or (not args.all and sid not in desk):
                stop_offset = consumed
                continue
            epoch = calendar.timegm(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
            if args.limit and written + retried_ok >= args.limit:
                break  # limit: bu satır sonraki koşuma (ofset ilerlemez)
            if now - epoch < MIN_AGE_S:
                # genç: hemen sorgulama; pending'e al, ofset ilerlesin (kayıp yok)
                pending.append({"generation_id": gen_id, "first_seen_ts": _iso(now),
                                "attempts": 0, "ts": ts, "model": model,
                                "agent_session_id": sid, "last_error": None})
                skip.add(gen_id)
                stop_offset = consumed
                continue
            data, err = fetch_generation(gen_id, key)
            if data is not None:
                write_row(ledger, ts, model, sid, gen_id, data)
                done.add(gen_id)
                written += 1
            else:
                pending.append({"generation_id": gen_id, "first_seen_ts": _iso(now),
                                "attempts": 1, "ts": ts, "model": model,
                                "agent_session_id": sid, "last_error": err})
                skip.add(gen_id)
            stop_offset = consumed

    state_file.write_text(json.dumps({"offset": stop_offset, "log": str(log_path)}),
                          encoding="utf-8")
    with pending_file.open("w", encoding="utf-8") as fh:
        for e in pending:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"işlendi: {written} yeni + {retried_ok} pending | pending: {len(pending)} | "
          f"missing'e bırakılan: {gave_up} | ofset {offset} -> {stop_offset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
