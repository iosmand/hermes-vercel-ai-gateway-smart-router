# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Çalışma zamanı planlayıcı — plugin'in içinden çağrılır.

`gwroute.py`'deki skor formülünün aynısını kullanır, ama CLI'ya bağlı değildir:

1. Sağlayıcı istatistikleri **10 dakika** diskte cache'lenir (`stats/<model>.json`).
   Cache bayatsa istek bekletilmez: eldeki veri kullanılır, yenileme arka planda yapılır.
2. Her oturum için bir sağlayıcı **kilitlenir** (`sessions.json`, 6 saat TTL).
   Kilit, o oturumun ilk isteğinde kurulur; sonraki turlarda aynı sağlayıcı başa yazılır
   (prompt cache affinity korunur).
3. Kilitli sağlayıcı hata verirse **strike** alır; 2 strike sonrası kilit düşer ve
   sağlayıcı 10 dakika **karantinaya** alınır (`demoted.json`). Sıradaki sağlayıcı başa geçer.
4. `order` listesi her zaman tam sıralamadır → gateway kilitli sağlayıcı ölürse
   sonrakine kendisi geçer (sunucu tarafı failover).

Bu modül asla istisna fırlatmamalı: her hata yutulur, yönlendirme devre dışı kalır.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import math
import os
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

BASE = "https://ai-gateway.vercel.sh/v1"
LAST_GATEWAY: dict[str, dict] = {}   # model → son üretilen order/only (1.4 log için)

DEFAULT_STATS_TTL = 600.0     # 10 dakika
DEFAULT_SESSION_TTL = 21600.0  # 6 h idle TTL; override per model with session_ttl_s
DEFAULT_DEMOTE_TTL = 600.0     # 10 dakika
STRIKES_TO_DEMOTE = 2
STRIKE_COOLDOWN_S = 30.0       # B3: aynı sağlayıcıya 30 sn'de en fazla 1 strike
STRIKE_BURST_WINDOW_S = 60.0   # B3: 2 farklı sağlayıcı penceresi
STRIKE_PAUSE_S = 300.0         # B3: küresel strike durdurma süresi

_LOCK = threading.Lock()
_REFRESHING: set[str] = set()
_ROUTING_WARN_TS: dict[str, float] = {}


def _warn_routing_once(model: str, msg: str, *args) -> None:
    """Route-off warnings are rate limited: at most one per model per minute."""
    now = time.time()
    with _LOCK:
        if now - float(_ROUTING_WARN_TS.get(model) or 0.0) < 60.0:
            return
        _ROUTING_WARN_TS[model] = now
    log.warning(msg, *args)


# ------------------------------------------------------------------ yollar
def home() -> Path:
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env).expanduser()
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        return Path.home() / ".hermes"


def state_dir() -> Path:
    d = home() / "ai-gateway-router"
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)      # ledger rows are billing data
    except OSError:
        pass
    return d


def rules_path() -> Path | None:
    env = os.environ.get("HERMES_AI_GATEWAY_RULES")
    cands = [Path(env).expanduser()] if env else []
    cands += [state_dir() / "rules.json"]
    for p in cands:
        if p.is_file():
            return p
    return None


def _stats_path(model: str) -> Path:
    d = state_dir() / "stats"
    d.mkdir(parents=True, exist_ok=True)
    return d / (model.replace("/", "__") + ".json")


KEY_SEP = "\x1f"   # (model, provider) composite key separator


def state_key(model: str | None, provider: str) -> str:
    """Composite key so a failure on model A never quarantines model B."""
    return f"{model}{KEY_SEP}{provider}" if model else provider


def split_key(key: str) -> tuple[str | None, str]:
    if KEY_SEP in key:
        model, provider = key.split(KEY_SEP, 1)
        return model, provider
    return None, key


def _json_state(name: str) -> dict:
    p = state_dir() / name
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except Exception:
        return {}


def _migrate_keys(data: dict) -> dict:
    """Rewrite legacy provider-only keys as (model, provider) keys, in memory.

    Old files stored ``{"particle": {...}}``; the record's own ``model`` field
    says which model the entry belongs to, so nothing is lost.
    """
    out: dict = {}
    for key, rec in (data or {}).items():
        if KEY_SEP in str(key):
            out[key] = rec
            continue
        model = (rec or {}).get("model") if isinstance(rec, dict) else None
        out[state_key(model, str(key))] = rec
    return out


def _strikes() -> dict:
    return _migrate_keys(_json_state("strikes.json"))


def _demoted_raw() -> dict:
    return _migrate_keys(_json_state("demoted.json"))


@contextlib.contextmanager
def state_lock(timeout: float = 5.0):
    """Cross-process lock for read-modify-write cycles on the state files."""
    path = state_dir() / ".state.lock"
    fh = path.open("a+")
    try:
        deadline = time.time() + timeout
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.time() > deadline:
                    log.warning("ai-gateway-router: state.lock alınamadı (%.1fs) — "
                                "kilitsiz devam ediliyor", timeout)
                    break
                time.sleep(0.02)
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        fh.close()


def _save_state(name: str, data: dict) -> None:
    """Atomic write: unique temp file in the state dir, fsync, then os.replace."""
    tmp = None
    try:
        d = state_dir()
        fd, tmp = tempfile.mkstemp(dir=str(d), prefix="." + name + "-")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, d / name)
    except Exception as exc:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        log.debug("ai-gateway-router: %s yazılamadı: %s", name, exc)


# ------------------------------------------------------------------ kurallar
RULE_KEYS = {
    "role", "precision", "precision_policy", "workload", "require_tools",
    "require_metrics", "missing_metric_penalty", "min_uptime_15m", "min_context",
    "model_context", "max_candidates", "time_value_cents_per_s",
    "session_affinity", "session_ttl_s", "stats_ttl_s", "quality",
}
PRECISION_MODES = {"auto", "strict", "exclude", "any"}
_VALIDATED_MTIME: dict[str, float] = {}


def validate_rules(data: dict) -> list[str]:
    """Return human-readable problems; an empty list means the file is valid."""
    problems: list[str] = []
    models = data.get("models")
    if not isinstance(models, dict):
        return ["models: bir nesne (object) olmalı"]
    for name, rule in models.items():
        if not isinstance(rule, dict):
            problems.append(f"models[{name}]: nesne olmalı")
            continue
        for key in sorted(set(rule) - RULE_KEYS):
            problems.append(f"models[{name}].{key}: bilinmeyen anahtar")
        prec = rule.get("precision")
        if prec is not None:
            if not isinstance(prec, dict):
                problems.append(f"models[{name}].precision: nesne olmalı")
            elif prec.get("mode", "auto") not in PRECISION_MODES:
                problems.append(f"models[{name}].precision.mode: "
                                f"{prec.get('mode')!r} geçersiz ({sorted(PRECISION_MODES)})")
        wl = rule.get("workload")
        if wl is not None and not isinstance(wl, dict):
            problems.append(f"models[{name}].workload: nesne olmalı")
        for key in ("require_tools", "require_metrics", "session_affinity"):
            if key in rule and not isinstance(rule[key], bool):
                problems.append(f"models[{name}].{key}: bool olmalı")
        for key in ("min_uptime_15m", "min_context", "max_candidates",
                    "time_value_cents_per_s", "stats_ttl_s", "session_ttl_s",
                    "missing_metric_penalty", "model_context"):
            if key in rule and rule[key] is not None:
                try:
                    float(rule[key])
                except (TypeError, ValueError):
                    problems.append(f"models[{name}].{key}: sayı olmalı")
    return problems


def load_rules() -> dict:
    p = rules_path()
    if p is None:
        return {}
    try:
        mtime = p.stat().st_mtime
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        log.debug("ai-gateway-router: rules.json okunamadı: %s", exc)
        return {}
    key = str(p)
    if _VALIDATED_MTIME.get(key) != mtime:
        _VALIDATED_MTIME[key] = mtime
        for problem in validate_rules(data):
            log.warning("ai-gateway-router: %s — %s", p.name, problem)
    models = data.get("models")
    return models if isinstance(models, dict) else {}


def rule_for(model: str) -> dict | None:
    """Rule for this model; the optional "*" entry is the default for unlisted ones."""
    rules = load_rules()
    rule = rules.get(model)
    if rule is None:
        rule = rules.get("*")
    return rule if isinstance(rule, dict) else None


# --------------------------------------------------------------- istatistik
def _api_key() -> str | None:
    key = os.environ.get("AI_GATEWAY_API_KEY")
    if key:
        return key
    try:
        for line in (home() / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("AI_GATEWAY_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return None


class _SameHostRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse cross-host/scheme redirects so the Authorization header never leaks."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old = urllib.parse.urlsplit(req.full_url)
        new = urllib.parse.urlsplit(newurl)
        if (old.scheme, old.netloc) != (new.scheme, new.netloc):
            raise urllib.error.HTTPError(newurl, code, "cross-host redirect refused", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_SameHostRedirect)


def _fetch_endpoints(model: str, timeout: float = 20.0) -> dict | None:
    key = _api_key()
    if not key:
        return None
    req = urllib.request.Request(
        f"{BASE}/models/{model}/endpoints",
        headers={"Authorization": "Bearer " + key, "User-Agent": "hermes-ai-gateway-router"},
    )
    with _OPENER.open(req, timeout=timeout) as resp:
        return json.loads(resp.read())["data"]


STATS_FETCH_TIMEOUT_S = 3.0      # cold start sits on the request path
STATS_REFRESH_TIMEOUT_S = 20.0   # background refresh may take its time
NEGATIVE_CACHE_S = 60.0          # do not retry a failed fetch for a minute
_NEG_CACHE_TS: dict[str, float] = {}
_SYNC_LOCKS: dict[str, threading.Lock] = {}


def _write_stats(path: Path, endpoints: list[dict], fetched_at: float | None = None) -> None:
    """Write the stats cache atomically (unique temp file in the same directory).

    A crash or a full disk must never leave a truncated JSON file: the reader
    would treat it as "no cache" and pay another synchronous fetch.
    """
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".stats-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"fetched_at": fetched_at or time.time(), "endpoints": endpoints},
                      fh, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_stats(path: Path) -> tuple[list[dict], float]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return (data.get("endpoints") or []), float(data.get("fetched_at") or 0.0)
    except Exception:
        return [], 0.0


def _sync_lock(model: str) -> threading.Lock:
    with _LOCK:
        return _SYNC_LOCKS.setdefault(model, threading.Lock())


def stats(model: str, ttl: float = DEFAULT_STATS_TTL) -> tuple[list[dict], bool]:
    """(endpoint list, stale?) — never blocks more than one cold fetch.

    Cache hit  → returned as-is; a stale file is refreshed in the background.
    Cache miss → exactly one thread per model fetches synchronously with a 3 s
                 timeout; every other thread gets ``[]`` at once. A failed or
                 empty fetch is negatively cached for 60 s, and an
                 ``endpoints: []`` file is not treated as a valid cache.
    """
    p = _stats_path(model)
    endpoints, fetched = _read_stats(p)
    if endpoints:
        fresh = (time.time() - fetched) < ttl
        if not fresh:
            _refresh_async(model, p)
        return endpoints, not fresh

    if time.time() - float(_NEG_CACHE_TS.get(model) or 0.0) < NEGATIVE_CACHE_S:
        return [], False

    lock = _sync_lock(model)
    if not lock.acquire(blocking=False):
        return [], False    # another thread is already fetching this model

    try:
        endpoints, fetched = _read_stats(p)      # someone may have filled it meanwhile
        if endpoints:
            return endpoints, (time.time() - fetched) >= ttl
        try:
            payload = _fetch_endpoints(model, timeout=STATS_FETCH_TIMEOUT_S)
            eps = (payload or {}).get("endpoints") or []
        except Exception as exc:
            log.debug("ai-gateway-router: %s ilk istatistik alınamadı: %s", model, exc)
            _NEG_CACHE_TS[model] = time.time()
            return [], False
        if not eps:
            _NEG_CACHE_TS[model] = time.time()
            return [], False
        _NEG_CACHE_TS.pop(model, None)
        try:
            _write_stats(p, eps)
        except Exception as exc:
            log.warning("ai-gateway-router: %s stats yazılamadı: %s", model, exc)
        return eps, False
    finally:
        lock.release()


def _refresh_async(model: str, path: Path) -> None:
    with _LOCK:
        if model in _REFRESHING:
            return
        _REFRESHING.add(model)

    def _work() -> None:
        try:
            data = _fetch_endpoints(model, timeout=STATS_REFRESH_TIMEOUT_S)
            eps = (data or {}).get("endpoints") or []
            if eps:
                _write_stats(path, eps)
                _NEG_CACHE_TS.pop(model, None)
        except Exception as exc:
            log.debug("ai-gateway-router: %s istatistik yenilenemedi: %s", model, exc)
        finally:
            with _LOCK:
                _REFRESHING.discard(model)

    threading.Thread(target=_work, name=f"gwstats-{model}", daemon=True).start()


# ------------------------------------------------------------------ skorlama
def _num(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _uptime(e: dict) -> float | None:
    parts = [(e.get("uptime_last_15m"), 0.5), (e.get("uptime_last_1h"), 0.3), (e.get("uptime_last_1d"), 0.2)]
    have = [(v, w) for v, w in ((_num(v), w) for v, w in parts) if v is not None]
    if not have:
        return None
    tw = sum(w for _, w in have)
    return sum(v * w for v, w in have) / tw


def _median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v is not None and math.isfinite(v) and v > 0)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


DEFAULT_PRECISION_POLICY = {
    "strict_min_pool": 3,           # strict FP8 için gereken en az FP8 sağlayıcı
    "strict_max_unknown_share": 0.30,  # strict için izin verilen en fazla bildirimsiz oran
}

DEFAULT_QUALITY = {
    "enabled": True,
    "unknown_precision_penalty": 0.97,  # beyan edilmiş FP8/FP4 varken "bilinmeyen" hafif geriye düşer
    "fp4_penalty": 0.80,          # FP4 ağırlık → aynı fiyatta %25 daha pahalı sayılır
    "missing_structured_output": 0.97,
    "context_shrink": 0.98,
    "context_shrink_min_ratio": 0.90,   # model context'inin %90'ından küçükse ceza
}


def _quality(e: dict, model_ctx: int | None, cfg: dict, precision_known: bool = False) -> float:
    """Sağlayıcı kalite çarpanı (1.0 = tam kalite). Düşük çarpan skoru büyütür.

    Precision yalnız BEYAN EDİLDİĞİNDE cezalandırılır: kimse bildirmiyorsa
    (minimax-m3.1-flash gibi) filtre uygulamak yerine burada yalnız bilinen
    kötü format (fp4) ceza alır, bilinmeyen nötr kalır.
    """
    if not cfg.get("enabled", True):
        return 1.0
    q = 1.0
    quant = (e.get("quantization") or "").lower()
    if quant in {"fp4", "int4", "int3", "int2"}:
        q *= float(cfg.get("fp4_penalty", 0.80))
    elif not quant and precision_known:
        # Piyasada FP8/FP4 beyanı varken "bilinmeyen" hafif geriye düşer.
        q *= float(cfg.get("unknown_precision_penalty", 1.00))
    tags = e.get("tags") or []
    if "structured-output" not in tags:
        q *= float(cfg.get("missing_structured_output", 0.97))
    ctx = e.get("context_length")
    if model_ctx and ctx and int(ctx) < int(model_ctx) * float(cfg.get("context_shrink_min_ratio", 0.90)):
        q *= float(cfg.get("context_shrink", 0.98))
    return max(q, 0.5)


DEFAULT_MISSING_METRIC_PENALTY = 2.0


def _score(e: dict, wl: dict, tv: float, quality_cfg: dict | None = None,
           model_ctx: int | None = None, precision_known: bool = False,
           metrics: dict | None = None) -> float | None:
    pr = e.get("pricing") or {}
    p_in, p_out = _num(pr.get("prompt")), _num(pr.get("completion"))
    if p_in is None or p_out is None:
        return None
    p_cache = _num(pr.get("input_cache_read"))
    p_cache = p_in if p_cache is None else p_cache
    n_in, n_out = float(wl.get("input_tokens", 8000)), float(wl.get("output_tokens", 1200))
    ratio = float(wl.get("cached_ratio", 0.0))
    # 4.4: implicit-caching desteklemeyen sağlayıcıda cache kazancı yoktur
    if "implicit-caching" not in (e.get("tags") or []):
        ratio = 0.0
    cached = n_in * ratio
    cost = (n_in - cached) * p_in + cached * p_cache + n_out * p_out

    # F2a: a missing or non-positive metric must not look like "no waiting".
    # The peer median is the estimate; the penalty makes it pessimistic
    # (slower latency and lower throughput both mean more seconds).
    metrics = metrics or {}
    pen = float(metrics.get("penalty") or DEFAULT_MISSING_METRIC_PENALTY)
    if pen <= 1.0:
        pen = DEFAULT_MISSING_METRIC_PENALTY
    lat = _num((e.get("latency_last_1h") or {}).get("p50"))
    tps = _num((e.get("throughput_last_1h") or {}).get("p50"))
    if lat is None or lat <= 0 or not math.isfinite(lat):
        lat = (metrics.get("lat") or 0.0) * pen
    if tps is None or tps <= 0 or not math.isfinite(tps):
        med_tps = metrics.get("tps") or 0.0
        tps = (med_tps / pen) if med_tps else 0.0
    secs = 0.0
    if lat:
        secs += float(lat) / 1000.0
    if tps:
        secs += n_out / float(tps)

    # F2b: unknown uptime stays neutral (risk 1.0); a measured 0% or NaN uptime is
    # the worst case and must never beat a healthy provider.
    up = _uptime(e)
    if up is None:
        risk = 1.0
    elif not math.isfinite(up) or up <= 0:
        risk = 100.0
    else:
        risk = 100.0 / up
    q = _quality(e, model_ctx, quality_cfg or DEFAULT_QUALITY, precision_known)
    return (cost * 100.0 + tv * secs) * risk / q


def _status_num(e: dict) -> int:
    """Numeric endpoint status; unknown/garbage is treated as 'not live' (-1)."""
    try:
        raw = e.get("status")
        return int(raw) if raw not in (None, "") else 0
    except (TypeError, ValueError, AttributeError):
        return -1


def resolve_precision(rule: dict, endpoints: list[dict]) -> dict:
    """`mode: auto` → katalogdaki beyan dağılımına göre politikayı türet.

    Kurallar:
      • Hiç kimse precision bildirmiyorsa → any (filtre anlamsız)
      • FP8 havuzu yeterince büyük ve bildirimsiz oran düşükse → strict fp8
        (strict'in maliyeti az: aday kaybı yok)
      • Aksi halde beyan edilen kötü formatı dışla → exclude fp4
        (bildirimsizler aday kalır, kalite çarpanı onları hafifçe geriye atar)
    """
    prec = rule.get("precision") or {}
    mode = prec.get("mode", "auto")
    if mode != "auto":
        return {"mode": mode, "require": prec.get("require", "fp8"),
                "exclude": tuple(prec.get("exclude") or ("fp4",)),
                "keep_unknown": mode != "strict" or bool(prec.get("allow_unknown")),
                "source": "explicit"}

    live = [e for e in endpoints if isinstance(e, dict) and _status_num(e) == 0]
    if not live:
        return {"mode": "any", "require": None, "exclude": (), "keep_unknown": True,
                "source": "auto:aday-yok"}
    declared = [q for q in ((e.get("quantization") or "").lower() or None for e in live) if q]
    unknown_share = 1.0 - (len(declared) / len(live))
    fp8_n = sum(1 for q in declared if q == "fp8")
    bad_n = sum(1 for q in declared if q in {"fp4", "int4", "int3", "int2"})
    policy = {**DEFAULT_PRECISION_POLICY, **(rule.get("precision_policy") or {})}

    if not declared:
        return {"mode": "any", "require": None, "exclude": (), "keep_unknown": True,
                "source": "auto:beyan-yok"}
    if fp8_n >= int(policy["strict_min_pool"]) and unknown_share <= float(policy["strict_max_unknown_share"]):
        return {"mode": "strict", "require": "fp8", "exclude": (), "keep_unknown": False,
                "source": f"auto:strict(fps8={fp8_n},bildirimsiz={unknown_share:.0%})"}
    if bad_n:
        return {"mode": "exclude", "require": None, "exclude": ("fp4", "int4"), "keep_unknown": True,
                "source": f"auto:exclude-fp4(fp8={fp8_n},fp4={bad_n},bildirimsiz={unknown_share:.0%})"}
    return {"mode": "any", "require": None, "exclude": (), "keep_unknown": True,
            "source": f"auto:any(fp8={fp8_n},bildirimsiz={unknown_share:.0%})"}


def rank(model: str, rule: dict, endpoints: list[dict], demoted: dict) -> list[dict]:
    """Apply the rule and rank providers by score.

    Hard filters (precision, tool-use, quarantine) are never relaxed. The soft
    floors (``min_uptime_15m``, ``min_context``) are dropped once, in a second
    pass, only when the first pass leaves no candidate at all.
    """
    rp = resolve_precision(rule, endpoints)
    mode, require = rp["mode"], rp["require"]
    exclude = set(rp["exclude"])
    keep_unknown = rp["keep_unknown"]
    precision_known = any((e.get("quantization") or "").lower() in {"fp8", "fp4"} for e in endpoints)

    min_up = rule.get("min_uptime_15m")
    min_ctx = rule.get("min_context")
    need_tools = rule.get("require_tools", True)
    require_metrics = bool(rule.get("require_metrics", False))
    wl = rule.get("workload") or {}
    tv = float(rule.get("time_value_cents_per_s", 0.15))
    quality_cfg = {**DEFAULT_QUALITY, **(rule.get("quality") or {})}

    def _ctx_of(e) -> int:
        try:
            return int(e.get("context_length") or 0)
        except (TypeError, ValueError):
            return 0

    model_ctx = rule.get("model_context") or max((_ctx_of(e) for e in endpoints), default=0) or None

    # F2a: peer medians are the estimate for providers that report no metrics.
    metrics = {
        "lat": _median([_num((e.get("latency_last_1h") or {}).get("p50")) or 0.0 for e in endpoints]),
        "tps": _median([_num((e.get("throughput_last_1h") or {}).get("p50")) or 0.0 for e in endpoints]),
        "penalty": float(rule.get("missing_metric_penalty") or DEFAULT_MISSING_METRIC_PENALTY),
    }

    now = time.time()

    def collect(up_floor, ctx_floor) -> list[dict]:
        picked: list[dict] = []
        for e in endpoints:
            try:
                if not isinstance(e, dict):
                    raise TypeError(f"endpoint is {type(e).__name__}, not dict")
                # F2c: comparison only fails on weird input, but do not trust it
                # with mixed types (a string status must not kill the model).
                status = e.get("status")
                status_n = int(status) if status not in (None, "") else 0
                if status_n != 0:
                    continue
                # F2d: quantization is matched case-insensitively.
                quant = (e.get("quantization") or "").strip().lower() or None
                if mode == "strict" and require:
                    if quant != (require or "").lower() and not (quant is None and keep_unknown):
                        continue
                elif mode == "exclude" and quant in {str(x).lower() for x in exclude}:
                    continue
                if need_tools and "tool-use" not in (e.get("tags") or []):
                    continue
                lat_raw = _num((e.get("latency_last_1h") or {}).get("p50"))
                tps_raw = _num((e.get("throughput_last_1h") or {}).get("p50"))
                if require_metrics and not ((lat_raw and lat_raw > 0) and (tps_raw and tps_raw > 0)):
                    continue
                up = _uptime(e)
                if up_floor is not None:
                    if up is None or not math.isfinite(up) or up <= 0 or up < float(up_floor):
                        continue
                if ctx_floor and _ctx_of(e) < int(ctx_floor):
                    continue
                prov = e.get("provider_name")
                penalty_until = (demoted.get(state_key(model, prov)) or demoted.get(prov) or {}).get("until", 0)
                if penalty_until and penalty_until > now:
                    continue
                score = _score(e, wl, tv, quality_cfg, model_ctx, precision_known, metrics)
                if score is None:
                    continue
                picked.append({"provider": prov, "score": score,
                               "quality": round(_quality(e, model_ctx, quality_cfg), 3)})
            except Exception as exc:
                log.warning("ai-gateway-router: %s — skipping malformed endpoint %r: %s",
                            model, (e or {}).get("provider_name"), exc)
        picked.sort(key=lambda x: x["score"])
        return picked

    out = collect(min_up, min_ctx)
    if not out and (min_up is not None or min_ctx):
        out = collect(None, None)
        if out:
            log.info("ai-gateway-router: %s — uptime/context floors relaxed "
                     "(precision, tool-use and quarantine kept)", model)
    return out   # the cut happens in build_gateway (the `only` set comes from here)


def has_server_filter(rule: dict, endpoints: list[dict] | None = None) -> list[str]:
    """Translate the rule's precision policy into the gateway ``has`` list.

    ``has`` must never contradict ``only``: in strict mode with
    ``allow_unknown`` the pool deliberately keeps undeclared providers, and no
    ``has`` expression can say "fp8 or unknown" — so no filter is sent and the
    ``only`` list carries the policy.
    """
    if endpoints is not None:
        rp = resolve_precision(rule, endpoints)
    else:
        prec = rule.get("precision") or {}
        rp = {"mode": prec.get("mode", "auto"), "require": prec.get("require", "fp8"),
              "exclude": tuple(prec.get("exclude") or ("fp4", "int4")),
              "keep_unknown": prec.get("mode", "auto") != "strict" or bool(prec.get("allow_unknown"))}
    if rp.get("mode") == "strict" and rp.get("require") and not rp.get("keep_unknown"):
        return [f"quantization:{rp['require']}"]
    if rp.get("mode") == "exclude":
        return [f"!quantization:{v}" for v in rp["exclude"]]
    return []


# Backwards-compatible alias (pre-existing callers/tests use has_filter).
has_filter = has_server_filter


# ------------------------------------------------------------------ oturumlar
def _sessions() -> dict:
    return _json_state("sessions.json")


def locked_provider(model: str, session_key: str, ttl: float | None = None) -> str | None:
    """Pinned provider for this session, or None once the idle TTL has passed.

    ``ttl`` comes from the rule's ``session_ttl_s`` (default DEFAULT_SESSION_TTL);
    the timestamp is refreshed by touch_provider() while the pin is in use, so the
    value is an *idle* TTL, not an absolute lifetime.
    """
    rec = (_sessions().get(session_key) or {})
    if rec.get("model") != model:
        return None
    limit = DEFAULT_SESSION_TTL if ttl is None else float(ttl)
    if time.time() - float(rec.get("ts") or 0) > limit:
        return None
    return rec.get("provider")


def touch_provider(model: str, session_key: str, provider: str, min_age: float = 60.0) -> None:
    """Refresh the lock timestamp (idle TTL): a session that keeps talking keeps
    its provider; only a session idle for DEFAULT_SESSION_TTL loses it."""
    if not session_key:
        return
    with _LOCK, state_lock():
        state = _sessions()
        rec = state.get(session_key) or {}
        if rec.get("model") != model or rec.get("provider") != provider:
            return
        if time.time() - float(rec.get("ts") or 0) < min_age:
            return
        rec["ts"] = time.time()
        state[session_key] = rec
        _save_state("sessions.json", state)


def lock_provider(model: str, session_key: str, provider: str,
                  ttl: float | None = None) -> None:
    with _LOCK, state_lock():
        state = _sessions()
        state[session_key] = {"model": model, "provider": provider, "ts": time.time()}
        now = time.time()
        limit = DEFAULT_SESSION_TTL if ttl is None else float(ttl)
        state = {k: v for k, v in state.items() if now - float(v.get("ts") or 0) <= limit}
        _save_state("sessions.json", state)


def release_provider(session_key: str) -> None:
    with _LOCK, state_lock():
        state = _sessions()
        if session_key in state:
            state.pop(session_key, None)
            _save_state("sessions.json", state)


# NOTE: the old ``freshest_session()`` fallback was removed. It had no TTL and
# no session scope, so a request without strike context could strike the provider
# of an unrelated conversation. Without context we record nothing.


def candidate_pool(model: str) -> list[str]:
    """Providers currently allowed for this model.

    ``LAST_GATEWAY`` is process memory and is empty right after a restart, so the
    pool is recomputed from the cached stats when the in-memory copy is missing.
    """
    rec = (LAST_GATEWAY.get(model) or {}).get("only") or []
    if rec:
        return list(rec)
    try:
        rule = rule_for(model)
        if rule is None:
            return []
        endpoints, _stale = stats(model, float(rule.get("stats_ttl_s", DEFAULT_STATS_TTL)))
        ranked = rank(model, rule, endpoints, demoted())
        if not ranked:
            return []
        n = int(rule.get("max_candidates", 4))
        return [r["provider"] for r in (ranked[:n] if n else ranked)]
    except Exception:
        return []


def record_strike(model: str, session_key: str | None, provider: str | None, reason: str = "") -> str | None:
    """Record a provider failure for (model, provider).

    B3: at most one strike per provider per 30 s (retries are not new failures);
    two different providers inside 60 s pause all strikes for 5 minutes.
    Strikes and quarantine are keyed by (model, provider): a failure on model A
    must never quarantine that provider for model B.

    Returns the quarantined provider name, or None.
    """
    if not provider:
        return None
    with _LOCK, state_lock():
        now = time.time()
        pause = _json_state("strike_pause.json")
        if float(pause.get("until") or 0) > now:
            log.debug("ai-gateway-router: strike duraklaması aktif (%.0f sn) — %s yok sayıldı",
                      float(pause["until"]) - now, provider)
            return None
        key = state_key(model, provider)
        strikes = _strikes()
        rec = strikes.get(key) or {"count": 0, "ts": 0, "model": model}
        if now - float(rec.get("ts") or 0) > DEFAULT_DEMOTE_TTL:
            rec = {"count": 0, "ts": 0, "model": model}
        # B3: 30 sn içinde aynı sağlayıcıya tekrar — retry'dır, strike sayılmaz
        if int(rec.get("count") or 0) and now - float(rec.get("ts") or 0) < STRIKE_COOLDOWN_S:
            log.debug("ai-gateway-router: %s için 30 sn içinde tekrar hata — strike sayılmadı (%s)",
                      provider, reason)
            return None
        rec["count"] = int(rec.get("count") or 0) + 1
        rec["ts"] = now
        rec["model"] = model
        rec["reason"] = reason
        # B3: 60 sn içinde 2 farklı sağlayıcı strike aldıysa 5 dk strike dur
        others = sorted({split_key(k)[1] for k, r in strikes.items()
                         if k != key and now - float((r or {}).get("ts") or 0) <= STRIKE_BURST_WINDOW_S})
        strikes[key] = rec
        if others:
            _save_state("strike_pause.json", {
                "until": now + STRIKE_PAUSE_S, "reason": reason, "model": model,
                "providers": sorted(others + [provider])})
            log.warning("ai-gateway-router: 60 sn içinde %d farklı sağlayıcı strike aldı (%s) — "
                        "strike'lar 5 dk durduruldu", len(others) + 1, sorted(others + [provider]))
        hit = False
        if rec["count"] >= STRIKES_TO_DEMOTE:
            demoted_state = _demoted_raw()
            # B4: only kümesinde karantina sonrası 2'den az aday kalıyorsa karantina YOK
            only = (LAST_GATEWAY.get(model) or {}).get("only") or candidate_pool(model)
            active = {split_key(k)[1] for k, r in demoted_state.items()
                      if (split_key(k)[0] in (None, model)) and float((r or {}).get("until") or 0) > now}
            remaining = [p for p in only if p != provider and p not in active]
            if only and len(remaining) < 2:
                log.warning("ai-gateway-router: karantina atlandı (%s) — only'de %d aday "
                            "kalıyor (<2): %s", provider, len(remaining), only)
            else:
                demoted_state[key] = {"until": now + DEFAULT_DEMOTE_TTL, "reason": reason, "model": model}
                _save_state("demoted.json", demoted_state)
                strikes.pop(key, None)
                hit = True
        _save_state("strikes.json", strikes)
        if hit:
            state = _sessions()
            if session_key and (state.get(session_key) or {}).get("provider") == provider:
                state.pop(session_key, None)
                _save_state("sessions.json", state)
            log.warning("ai-gateway-router: %s karantinaya alındı (%s) — sıradaki sağlayıcıya geçilecek",
                        provider, reason or "hata")
        return provider if hit else None


def demoted(model: str | None = None) -> dict:
    """Active quarantine entries, keyed by (model, provider).

    ``model`` filters to one model; legacy provider-only keys are matched only
    when their record carries no model.
    """
    d = _demoted_raw()
    now = time.time()
    out = {}
    for k, v in d.items():
        if float((v or {}).get("until") or 0) <= now:
            continue
        m, _p = split_key(k)
        if model is not None and m not in (None, model):
            continue
        out[k] = v
    return out


def infer_provider(model: str, input_tokens: int, output_tokens: int,
                   cost_usd, cache_read_tokens: int = 0, tolerance: float = 0.02) -> dict:
    """Yanıttaki `usage.cost`'u sağlayıcı fiyatlarıyla eşleştirerek kimin servis ettiğini çıkar.

    Gateway yanıtı `provider_name` ALANINI DÖNDÜRMEZ (yalnız `generationId`).
    Fiyatlar sağlayıcıya özgü olduğu için maliyet aritmetiği çoğu zaman tek eşleşme verir;
    fiyatları aynı olan sağlayıcılarda sonuç belirsizdir (candidates > 1).
    """
    try:
        cost = float(cost_usd)
    except (TypeError, ValueError):
        return {"provider": None, "candidates": [], "ambiguous": True, "reason": "maliyet yok"}
    endpoints, _ = stats(model, float(os.environ.get("HERMES_AI_GATEWAY_STATS_TTL", 600)))
    hits = []
    for e in endpoints:
        pr = e.get("pricing") or {}
        p_in, p_out = _num(pr.get("prompt")), _num(pr.get("completion"))
        if p_in is None or p_out is None:
            continue
        p_cache = _num(pr.get("input_cache_read"))
        p_cache = p_in if p_cache is None else p_cache
        fresh = max(0, int(input_tokens or 0) - int(cache_read_tokens or 0))
        est = fresh * p_in + int(cache_read_tokens or 0) * p_cache + int(output_tokens or 0) * p_out
        if abs(est - cost) <= max(abs(cost) * tolerance, 1e-9):
            hits.append({"provider": e.get("provider_name"), "estimated": round(est, 10)})
    return {"provider": hits[0]["provider"] if len(hits) == 1 else None,
            "candidates": [h["provider"] for h in hits],
            "ambiguous": len(hits) != 1,
            "reason": "tek eşleşme" if len(hits) == 1 else f"{len(hits)} sağlayıcı aynı fiyatı veriyor"}


def lookup_generation(generation_id: str, retries: int = 6, delay: float = 5.0) -> dict | None:
    """Gateway usage event'i gecikmeli düşer (~10-20 sn) → gerçek sağlayıcı adı."""
    key = _api_key()
    if not key or not generation_id:
        return None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{BASE}/generation?id={generation_id}",
                headers={"Authorization": "Bearer " + key, "User-Agent": "hermes-ai-gateway-router"},
            )
            with _OPENER.open(req, timeout=15) as resp:
                data = json.loads(resp.read()).get("data") or {}
            if data.get("provider_name"):
                return data
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(delay)
    return None


def _passes_has(e: dict, has: list[str]) -> bool:
    """True when the endpoint satisfies every entry of a gateway ``has`` list."""
    quant = (e.get("quantization") or "").strip().lower()
    for item in has:
        neg = item.startswith("!")
        token = item[1:] if neg else item
        if ":" not in token:
            return False
        key, want = token.split(":", 1)
        if key != "quantization":
            return False
        ok = quant == (want or "").strip().lower()
        if ok == neg:
            return False
    return True


# ------------------------------------------------------------------ ana giriş
def build_gateway(model: str, session_key: str | None = None) -> dict:
    """Model için gateway bloğu: {order, has, sessionId, sessionAffinity}."""
    try:
        rule = rule_for(model)
        if rule is None:
            return {}
        endpoints, stale = stats(model, float(rule.get("stats_ttl_s", DEFAULT_STATS_TTL)))
        if not endpoints:
            return {}
        dem = demoted(model)
        ranked = rank(model, rule, endpoints, dem)
        if not ranked:
            live = [e for e in endpoints if isinstance(e, dict) and _status_num(e) == 0]
            if live and all(state_key(model, e.get("provider_name")) in dem for e in live):
                # F1d: quarantining everything must not drop the precision policy.
                # Without an order/only set the gateway picks any provider, so we
                # keep sending `has` (precision stays enforced) and only lose the
                # ranking. Whitelisted providers are back after 10 minutes.
                _warn_routing_once(
                    model, "ai-gateway-router: %s — every provider is quarantined; ranking is OFF "
                           "for this model until a quarantine expires (has=%s still sent)",
                    model, has_server_filter(rule, endpoints) or None)
                h = has_server_filter(rule, endpoints)
                return {"has": h} if h else {}
            else:
                _warn_routing_once(
                    model, "ai-gateway-router: %s — no provider passes the rule filters "
                           "(precision/tool-use/uptime/context); routing is OFF for this model", model)
            return {}
        max_cand = int(rule.get("max_candidates", 4))
        top = ranked[:max_cand] if max_cand else list(ranked)
        # 1.3: 2'den az sağlayıcı kaldıysa filtreden geçmiş sonraki en iyi adayları ekle
        # (hard filtre elenmiş sağlayıcı ASLA eklenmez — ranked zaten filtrelenmiş havuz).
        if len(top) < 2:
            top = ranked[:2]
        order = [r["provider"] for r in top]

        # `has` ve `only` çelişmemeli: sunucu tarafı filtrenin reddedeceği sağlayıcı
        # `only` içinde kalırsa istek "no_capable_providers_available" ile düşer.
        h = has_server_filter(rule, endpoints)
        if h:
            allowed = {e.get("provider_name") for e in endpoints if _passes_has(e, h)}
            kept = [p for p in order if p in allowed]
            if not kept:
                _warn_routing_once(
                    model, "ai-gateway-router: %s — has=%s excludes the whole only set; "
                           "routing is OFF for this model", model, h)
                return {}
            if len(kept) != len(order):
                log.info("ai-gateway-router: %s — %d provider(s) dropped from only: "
                         "inconsistent with has=%s", model, len(order) - len(kept), h)
            order = kept

        if session_key and rule.get("session_affinity", True):
            ttl = rule.get("session_ttl_s")
            pinned = locked_provider(model, session_key, ttl)
            if pinned and pinned in order:
                order = [pinned] + [p for p in order if p != pinned]
                touch_provider(model, session_key, pinned)
            elif pinned:
                release_provider(session_key)
                pinned = None
            if not pinned:
                lock_provider(model, session_key, str(order[0]), ttl)

        # 1.2: only == order (filtre sonrası, max_candidates adet). Kuyruk kapanır;
        # 1.4: hepsi başarısız olursa istek düşer, küme asla genişletilmez.
        gw = {"order": order, "only": list(order),
              "sessionAffinity": bool(rule.get("session_affinity", True))}
        LAST_GATEWAY[model] = {"order": list(order), "only": list(order)}
        if h:
            gw["has"] = h
        if stale:
            gw["_stale"] = True   # yalnız log/telemetri için; gateway'e gönderilmez
        return gw
    except Exception as exc:
        log.debug("ai-gateway-router: plan üretilemedi (%s): %s", model, exc)
        return {}