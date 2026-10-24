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

import json
import logging
import math
import os
import threading
import time
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

BASE = "https://ai-gateway.vercel.sh/v1"
LAST_GATEWAY: dict[str, dict] = {}   # model → son üretilen order/only (1.4 log için)

DEFAULT_STATS_TTL = 600.0     # 10 dakika
DEFAULT_SESSION_TTL = 21600.0  # 6 saat
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


def _json_state(name: str) -> dict:
    p = state_dir() / name
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except Exception:
        return {}


def _save_state(name: str, data: dict) -> None:
    try:
        p = state_dir() / name
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)
    except Exception as exc:
        log.debug("ai-gateway-router: %s yazılamadı: %s", name, exc)


# ------------------------------------------------------------------ kurallar
def load_rules() -> dict:
    p = rules_path()
    if p is None:
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("models") or {}
    except Exception as exc:
        log.debug("ai-gateway-router: rules.json okunamadı: %s", exc)
        return {}


def rule_for(model: str) -> dict | None:
    rules = load_rules()
    rule = rules.get(model)
    if rule is None:
        return None
    return rule


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


def _fetch_endpoints(model: str) -> dict | None:
    key = _api_key()
    if not key:
        return None
    req = urllib.request.Request(
        f"{BASE}/models/{model}/endpoints",
        headers={"Authorization": "Bearer " + key, "User-Agent": "hermes-ai-gateway-router"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())["data"]


def stats(model: str, ttl: float = DEFAULT_STATS_TTL) -> tuple[list[dict], bool]:
    """(endpoint listesi, bayat_mı). Bayatsa arka planda yenilenir."""
    p = _stats_path(model)
    data = {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    fetched = float(data.get("fetched_at") or 0)
    endpoints = data.get("endpoints") or []
    if not endpoints:
        # Hiç cache yok: ilk istek beklemeyi göze alır (tek seferlik, ~0.3s).
        try:
            payload = _fetch_endpoints(model)
            if payload:
                endpoints = payload.get("endpoints") or []
                p.write_text(json.dumps({"fetched_at": time.time(), "endpoints": endpoints},
                                        ensure_ascii=False), encoding="utf-8")
                return endpoints, False
        except Exception as exc:
            log.debug("ai-gateway-router: %s ilk istatistik alınamadı: %s", model, exc)
        return [], False
    fresh = (time.time() - fetched) < ttl
    if not fresh:
        _refresh_async(model, p)
    return endpoints, not fresh


def _refresh_async(model: str, path: Path) -> None:
    with _LOCK:
        if model in _REFRESHING:
            return
        _REFRESHING.add(model)

    def _work() -> None:
        try:
            data = _fetch_endpoints(model)
            if data:
                path.write_text(json.dumps({"fetched_at": time.time(), "endpoints": data["endpoints"]},
                                           ensure_ascii=False), encoding="utf-8")
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
                penalty_until = (demoted.get(prov) or {}).get("until", 0)
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


def locked_provider(model: str, session_key: str) -> str | None:
    rec = (_sessions().get(session_key) or {})
    if rec.get("model") != model:
        return None
    if time.time() - float(rec.get("ts") or 0) > DEFAULT_SESSION_TTL:
        return None
    return rec.get("provider")


def lock_provider(model: str, session_key: str, provider: str) -> None:
    with _LOCK:
        state = _sessions()
        state[session_key] = {"model": model, "provider": provider, "ts": time.time()}
        now = time.time()
        state = {k: v for k, v in state.items() if now - float(v.get("ts") or 0) <= DEFAULT_SESSION_TTL}
        _save_state("sessions.json", state)


def release_provider(session_key: str) -> None:
    with _LOCK:
        state = _sessions()
        if session_key in state:
            state.pop(session_key, None)
            _save_state("sessions.json", state)


def freshest_session(model: str) -> tuple[str, str] | None:
    """Bir model için en son kullanılan (session_key, sağlayıcı) çifti."""
    entries = [(k, v) for k, v in _sessions().items() if v.get("model") == model]
    if not entries:
        return None
    key, rec = max(entries, key=lambda kv: float(kv[1].get("ts") or 0))
    return key, rec.get("provider")


def record_strike(model: str, session_key: str | None, provider: str | None, reason: str = "") -> str | None:
    """Kilitli sağlayıcıya hata yaz. Eşik aşılırsa karantinaya alıp kilidi bırakır.

    B3: aynı sağlayıcıya 30 sn'de en fazla 1 strike (retry'lar ayrı sayılmaz);
    60 sn içinde 2 farklı sağlayıcı strike alırsa tüm strike'lar 5 dk durdurulur.

    Dönüş: karantinaya alınan sağlayıcı adı (yoksa None).
    """
    if not provider:
        return None
    with _LOCK:
        now = time.time()
        pause = _json_state("strike_pause.json")
        if float(pause.get("until") or 0) > now:
            log.debug("ai-gateway-router: strike duraklaması aktif (%.0f sn) — %s yok sayıldı",
                      float(pause["until"]) - now, provider)
            return None
        strikes = _json_state("strikes.json")
        rec = strikes.get(provider) or {"count": 0, "ts": 0, "model": model}
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
        others = sorted(p for p, r in strikes.items()
                        if p != provider and now - float((r or {}).get("ts") or 0) <= STRIKE_BURST_WINDOW_S)
        strikes[provider] = rec
        if others:
            _save_state("strike_pause.json", {
                "until": now + STRIKE_PAUSE_S, "reason": reason, "model": model,
                "providers": sorted(others + [provider])})
            log.warning("ai-gateway-router: 60 sn içinde %d farklı sağlayıcı strike aldı (%s) — "
                        "strike'lar 5 dk durduruldu", len(others) + 1, sorted(others + [provider]))
        hit = False
        if rec["count"] >= STRIKES_TO_DEMOTE:
            demoted = _json_state("demoted.json")
            # B4: only kümesinde karantina sonrası 2'den az aday kalıyorsa karantina YOK
            only = (LAST_GATEWAY.get(model) or {}).get("only") or []
            active = [p for p, r in demoted.items() if float((r or {}).get("until") or 0) > now]
            remaining = [p for p in only if p != provider and p not in active]
            if only and len(remaining) < 2:
                log.warning("ai-gateway-router: karantina atlandı (%s) — only'de %d aday "
                            "kalıyor (<2): %s", provider, len(remaining), only)
            else:
                demoted[provider] = {"until": now + DEFAULT_DEMOTE_TTL, "reason": reason, "model": model}
                _save_state("demoted.json", demoted)
                strikes.pop(provider, None)
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


def demoted() -> dict:
    d = _json_state("demoted.json")
    now = time.time()
    return {k: v for k, v in d.items() if float(v.get("until") or 0) > now}


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
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read()).get("data") or {}
            if data.get("provider_name"):
                return data
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(delay)
    return None


def record_truth(generation_id: str, intended: str | None, model: str) -> None:
    """Arka planda gerçek sağlayıcıyı doğrula ve generation_truth.jsonl'e yaz."""
    def _work() -> None:
        data = lookup_generation(generation_id)
        if not data:
            return
        try:
            path = state_dir() / "generation_truth.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "model": model,
                    "generation_id": generation_id,
                    "provider_actual": data.get("provider_name"),
                    "provider_intended": intended,
                    "match": (data.get("provider_name") == intended) if intended else None,
                    "cost_usd": data.get("total_cost"),
                    "latency_ms": data.get("latency"),
                    "generation_time_ms": data.get("generation_time"),
                    "finish_reason": data.get("finish_reason"),
                    "streamed": data.get("streamed"),
                }, ensure_ascii=False) + "\n")
        except Exception as exc:
            log.debug("ai-gateway-router: truth yazılamadı: %s", exc)

    threading.Thread(target=_work, name="gw-truth", daemon=True).start()


def _passes_has(e: dict, has: list[str]) -> bool:
    """True when the endpoint satisfies every entry of a gateway ``has`` list."""
    quant = (e.get("quantization") or "").lower()
    for item in has:
        neg = item.startswith("!")
        token = item[1:] if neg else item
        if ":" not in token:
            return False
        key, want = token.split(":", 1)
        if key != "quantization":
            return False
        ok = quant == (want or "").lower()
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
        dem = demoted()
        ranked = rank(model, rule, endpoints, dem)
        if not ranked:
            live = [e for e in endpoints if isinstance(e, dict) and _status_num(e) == 0]
            if live and all(e.get("provider_name") in dem for e in live):
                _warn_routing_once(
                    model, "ai-gateway-router: %s — every provider is quarantined; routing is OFF "
                           "for this model until a quarantine expires", model)
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
            pinned = locked_provider(model, session_key)
            if pinned and pinned in order:
                order = [pinned] + [p for p in order if p != pinned]
            elif pinned:
                release_provider(session_key)
                pinned = None
            if not pinned:
                lock_provider(model, session_key, order[0])

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