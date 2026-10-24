#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""gwroute — Vercel AI Gateway için maliyet + gecikme + uptime temelli sağlayıcı seçici.

v2 modeli: her sağlayıcı için "bu iş yükünü çalıştırmanın toplam bedeli" tek bir
sayıya indirilir ve o sayı üzerinden sıralanır.

    skor(sent) = iş_maliyeti(sent)  +  zaman_değeri(sent/sn) × beklenen_süre(sn)

    iş_maliyeti   = girdi×in_price + çıktı×out_price   (cache'li kısım cache fiyatından)
    beklenen_süre = (ttft + çıktı/tps) / (uptime/100)     ← uptime = başarısızlık+retry cezası

Böylece "en ucuz sağlayıcı 30 tps, bir sonraki 60 tps ve arada 2-3 sent varsa
ikincisini seç" kuralı doğal olarak ortaya çıkar: 2-3 sentlik fark, kazanılan
saniyelerin zaman değerinden küçükse hızlı olan kazanır.

Kullanım:
  python3 gwroute.py --model zai/glm-5.3-flash --quant fp8
  python3 gwroute.py --model zai/glm-5.3 --quant fp8 --in 8000 --out 1500 --time-value 0.5
  python3 gwroute.py --model zai/glm-5.3-flash --quant fp8 --json
  python3 gwroute.py --list-models                        # katalogdaki tüm modeller
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from dataclasses import dataclass

BASE = "https://ai-gateway.vercel.sh/v1"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_TTL = 300


# --------------------------------------------------------------------- API
def api_key() -> str:
    k = os.environ.get("AI_GATEWAY_API_KEY")
    if k:
        return k
    env_file = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / ".env"
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.split("=", 1)[0].strip() == "AI_GATEWAY_API_KEY":
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("AI_GATEWAY_API_KEY yok")


def _get(path: str) -> dict:
    os.makedirs(CACHE, exist_ok=True)
    cf = os.path.join(CACHE, path.strip("/").replace("/", "_") + ".json")
    if os.path.exists(cf) and time.time() - os.path.getmtime(cf) < CACHE_TTL:
        return json.load(open(cf, encoding="utf-8"))
    req = urllib.request.Request(BASE + path, headers={
        "Authorization": "Bearer " + api_key(), "User-Agent": "hermes-gwroute"})
    with urllib.request.urlopen(req, timeout=45) as r:
        data = json.loads(r.read())
    json.dump(data, open(cf, "w", encoding="utf-8"))
    return data


# ------------------------------------------------------------------ adaylar
@dataclass
class EP:
    provider: str
    quant: str | None
    p_in: float          # USD / token
    p_out: float
    p_cache: float
    tps: float | None
    ttft_ms: float | None
    up15: float | None
    up1h: float | None
    up1d: float | None
    status: int
    tools: bool
    ctx: int | None
    zdr: bool
    tags: list
    # hesaplanan
    turn_s: float = 0.0
    turn_cents: float = 0.0
    wait_cents: float = 0.0
    score: float = 0.0
    raw_cents: float = 0.0     # risk öncesi iş maliyeti
    raw_s: float = 0.0         # risk öncesi beklenen süre
    risk: float = 1.0
    quality: float = 1.0       # kalite çarpanı (1.0 = tam)

    @property
    def uptime(self) -> float | None:
        vals = [(self.up15, 0.5), (self.up1h, 0.3), (self.up1d, 0.2)]
        have = [(v, w) for v, w in vals if v is not None]
        if not have:
            return None
        tw = sum(w for _, w in have)
        return sum(v * w for v, w in have) / tw

    @property
    def per_1m_in(self) -> float:
        return self.p_in * 1e6

    @property
    def per_1m_out(self) -> float:
        return self.p_out * 1e6


def load(model: str) -> tuple[dict, list[EP]]:
    d = _get(f"/models/{model}/endpoints")["data"]
    out = []
    for e in d["endpoints"]:
        p = e.get("pricing") or {}
        tp = (e.get("throughput_last_1h") or {}).get("p50")
        la = (e.get("latency_last_1h") or {}).get("p50")
        out.append(EP(
            provider=e["provider_name"],
            quant=e.get("quantization") or None,
            p_in=float(p.get("prompt") or 0),
            p_out=float(p.get("completion") or 0),
            p_cache=float(p.get("input_cache_read") or p.get("prompt") or 0),
            tps=tp, ttft_ms=la,
            up15=e.get("uptime_last_15m"), up1h=e.get("uptime_last_1h"), up1d=e.get("uptime_last_1d"),
            status=int(e.get("status") or 0),
            tools="tool-use" in (e.get("tags") or []),
            ctx=e.get("context_length"), zdr=bool(e.get("has_zdr")),
            tags=list(e.get("tags") or []),
        ))
    return d, out


# ------------------------------------------------------------------ iş profili
DEFAULT_PRECISION_POLICY = {"strict_min_pool": 3, "strict_max_unknown_share": 0.30}

DEFAULT_QUALITY = {
    "enabled": True,
    "fp4_penalty": 0.80,
    "unknown_precision_penalty": 0.97,
    "missing_structured_output": 0.97,
    "context_shrink": 0.98,
    "context_shrink_min_ratio": 0.90,
}


@dataclass
class Workload:
    input_tokens: int = 8000
    output_tokens: int = 1200
    cached_ratio: float = 0.0      # girdi token'larının cache'ten okunan oranı
    time_value_cents_per_s: float = 0.5   # 1 saniye gecikmenin senin için değeri (sent)
    quality_cfg: dict | None = None
    model_ctx: int | None = None
    precision_known: bool = False

    def apply(self, ep: EP) -> EP:
        cached = self.input_tokens * self.cached_ratio
        fresh = self.input_tokens - cached
        cost = fresh * ep.p_in + cached * ep.p_cache + self.output_tokens * ep.p_out
        ep.raw_cents = cost * 100
        ep.turn_cents = cost * 100
        secs = 0.0
        if ep.ttft_ms is not None:
            secs += ep.ttft_ms / 1000.0
        if ep.tps:
            secs += self.output_tokens / ep.tps
        ep.turn_s = secs
        up = ep.uptime
        # uptime: başarısız istek = tekrar deneme. Beklenen maliyet ve süre 1/p ile büyür.
        risk = (100.0 / up) if up and up > 0 else 1.0
        ep.raw_s = secs
        ep.risk = risk
        ep.turn_cents = ep.turn_cents * risk
        ep.wait_cents = self.time_value_cents_per_s * secs * risk
        ep.quality = self.quality(ep)
        ep.score = (ep.turn_cents + ep.wait_cents) / ep.quality
        return ep

    def quality(self, ep: EP) -> float:
        """Kalite çarpanı — yalnız BEYAN EDİLEN kötü format cezalanır.

        fp4/int4 ağırlık, structured-output eksikliği, model context'inden küçük
        bağlam. Precision bildirilmiyorsa (None) ceza yok: bilinmeyen nötrdür.
        """
        cfg = {**DEFAULT_QUALITY, **(self.quality_cfg or {})}
        if not cfg.get("enabled", True):
            return 1.0
        q = 1.0
        quant = (ep.quant or "").lower()
        if quant in {"fp4", "int4", "int3", "int2"}:
            q *= float(cfg.get("fp4_penalty", 0.80))
        elif not quant and self.precision_known:
            q *= float(cfg.get("unknown_precision_penalty", 1.00))
        if "structured-output" not in (ep.tags or []):
            q *= float(cfg.get("missing_structured_output", 0.97))
        ctx = self.model_ctx or 0
        floor = float(cfg.get("context_shrink_min_ratio", 0.90))
        if ctx and ep.ctx and int(ep.ctx) < int(ctx) * floor:
            q *= float(cfg.get("context_shrink", 0.98))
        return max(q, 0.5)
        return ep


# -------------------------------------------------------------------- seçim
def resolve_precision(eps: list[EP], mode: str, require: str = "fp8",
                      exclude: tuple = ("fp4", "int4"),
                      policy: dict | None = None) -> dict:
    """`mode: auto` → beyan dağılımına göre politikayı türet (planning.py ile aynı)."""
    if mode != "auto":
        return {"mode": mode, "require": require if mode == "strict" else None,
                "exclude": tuple(exclude) if mode == "exclude" else (),
                "keep_unknown": mode != "strict", "source": "explicit"}
    live = [e for e in eps if e.status == 0]
    if not live:
        return {"mode": "any", "require": None, "exclude": (), "keep_unknown": True, "source": "auto:aday-yok"}
    declared = [(e.quant or "").lower() for e in live if e.quant]
    unknown_share = 1.0 - len(declared) / len(live)
    fp8_n = sum(1 for q in declared if q == "fp8")
    bad_n = sum(1 for q in declared if q in {"fp4", "int4", "int3", "int2"})
    cfg = {**DEFAULT_PRECISION_POLICY, **(policy or {})}
    if not declared:
        return {"mode": "any", "require": None, "exclude": (), "keep_unknown": True, "source": "auto:beyan-yok"}
    if fp8_n >= int(cfg["strict_min_pool"]) and unknown_share <= float(cfg["strict_max_unknown_share"]):
        return {"mode": "strict", "require": "fp8", "exclude": (), "keep_unknown": False,
                "source": f"auto:strict(fp8={fp8_n},bildirimsiz={unknown_share:.0%})"}
    if bad_n:
        return {"mode": "exclude", "require": None, "exclude": ("fp4", "int4"), "keep_unknown": True,
                "source": f"auto:exclude-fp4(fp8={fp8_n},fp4={bad_n},bildirimsiz={unknown_share:.0%})"}
    return {"mode": "any", "require": None, "exclude": (), "keep_unknown": True,
            "source": f"auto:any(fp8={fp8_n},bildirimsiz={unknown_share:.0%})"}


def select(eps: list[EP], wl: Workload, *, quant=None, require_tools=True,
           min_uptime=None, min_ctx=None, require_zdr=False,
           sticky=None, sticky_margin_pct=None, max_candidates=None,
           allow_unknown_quant=False, min_candidates: int = 2,
           precision_mode: str | None = None,
           precision_exclude: tuple = ("fp4", "int4"),
           precision_policy: dict | None = None) -> dict:
    rejected = {}
    healthy = [e for e in eps if e.status == 0]
    rejected["status"] = len(eps) - len(healthy)

    # Precision politikası: "strict" → yalnız bildirilen format,
    # "exclude" → bildirilen kötü formatlar dışlanır (bildirimsizler kalır),
    # "any" → filtre yok. Sunucu tarafında aynısı `has` ile uygulanır.
    if precision_mode is None:
        precision_mode = "strict" if quant else "auto"
        if allow_unknown_quant and quant:
            precision_mode = "exclude"
    rp = resolve_precision(eps, precision_mode, require=quant or "fp8",
                           exclude=precision_exclude, policy=precision_policy)
    eff = rp["mode"]
    wl.precision_known = any((e.quant or "").lower() in {"fp8", "fp4"} for e in eps)
    if eff == "strict" and rp["require"]:
        has = [f"quantization:{rp['require']}"]
        precision_pred = (lambda e: (e.quant or "").lower() == rp["require"]
                          or (rp["keep_unknown"] and not e.quant))
    elif eff == "exclude":
        has = [f"!quantization:{v}" for v in rp["exclude"]]
        precision_pred = (lambda e: (e.quant or "").lower() not in set(rp["exclude"]))
    else:
        has = []
        precision_pred = None

    filters = []
    if precision_pred is not None:
        filters.append(("quant", precision_pred))
    if require_tools:
        filters.append(("tools", lambda e: e.tools))
    if min_uptime is not None:
        filters.append(("uptime", lambda e: (e.uptime or 0) >= min_uptime))
    if min_ctx:
        filters.append(("context", lambda e: (e.ctx or 0) >= min_ctx))
    if require_zdr:
        filters.append(("zdr", lambda e: e.zdr))

    def apply_filters(skip: set[str]) -> list[EP]:
        pool = list(healthy)
        for label, pred in filters:
            if label in skip:
                continue
            keep = [e for e in pool if pred(e)]
            if len(keep) != len(pool):
                rejected[label] = len(pool) - len(keep)
            pool = keep
        return pool

    pool = apply_filters(set())
    relaxed: list[str] = []
    # Tek sağlayıcıya kilitlenmeyi önle: sırayla opsiyonel filtreleri gevşet.
    # quant (precision) ASLA gevşetilmez: kalite sözleşmesi. Sadece operasyonel
    # filtreler gevşer; tek sağlayıcı kalırsa kilit kabul edilir ve işaretlenir.
    for label in ("uptime", "context", "zdr"):
        if len(pool) >= min_candidates:
            break
        if not any(l == label for l, _ in filters) or label in relaxed:
            continue
        trial = apply_filters(set(relaxed) | {label})
        if len(trial) > len(pool):
            relaxed.append(label)
            pool = trial
    for label in relaxed:
        rejected.pop(label, None)

    for e in pool:
        wl.apply(e)
    ranked = sorted(pool, key=lambda e: e.score)
    if not ranked:
        return {"error": "filtrelerden geçen sağlayıcı yok", "rejected": rejected}

    chosen, note = ranked[0], "en düşük toplam bedel"
    if sticky:
        for e in ranked:
            if e.provider == sticky:
                if chosen.score <= 0 or (e.score - chosen.score) / chosen.score <= (sticky_margin_pct or 0) / 100:
                    chosen, note = e, f"sticky {sticky} korundu"
                else:
                    note = f"sticky {sticky} bırakıldı (+%{100*(e.score/chosen.score-1):.1f} bedel)"
                break

    order = [chosen.provider] + [e.provider for e in ranked if e.provider != chosen.provider]
    if max_candidates:
        order = order[:max_candidates]
    return {"chosen": chosen, "note": note, "ranked": ranked, "rejected": rejected,
            "relaxed": relaxed,
            "single_provider": len(ranked) == 1,
            "precision": rp,
            "gateway": {"order": order,
                        "has": list(has),
                        "precision_mode": eff}}


def crossover(a: EP, b: EP) -> float | None:
    """İki sağlayıcının skorunu eşitleyen zaman değeri (¢/sn).

    Workload.apply() ikisine de uygulanmış olmalı.
    Dönüş: tv* — bu değerin ÜSTÜNDE `b` kazanır, altında `a`.
    None → beklenen süreleri eşit; seçim zaman değerinden bağımsız.
    """
    da = a.raw_s * a.risk
    db = b.raw_s * b.risk
    if abs(da - db) < 1e-12:
        return None
    # a: ca + tv*da  =  b: cb + tv*db   →   tv = (cb-ca)/(da-db)
    tv = (b.raw_cents * b.risk - a.raw_cents * a.risk) / (da - db)
    return max(tv, 0.0)


def table(ranked: list[EP], wl: Workload) -> str:
    h = (f"{'#':>2} {'provider':14s} {'q':4s} {'$/1M in':>8s} {'$/1M out':>9s} "
         f"{'tps':>6s} {'ttft':>6s} {'up%':>6s} {'iş¢':>7s} {'bekle¢':>7s} {'skor¢':>7s}")
    lines = [h, "-" * len(h)]
    for i, e in enumerate(ranked, 1):
        lines.append(
            f"{i:>2} {e.provider:14s} {e.quant or '-':4s} {e.per_1m_in:8.2f} {e.per_1m_out:9.2f} "
            f"{(e.tps if e.tps is not None else 0):6.0f} {(e.ttft_ms if e.ttft_ms is not None else 0):6.0f} "
            f"{e.uptime:6.2f} {e.turn_cents:7.3f} {e.wait_cents:7.3f} {e.score:7.3f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--quant", default=None, help="fp8 / fp4 / bf16 …")
    ap.add_argument("--in", dest="inp", type=int, default=8000)
    ap.add_argument("--out", dest="outp", type=int, default=1200)
    ap.add_argument("--cached", type=float, default=0.0, help="girdinin cache'ten gelen oranı 0-1")
    ap.add_argument("--time-value", type=float, default=0.5, help="sent/saniye")
    ap.add_argument("--min-uptime", type=float, default=None)
    ap.add_argument("--min-ctx", type=int, default=None)
    ap.add_argument("--zdr", action="store_true")
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--sticky", default=None)
    ap.add_argument("--sticky-margin-pct", type=float, default=None)
    ap.add_argument("--max-candidates", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--crossover", default=None, help="A,B → iki sağlayıcı arasındaki denge zaman-değeri")
    a = ap.parse_args()

    if a.list_models:
        cat = _get("/models")["data"]
        for m in sorted(cat, key=lambda x: x["id"]):
            if m.get("type") == "language":
                print(m["id"])
        return
    if not a.model:
        sys.exit("--model gerekli")

    wl = Workload(a.inp, a.outp, a.cached, a.time_value)
    meta, eps = load(a.model)

    if a.crossover:
        pa, pb = [x.strip() for x in a.crossover.split(",")]
        by = {e.provider: e for e in eps}
        for name in (pa, pb):
            if name not in by:
                sys.exit(f"sağlayıcı yok: {name} (mevcut: {sorted(by)})")
        ea, eb = wl.apply(by[pa]), wl.apply(by[pb])
        tv = crossover(ea, eb)
        if tv is None:
            print(f"{pa} ≈ {pb}: beklenen süreler eşit, seçim zaman değerinden bağımsız.")
        else:
            print(f"{pa} ({ea.raw_s:.1f}s) vs {pb} ({eb.raw_s:.1f}s)")
            print(f"  denge zaman-değeri ≈ {tv:.3f} ¢/sn")
            print(f"  zaman değerin bunun ALTINDAysa → {pa} daha iyi")
            print(f"  zaman değerin bunun ÜSTÜNDEyse → {pb} daha iyi")
        return
    res = select(eps, wl, quant=a.quant, require_tools=not a.no_tools,
                 min_uptime=a.min_uptime, min_ctx=a.min_ctx, require_zdr=a.zdr,
                 sticky=a.sticky, sticky_margin_pct=a.sticky_margin_pct,
                 max_candidates=a.max_candidates)

    if a.json:
        print(json.dumps({
            "model": a.model,
            "workload": {"in": wl.input_tokens, "out": wl.output_tokens,
                         "cached": wl.cached_ratio, "time_value_cents_per_s": wl.time_value_cents_per_s},
            "chosen": res["chosen"].provider if "chosen" in res else None,
            "note": res.get("note"),
            "gateway": res.get("gateway"),
            "rejected": res.get("rejected"),
            "ranked": [{"provider": e.provider, "quant": e.quant,
                        "per_1m_in": round(e.per_1m_in, 4), "per_1m_out": round(e.per_1m_out, 4),
                        "tps": e.tps, "ttft_ms": e.ttft_ms,
                        "uptime": round(e.uptime, 3) if e.uptime else None,
                        "turn_cents": round(e.turn_cents, 4),
                        "wait_cents": round(e.wait_cents, 4),
                        "score_cents": round(e.score, 4)} for e in res.get("ranked", [])],
        }, indent=1, ensure_ascii=False))
        return

    print(f"\n{meta.get('name')} ({a.model})")
    print(f"iş yükü: {wl.input_tokens} girdi / {wl.output_tokens} çıktı "
          f"| cache oranı {wl.cached_ratio:.0%} | zaman değeri {wl.time_value_cents_per_s} ¢/sn")
    print(f"filtre: quant={a.quant or '-'} tools={not a.no_tools} "
          f"min_uptime={a.min_uptime} min_ctx={a.min_ctx} zdr={a.zdr}")
    print("elenen:", res.get("rejected"))
    print()
    print(table(res.get("ranked", []), wl))
    if "chosen" in res:
        c = res["chosen"]
        print(f"\nSEÇİM: {c.provider}  ({res['note']})")
        print(f"  iş {c.turn_cents:.3f}¢ + bekleme {c.wait_cents:.3f}¢ = {c.score:.3f}¢"
              f"   (~{c.turn_s:.2f}s beklenen süre)")
        if len(res["ranked"]) > 1:
            s = res["ranked"][1]
            print(f"  2. {s.provider}: {s.score:.3f}¢  (fark {s.score-c.score:+.3f}¢)")
        print("\ngateway:", json.dumps(res["gateway"], ensure_ascii=False))


if __name__ == "__main__":
    main()
