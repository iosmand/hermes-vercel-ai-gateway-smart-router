#!/usr/bin/env python3
"""AI Gateway provider seçici — prototip (keşif aşaması).

Amaç: bir model için sağlayıcıları (endpoint'leri) sert filtrelerden geçirip
ağırlıklı puanla sıralamak, sonucu `providerOptions.gateway.order/only`
planına çevirmek.

Katalog + metrikler canlı API'den gelir:
  GET /v1/models                              → model kayıtları
  GET /v1/models/{creator}/{model}/endpoints   → provider, quantization,
                                                 pricing, tps, ttft, uptime

Kullanım:
  python3 gwscore.py --model zai/glm-5.3 --quant fp8
  python3 gwscore.py --model zai/glm-5.3-flash --quant fp8 --weights price=0.5,tps=0.35,ttft=0.05,uptime=0.10
  python3 gwscore.py --model zai/glm-5.3 --quant fp8 --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field

BASE = "https://ai-gateway.vercel.sh/v1"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_TTL = 300  # saniye


# ---------------------------------------------------------------- API katmanı
def api_key() -> str:
    key = os.environ.get("AI_GATEWAY_API_KEY")
    if key:
        return key
    env_file = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / ".env"
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.split("=", 1)[0].strip() == "AI_GATEWAY_API_KEY":
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("AI_GATEWAY_API_KEY bulunamadı")


def _get(path: str) -> dict:
    os.makedirs(CACHE, exist_ok=True)
    slug = path.strip("/").replace("/", "_") or "root"
    cf = os.path.join(CACHE, slug + ".json")
    if os.path.exists(cf) and time.time() - os.path.getmtime(cf) < CACHE_TTL:
        return json.load(open(cf, encoding="utf-8"))
    req = urllib.request.Request(BASE + path, headers={
        "Authorization": "Bearer " + api_key(),
        "User-Agent": "hermes-router-proto",
    })
    with urllib.request.urlopen(req, timeout=45) as r:
        data = json.loads(r.read())
    json.dump(data, open(cf, "w", encoding="utf-8"))
    return data


def endpoints(model: str) -> dict:
    return _get(f"/models/{model}/endpoints")["data"]


# ------------------------------------------------------------------- modeller
@dataclass
class Candidate:
    provider: str
    quant: str | None
    in_price: float        # USD / 1M prompt token
    out_price: float       # USD / 1M completion token
    cache_price: float
    tps: float | None
    ttft: float | None
    up15: float | None
    up1h: float | None
    up1d: float | None
    status: int
    tools: bool
    context: int | None
    tags: list = field(default_factory=list)
    score: float = 0.0
    parts: dict = field(default_factory=dict)

    @property
    def blend(self) -> float:
        """5 input : 1 output iş yükü için 1M karma token maliyeti."""
        return (5 * self.in_price + self.out_price) / 6

    @property
    def uptime(self) -> float:
        """Kısa vadeye ağırlıklı uptime (15m %50, 1h %30, 1d %20)."""
        vals = [(self.up15, 0.5), (self.up1h, 0.3), (self.up1d, 0.2)]
        have = [(v, w) for v, w in vals if v is not None]
        if not have:
            return 0.0
        tw = sum(w for _, w in have)
        return sum(v * w for v, w in have) / tw


def load(model: str) -> list[Candidate]:
    out = []
    for e in endpoints(model)["endpoints"]:
        p = e.get("pricing") or {}
        tp = e.get("throughput_last_1h") or {}
        la = e.get("latency_last_1h") or {}
        out.append(Candidate(
            provider=e["provider_name"],
            quant=(e.get("quantization") or None),
            in_price=float(p.get("prompt") or 0) * 1e6,
            out_price=float(p.get("completion") or 0) * 1e6,
            cache_price=float(p.get("input_cache_read") or 0) * 1e6,
            tps=tp.get("p50"),
            ttft=la.get("p50"),
            up15=e.get("uptime_last_15m"),
            up1h=e.get("uptime_last_1h"),
            up1d=e.get("uptime_last_1d"),
            status=int(e.get("status") or 0),
            tools="tool-use" in (e.get("tags") or []),
            context=e.get("context_length"),
            tags=list(e.get("tags") or []),
        ))
    return out


# ------------------------------------------------------------------ puanlama
DEFAULT_WEIGHTS = {"price": 0.45, "tps": 0.35, "ttft": 0.10, "uptime": 0.10}


def _norm(value: float, lo: float, hi: float, invert: bool = False) -> float:
    """0 = en iyi, 1 = en kötü. Tüm adaylar eşitse 0 döner."""
    if hi - lo < 1e-12:
        return 0.0
    x = (value - lo) / (hi - lo)
    return 1.0 - x if invert else x


def rank(cands: list[Candidate], weights: dict[str, float]) -> list[Candidate]:
    """Ağırlıklı min-max puanlama; düşük puan = tercih edilen sağlayıcı."""
    if not cands:
        return []
    blends = [c.blend for c in cands]
    tpss = [c.tps for c in cands if c.tps]
    ttfts = [c.ttft for c in cands if c.ttft]
    ups = [c.uptime for c in cands if c.uptime]

    p_lo, p_hi = min(blends), max(blends)
    t_lo, t_hi = (min(tpss), max(tpss)) if tpss else (0, 0)
    a_lo, a_hi = (min(ttfts), max(ttfts)) if ttfts else (0, 0)
    u_lo, u_hi = (min(ups), max(ups)) if ups else (0, 0)

    for c in cands:
        parts = {
            "price": _norm(c.blend, p_lo, p_hi),                       # ucuz iyi
            "tps": _norm(c.tps, t_lo, t_hi, invert=True) if c.tps else 1.0,
            "ttft": _norm(c.ttft, a_lo, a_hi) if c.ttft else 1.0,      # düşük ms iyi
            "uptime": _norm(c.uptime, u_lo, u_hi, invert=True) if c.uptime else 1.0,
        }
        c.parts = {k: round(v, 4) for k, v in parts.items()}
        c.score = round(sum(weights[k] * v for k, v in parts.items()), 6)
    return sorted(cands, key=lambda c: (c.score, c.blend))


def select(cands: list[Candidate], *, quant: str | None = None,
           require_tools: bool = True, min_uptime_15m: float | None = None,
           min_context: int | None = None, allow_unknown_quant: bool = False,
           weights: dict[str, float] | None = None,
           sticky: str | None = None, sticky_margin: float = 0.05) -> dict:
    weights = weights or DEFAULT_WEIGHTS
    pool = [c for c in cands if c.status == 0]
    rejected = {"status": len(cands) - len(pool)}

    if quant:
        if allow_unknown_quant:
            keep = [c for c in pool if c.quant == quant or c.quant is None]
        else:
            keep = [c for c in pool if c.quant == quant]
        rejected["quant"] = len(pool) - len(keep)
        pool = keep
    if require_tools:
        keep = [c for c in pool if c.tools]
        rejected["tools"] = len(pool) - len(keep)
        pool = keep
    if min_uptime_15m is not None:
        keep = [c for c in pool if (c.up15 or 0) >= min_uptime_15m]
        rejected["uptime15"] = len(pool) - len(keep)
        pool = keep
    if min_context is not None:
        keep = [c for c in pool if (c.context or 0) >= min_context]
        rejected["context"] = len(pool) - len(keep)
        pool = keep

    ranked = rank(pool, weights)
    if not ranked:
        return {"error": "filtrelerden geçen sağlayıcı yok", "rejected": rejected}

    chosen = ranked[0]
    note = "en iyi puan"
    if sticky:
        for c in ranked:
            if c.provider == sticky:
                if c.score - chosen.score <= sticky_margin:
                    chosen, note = c, f"sticky: {sticky} (marj içinde)"
                else:
                    note = f"sticky bırakıldı: {sticky} puanı {c.score:.3f} > {chosen.score:.3f}+{sticky_margin}"
                break

    others = [c.provider for c in ranked if c.provider != chosen.provider]
    return {
        "model": None,
        "chosen": chosen,
        "note": note,
        "ranked": ranked,
        "gateway": {
            "order": [chosen.provider] + others,
            "only": sorted({c.provider for c in ranked}),
        },
        "rejected": rejected,
    }


# ----------------------------------------------------------------------- CLI
def fmt_table(ranked: list[Candidate]) -> str:
    head = (f"{'#':>2} {'provider':15s} {'quant':5s} {'blend':>7s} {'in':>6s} {'out':>6s} "
            f"{'tps':>6s} {'ttft':>6s} {'up%':>7s} {'score':>6s}  price  tps ttft   up")
    lines = [head, "-" * len(head)]
    for i, c in enumerate(ranked, 1):
        q = c.quant or "-"
        lines.append(
            f"{i:>2} {c.provider:15s} {q:5s} {c.blend:7.3f} {c.in_price:6.2f} {c.out_price:6.2f} "
            f"{c.tps if c.tps is not None else '-':>6} {c.ttft if c.ttft is not None else '-':>6} "
            f"{c.uptime:7.2f} {c.score:6.3f}  "
            f"{c.parts['price']:.2f}  {c.parts['tps']:.2f} {c.parts['ttft']:.2f} {c.parts['uptime']:.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default=None, help="fp8 / fp4 / bf16 ... (orijinal precision)")
    ap.add_argument("--allow-unknown-quant", action="store_true")
    ap.add_argument("--weights", default=None, help="price=0.45,tps=0.35,ttft=0.1,uptime=0.1")
    ap.add_argument("--min-uptime-15m", type=float, default=None)
    ap.add_argument("--min-context", type=int, default=None)
    ap.add_argument("--no-require-tools", action="store_true")
    ap.add_argument("--sticky", default=None)
    ap.add_argument("--sticky-margin", type=float, default=0.05)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    weights = dict(DEFAULT_WEIGHTS)
    if a.weights:
        for kv in a.weights.split(","):
            k, v = kv.split("=")
            weights[k.strip()] = float(v)
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

    cands = load(a.model)
    res = select(cands, quant=a.quant, require_tools=not a.no_require_tools,
                 min_uptime_15m=a.min_uptime_15m, min_context=a.min_context,
                 allow_unknown_quant=a.allow_unknown_quant,
                 weights=weights, sticky=a.sticky, sticky_margin=a.sticky_margin)
    res["model"] = a.model

    if a.json:
        print(json.dumps({
            "model": a.model,
            "weights": weights,
            "chosen": res["chosen"].provider if "chosen" in res else None,
            "note": res.get("note"),
            "gateway": res.get("gateway"),
            "rejected": res.get("rejected"),
            "ranked": [{"provider": c.provider, "quant": c.quant, "blend": round(c.blend, 4),
                        "tps": c.tps, "ttft": c.ttft, "uptime": round(c.uptime, 3),
                        "score": c.score, "parts": c.parts} for c in res.get("ranked", [])],
        }, indent=1, ensure_ascii=False))
        return

    print(f"\n{res['model']}  |  filtre: quant={a.quant or 'yok'} tools={not a.no_require_tools} "
          f"|  ağırlık: " + " ".join(f"{k}={v:.2f}" for k, v in weights.items()))
    print("elenen:", res.get("rejected"))
    print()
    print(fmt_table(res.get("ranked", [])))
    if "chosen" in res:
        g = res["gateway"]
        print(f"\nseçim: {res['chosen'].provider}  ({res['note']})")
        print("gateway planı:", json.dumps(g, ensure_ascii=False))
        print("\norder =", g["order"][:5], "..." if len(g["order"]) > 5 else "")


if __name__ == "__main__":
    main()
