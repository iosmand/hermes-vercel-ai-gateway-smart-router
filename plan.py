#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""plan.py — router.yaml'daki kuralları her model için gateway planına çevirir.

  python3 plan.py                          # tüm modeller için plan tablosu
  python3 plan.py --model zai/glm-5.3-flash
  python3 plan.py --write plans.json       # plugin'in okuyacağı plan dosyası
  python3 plan.py --live zai/glm-5.3-flash # planı gerçek istekle dene (plan vs en ucuz)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

from gwroute import (BASE, EP, Workload, api_key, crossover, load, select, table)

HERE = os.path.dirname(os.path.abspath(__file__))


def _yaml():
    """PyYAML is only needed for router.yaml input, not for --check-rules."""
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML gerekli: python3 -m pip install pyyaml")
    return yaml


def merge(defaults: dict, model_cfg: dict) -> dict:
    out = json.loads(json.dumps(defaults))  # derin kopya
    for k, v in (model_cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def export_rules(cfg_path: str, out_path: str) -> dict:
    """router.yaml → rules.json (plugin'in çalışma zamanında okuduğu dosya)."""
    cfg = _yaml().safe_load(open(cfg_path, encoding="utf-8"))
    defaults = cfg.get("defaults") or {}
    models = {}
    for model, mcfg in (cfg.get("models") or {}).items():
        c = merge(defaults, mcfg or {})
        models[model] = {
            "role": c.get("role"),
            "precision": c.get("precision") or {"mode": "any"},
            "workload": c.get("workload") or {},
            "require_tools": bool(c.get("require_tools", True)),
            "min_uptime_15m": c.get("min_uptime_15m"),
            "min_context": c.get("min_context"),
            "max_candidates": int(c.get("max_candidates", 4)),
            "time_value_cents_per_s": float(c.get("time_value_cents_per_s", 0.15)),
            "session_affinity": bool(c.get("session_affinity", True)),
            "stats_ttl_s": float(c.get("stats_ttl_s", 600)),
            "precision_policy": c.get("precision_policy") or {},
            "quality": c.get("quality") or {},
            "model_context": c.get("model_context"),
        }
    payload = {"generated_from": cfg_path, "models": models}
    json.dump(payload, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return payload


def build(cfg_path: str) -> dict:
    cfg = _yaml().safe_load(open(cfg_path, encoding="utf-8"))
    defaults = cfg.get("defaults") or {}
    plans = {}
    for model, mcfg in (cfg.get("models") or {}).items():
        c = merge(defaults, mcfg or {})
        wl_cfg = c.get("workload") or {}
        meta, eps = load(model)
        model_ctx = c.get("model_context") or max(
            (e.ctx or 0 for e in eps), default=0) or None
        wl = Workload(
            input_tokens=int(wl_cfg.get("input_tokens", 8000)),
            output_tokens=int(wl_cfg.get("output_tokens", 1200)),
            cached_ratio=float(wl_cfg.get("cached_ratio", 0.0)),
            time_value_cents_per_s=float(c.get("time_value_cents_per_s", 0.35)),
            quality_cfg=c.get("quality") or {},
            model_ctx=model_ctx,
        )
        prec = c.get("precision") or {}
        mode = prec.get("mode", "auto")
        res = select(
            eps, wl,
            quant=prec.get("require", "fp8"),
            precision_mode=mode,
            precision_exclude=tuple(prec.get("exclude") or ["fp4", "int4"]),
            precision_policy=c.get("precision_policy"),
            require_tools=bool(c.get("require_tools", True)),
            min_uptime=c.get("min_uptime_15m"),
            min_ctx=c.get("min_context"),
            require_zdr=bool(c.get("require_zdr", False)),
            sticky=c.get("sticky"),
            sticky_margin_pct=c.get("sticky_margin_pct"),
            max_candidates=c.get("max_candidates"),
        )
        if "error" in res:
            plans[model] = {"model": model, "role": c.get("role"), "error": res["error"],
                            "rejected": res.get("rejected")}
            continue
        ch = res["chosen"]
        plans[model] = {
            "model": model,
            "role": c.get("role"),
            "precision": res.get("precision") or prec,
            "workload": {"in": wl.input_tokens, "out": wl.output_tokens,
                         "cached": wl.cached_ratio, "time_value": wl.time_value_cents_per_s},
            "chosen": ch.provider,
            "reason": res["note"],
            "score_cents": round(ch.score, 4),
            "work_cents": round(ch.turn_cents, 4),
            "wait_cents": round(ch.wait_cents, 4),
            "expected_s": round(ch.raw_s * ch.risk, 2),
            "runner_up": (res["ranked"][1].provider if len(res["ranked"]) > 1 else None),
            "margin_cents": (round(res["ranked"][1].score - ch.score, 4)
                             if len(res["ranked"]) > 1 else None),
            "gateway": {
                "order": res["gateway"]["order"],
                "has": res["gateway"]["has"],
                "sessionAffinity": bool(c.get("session_affinity", True)),
            },
            "rejected": res["rejected"],
            "relaxed": res.get("relaxed") or [],
            "ranked": [{"provider": e.provider, "score": round(e.score, 4), "quality": round(e.quality, 3),
                        "tps": e.tps, "ttft_ms": e.ttft_ms} for e in res["ranked"]],
        }
    return plans


# ------------------------------------------------------------------ canlı test
PROMPT = ("Bir Python fonksiyonundaki üç hata durumunu sırayla açıkla ve her biri için "
          "düzeltme öner. Sonunda refactor edilmiş tek sürüm yaz.")


def live(model: str, plan: dict, max_tokens: int = 400) -> None:
    """Planı ve 'yalnız en ucuz' alternatifini gerçek istekle ölçer."""
    from bench import stream_call

    order = plan["gateway"]["order"]
    cheapest = plan["ranked"][-1]["provider"] if plan.get("ranked") else None
    runs = [("plan", order[0], order), ("en ucuz", cheapest, [cheapest])]
    print(f"\n{model} — canlı karşılaştırma (max_tokens={max_tokens})")
    print(f"plan order: {order[:plan['gateway'].get('max', 4)]}")
    for label, provider, ordered in runs:
        payload_order = ordered if label == "plan" else [cheapest]
        try:
            r = stream_call(model, provider, max_tokens, reasoning_effort="low",
                            order_override=payload_order)
        except Exception as e:
            print(f"  {label:10s} HATA {type(e).__name__}: {e}")
            continue
        print(f"  {label:10s} ilk={provider:12s} ttft={r['ttft_s']:.2f}s "
              f"toplam={r['total_s']:.2f}s tps={r.get('observed_tps')} "
              f"cost=${r['cost_usd']} karakter={r['chars']}")


def _planning_module():
    """Load the plugin's planning module for rule validation."""
    import importlib.util
    path = os.path.join(HERE, "plugin", "planning.py")
    spec = importlib.util.spec_from_file_location("ai_gateway_router_planning_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_rules(path: str | None = None) -> int:
    """Validate a rules.json file; exit 1 (and list problems) when it is invalid."""
    if not path:
        home = os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")
        path = os.path.join(home, "ai-gateway-router", "rules.json")
    if not os.path.isfile(path):
        print(f"yok: {path}")
        return 1
    try:
        data = json.loads(open(path, encoding="utf-8").read())
    except ValueError as exc:
        print(f"geçersiz JSON: {exc}")
        return 1
    problems = _planning_module().validate_rules(data)
    if problems:
        print(f"{path}: {len(problems)} sorun")
        for p in problems:
            print("  -", p)
        return 1
    models = data.get("models") or {}
    print(f"{path}: geçerli ({len(models)} model kuralı)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "router.yaml"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--write", default=None)
    ap.add_argument("--force", action="store_true",
                    help="mevcut rules.json'u --write sırasında ez")
    ap.add_argument("--check-rules", dest="check_rules", nargs="?", const="",
                    default=None, metavar="RULES_JSON",
                    help="rules.json doğrula ve çık (yol verilmezse $HERMES_HOME/ai-gateway-router/rules.json)")
    ap.add_argument("--live", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.check_rules is not None:
        sys.exit(check_rules(a.check_rules))

    if a.live:
        cfg = _yaml().safe_load(open(a.config, encoding="utf-8"))
        c = merge(cfg.get("defaults") or {}, (cfg.get("models") or {}).get(a.live) or {})
        plans = build(a.config)
        live(a.live, plans[a.live])
        return

    plans = build(a.config)
    if a.model:
        plans = {a.model: plans[a.model]} if a.model in plans else sys.exit(f"yok: {a.model}")
    if a.write:
        rules_out = os.path.join(os.path.dirname(os.path.abspath(a.write)), "rules.json")
        if os.path.exists(rules_out) and not a.force:
            sys.exit(f"{rules_out} zaten var — ezmek için --force kullanın")
        json.dump(plans, open(a.write, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        export_rules(a.config, rules_out)
        print("yazıldı:", a.write, "+", os.path.basename(rules_out))
    if a.json:
        print(json.dumps(plans, indent=1, ensure_ascii=False))
        return

    for model, p in plans.items():
        print(f"\n{'='*78}\n{model}  —  {p.get('role') or ''}")
        if "error" in p:
            print("  HATA:", p["error"], p.get("rejected"))
            continue
        w = p["workload"]
        print(f"  iş yükü: {w['in']}/{w['out']} tok, cache {w['cached']:.0%}, "
              f"zaman değeri {w['time_value']} ¢/sn")
        print(f"  SEÇİM: {p['chosen']}  ({p['reason']})")
        print(f"    iş {p['work_cents']}¢ + bekleme {p['wait_cents']}¢ = {p['score_cents']}¢  "
              f"(~{p['expected_s']}s)   yedek: {p['runner_up']} (+{p['margin_cents']}¢)")
        print(f"    order: {p['gateway']['order']}")
        print(f"    precision politikası: {p['precision'].get('source')} → mod={p['precision'].get('mode')}")
        print(f"    elenen: {p['rejected']}" + (f"   gevşetilen filtre: {p['relaxed']}" if p.get("relaxed") else ""))


if __name__ == "__main__":
    main()
