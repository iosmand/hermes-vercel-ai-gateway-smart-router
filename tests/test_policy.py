#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Policy tests for the production ranking: ``planning.rank()``.

Every case is a synthetic catalog plus a rule — no network, no state, no API key.
Run with ``python3 tests/test_policy.py`` or through ``tests/run_all.sh``.
"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FAILS: list[str] = []


def _load_planning():
    spec = importlib.util.spec_from_file_location("gw_planning_policy",
                                                  REPO / "plugin" / "planning.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


planning = _load_planning()

BASE_RULE = {
    "precision": {"mode": "strict", "require": "fp8"},
    "workload": {"input_tokens": 8000, "output_tokens": 1200, "cached_ratio": 0.0},
    "time_value_cents_per_s": 0.15,
    "require_tools": True,
    "max_candidates": 4,
    "session_affinity": False,
}


def ep(name, *, pin_cents=0.30, pout_cents=1.50, tps: float | None = 60,
       ttft_ms: float | None = 800, uptime: float | None = 100.0,
       quant="fp8", tools=True, cache_cents=None, ctx=10 ** 6, status=0):
    pricing = {"prompt": pin_cents / 100 / 1e6, "completion": pout_cents / 100 / 1e6}
    if cache_cents is not None:
        pricing["input_cache_read"] = cache_cents / 100 / 1e6
    tags = ["structured-output", "implicit-caching"] + (["tool-use"] if tools else [])
    return {"provider_name": name, "quantization": quant, "status": status,
            "pricing": pricing,
            "uptime_last_15m": uptime, "uptime_last_1h": uptime, "uptime_last_1d": uptime,
            "throughput_last_1h": {"p50": tps} if tps is not None else {},
            "latency_last_1h": {"p50": ttft_ms} if ttft_ms is not None else {},
            "tags": tags, "context_length": ctx}


def order(eps, **rule):
    ranked = planning.rank("test/model", {**BASE_RULE, **rule}, eps, {})
    return [r["provider"] for r in ranked]


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def main() -> int:
    # 1) a small price gap loses against the time value: fast beats cheap-but-slow
    got = order([ep("cheap-slow", pin_cents=0.30, tps=30),
                 ep("fast", pin_cents=0.60, tps=90)])
    check("zaman değeri küçük fiyat farkını yener (hızlı önde)", got[0] == "fast", str(got))

    # 2) a large price gap keeps the cheap provider on top: 10× the price buys
    #    only 2.8 s, which is worth 0.42 ¢ at 0.15 ¢/s — the 2.5 ¢ gap wins.
    got = order([ep("cheap", pin_cents=30.0, pout_cents=30.0, tps=100),
                 ep("pricey-fast", pin_cents=300.0, pout_cents=300.0, tps=130)])
    check("büyük fiyat farkı ucuzu korur", got[0] == "cheap", str(got))

    # 3) with zero time value money alone decides
    got = order([ep("cheap-slow", pin_cents=0.30, tps=20),
                 ep("fast", pin_cents=0.60, tps=200)],
                time_value_cents_per_s=0.0)
    check("zaman değeri 0 → en ucuz kazanır", got[0] == "cheap-slow", str(got))

    # 4) a failing provider is penalised (100% vs 50% uptime, same price/speed)
    got = order([ep("flaky", uptime=50.0), ep("solid", uptime=100.0)])
    check("uptime düşük sağlayıcı geride kalır", got[0] == "solid", str(got))

    # 5) the uptime floor is a hard filter
    got = order([ep("flaky", uptime=80.0), ep("solid", uptime=100.0)],
                min_uptime_15m=90)
    check("min_uptime_15m zayıf sağlayıcıyı eler", got == ["solid"], str(got))

    # 6) strict precision excludes fp4 even when it is cheapest and fastest
    got = order([ep("fp4-cheap", pin_cents=0.01, quant="fp4", tps=200),
                 ep("fp8", pin_cents=0.30)])
    check("strict precision fp4'ü eler", got == ["fp8"], str(got))

    # 7) a provider without metrics loses to a measured one at the same price
    got = order([ep("measured", pin_cents=0.30, tps=60),
                 ep("unmeasured", pin_cents=0.30, tps=None, ttft_ms=None)])
    check("ölçüsüz sağlayıcı ölçülüye kaybeder", got[0] == "measured", str(got))

    # 8) require_tools is a hard filter
    got = order([ep("no-tools", pin_cents=0.01, tools=False), ep("tools", pin_cents=0.30)],
                require_tools=True)
    check("require_tools araçsızı eler", got == ["tools"], str(got))

    # 9) a cache-heavy workload prefers the cheaper cache price
    eps = [ep("no-cache-price", pin_cents=0.30, cache_cents=0.30),
           ep("cache-cheap", pin_cents=0.30, cache_cents=0.03)]
    got = order(eps, workload={"input_tokens": 8000, "output_tokens": 100,
                               "cached_ratio": 0.9})
    check("cache ağırlıklı iş yükü cache fiyatını ödüllendirir", got[0] == "cache-cheap", str(got))

    # 10) a provider without pricing is dropped instead of scoring as free
    no_price = {"provider_name": "free-looking", "quantization": "fp8", "status": 0,
                "pricing": {}, "uptime_last_15m": 100, "uptime_last_1h": 100,
                "uptime_last_1d": 100, "throughput_last_1h": {"p50": 60},
                "latency_last_1h": {"p50": 800}, "tags": ["tool-use"],
                "context_length": 10 ** 6}
    got = order([no_price, ep("priced", pin_cents=5.0)])
    check("fiyatsız sağlayıcı elenir (bedava sayılmaz)", got == ["priced"], str(got))

    # 11) unknown uptime is neutral (risk 1.0) — the provider stays a candidate
    got = order([ep("unknown-up", uptime=None), ep("solid", uptime=100.0)])
    check("uptime bilinmiyorsa nötr (aday kalır)", "unknown-up" in got, str(got))

    print()
    if FAILS:
        print(f"{len(FAILS)} test BAŞARISIZ: {', '.join(FAILS)}")
        return 1
    print(f"{11}/{11} test geçti")
    return 0


if __name__ == "__main__":
    sys.exit(main())
