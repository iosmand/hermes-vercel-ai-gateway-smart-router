#!/usr/bin/env python3
"""Çalışma zamanı planlayıcı testleri (ağsız).

Kendi geçici HERMES_HOME'unu kurar, sentetik istatistiklerle çalışır:

    PYTHONPATH=/usr/local/lib/hermes-agent:$PYTHONPATH \
      python3 tests/test_runtime.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def ep(provider, in1m, out1m, tps, ttft, quant="fp8", up=100.0, status=0, tools=True):
    return {
        "provider_name": provider, "quantization": quant,
        "pricing": {"prompt": str(in1m / 1e6), "completion": str(out1m / 1e6),
                    "input_cache_read": str(in1m / 4e6)},
        "throughput_last_1h": {"p50": tps}, "latency_last_1h": {"p50": ttft},
        "uptime_last_15m": up, "uptime_last_1h": up, "uptime_last_1d": up,
        "status": status, "tags": (["tool-use"] if tools else []), "context_length": 1_000_000,
        "has_zdr": True,
    }


MODEL = "test/model"


def write_stats(root: Path, endpoints, age_s: float = 0.0) -> None:
    d = root / "ai-gateway-router" / "stats"
    d.mkdir(parents=True, exist_ok=True)
    (d / (MODEL.replace("/", "__") + ".json")).write_text(
        json.dumps({"fetched_at": time.time() - age_s, "endpoints": endpoints}), encoding="utf-8")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="gwrouter-test-"))
    os.environ["HERMES_HOME"] = str(tmp)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugin"))
    import planning

    # ağ yok: her fetch çağrısı sayılır
    calls = {"n": 0}

    def fake_fetch(model, timeout=None):
        calls["n"] += 1
        return None

    planning._fetch_endpoints = fake_fetch

    (tmp / "ai-gateway-router").mkdir(parents=True, exist_ok=True)
    (tmp / "ai-gateway-router" / "rules.json").write_text(json.dumps({"models": {MODEL: {
        "precision": {"mode": "strict", "require": "fp8"},
        "workload": {"input_tokens": 8000, "output_tokens": 1200, "cached_ratio": 0.0},
        "time_value_cents_per_s": 0.5, "require_tools": True,
        "max_candidates": 4, "session_affinity": True, "stats_ttl_s": 600,
    }}}), encoding="utf-8")

    cheap_slow = ep("cheap-slow", 1.0, 3.0, 30, 800)
    fast = ep("fast", 1.05, 3.15, 60, 800)
    fp4 = ep("fp4-prov", 0.5, 1.5, 90, 700, quant="fp4")
    extra = ep("extra", 5.0, 15.0, 10, 2000)   # B4: only'de 3. aday (karantina için)
    write_stats(tmp, [cheap_slow, fast, fp4, extra])

    # 1) skorlama + has filtresi
    gw = planning.build_gateway(MODEL, "sess-1")
    check("yönlendirme: order dolu", bool(gw.get("order")), json.dumps(gw))
    check("yönlendirme: has strict fp8", gw.get("has") == ["quantization:fp8"], json.dumps(gw))
    check("yönlendirme: FP4 aday elendi", "fp4-prov" not in gw.get("order", []), json.dumps(gw))
    first = gw["order"][0]
    check("skor: küçük fiyat farkında hızlı sağlayıcı kazanır", first == "fast", json.dumps(gw))

    # 2) oturum kilidi: veri tersine dönse de kilit korunur
    # cheap-slow artık hem daha ucuz hem daha hızlı → taze oturum onu seçer
    write_stats(tmp, [ep("cheap-slow", 0.5, 1.5, 120, 700), fast, fp4, extra])
    gw2 = planning.build_gateway(MODEL, "sess-1")
    check("kilit: aynı oturum aynı sağlayıcıda kalır", gw2["order"][0] == "fast", json.dumps(gw2))
    gw2b = planning.build_gateway(MODEL, "sess-2")
    check("kilit: yeni oturum yeniden seçer", gw2b["order"][0] == "cheap-slow", json.dumps(gw2b))

    # 3) strike → karantina → fallback
    hit1 = planning.record_strike(MODEL, "sess-1", "fast", "HTTP 503")
    check("strike: ilk hatada karantina yok", hit1 is None)
    # B3: 30 sn soğuma — 2. strike'ın sayılması için kaydı geriye al
    st = planning._json_state("strikes.json")
    st["fast"]["ts"] = time.time() - planning.STRIKE_COOLDOWN_S - 1
    planning._save_state("strikes.json", st)
    hit2 = planning.record_strike(MODEL, "sess-1", "fast", "HTTP 503")
    check("strike: ikinci hatada karantina", hit2 == "fast")
    gw3 = planning.build_gateway(MODEL, "sess-1")
    check("fallback: karantinalı sağlayıcı sıralamada yok", "fast" not in gw3["order"], json.dumps(gw3))
    check("fallback: kilit sıradakine geçti", gw3["order"][0] == "cheap-slow", json.dumps(gw3))

    # 4) karantina süresi dolunca geri döner
    dem = tmp / "ai-gateway-router" / "demoted.json"
    dem.write_text(json.dumps({"fast": {"until": time.time() - 1, "reason": "test"}}), encoding="utf-8")
    gw4 = planning.build_gateway(MODEL, "sess-9")
    check("karantina: süre dolunca aday geri gelir", "fast" in gw4["order"], json.dumps(gw4))

    # 5) TTL: bayat veri anında kullanılır, yenileme arka planda
    calls["n"] = 0
    write_stats(tmp, [cheap_slow, fast], age_s=3600)
    gw5 = planning.build_gateway(MODEL, "sess-3")
    check("TTL: bayat cache yine de kullanıldı", bool(gw5.get("order")), json.dumps(gw5))
    time.sleep(0.4)
    check("TTL: arka plan yenilemesi tetiklendi", calls["n"] >= 1, f"fetch çağrısı={calls['n']}")

    # 6) taze veri yenileme tetiklemez
    calls["n"] = 0
    write_stats(tmp, [cheap_slow, fast], age_s=0)
    planning.build_gateway(MODEL, "sess-4")
    time.sleep(0.3)
    check("TTL: taze cache yenileme yapmaz", calls["n"] == 0, f"fetch çağrısı={calls['n']}")

    # 7) kural yoksa yönlendirme kapalı
    check("kural yoksa boş plan", planning.build_gateway("bilinmeyen/model", "s") == {})

    # 8) kill switch plugin seviyesinde test edilir; burada yalnız oturum TTL'i
    sessions = json.loads((tmp / "ai-gateway-router" / "sessions.json").read_text())
    old_ts = time.time() - 999999
    (tmp / "ai-gateway-router" / "sessions.json").write_text(
        json.dumps({"old": {"model": MODEL, "provider": "cheap-slow", "ts": old_ts}}))
    gw8 = planning.build_gateway(MODEL, "old")
    rec = json.loads((tmp / "ai-gateway-router" / "sessions.json").read_text())["old"]
    check("oturum TTL: eski kilit yok sayıldı ve yeniden kilitlendi",
          rec["ts"] > old_ts + 1 and rec["provider"] == gw8["order"][0], json.dumps({"rec": rec, "gw": gw8}))

    # 9) otomatik precision politikası (resolve_precision)
    def eps_of(*specs):
        return [ep(f"p{i}", 1, 3, 50, 800, quant=q) for i, q in enumerate(specs)]

    rule_auto = {"precision": {"mode": "auto"}}
    cases = [
        ("hiç beyan yok → any", eps_of(None, None, None), "any"),
        ("fp8=6, bildirimsiz %14 → strict",
         eps_of("fp8", "fp8", "fp8", "fp8", "fp8", "fp8", None), "strict"),
        ("fp8=2, fp4=5, bildirimsiz %71 → exclude", eps_of("fp8", "fp8", "fp4", "fp4", "fp4", "fp4", "fp4",
                                                           None, None, None, None, None, None, None, None, None), "exclude"),
        ("fp8=4, fp4 yok, bildirimsiz %79 → any", eps_of("fp8", "fp8", "fp8", "fp8", None, None, None,
                                                          None, None, None, None, None, None), "any"),
    ]
    for name, ends, want in cases:
        got = planning.resolve_precision(rule_auto, ends)
        check(f"auto politika: {name}", got["mode"] == want, f"{got}")

    rp = planning.resolve_precision(rule_auto, eps_of("fp8", "fp8", "fp4", None, None))
    check("auto politika: exclude fp4+int4 listesi",
          rp["mode"] == "exclude" and "fp4" in rp["exclude"] and rp["require"] is None, json.dumps(rp))

    # 10) bilinmeyen precision cezası yalnız beyan varken uygulanır
    q_unknown = planning._quality({"quantization": None, "tags": ["structured-output"],
                                   "context_length": 1_000_000}, None, planning.DEFAULT_QUALITY, True)
    q_fp8 = planning._quality({"quantization": "fp8", "tags": ["structured-output"],
                               "context_length": 1_000_000}, None, planning.DEFAULT_QUALITY, True)
    check("kalite: beyan varken bilinmeyen 0.97, fp8 1.00",
          abs(q_unknown - 0.97) < 1e-9 and abs(q_fp8 - 1.0) < 1e-9, f"{q_unknown} / {q_fp8}")
    q_none_market = planning._quality({"quantization": None, "tags": ["structured-output"],
                                       "context_length": 1_000_000}, None, planning.DEFAULT_QUALITY, False)
    check("kalite: piyasada beyan yoksa bilinmeyen cezasız", abs(q_none_market - 1.0) < 1e-9, str(q_none_market))

    # 11) 4.4: implicit-caching yoksa cache kazancı sıfır
    wl = {"input_tokens": 1000, "output_tokens": 100, "cached_ratio": 0.9}
    qcfg = dict(planning.DEFAULT_QUALITY)
    e_with = ep("cache-var", 1.0, 4.0, 100, 100, quant="fp8")
    e_with["tags"] = ["tool-use", "structured-output", "implicit-caching"]
    e_without = ep("cache-yok", 1.0, 4.0, 100, 100, quant="fp8")
    e_without["tags"] = ["tool-use", "structured-output"]
    s_with = planning._score(e_with, wl, 0.15, qcfg, None, True)
    s_without = planning._score(e_without, wl, 0.15, qcfg, None, True)
    pin = planning._num(e_with["pricing"]["prompt"])
    pcache = planning._num(e_with["pricing"].get("input_cache_read"))
    pcache = pin if pcache is None else pcache
    expected = -1000 * 0.9 * (pin - pcache) * 100  # caching ucuzlatır (skor farkı negatif)
    check("4.4: implicit-caching olmayanda cache kazancı yok",
          abs((s_with - s_without) - expected) < 1e-9,
          f"delta={s_with - s_without:.8f} beklenen={expected:.8f}")

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(FAILS)} hata" if FAILS else "\ntüm testler geçti")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())