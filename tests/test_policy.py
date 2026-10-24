#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Politika doğrulama testleri — sentetik sağlayıcılarla seçim kurallarını sınar.

Çalıştır: python3 tests/test_policy.py   (veya depo kökünden python3 -m tests.test_policy)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gwroute import EP, Workload, crossover, select, table


def ep(prov, in1m, out1m, tps, ttft_ms, up=100.0, quant="fp8", tools=True, status=0):
    return EP(provider=prov, quant=quant, p_in=in1m / 1e6, p_out=out1m / 1e6, p_cache=in1m / 1e6 / 4,
              tps=tps, ttft_ms=ttft_ms, up15=up, up1h=up, up1d=up,
              status=status, tools=tools, ctx=1_000_000, zdr=True, tags=["tool-use"])


def run(name, eps, wl, expect, **kw):
    res = select(eps, wl, quant="fp8", require_tools=True, **kw)
    got = res["chosen"].provider
    ok = got == expect
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: seçim={got} beklenen={expect} ({res['note']})")
    if not ok:
        print(table(res["ranked"], wl))
    return ok


WL = Workload(input_tokens=8000, output_tokens=1200, time_value_cents_per_s=0.5)

results = []

# 1) Kullanıcının örneği: en ucuz 30 tps, ikinci 60 tps, fiyat farkı küçük → hızlı olan kazanmalı
results.append(run("ucuz+yavaş vs pahalı+hızlı (küçük fark)",
    [ep("cheap-slow", 1.00, 3.00, 30, 800),
     ep("fast", 1.05, 3.15, 60, 800)],
    WL, "fast"))

# 2) Kullanıcı kuralı: fark ~2.3¢ ve hızlı olan 2× tps → hızlı olan tercih edilmeli
results.append(run("~2.3¢ fark + 2× tps → hızlı",
    [ep("cheap-slow", 1.00, 3.00, 30, 800),
     ep("fast", 3.00, 9.00, 60, 800)],
    Workload(8000, 1200, 0.0, 0.5), "fast"))

# 2b) Aynı çift, zaman değeri sıfıra yakın → ucuz olan kazanmalı (monoton geçiş)
results.append(run("aynı çift, zaman değeri ≈0 → ucuz",
    [ep("cheap-slow", 1.00, 3.00, 30, 800),
     ep("fast", 3.00, 9.00, 60, 800)],
    Workload(8000, 1200, 0.0, 0.001), "cheap-slow"))

# 2c) Denge noktası gerçekten arada bir yerde olmalı
_e1, _e2 = ep("cheap-slow", 1.00, 3.00, 30, 800), ep("fast", 3.00, 9.00, 60, 800)
_w = Workload(8000, 1200, 0.0, 0.5)
_w.apply(_e1); _w.apply(_e2)
_tv = crossover(_e1, _e2)
_ok = _tv is not None and 0.001 < _tv < 0.5
print(f"[{'PASS' if _ok else 'FAIL'}] denge zaman-değeri 0.001 < {_tv:.4f} ¢/sn < 0.5")
results.append(_ok)

# 3) Uptime cezası: %70 uptime'lı ucuz sağlayıcı, sağlıklı olana kaybetmeli
results.append(run("uptime cezası",
    [ep("flaky", 0.90, 2.70, 60, 800, up=70.0),
     ep("healthy", 1.00, 3.00, 55, 850, up=100.0)],
    WL, "healthy"))

# 4) 15 dakikalık uptime çöküşü anında filtre devreye girmeli
results.append(run("min_uptime filtresi",
    [ep("degraded", 0.80, 2.40, 60, 800, up=88.0),
     ep("healthy", 1.00, 3.00, 55, 850, up=100.0)],
    WL, "healthy", min_uptime=95.0))

# 5) Aynı fiyat + aynı hız ama biri FP4 → quant filtresi FP4'ü elemeli
results.append(run("quant filtresi (fp4 elenir)",
    [ep("fp4-prov", 0.70, 2.10, 90, 600, quant="fp4"),
     ep("fp8-prov", 1.00, 3.00, 55, 850)],
    WL, "fp8-prov"))

# 6) Sticky: mevcut sağlayıcı %5 marj içindeyse korunmalı
eps6 = [ep("incumbent", 1.00, 3.00, 55, 850),
        ep("slightly-better", 0.98, 2.94, 58, 840)]
results.append(run("sticky korunur (marj %5)",
    eps6, WL, "incumbent", sticky="incumbent", sticky_margin_pct=5.0))

# 7) Sticky marjı aşılırsa bırakılmalı
eps7 = [ep("incumbent", 2.50, 7.50, 30, 1500),
        ep("much-better", 1.00, 3.00, 60, 700)]
results.append(run("sticky bırakılır (marj aşıldı)",
    eps7, WL, "much-better", sticky="incumbent", sticky_margin_pct=5.0))

# 8) Zaman değeri 0 → saf maliyet optimizasyonu (en ucuz kazanır)
results.append(run("zaman değeri 0 → en ucuz",
    [ep("cheap-slow", 0.80, 2.40, 25, 1800),
     ep("fast", 1.00, 3.00, 90, 600)],
    Workload(8000, 1200, 0.0, 0.0), "cheap-slow"))

# 9) Cache oranı yüksekse cache fiyatı baskın olmalı
eps9 = [ep("cache-friendly", 2.00, 4.00, 60, 700),
        ep("no-cache", 1.20, 4.00, 60, 700)]
eps9[0].p_cache = 0.05 / 1e6   # cache okuma neredeyse bedava
eps9[1].p_cache = 1.20 / 1e6
results.append(run("cache-ağırlıklı iş yükü",
    eps9, Workload(20000, 1000, 0.85, 0.5), "cache-friendly"))

print(f"\n{sum(results)}/{len(results)} test geçti")
raise SystemExit(0 if all(results) else 1)
