"""test_strike.py — strike hedefi testleri (ağ yok; planning.stats taklitli).

3.5: iki paralel oturum, aynı model. Oturum B 503 alır → yalnız B'nin sağlayıcısı strike alır.
3.2: hata gövdesi sağlayıcıyı adlandırıyorsa o sağlayıcı strike alır.

Koşum:  PYTHONPATH=/usr/local/lib/hermes-agent:$PYTHONPATH HERMES_HOME=<temp> \
            python3 tests/test_strike.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="gw-strike-test-"))
os.environ["HERMES_HOME"] = str(TMP)
sys.path.insert(0, str(REPO / "plugin"))
sys.path.insert(0, str(REPO))

import planning  # noqa: E402

FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def ep(name: str, price_in: float, price_out: float, tps: float, ttft: float):
    return {"provider_name": name, "quantization": "fp8", "status": 0,
            "pricing": {"prompt": price_in, "completion": price_out, "input_cache_read": price_in / 10},
            "uptime_last_15m": 100, "uptime_last_1h": 100, "uptime_last_1d": 100,
            "throughput_last_1h": {"p50": tps}, "latency_last_1h": {"p50": ttft},
            "tags": ["tool-use", "structured-output"], "context_length": 1_000_000}


FAKE = [ep("particle", 1e-7, 4e-7, 250, 900), ep("friendli", 1.7e-7, 6e-7, 200, 300),
        ep("togetherai", 1.5e-7, 5e-7, 200, 1000)]
planning.stats = lambda model, ttl=600.0: (FAKE, False)
planning.rule_for = lambda model: {
    "workload": {"input_tokens": 1000, "output_tokens": 200, "cached_ratio": 0.5},
    "max_candidates": 4, "session_affinity": True, "time_value_cents_per_s": 0.15,
    "require_tools": True, "min_uptime_15m": 95,
    "precision": {"mode": "auto", "require": "fp8", "exclude": ["fp4", "int4"]},
}

sys.path.insert(0, str(REPO / "plugin"))
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("gw_plugin_under_test", REPO / "plugin" / "__init__.py")
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)

prof = plugin._profile
MODEL = "zai/glm-5.3-flash"
key_a = plugin._session_key("oturum-A")
key_b = plugin._session_key("oturum-B")


def session_b_worker():
    # B oturumu friendli'ye kilitli
    prof.build_extra_body(session_id="oturum-B", model=MODEL)
    prof.classify_api_error(None, status_code=503, message="HTTP 503", body=None, model=MODEL)


# --- 3.5: iki paralel oturum -------------------------------------------------
planning.lock_provider(MODEL, key_a, "particle")
planning.lock_provider(MODEL, key_b, "friendli")
planning._save_state("strikes.json", {})

prof.build_extra_body(session_id="oturum-A", model=MODEL)   # A hata ALMAZ
t = threading.Thread(target=session_b_worker)
t.start(); t.join()

strikes = planning._json_state("strikes.json")
check("3.5: yalnız B'nin sağlayıcısı strike aldı",
      list(strikes) == ["friendli"] and strikes["friendli"]["count"] == 1, str(strikes))
check("3.5: A'nın sağlayıcısına strike yok", "particle" not in strikes, str(sorted(strikes)))
check("3.5: karantina yok (tek strike)", planning.demoted() == {}, str(planning.demoted()))

# --- 3.2: hata gövdesi sağlayıcıyı adlandırıyor -------------------------------
planning._save_state("strikes.json", {})
prof.build_extra_body(session_id="oturum-A", model=MODEL)   # ctx: kilitli = particle
prof.classify_api_error(
    None, status_code=503, message="HTTP 503",
    body={"error": {"message": "upstream failed", "provider_name": "runinfra"}}, model=MODEL)
strikes = planning._json_state("strikes.json")
check("3.2: gövdedeki sağlayıcı strike aldı (kilitli değil)",
      list(strikes) == ["runinfra"], str(strikes))

# --- 3.4: gateway seviyesi 429 strike DEĞİL -----------------------------------
planning._save_state("strikes.json", {})
prof.build_extra_body(session_id="oturum-A", model=MODEL)
prof.classify_api_error(
    None, status_code=429, message="HTTP 429: could not verify available credits",
    body={"error": {"message": "credits"}}, model=MODEL)
check("3.4: sağlayıcı adlandırmayan 429 strike değil",
      planning._json_state("strikes.json") == {}, str(planning._json_state("strikes.json")))

# --- 3.4: sağlayıcı adlandıran 429 strike -------------------------------------
prof.classify_api_error(
    None, status_code=429, message="HTTP 429",
    body={"provider": "parasail", "error": {"message": "rate limited"}}, model=MODEL)
strikes = planning._json_state("strikes.json")
check("3.4: sağlayıcı adlandıran 429 strike sayılır",
      list(strikes) == ["parasail"], str(strikes))

if FAILED:
    print(f"\n{len(FAILED)} test BAŞARISIZ: {FAILED}")
    sys.exit(1)
print("\ntüm testler geçti")
