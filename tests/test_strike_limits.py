"""test_strike_limits.py — B1-B5: strike limitleri, GERÇEK hata yolu üzerinden.

Yol: yerel mock HTTP 5xx → gerçek openai APIStatusNoktası (istisna) →
     agent.error_classifier.classify_api_error (sohbet döngüsünün gerçek girişi)
     → profil kanca _profile_verdict → planning.record_strike.

Hiçbir yerde profil.sınıf_metodu doğrudan çağrılmaz; isimler yalnız mock
HTTP gövdelerinde geçer.

Koşum:  PYTHONPATH=/usr/local/lib/hermes-agent:$PYTHONPATH HERMES_HOME=<temp> \
            python3 tests/test_strike_limits.py [all|a|b|c|d|e]
"""
from __future__ import annotations

import http.server
import importlib.util
import json
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace as NS

REPO = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="gw-strike-limits-"))
os.environ["HERMES_HOME"] = str(TMP)
sys.path.insert(0, "/usr/local/lib/hermes-agent")
sys.path.insert(0, str(REPO / "plugin"))

import planning  # noqa: E402

MODEL = "zai/glm-5.3-flash"
FAILED: list[str] = []
LOGS: list[str] = []


class Collector(logging.Handler):
    def emit(self, record):
        LOGS.append(record.getMessage())


logging.getLogger().addHandler(Collector())


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def has_log(needle: str) -> bool:
    return any(needle in m for m in LOGS)


# ---------------------------------------------------------------- sahte katalog
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

# ------------------------------------------------------------- sahte saat (B3+)
CLOCK = {"t": 1_000_000.0}
planning.time = NS(time=lambda: CLOCK["t"])

spec = importlib.util.spec_from_file_location("gw_plugin_under_test", REPO / "plugin" / "__init__.py")
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)
prof = plugin._profile

from agent import error_classifier  # noqa: E402  (gerçek giriş noktası)

# ------------------------------------------------------- yerel mock (gerçek HTTP)
MOCK = {"status": 503, "body": {"error": {"message": "mock upstream unavailable"}}}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        raw = json.dumps(MOCK["body"]).encode()
        self.send_response(MOCK["status"])
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
PORT = server.server_address[1]

import openai  # noqa: E402

client = openai.OpenAI(api_key="test-key", base_url=f"http://127.0.0.1:{PORT}/v1", max_retries=0)


def real_error(body: dict, status: int = 503):
    """Mock'a gerçek istek at → gerçek openai istisnası döndür (gövde mock'tan)."""
    MOCK["body"], MOCK["status"] = body, status
    try:
        client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "x"}],
                                       max_tokens=5)
    except Exception as exc:
        return exc
    raise RuntimeError("mock beklenen hatayı vermedi")


def route_body(*providers: str) -> dict:
    """Gerçekçi gateway hata gövdesi: modelAttempts[].providerAttempts[]."""
    return {
        "error": {"message": "Provider returned error", "type": "provider_error", "code": 503},
        "providerMetadata": {"gateway": {"routing": {
            "modelAttempts": [{"canonicalSlug": MODEL, "success": False,
                               "providerAttempts": [{"provider_name": p, "statusCode": 503}
                                                    for p in providers]}],
            "totalProviderAttemptCount": len(providers)}}},
    }


def reset():
    LOGS.clear()
    planning._save_state("strikes.json", {})
    planning._save_state("demoted.json", {})
    planning._save_state("strike_pause.json", {})
    planning._save_state("sessions.json", {})


def strike_of(provider: str) -> dict:
    return planning._json_state("strikes.json").get(planning.state_key(MODEL, provider)) or {}


def struck_providers() -> list[str]:
    """Provider names from (model, provider)-keyed strike state."""
    return sorted(planning.split_key(k)[1] for k in planning._json_state("strikes.json"))


def demoted_providers() -> list[str]:
    return sorted(planning.split_key(k)[1] for k in planning._json_state("demoted.json"))


def run_case(case: str):
    reset()
    # bağlam: build_extra_body gerçek yolu (kilit + oturum anahtarı)
    prof.build_extra_body(body={}, model=MODEL, session_id=f"test-{case}")

    if case == "a":   # B5(a): gövde 1 sağlayıcı adlandırıyor → 1 strike
        exc = real_error(route_body("provA"))
        error_classifier.classify_api_error(exc, provider="ai-gateway", model=MODEL)
        check("B5(a): gövdedeki sağlayıcı 1 strike aldı",
              struck_providers() == ["provA"] and strike_of("provA").get("count") == 1,
              str(planning._json_state("strikes.json")))

    elif case == "b":  # B5(b): 2 sağlayıcı → strike yok, global failure suspected
        exc = real_error(route_body("provA", "provB"))
        error_classifier.classify_api_error(exc, provider="ai-gateway", model=MODEL)
        s = planning._json_state("strikes.json")
        check("B5(b): 2 sağlayıcılı hata strike üretmedi", s == {}, str(s))
        check("B5(b): 'global failure suspected' loglandı", has_log("global failure suspected"))

    elif case == "c":  # B5(c): 10 sn içinde 3 deneme → 1 strike
        for _ in range(3):
            exc = real_error(route_body("provC"))
            error_classifier.classify_api_error(exc, provider="ai-gateway", model=MODEL)
            CLOCK["t"] += 3      # 3 deneme = 9 sn
        check("B5(c): 3 deneme/10 sn → tek strike",
              struck_providers() == ["provC"] and strike_of("provC").get("count") == 1,
              str(planning._json_state("strikes.json")))

    elif case == "d":  # B5(d): 60 sn'de 2 farklı sağlayıcı → strike'lar durur
        exc = real_error(route_body("provD"))
        error_classifier.classify_api_error(exc, provider="ai-gateway", model=MODEL)
        CLOCK["t"] += 5
        exc = real_error(route_body("provE"))
        error_classifier.classify_api_error(exc, provider="ai-gateway", model=MODEL)
        pause = planning._json_state("strike_pause.json")
        CLOCK["t"] += 10
        exc = real_error(route_body("provF"))
        error_classifier.classify_api_error(exc, provider="ai-gateway", model=MODEL)
        check("B5(d): ilk iki sağlayıcı strike aldı",
              set(struck_providers()) == {"provD", "provE"}, str(struck_providers()))
        check("B5(d): üçüncü sağlayıcıya strike durdu", "provF" not in struck_providers(),
              str(struck_providers()))
        check("B5(d): 5 dk küresel duraklama yazıldı",
              pause.get("until", 0) > CLOCK["t"] and pause.get("until", 0) <= CLOCK["t"] + 300, str(pause))

    elif case == "e":  # B5(e): only'de 2 aday kalıyorsa karantina atlanır
        planning.LAST_GATEWAY[MODEL] = {"only": ["provX", "provY"], "order": ["provX", "provY"]}
        exc = real_error(route_body("provX"))
        error_classifier.classify_api_error(exc, provider="ai-gateway", model=MODEL)
        CLOCK["t"] += 31        # B3 soğuması dolsun ki 2. strike sayılsın
        exc = real_error(route_body("provX"))
        error_classifier.classify_api_error(exc, provider="ai-gateway", model=MODEL)
        d = planning._json_state("demoted.json")
        check("B5(e): 2 strike sayıldı", strike_of("provX").get("count") == 2,
              str(planning._json_state("strikes.json")))
        check("B5(e): karantina ATLANDI (only'de <2 aday kalıyor)", d == {}, str(d))
        check("B5(e): atlandı loglandı", has_log("karantina atlandı"))
        # kontrol: 3 aday varsa karantina uygulanır
        reset()
        planning.LAST_GATEWAY[MODEL] = {"only": ["provX", "provY", "provZ"], "order": []}
        prof.build_extra_body(body={}, model=MODEL, session_id="test-e2")
        exc = real_error(route_body("provX"))
        error_classifier.classify_api_error(exc, provider="ai-gateway", model=MODEL)
        CLOCK["t"] += 31
        exc = real_error(route_body("provX"))
        error_classifier.classify_api_error(exc, provider="ai-gateway", model=MODEL)
        check("B5(e-kontrol): 3 adayda karantina UYGULANIR", "provX" in demoted_providers(),
              str(planning._json_state("demoted.json")))


if __name__ == "__main__":
    want = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    for c in ("a", "b", "c", "d", "e"):
        if want in ("all", c):
            print(f"--- vaka {c} ---")
            run_case(c)
    server.shutdown()
    if FAILED:
        print(f"\n{len(FAILED)} test BAŞARISIZ: {FAILED}")
        sys.exit(1)
    print("\ntüm testler geçti")
