#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Plugin sözleşme testleri.

Hermes'in kendi kayıt mekanizmasını (providers registry) kullanarak çalışır:

    bash deploy.sh && python3 tests/test_plugin.py

Testler canlı ağ çağrısı yapmaz; sadece üretilen istek gövdesini doğrular.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

# A throwaway HERMES_HOME with seeded rules + stats: the test never depends on
# (and never writes to) the real installation.
HOME = Path(tempfile.mkdtemp(prefix="gw-plugin-test-"))
os.environ["HERMES_HOME"] = str(HOME)
STATE = HOME / "ai-gateway-router"
(STATE / "stats").mkdir(parents=True, exist_ok=True)
MODEL = "zai/glm-5.3-flash"

def _ep(name: str, quant: str, pin: float = 1e-7, pout: float = 4e-7) -> dict:
    return {"provider_name": name, "quantization": quant, "status": 0,
            "pricing": {"prompt": pin, "completion": pout, "input_cache_read": pin / 10},
            "uptime_last_15m": 100, "uptime_last_1h": 100, "uptime_last_1d": 100,
            "throughput_last_1h": {"p50": 100}, "latency_last_1h": {"p50": 800},
            "tags": ["tool-use", "structured-output"], "context_length": 10 ** 6}

(STATE / "rules.json").write_text(json.dumps({"models": {MODEL: {
    "precision": {"mode": "strict", "require": "fp8"},
    "workload": {"input_tokens": 8000, "output_tokens": 1200, "cached_ratio": 0.5},
    "require_tools": True, "min_uptime_15m": 95, "max_candidates": 4,
    "session_affinity": True, "stats_ttl_s": 600,
}}}), encoding="utf-8")
(STATE / "stats" / (MODEL.replace("/", "__") + ".json")).write_text(json.dumps({
    "fetched_at": time.time(),
    "endpoints": [_ep("particle", "fp8"), _ep("friendli", "fp8", 1.2e-7, 4.8e-7),
                  _ep("fp4-prov", "fp4", 0.6e-7, 2.4e-7)],
}), encoding="utf-8")

# Tell the plugin loader that this process is already in-process: without it the
# loader reads the profile config (and hermes_yaml → ruamel), which a bare test
# environment may not ship.
os.environ.setdefault("HERMES_PLUGIN_HOST_PROCESS", "1")

# Install the plugin into the throwaway home so the registry picks it up
# (same layout deploy.sh creates).
_PLUGIN_SRC = Path(__file__).resolve().parents[1] / "plugin"
_PLUGIN_DST = HOME / "plugins" / "model-providers" / "ai-gateway"
_PLUGIN_DST.mkdir(parents=True, exist_ok=True)
for _name in ("__init__.py", "planning.py", "transport.py", "plugin.yaml"):
    (_PLUGIN_DST / _name).write_text((_PLUGIN_SRC / _name).read_text(encoding="utf-8"),
                                     encoding="utf-8")

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    raw_usage: dict | None = None


def main() -> int:
    from providers import get_provider_profile

    p = get_provider_profile("ai-gateway")
    check("registry: ai-gateway profili override edildi", type(p).__name__ == "AIGatewayRouterProfile",
          type(p).__name__)
    check("registry: alias 'vercel' aynı profili verir", get_provider_profile("vercel") is p)

    # 1) plan uygulanır
    body = p.build_extra_body(session_id="sess-abc", model="zai/glm-5.3-flash")
    gw = body.get("providerOptions", {}).get("gateway", {})
    check("plan: order enjekte edildi", bool(gw.get("order")), json.dumps(gw))
    has = gw.get("has") or []
    check("plan: has enjekte edildi (fp8 zorunlu ya da fp4 dışlanmış)",
          "quantization:fp8" in has or "!quantization:fp4" in has, json.dumps(gw))
    check("plan: sessionAffinity true", gw.get("sessionAffinity") is True)
    check("plan: sessionId opak ve kararlı",
          gw.get("sessionId", "").startswith("hermes-")
          and gw.get("sessionId") == p.build_extra_body(session_id="sess-abc", model="zai/glm-5.3-flash")
                                              ["providerOptions"]["gateway"]["sessionId"])

    # 2) plansız model → gateway bloğu yok, sadece sessionId
    body2 = p.build_extra_body(session_id="s", model="openai/not-a-plan")
    gw2 = body2.get("providerOptions", {}).get("gateway", {})
    check("plan yoksa order/has yok", "order" not in gw2 and "has" not in gw2, json.dumps(gw2))

    # 3) provider_preferences planı ezer
    body3 = p.build_extra_body(session_id="s", model="zai/glm-5.3-flash",
                               provider_preferences={"sort": "ttft"})
    gw3 = body3["providerOptions"]["gateway"]
    check("provider_preferences öncelikli", gw3.get("sort") == "ttft", json.dumps(gw3))

    # 4) beyaz liste: desteklenmeyen anahtar atılır
    body4 = p.build_extra_body(session_id="s", model="zai/glm-5.3-flash",
                               provider_preferences={"ignore": ["x"], "totallyBogus": 1})
    gw4 = body4["providerOptions"]["gateway"]
    check("beyaz liste: desteklenmeyen anahtarlar düşürüldü",
          "ignore" not in gw4 and "totallyBogus" not in gw4, json.dumps(gw4))

    # 5) kill switch
    os.environ["HERMES_AI_GATEWAY_ROUTING"] = "off"
    try:
        body5 = p.build_extra_body(session_id="s", model="zai/glm-5.3-flash")
        check("kill switch: yönlendirme kapalı", "providerOptions" not in body5, json.dumps(body5))
    finally:
        os.environ.pop("HERMES_AI_GATEWAY_ROUTING", None)

    # 6) reasoning passthrough korunuyor (bundled davranış)
    extra, top = p.build_api_kwargs_extras(reasoning_config={"effort": "high"}, supports_reasoning=True)
    check("reasoning passthrough korundu", extra.get("reasoning") == {"effort": "high"}, json.dumps(extra))

    # 7) gerçek maliyet usage.cost'tan okunur
    cost = p.get_usage_cost("zai/glm-5.3-flash", FakeUsage(raw_usage={"cost": 0.0001234, "id": "gen_x"}))
    check("maliyet: usage.cost okundu",
          cost is not None and cost.amount_usd == Decimal("0.0001234") and cost.status == "actual",
          str(cost))
    check("maliyet: maliyet yoksa None", p.get_usage_cost("m", FakeUsage(raw_usage={})) is None)

    # 8) başka sağlayıcılar etkilenmiyor
    try:
        orp = get_provider_profile("openrouter")
    except ImportError as exc:      # a bare environment without ruamel.yaml
        print(f"[SKIP] izolasyon kontrolü: {exc}")
        orp = None
    if orp is not None and type(orp).__name__ != "AIGatewayRouterProfile":
        body8 = orp.build_extra_body(session_id="s", model="anthropic/claude-opus-5")
        check("izolasyon: openrouter gövdesinde providerOptions yok",
              "providerOptions" not in (body8 or {}), json.dumps(body8 or {}))

    print(f"\n{len(FAILS)} hata" if FAILS else "\ntüm testler geçti")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())