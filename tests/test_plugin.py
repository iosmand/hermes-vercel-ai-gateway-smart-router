#!/usr/bin/env python3
"""Plugin sözleşme testleri.

Hermes'in kendi kayıt mekanizmasını (providers registry) kullanarak çalışır:

    bash deploy.sh && python3 tests/test_plugin.py

Testler canlı ağ çağrısı yapmaz; sadece üretilen istek gövdesini doğrular.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from decimal import Decimal

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
    orp = get_provider_profile("openrouter")
    if orp is not None and type(orp).__name__ != "AIGatewayRouterProfile":
        body8 = orp.build_extra_body(session_id="s", model="anthropic/claude-opus-5")
        check("izolasyon: openrouter gövdesinde providerOptions yok",
              "providerOptions" not in (body8 or {}), json.dumps(body8 or {}))

    print(f"\n{len(FAILS)} hata" if FAILS else "\ntüm testler geçti")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())