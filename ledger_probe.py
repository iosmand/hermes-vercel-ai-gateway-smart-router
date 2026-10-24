#!/usr/bin/env python3
"""ledger_probe.py — gerçek isteklerle usage.jsonl + generation_truth.jsonl üretir.

Hermes'in GERÇEK kod yolunu kullanır:
  providers.get_provider_profile("vercel").build_extra_body(...)  → istek gövdesi
  agent.usage_pricing.normalize_usage(...)                        → CanonicalUsage
  profile.get_usage_cost(model, usage)                            → ledger satırı
  planning.record_truth(...)                                      → 20 sn sonra gerçek provider

Kullanım:
  PYTHONPATH=/usr/local/lib/hermes-agent:$PYTHONPATH HERMES_HOME=~/.hermes \
    HERMES_AI_GATEWAY_LEDGER=1 python3 ledger_probe.py

Çıktı: ~/.hermes/ai-gateway-router/usage.jsonl + generation_truth.jsonl
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://ai-gateway.vercel.sh/v1"
MODELS = ["zai/glm-5.3-flash", "minimax/minimax-m3.1-flash", "deepseek/deepseek-v4.1-flash-1017"]


def key() -> str:
    value = os.environ.get("AI_GATEWAY_API_KEY")
    if value:
        return value
    home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    for line in (home / ".env").read_text(encoding="utf-8").splitlines():
        if line.split("=", 1)[0].strip() == "AI_GATEWAY_API_KEY":
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("AI_GATEWAY_API_KEY bulunamadı")


def main() -> int:
    os.environ["HERMES_AI_GATEWAY_LEDGER"] = "1"
    from providers import get_provider_profile
    from agent.usage_pricing import normalize_usage

    profile = get_provider_profile("vercel")
    for model in MODELS:
        body = profile.build_extra_body(session_id="ledger-probe-" + model.split("/")[-1], model=model)
        payload = {
            "model": model,
            "messages": [{"role": "user",
                          "content": "Üç maddede Python'da liste ile tuple farkını anlat."}],
            "max_tokens": 220,
            "reasoning": {"effort": "low"},
            **body,
        }
        req = urllib.request.Request(
            BASE + "/chat/completions", data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + key(), "Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read())
        usage = normalize_usage(raw.get("usage"), provider="ai-gateway", api_mode="chat_completions")
        # usage.cost → CanonicalUsage.raw_usage içinde; sağlayıcıya sor:
        cost = profile.get_usage_cost(model, usage)
        print(f"{model:32s} prompt={usage.input_tokens} out={usage.output_tokens} "
              f"reason={usage.reasoning_tokens} cache={usage.cache_read_tokens} "
              f"cost={cost.amount_usd if cost else None} gw={body['providerOptions']['gateway']['order'][0]}")
    print("\n20 sn bekleniyor (gateway usage event gecikmesi)…")
    time.sleep(22)
    return 0


if __name__ == "__main__":
    sys.exit(main())