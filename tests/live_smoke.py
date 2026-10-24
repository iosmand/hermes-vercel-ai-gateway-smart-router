#!/usr/bin/env python3
"""Canlı duman testi — plugin'in ürettiği gövdeyi gerçek gateway'e gönderir.

    PYTHONPATH=/usr/local/lib/hermes-agent:$PYTHONPATH HERMES_HOME=~/.hermes \
      python3 tests/live_smoke.py

Plan dosyasındaki her model için tek küçük istek atar; HTTP 200 + `usage.cost`
bekler. Maliyet ~1-2 mikrodolar.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://ai-gateway.vercel.sh/v1"


def hermes_home() -> Path:
    import os
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def key() -> str:
    import os
    value = os.environ.get("AI_GATEWAY_API_KEY")
    if value:
        return value
    for line in (hermes_home() / ".env").read_text(encoding="utf-8").splitlines():
        if line.split("=", 1)[0].strip() == "AI_GATEWAY_API_KEY":
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("AI_GATEWAY_API_KEY bulunamadı")


def main() -> int:
    from providers import get_provider_profile

    p = get_provider_profile("vercel")
    plans = json.load(open(hermes_home() / "ai-gateway-router" / "plans.json", encoding="utf-8"))
    fails = 0
    for model, plan in plans.items():
        if plan.get("error"):
            print(f"{model:32s} ATLANDI (plan hatası)")
            continue
        body = p.build_extra_body(session_id="live-smoke", model=model)
        gw = body.get("providerOptions", {}).get("gateway", {})
        payload = {"model": model, "messages": [{"role": "user", "content": "Say OK"}],
                   "max_tokens": 32, "reasoning": {"effort": "low"}, **body}
        req = urllib.request.Request(
            BASE + "/chat/completions", data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + key(), "Content-Type": "application/json"},
            method="POST")
        t = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                b = json.loads(r.read())
            print(f"{model:32s} 200 {(time.time()-t):5.2f}s cost=${b['usage'].get('cost'):<10}"
                  f" first={gw.get('order', ['-'])[0]:<12s} has={gw.get('has')}")
        except urllib.error.HTTPError as e:
            fails += 1
            print(f"{model:32s} {e.code} {e.read()[:180].decode(errors='replace')}")
    print("\nHATA YOK" if not fails else f"\n{fails} modelde hata")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())