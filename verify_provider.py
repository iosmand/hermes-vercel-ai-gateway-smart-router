#!/usr/bin/env python3
"""verify_provider.py — gerçek sağlayıcıyı gateway'e sorar (authoritative).

Gateway yanıtı `provider_name` döndürmez; sadece `generationId` verir.
`GET /v1/generation?id=<id>` çağrısı ~10-20 sn sonra gerçek `provider_name`,
`latency`, `generation_time` ve `total_cost` alanlarını verir.

Kullanım:
  PYTHONPATH=/usr/local/lib/hermes-agent:$PYTHONPATH HERMES_HOME=~/.hermes \
    python3 verify_provider.py [model ...]
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://ai-gateway.vercel.sh/v1"
DEFAULT_MODELS = ["zai/glm-5.3-flash", "minimax/minimax-m3.1-flash", "deepseek/deepseek-v4.1-flash-1017"]


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def key() -> str:
    value = os.environ.get("AI_GATEWAY_API_KEY")
    if value:
        return value
    for line in (hermes_home() / ".env").read_text(encoding="utf-8").splitlines():
        if line.split("=", 1)[0].strip() == "AI_GATEWAY_API_KEY":
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("AI_GATEWAY_API_KEY bulunamadı")


def main() -> int:
    from providers import get_provider_profile

    sys.path.insert(0, str(hermes_home() / "plugins" / "model-providers" / "ai-gateway"))
    import planning

    profile = get_provider_profile("vercel")
    models = sys.argv[1:] or DEFAULT_MODELS
    pending = []

    for model in models:
        body = profile.build_extra_body(session_id="verify-" + model.split("/")[-1], model=model)
        gw = body["providerOptions"]["gateway"]
        payload = {"model": model,
                   "messages": [{"role": "user", "content": "Tek cümle: tuple neden immutable?"}],
                   "max_tokens": 120, "reasoning": {"effort": "low"}, **body}
        req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Authorization": "Bearer " + key(),
                                              "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = json.loads(resp.read())
        gid = raw.get("generationId") or raw.get("id")
        u = raw.get("usage") or {}
        inferred = planning.infer_provider(
            model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
            u.get("cost"), (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
        print(f"{model:32s} plan={gw['order'][0]:<12s} cost=${u.get('cost')} "
              f"inferred={inferred.get('provider')} ({'belirsiz: ' + ','.join(inferred.get('candidates') or []) if inferred.get('ambiguous') else 'tek eşleşme'})")
        pending.append((model, gw["order"][0], gid))

    print("\ngateway usage event'i bekleniyor (~20 sn)…")
    time.sleep(22)

    rows = []
    for model, intended, gid in pending:
        d = planning.lookup_generation(gid, retries=4, delay=5) or {}
        row = {"model": model, "generation_id": gid, "provider_intended": intended,
               "provider_actual": d.get("provider_name"),
               "match": d.get("provider_name") == intended,
               "cost_usd": d.get("total_cost"), "latency_ms": d.get("latency"),
               "generation_time_ms": d.get("generation_time"), "finish_reason": d.get("finish_reason")}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    out = planning.state_dir() / "generation_truth.jsonl"
    with out.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("\n->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())