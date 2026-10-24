#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""FP8 provider benchmark — streaming ile gerçek TTFT / throughput / maliyet ölçümü.

Katalog metrikleri (throughput_last_1h, latency_last_1h) Vercel'in kendi
gözleminden gelir; bu betik aynı prompt'u her sağlayıcıya `only:[provider]`
ile kilitleyip ölçer, böylece katalog metriği ile gerçek deneyim kıyaslanır.

Kullanım:
  python3 bench.py --model zai/glm-5.3-flash --quant fp8
  python3 bench.py --model zai/glm-5.3-flash --providers gmicloud,particle --max-tokens 256
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://ai-gateway.vercel.sh/v1"


def api_key() -> str:
    k = os.environ.get("AI_GATEWAY_API_KEY")
    if k:
        return k
    env_file = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / ".env"
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.split("=", 1)[0].strip() == "AI_GATEWAY_API_KEY":
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("AI_GATEWAY_API_KEY yok")


K = api_key()

PROMPT = (
    "Aşağıdaki Python fonksiyonunu analiz et ve geliştirmeleri listele, sonra "
    "düzeltilmiş sürümü yaz. Kısa madde işaretleri kullan.\n\n"
    "def parse(items):\n"
    "    out = []\n"
    "    for i in items:\n"
    "        if i['type'] == 'a':\n"
    "            out.append(i['value'] * 2)\n"
    "        elif i['type'] == 'b':\n"
    "            out.append(str(i['value']))\n"
    "    return out\n\n"
    "Bu fonksiyon 3 farklı hata durumunda ne yapar? Hepsini tek tek açıkla ve "
    "her biri için düzeltme öner. Sonunda tek bir refactor edilmiş sürüm ver."
)


def stream_call(model: str, provider: str, max_tokens: int = 256,
                reasoning_effort: str | None = "low",
                order_override: list[str] | None = None) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "providerOptions": {"gateway": {
            "only": sorted({provider} | set(order_override or [])),
            "order": order_override or [provider],
        }},
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + K, "Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    ttft = None
    chunks = 0
    usage = {}
    text = []
    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                delta = (ch.get("delta") or {}).get("content")
                if delta:
                    if ttft is None:
                        ttft = time.time() - t0
                    chunks += 1
                    text.append(delta)
    total = time.time() - t0
    ct = usage.get("completion_tokens") or chunks
    gen_window = max(total - (ttft or 0), 1e-6)
    return {
        "provider": provider,
        "ttft_s": round(ttft or 0, 2),
        "total_s": round(total, 2),
        "completion_tokens": ct,
        "observed_tps": round(ct / gen_window, 1) if ttft else None,
        "cost_usd": usage.get("cost"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "chars": len("".join(text)),
        "ok": bool(text),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zai/glm-5.3-flash")
    ap.add_argument("--providers", default=None, help="virgülle; boşsa katalogdan fp8 olanlar")
    ap.add_argument("--quant", default="fp8")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--reasoning-effort", default="low", choices=["low", "high", "max", "off"])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.providers:
        provs = [p.strip() for p in a.providers.split(",") if p.strip()]
    else:
        r = urllib.request.Request(f"{BASE}/models/{a.model}/endpoints",
                                   headers={"Authorization": "Bearer " + K})
        with urllib.request.urlopen(r, timeout=45) as resp:
            eps = json.loads(resp.read())["data"]["endpoints"]
        provs = [e["provider_name"] for e in eps
                 if (e.get("quantization") or None) == a.quant and e.get("status") == 0]

    print(f"model={a.model} quant={a.quant} provider={len(provs)} max_tokens={a.max_tokens}\n")
    rows = []
    for p in provs:
        for i in range(a.repeat):
            try:
                res = stream_call(a.model, p, a.max_tokens,
                                  None if a.reasoning_effort == "off" else a.reasoning_effort)
            except Exception as e:
                res = {"provider": p, "error": f"{type(e).__name__}: {e}"}
            rows.append(res)
            print(json.dumps(res, ensure_ascii=False))

    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1, ensure_ascii=False)
        print("\n->", a.out)
    ok = [r for r in rows if r.get("ok")]
    if ok:
        print("\nözet:")
        for r in sorted(ok, key=lambda x: x["ttft_s"]):
            print(f"  {r['provider']:14s} ttft={r['ttft_s']:6.2f}s total={r['total_s']:6.2f}s "
                  f"tps={r['observed_tps']:7.1f} cost={r['cost_usd']}")


if __name__ == "__main__":
    main()
