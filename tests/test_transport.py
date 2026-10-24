# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""test_transport.py — transport sarmalayıcısının birim testleri (ağ yok).

Koşum:  PYTHONPATH=/usr/local/lib/hermes-agent:$PYTHONPATH HERMES_HOME=<temp> \
            python3 tests/test_transport.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

REPO = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="gw-transport-test-"))
os.environ["HERMES_HOME"] = str(TMP)
sys.path.insert(0, str(REPO / "plugin"))

spec = importlib.util.spec_from_file_location("gw_transport", REPO / "plugin" / "transport.py")
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)

FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def delta_chunk(content=None, reasoning=None, usage=None, gen="gen_X", resolved="particle",
                routing=None):
    # gerçek yapı: provider_metadata choices[0].delta üzerinde, generationId chunk üzerinde
    rt = routing if routing is not None else {"resolvedProvider": resolved, "fallbacksAvailable": []}
    d = NS(content=content, reasoning_content=reasoning, tool_calls=None,
           model_extra={"provider_metadata": {"gateway": {"routing": rt}}})
    return NS(choices=[NS(delta=d, finish_reason=None)], usage=usage, model_extra=None,
              generationId=gen)


def usage_obj(inp, out, cached=None, cache_write=None, reason=None):
    return NS(prompt_tokens=inp, completion_tokens=out,
              prompt_tokens_details=NS(cached_tokens=cached, cache_creation_tokens=cache_write,
                                       model_extra=None),
              completion_tokens_details=NS(reasoning_tokens=reason, model_extra=None),
              model_extra=None)


# ---------------------------------------------------------------- 1) stream tek satır
def test_stream_row():
    chunks = [delta_chunk(reasoning="düşün", usage=None),
              delta_chunk(content="cevap", usage=usage_obj(26, 97, 0, None, 1), gen="gen_STREAM1")]
    inner = iter(chunks)
    s = transport._InstrumentedStream(inner, {"ts": "T", "model": "m", "session_id": "hermes-x",
                                              "session_key": "hermes-x", "provider_intended": "particle",
                                              "provider_actual": None, "only_set": ["particle"]}, 0.0)
    list(s)
    rows = [json.loads(l) for l in open(TMP / "ai-gateway-router" / "ledger.jsonl")]
    r = rows[-1]
    fields = list(r.keys())
    check("stream: ledger alanları tam",
          fields == list(transport.LEDGER_FIELDS), json.dumps(fields))
    check("stream: bir satır yazıldı", len(rows) == 1, f"{len(rows)} satır")
    check("stream: ttft ölçüldü", r["ttft_ms"] is not None and r["ttft_ms"] >= 0, str(r["ttft_ms"]))
    check("stream: provider_actual okundu", r["provider_actual"] == "particle", str(r["provider_actual"]))
    check("stream: generation_id okundu", r["generation_id"] == "gen_STREAM1", str(r["generation_id"]))
    check("stream: tokenlar okundu", (r["input_tokens"], r["output_tokens"],
                                      r["reasoning_tokens"]) == (26, 97, 1),
          json.dumps([r["input_tokens"], r["output_tokens"], r["reasoning_tokens"]]))


# ---------------------------------------------------------------- 2) failover: kilit taşınır, strike YOK
def test_failover_moves_lock():
    import planning
    planning.lock_provider("m", "hermes-y", "particle")
    before = planning._json_state("strikes.json")
    stream = transport._InstrumentedStream(
        iter([delta_chunk(content="x", usage=usage_obj(10, 5), resolved="friendli", gen="gen_F")]),
        {"ts": "T", "model": "m", "session_id": "hermes-y", "session_key": "hermes-y",
         "provider_intended": "particle", "provider_actual": None,
         "only_set": ["particle", "friendli"]}, 0.0, planning)
    list(stream)
    rows = [json.loads(l) for l in open(TMP / "ai-gateway-router" / "ledger.jsonl")]
    r = rows[-1]
    check("failover: provider_actual != intended yazıldı",
          r["provider_intended"] == "particle" and r["provider_actual"] == "friendli",
          json.dumps([r["provider_intended"], r["provider_actual"]]))
    check("failover: kilit friendli'ye taşındı",
          planning.locked_provider("m", "hermes-y") == "friendli",
          str(planning.locked_provider("m", "hermes-y")))
    check("failover: strike yazılmadı", planning._json_state("strikes.json") == before,
          json.dumps(planning._json_state("strikes.json")))


# ---------------------------------------------------------------- 3) failover: only dışındaysa kilit TAŞINMAZ
def test_failover_outside_only_keeps_lock():
    import planning
    planning.lock_provider("m", "hermes-z", "particle")
    stream = transport._InstrumentedStream(
        iter([delta_chunk(content="x", usage=usage_obj(10, 5), resolved="runinfra", gen="gen_G")]),
        {"ts": "T", "model": "m", "session_id": "hermes-z", "session_key": "hermes-z",
         "provider_intended": "particle", "provider_actual": None,
         "only_set": ["particle", "friendli"]}, 0.0, planning)
    list(stream)
    check("failover: only dışındaki sağlayıcıya kilit TUTULMAZ",
          planning.locked_provider("m", "hermes-z") == "particle",
          str(planning.locked_provider("m", "hermes-z")))


# ---------------------------------------------------------------- 4) non-stream satırı
def test_nonstream_row():
    class FakeCompletions:
        def create(self, **kw):
            msg = NS(model_extra={"provider_metadata": {"gateway": {"routing":
                     {"resolvedProvider": "friendli", "fallbacksAvailable": []}}}})
            return NS(choices=[NS(message=msg)], usage=usage_obj(5, 7, 3, None, 2),
                      model_extra={"generationId": "gen_NS"})

    comp = transport._InstrumentedCompletions(FakeCompletions())
    comp.create(model="m2", extra_body={"providerOptions": {"gateway":
                {"order": ["friendli"], "only": ["friendli"], "sessionId": "hermes-n"}}})
    rows = [json.loads(l) for l in open(TMP / "ai-gateway-router" / "ledger.jsonl")]
    r = rows[-1]
    check("non-stream: provider_actual/generation okundu",
          r["provider_actual"] == "friendli" and r["generation_id"] == "gen_NS",
          json.dumps([r["provider_actual"], r["generation_id"]]))
    check("non-stream: ttft ölçülemez → null", r["ttft_ms"] is None, str(r["ttft_ms"]))
    check("non-stream: cache_read okundu", r["cache_read_tokens"] == 3, str(r["cache_read_tokens"]))


# ---------------------------------------------------------------- 5) affinity alanları (stream)
def test_affinity_stream():
    rt = {"resolvedProvider": "xiaomi", "fallbacksAvailable": [],
          "clientSessionId": "hermes-ba98", "clientSessionIdSource": "explicit",
          "affinity": {"outcome": "repinned", "pinnedProvider": "xiaomi",
                       "displacedProvider": "novita"}}
    s = transport._InstrumentedStream(
        iter([delta_chunk(content="x", usage=usage_obj(3, 4), gen="gen_AFFS", routing=rt)]),
        {"ts": "T", "model": "m", "session_id": "hermes-x", "session_key": "hermes-x",
         "provider_intended": "xiaomi", "provider_actual": None,
         "only_set": ["xiaomi"]}, 0.0)
    list(s)
    r = [json.loads(l) for l in open(TMP / "ai-gateway-router" / "ledger.jsonl")][-1]
    check("affinity stream: 5 alan yazıldı",
          [r["client_session_id"], r["client_session_id_source"], r["affinity_outcome"],
           r["affinity_pinned_provider"], r["affinity_displaced_provider"]]
          == ["hermes-ba98", "explicit", "repinned", "xiaomi", "novita"],
          json.dumps({k: r[k] for k in transport.LEDGER_FIELDS[-5:]}))


# ---------------------------------------------------------------- 6) affinity alanları (non-stream)
def test_affinity_nonstream():
    rt = {"resolvedProvider": "xiaomi", "fallbacksAvailable": [],
          "clientSessionId": "hermes-402f", "clientSessionIdSource": "explicit",
          "affinity": {"outcome": "confirmed", "pinnedProvider": "xiaomi",
                       "displacedProvider": "boundless"}}

    class FakeCompletions:
        def create(self, **kw):
            msg = NS(model_extra={"provider_metadata": {"gateway": {"routing": rt}}})
            return NS(choices=[NS(message=msg)], usage=usage_obj(5, 6),
                      model_extra={"generationId": "gen_AFFN"})

    comp = transport._InstrumentedCompletions(FakeCompletions())
    comp.create(model="m2", extra_body={"providerOptions": {"gateway":
                {"order": ["xiaomi"], "sessionId": "hermes-402f"}}})
    r = [json.loads(l) for l in open(TMP / "ai-gateway-router" / "ledger.jsonl")][-1]
    check("affinity non-stream: 5 alan yazıldı",
          [r["client_session_id"], r["client_session_id_source"], r["affinity_outcome"],
           r["affinity_pinned_provider"], r["affinity_displaced_provider"]]
          == ["hermes-402f", "explicit", "confirmed", "xiaomi", "boundless"],
          json.dumps({k: r[k] for k in transport.LEDGER_FIELDS[-5:]}))


# ---------------------------------------------------------------- 7) eksik alanlar → null
def test_affinity_missing_nulls():
    # routing'te affinity bloğu ve clientSessionId YOK (eski yanıt şekli)
    s = transport._InstrumentedStream(
        iter([delta_chunk(content="x", usage=usage_obj(1, 1), gen="gen_AFF0",
                          routing={"resolvedProvider": "particle", "fallbacksAvailable": []})]),
        {"ts": "T", "model": "m", "session_id": "hermes-x", "session_key": "hermes-x",
         "provider_intended": "particle", "provider_actual": None,
         "only_set": ["particle"]}, 0.0)
    list(s)
    r = [json.loads(l) for l in open(TMP / "ai-gateway-router" / "ledger.jsonl")][-1]
    AFFINITY = ("client_session_id", "client_session_id_source", "affinity_outcome",
                "affinity_pinned_provider", "affinity_displaced_provider")
    check("affinity eksik: 5 alan null",
          all(r[k] is None for k in AFFINITY),
          json.dumps({k: r[k] for k in AFFINITY}))
    # sütunlar her satırda var (18 alan + sonuç kolonu)
    check("ledger: alan listesi 19 kolon (18 + outcome)",
          len(transport.LEDGER_FIELDS) == 19 and list(r.keys()) == list(transport.LEDGER_FIELDS),
          str(len(transport.LEDGER_FIELDS)))


test_stream_row()
test_failover_moves_lock()
test_failover_outside_only_keeps_lock()
test_nonstream_row()
test_affinity_stream()
test_affinity_nonstream()
test_affinity_missing_nulls()

if FAILED:
    print(f"\n{len(FAILED)} test BAŞARISIZ: {FAILED}")
    sys.exit(1)
print("\ntüm testler geçti")
