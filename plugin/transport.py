# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""transport.py — gerçek sağlayıcıyı yanıttan okuyan instrumented OpenAI istemcisi.

2.1: en alt katman iki yerde görünüyor:
  (a) çekirdek: agent/chat_completion_helpers.py:787  (non-stream: agent.client.chat.completions.create)
                      agent/chat_completion_helpers.py:2959 (stream: request_client.chat.completions.create)
  (b) plugin:  providers/base.py:329 ProviderProfile.create_client — çağrı yeri
               agent/agent_runtime_helpers.py:1912 (_provider_supplied_client)
Seçim: (b) — çekirdek dosyasına dokunmadan tek dosya. Bedeli: profil yalnız in-process çalışır
(hermes_cli/plugin_host_child.py:520-523 create_client override'ında PluginHostUnsupported fırlatır).

Yanıt gövdesindeki gerçek bilgi:
  non-stream : choices[0].message.provider_metadata.gateway.routing.resolvedProvider
  stream     : son chunk choices[0].delta.provider_metadata.gateway.routing.resolvedProvider
  her ikisi  : generationId (üst düzey / son chunk)

Ledger: istek başına BİR satır. Prompt/yanıt metni ASLA yazılmaz.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

LEDGER_LOCK = threading.Lock()


def _state_dir() -> Path:
    import os
    home = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
    d = home / "ai-gateway-router"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _extra(obj: Any, *names: str) -> Any:
    """pydantic model_extra / dict / getattr üçlüsünden alan oku."""
    if obj is None:
        return None
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    me = getattr(obj, "model_extra", None)
    if isinstance(me, dict):
        for n in names:
            if me.get(n) is not None:
                return me[n]
    if isinstance(obj, dict):
        for n in names:
            if obj.get(n) is not None:
                return obj[n]
    return None


def _routing_dict(payload: Any) -> dict:
    """provider_metadata.gateway.routing sözlüğü (yoksa boş)."""
    pm = _extra(payload, "provider_metadata") or {}
    gw = (pm.get("gateway") or {}) if isinstance(pm, dict) else {}
    rt = gw.get("routing") or {}
    return rt if isinstance(rt, dict) else {}


def _routing(payload: Any) -> tuple[str | None, str | None, list[str]]:
    """(resolvedProvider, generationId, fallbacksAvailable) — provider_metadata içinden."""
    rt = _routing_dict(payload)
    resolved = rt.get("resolvedProvider")
    fallbacks = list(rt.get("fallbacksAvailable") or [])
    return resolved, None, fallbacks


def _affinity_fields(routing: dict) -> dict:
    """3.1: routing/affinity alanları → ledger kolonları; eksik alan null."""
    aff = routing.get("affinity") or {}
    if not isinstance(aff, dict):
        aff = {}
    return {
        "client_session_id": routing.get("clientSessionId"),
        "client_session_id_source": routing.get("clientSessionIdSource"),
        "affinity_outcome": aff.get("outcome"),
        "affinity_pinned_provider": aff.get("pinnedProvider"),
        "affinity_displaced_provider": aff.get("displacedProvider"),
    }


def _usage_fields(usage: Any) -> dict:
    if usage is None:
        return {}
    pd = _extra(usage, "prompt_tokens_details") or {}
    cd = _extra(usage, "completion_tokens_details") or {}
    if not isinstance(pd, dict):
        pd = {"cached_tokens": _extra(pd, "cached_tokens"), "cache_creation_tokens": _extra(pd, "cache_creation_tokens")}
    if not isinstance(cd, dict):
        cd = {"reasoning_tokens": _extra(cd, "reasoning_tokens")}
    return {
        "input_tokens": _extra(usage, "prompt_tokens", "input_tokens"),
        "output_tokens": _extra(usage, "completion_tokens", "output_tokens"),
        "cache_read_tokens": pd.get("cached_tokens"),
        "cache_write_tokens": pd.get("cache_creation_tokens"),
        "reasoning_tokens": cd.get("reasoning_tokens"),
    }


LEDGER_FIELDS = ("ts", "model", "session_id", "provider_intended", "provider_actual",
                 "generation_id", "ttft_ms", "total_ms", "input_tokens", "output_tokens",
                 "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
                 # 3.1: affinity kolonları — eski satırlar değişmez, eksik alan null
                 "client_session_id", "client_session_id_source",
                 "affinity_outcome", "affinity_pinned_provider", "affinity_displaced_provider",
                 # stream tüketilmeden kapandıysa "aborted"; eksik alan ok/None sayılır
                 "outcome")

LEDGER_MAX_BYTES = 5_000_000     # rotate to ledger.jsonl.1 past this size
LEDGER_MAX_ENV = "HERMES_AI_GATEWAY_LEDGER_MAX_BYTES"


def _ledger_max_bytes() -> int:
    import os
    try:
        return int(os.environ.get(LEDGER_MAX_ENV) or LEDGER_MAX_BYTES)
    except ValueError:
        return LEDGER_MAX_BYTES


def _rotate_if_needed(path: Path) -> None:
    """Keep one previous file; a runaway ledger must not fill the disk."""
    limit = _ledger_max_bytes()
    if limit <= 0:
        return
    try:
        if path.exists() and path.stat().st_size >= limit:
            path.replace(path.with_name(path.name + ".1"))
    except OSError:
        pass


def write_ledger_row(row: dict) -> None:
    try:
        row = {k: row.get(k) for k in LEDGER_FIELDS}
        path = _state_dir() / "ledger.jsonl"
        with LEDGER_LOCK:
            _rotate_if_needed(path)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as exc:  # asla isteği düşürme
        log.debug("ai-gateway-router: ledger yazılamadı: %s", exc)


def ledger_enabled() -> bool:
    import os
    return (os.environ.get("HERMES_AI_GATEWAY_LEDGER") or "on").strip().lower() not in {"off", "0", "false", "no"}


class _InstrumentedStream:
    """openai Stream sarmalayıcısı: chunk'ları aynen geçirir, ölçüm alır."""

    def __init__(self, inner: Any, meta: dict, t0: float, planning: Any = None):
        self._inner = inner
        self._it = iter(inner)
        self._meta = meta
        self._t0 = t0
        self._planning = planning
        self._ttft_ms: float | None = None
        self._last_chunk: Any = None
        self._done = False
        self._completed = False
        self._aborted = False

    # --- SDK yüzeyi: her şey delege ---
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self._it)
        except StopIteration:
            self._completed = True
            self._finish()
            raise
        self._observe(chunk)
        return chunk

    def __enter__(self):
        if hasattr(self._inner, "__enter__"):
            self._inner.__enter__()
        return self

    def __exit__(self, exc_type=None, *exc):
        if exc_type is not None or not self._completed:
            self._aborted = True
        self._finish()
        if hasattr(self._inner, "__exit__"):
            return self._inner.__exit__(exc_type, *exc)
        return None

    def __del__(self):  # finalizer: never lose a billed row on an abandoned stream
        try:
            if not self._done:
                self._aborted = True
                self._finish()
        except Exception:
            pass

    def close(self):
        self._finish()
        if hasattr(self._inner, "close"):
            self._inner.close()

    def _observe(self, chunk: Any) -> None:
        self._last_chunk = chunk
        if self._ttft_ms is None:
            try:
                choice = (getattr(chunk, "choices", None) or [None])[0]
                delta = getattr(choice, "delta", None) if choice is not None else None
                if delta is not None and (_extra(delta, "content") or _extra(delta, "reasoning_content")
                                          or _extra(delta, "reasoning")):
                    self._ttft_ms = (time.perf_counter() - self._t0) * 1000.0
            except Exception:
                pass

    def _finish(self) -> None:
        """Ledger bookkeeping — must never break a billed response."""
        if self._done:
            return
        self._done = True
        try:
            self._write_row()
        except Exception:
            log.warning("ai-gateway-router: ledger satırı yazılamadı (stream)", exc_info=True)

    def _write_row(self) -> None:
        total_ms = (time.perf_counter() - self._t0) * 1000.0
        chunk = self._last_chunk
        row = dict(self._meta)
        row.update(_usage_fields(_extra(chunk, "usage") if chunk is not None else None))
        row["ttft_ms"] = round(self._ttft_ms, 1) if self._ttft_ms is not None else None
        row["total_ms"] = round(total_ms, 1)
        if chunk is not None:
            choices = getattr(chunk, "choices", None) or []
            delta = getattr(choices[0], "delta", None) if choices else None
            resolved, _, _ = _routing(delta)
            row["generation_id"] = _extra(chunk, "generationId", "id")
            row["provider_actual"] = resolved or row.get("provider_actual")
            # 3.1: stream — affinity alanları SON chunk'ın delta'sı üzerinden
            row.update(_affinity_fields(_routing_dict(delta)))
        row["outcome"] = "aborted" if self._aborted else "ok"
        _after_request(row, self._meta, self._planning)


_WARNED = {"failover": False, "post": False}


def _warn_once(key: str, msg: str, *args) -> None:
    if _WARNED.get(key):
        return
    _WARNED[key] = True
    log.warning(msg, *args, exc_info=True)


def _after_request(row: dict, meta: dict, planning: Any = None) -> None:
    """2.4 ledger + 2.6 failover işleme (asla istisna fırlatmaz).

    ``planning`` is injected by the plugin; this module never imports it, so a
    package-style load cannot fail with ModuleNotFoundError.
    """
    try:
        if ledger_enabled():
            write_ledger_row(row)
        actual, intended = row.get("provider_actual"), meta.get("provider_intended")
        if not actual or not intended:
            return
        only = meta.get("only_set") or []
        session_key = meta.get("session_key")
        model = meta.get("model")
        if actual != intended:
            log.warning("ai-gateway-router: failover model=%s intended=%s actual=%s",
                        model, intended, actual)
            if session_key and model and actual in only:
                if planning is None:
                    _warn_once("failover", "ai-gateway-router: planning module not injected — "
                                           "session lock NOT moved after failover")
                else:
                    planning.lock_provider(model, session_key, actual)
                    log.warning("ai-gateway-router: kilit %s → %s taşındı (failover)",
                                intended, actual)
    except Exception:
        _warn_once("post", "ai-gateway-router: post-request işlenemedi (tek seferlik uyarı)")


class _InstrumentedCompletions:
    def __init__(self, inner: Any, planning: Any = None):
        self._inner = inner
        self._planning = planning

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def create(self, **kwargs: Any):
        t0 = time.perf_counter()
        meta = _request_meta(kwargs)
        result = self._inner.create(**kwargs)
        if hasattr(result, "__iter__") and not hasattr(result, "choices"):
            return _InstrumentedStream(result, meta, t0, self._planning)
        # non-stream: ttft ölçülemez (tek parça döner) → None
        try:
            row = dict(meta)
            row.update(_usage_fields(_extra(result, "usage")))
            row["ttft_ms"] = None
            row["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            choice = (getattr(result, "choices", None) or [None])[0]
            message = getattr(choice, "message", None) if choice is not None else None
            resolved, _, _ = _routing(message)
            row["provider_actual"] = resolved
            row["generation_id"] = _extra(result, "generationId", "id")
            # 3.1: non-stream — affinity alanları choices[0].message üzerinden
            row.update(_affinity_fields(_routing_dict(message)))
            row["outcome"] = "ok"
            _after_request(row, meta, self._planning)
        except Exception:
            log.warning("ai-gateway-router: ledger satırı yazılamadı (non-stream)", exc_info=True)
        return result


def _request_meta(kwargs: dict) -> dict:
    body = kwargs.get("extra_body") or {}
    gw = (body.get("providerOptions") or {}).get("gateway") or {}
    order = list(gw.get("order") or [])
    only = list(gw.get("only") or order)
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": kwargs.get("model"),
        "session_id": gw.get("sessionId"),
        "session_key": gw.get("sessionId"),
        "provider_intended": order[0] if order else None,
        "provider_actual": None,
        "only_set": only,
    }


class _InstrumentedChat:
    def __init__(self, inner: Any, planning: Any = None):
        self._inner = inner
        self.completions = _InstrumentedCompletions(inner.completions, planning)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def wrap_client(client: Any, planning: Any = None) -> Any:
    """Instrument an openai.OpenAI client; isinstance(client, OpenAI) is preserved.

    ``planning`` is the plugin's planning module (passed in, not imported).
    """
    inner_chat = client.chat
    client.chat = _InstrumentedChat(inner_chat, planning)  # type: ignore[assignment]
    return client
