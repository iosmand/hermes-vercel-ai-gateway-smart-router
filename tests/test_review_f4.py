#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Regression tests for review findings F4 (transport module import).

Loads the plugin the way Hermes does when the directory is a package, then
checks that the failover lock moves without a top-level ``import planning``.

Run:  HERMES_HOME=<temp> python3 tests/test_review_f4.py
No network, no API key.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

FAILS: list[str] = []
TMP = Path(tempfile.mkdtemp(prefix="gw-f4-"))
os.environ["HERMES_HOME"] = str(TMP)
PLUGIN = Path(__file__).resolve().parents[1] / "plugin"


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


# --- load as a package exactly like a plugin loader would ---
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "gwplug", PLUGIN / "__init__.py", submodule_search_locations=[str(PLUGIN)])
plugin = importlib.util.module_from_spec(spec)
sys.modules["gwplug"] = plugin
spec.loader.exec_module(plugin)

check("F4: planning resolves as a package submodule",
      getattr(plugin, "planning", None) is not None
      and plugin.planning.__name__ == "gwplug.planning",
      str(getattr(plugin, "planning", None)))
check("F4: no sys.path entry was added for the plugin directory",
      str(PLUGIN) not in sys.path, str(sys.path[:3]))

# --- failover lock moves with the planning module passed in ---
transport = plugin._load_transport()
check("F4: transport is loaded once and cached", transport is plugin._load_transport())

MODEL = "m/f4"
SESSION = "sess-f4"


def routing_response(provider):
    routed = {"gateway": {"routing": {"resolvedProvider": provider}}}
    msg = SimpleNamespace(content="ok", provider_metadata=routed)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None,
                           generationId="gen_test")


class _Completions:
    def __init__(self, provider):
        self.provider = provider

    def create(self, **kwargs):
        return routing_response(self.provider)


class _Chat:
    def __init__(self, provider):
        self.completions = _Completions(provider)


client = SimpleNamespace(chat=_Chat("bb"))
os.environ["HERMES_AI_GATEWAY_LEDGER"] = "off"
wrapped = transport.wrap_client(client, planning=plugin.planning)
wrapped.chat.completions.create(
    model=MODEL,
    extra_body={"providerOptions": {"gateway": {
        "order": ["aa"], "only": ["aa", "bb"], "sessionId": SESSION}}})

sessions = json.loads((TMP / "ai-gateway-router" / "sessions.json").read_text(encoding="utf-8"))
check("F4: failover moves the session lock (intended aa → actual bb)",
      sessions.get(SESSION, {}).get("provider") == "bb", json.dumps(sessions))

print(f"\n{len(FAILS)} hata" if FAILS else "\ntüm testler geçti")
sys.exit(1 if FAILS else 0)
