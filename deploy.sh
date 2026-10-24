#!/usr/bin/env bash
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# deploy.sh — install the plugin and runtime files into a Hermes home.
#
#   ./deploy.sh            # plugin + rules + verification
#   ./deploy.sh --plugin   # plugin only
#   ./deploy.sh --plans    # runtime files only
#
# Source: this repository. Target: $HERMES_HOME/plugins/model-providers/ai-gateway/
# and $HERMES_HOME/ai-gateway-router/. Every overwritten file is backed up first
# as <name>.bak-<timestamp>. Changes always go through git: commit → deploy.
set -euo pipefail

usage() {
  echo "kullanım: $0 [--plugin|--plans|--help]" >&2
  exit 2
}

REPO="$(cd "$(dirname "$0")" && pwd)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PLUGIN_DST="$HERMES_HOME/plugins/model-providers/ai-gateway"
PLANS_DST="$HERMES_HOME/ai-gateway-router"
STAMP="$(date +%Y%m%d-%H%M%S)"

do_plugin=1; do_plans=1
case "${1:-}" in
  "")        ;;
  --plugin)  do_plans=0 ;;
  --plans)   do_plugin=0 ;;
  --help|-h) usage ;;
  *) echo "bilinmeyen argüman: $1" >&2; usage ;;
esac
[ "$#" -gt 1 ] && { echo "fazla argüman: ${*:2}" >&2; usage; }

backup() {   # backup <file> — only when the target exists
  [ -f "$1" ] || return 0
  cp -f "$1" "$1.bak-$STAMP"
}

if [ "$do_plugin" = 1 ]; then
  mkdir -p "$PLUGIN_DST"
  for f in plugin.yaml __init__.py planning.py transport.py; do
    backup "$PLUGIN_DST/$f"
    cp -f "$REPO/plugin/$f" "$PLUGIN_DST/$f"
  done
  rm -rf "$PLUGIN_DST/__pycache__"
  echo "plugin  → $PLUGIN_DST"
fi

if [ "$do_plans" = 1 ]; then
  mkdir -p "$PLANS_DST"
  # rules.json is operator-owned and never committed: keep the existing one,
  # install the example only on a fresh setup.
  RULES_SRC="${HERMES_AI_GATEWAY_RULES_SOURCE:-$REPO/rules.json}"
  if [ -f "$RULES_SRC" ]; then
    backup "$PLANS_DST/rules.json"
    cp -f "$RULES_SRC" "$PLANS_DST/rules.json"
    echo "kurallar → $PLANS_DST/rules.json"
  elif [ -f "$PLANS_DST/rules.json" ]; then
    echo "kurallar → mevcut $PLANS_DST/rules.json korundu"
  else
    cp -f "$REPO/rules.example.json" "$PLANS_DST/rules.json"
    echo "kurallar → örnek $PLANS_DST/rules.json kuruldu"
  fi
  if [ -f "$PLANS_DST/rules.json" ]; then
    "$(command -v python3)" -m json.tool "$PLANS_DST/rules.json" > /dev/null \
      && echo "rules.json: geçerli JSON" \
      || { echo "rules.json: GEÇERSİZ JSON — düzeltin" >&2; exit 1; }
  fi
fi

# Hermes'in araç Python'u varsa onu kullan, yoksa PATH'teki python3.
PY=$(ls -d "$HERMES_HOME"/tools/python-3*/bin/python3 2>/dev/null | sort -V | tail -1 || true)
PY=${PY:-$(command -v python3)}
if [ "$do_plugin" = 1 ] && [ -x "$PY" ]; then
  # Paths are passed as argv, never interpolated into the Python source.
  "$PY" - "$PLUGIN_DST" <<'EOF'
import compileall, sys
ok = compileall.compile_dir(sys.argv[1], quiet=1, force=True)
print("sözdizimi OK" if ok else "sözdizimi HATASI")
sys.exit(0 if ok else 1)
EOF
  rm -rf "$PLUGIN_DST/__pycache__"
fi

echo
echo "Doğrulama:"
if [ -d /usr/local/lib/hermes-agent ] && [ "$do_plugin" = 1 ]; then
  # Verification runs against a throwaway HERMES_HOME so it cannot touch state.
  VERIFY_HOME="$(mktemp -d)"
  trap 'rm -rf "$VERIFY_HOME"' EXIT
  # Copy the plugin into the throwaway home so the registry loads OUR profile
  # while the verification itself cannot touch the real state directory.
  mkdir -p "$VERIFY_HOME/plugins/model-providers"
  cp -r "$PLUGIN_DST" "$VERIFY_HOME/plugins/model-providers/ai-gateway"
  PYTHONPATH="/usr/local/lib/hermes-agent${PYTHONPATH:+:$PYTHONPATH}" \
    HERMES_HOME="$VERIFY_HOME" HERMES_AI_GATEWAY_ROUTING=off \
    "$PY" - <<'EOF'
from providers import get_provider_profile
p = get_provider_profile("vercel")
print("  profil:", type(p).__name__, "| base_url:", p.base_url)
body = p.build_extra_body(session_id="test-session", model="zai/glm-5.3-flash")
print("  gövde :", body.get("providerOptions", {}))
EOF
else
  echo "  (hermes-agent kurulu değil ya da plugin kopyalanmadı; doğrulama atlandı)"
fi
