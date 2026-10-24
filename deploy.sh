#!/usr/bin/env bash
# deploy.sh — plugin'i ve planları Hermes'in canlı dizinlerine kopyalar.
#
#   ./deploy.sh            # plugin + plans.json + doğrulama
#   ./deploy.sh --plugin   # sadece plugin
#   ./deploy.sh --plans    # sadece plan dosyası
#
# Kaynak: bu depo. Hedef: ~/.hermes/plugins/model-providers/ai-gateway/
# Değişiklikler her zaman git ile: commit → deploy.
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PLUGIN_DST="$HERMES_HOME/plugins/model-providers/ai-gateway"
PLANS_DST="$HERMES_HOME/ai-gateway-router"

do_plugin=1; do_plans=1
case "${1:-}" in
  --plugin) do_plans=0 ;;
  --plans)  do_plugin=0 ;;
esac

if [ "$do_plugin" = 1 ]; then
  mkdir -p "$PLUGIN_DST"
  cp -f "$REPO/plugin/plugin.yaml" "$PLUGIN_DST/plugin.yaml"
  cp -f "$REPO/plugin/__init__.py" "$PLUGIN_DST/__init__.py"
  cp -f "$REPO/plugin/planning.py" "$PLUGIN_DST/planning.py"
  cp -f "$REPO/plugin/transport.py" "$PLUGIN_DST/transport.py"
  rm -rf "$PLUGIN_DST/__pycache__"
  echo "plugin  → $PLUGIN_DST"
fi

if [ "$do_plans" = 1 ]; then
  mkdir -p "$PLANS_DST"
  # rules.json operatöre özeldir ve git'e girmez. Kaynakta yoksa güvenli örnek kurulur.
  RULES_SRC="${HERMES_AI_GATEWAY_RULES_SOURCE:-$REPO/rules.json}"
  if [ -f "$RULES_SRC" ]; then
    cp -f "$RULES_SRC" "$PLANS_DST/rules.json"
    echo "kurallar → $PLANS_DST/rules.json"
  elif [ -f "$PLANS_DST/rules.json" ]; then
    echo "kurallar → mevcut $PLANS_DST/rules.json korundu"
  else
    cp -f "$REPO/rules.example.json" "$PLANS_DST/rules.json"
    echo "kurallar → örnek $PLANS_DST/rules.json kuruldu"
  fi
fi

# Sözdizimi kontrolü; Hermes'in araç Python'u varsa kullan, yoksa PATH'teki python3.
PY=$(ls -d "$HERMES_HOME"/tools/python-3*/bin/python3 2>/dev/null | sort -V | tail -1 || true)
PY=${PY:-$(command -v python3)}
if [ "$do_plugin" = 1 ] && [ -x "$PY" ]; then
  PYTHONPATH="/usr/local/lib/hermes-agent${PYTHONPATH:+:$PYTHONPATH}" HERMES_HOME="$HERMES_HOME" "$PY" \
    -c "import ast; ast.parse(open('$PLUGIN_DST/__init__.py').read()); print('sözdizimi OK')"
fi

echo
echo "Doğrulama:"
if [ -d /usr/local/lib/hermes-agent ]; then
PYTHONPATH="/usr/local/lib/hermes-agent${PYTHONPATH:+:$PYTHONPATH}" HERMES_HOME="$HERMES_HOME" "$PY" - <<'EOF' 
from providers import get_provider_profile
p = get_provider_profile("vercel")
print("  profil:", type(p).__name__, "| base_url:", p.base_url)
body = p.build_extra_body(session_id="test-session", model="zai/glm-5.3-flash")
import json; print("  gövde :", json.dumps(body, ensure_ascii=False))
EOF
else
  echo "  (hermes-agent kurulu değil; doğrulama atlandı)"
fi
