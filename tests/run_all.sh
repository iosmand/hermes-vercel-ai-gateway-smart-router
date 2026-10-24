#!/usr/bin/env bash
# Run every offline test in one go.
#
#   ./tests/run_all.sh
#
# Network tests are excluded: tests/live_smoke.py costs money and needs a key.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$(ls -d "$HOME/.hermes"/tools/python-3*/bin/python3 2>/dev/null | sort -V | tail -1 || true)}"
PY="${PY:-$(command -v python3)}"
export HERMES_HOME="${TEST_HERMES_HOME:-$(mktemp -d)}"
export PYTHONPATH="/usr/local/lib/hermes-agent${PYTHONPATH:+:$PYTHONPATH}"

pass=0
fail=0
failed=()

for t in "$REPO"/tests/test_*.py; do
  name="$(basename "$t")"
  printf '%-28s ' "$name"
  if out="$("$PY" "$t" 2>&1)"; then
    printf 'PASS  %s\n' "$(echo "$out" | tail -1)"
    pass=$((pass + 1))
  else
    printf 'FAIL\n'
    echo "$out" | sed 's/^/    /' | tail -20
    fail=$((fail + 1))
    failed+=("$name")
  fi
done

echo
echo "geçen: $pass   başarısız: $fail"
[ "$fail" -eq 0 ] || { printf 'başarısız: %s\n' "${failed[*]}"; exit 1; }
