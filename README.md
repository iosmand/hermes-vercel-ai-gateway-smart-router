# hermes-vercel-ai-gateway-smart-router

A Hermes model-provider plugin for the Vercel AI Gateway. For each model it ranks
the gateway's upstream providers and sends `order`, `only`, `has`, `sessionId`
and `sessionAffinity` in `providerOptions.gateway`, so a request lands on the
provider that is the best trade-off of price, latency, throughput, uptime and
declared precision — and stays there while the prompt cache is warm.

Türkçe sürüm: [README.tr.md](README.tr.md).

## What it does

1. Reads a per-model rule from `$HERMES_HOME/ai-gateway-router/rules.json`.
2. Reads the provider catalog from `GET /v1/models/{model}/endpoints`
   (cached in `stats/<model>.json`).
3. Filters providers that cannot serve the request: precision policy, the
   `tool-use` tag, uptime floor, minimum context, active quarantine.
4. Scores the rest and puts the best `max_candidates` into `only` (in order).
5. Pins one provider per session (`sessions.json`) so implicit prompt caching
   keeps hitting the same upstream.
6. Watches the response for the provider that actually served it; on a real
   failover it moves the session lock (`ledger.jsonl` records the move).
7. Records provider failures as strikes and quarantines a failing provider for
   `(model, provider)` only.

## Requirements

| Item | Value |
|---|---|
| Hermes Agent | unknown — the plugin relies on `providers.ProviderProfile`, `create_client()` and `classify_api_error()`; it was developed and tested against the version that ships these hooks |
| Python | >= 3.10 |
| PyYAML | only for `plan.py --config router.yaml` (not needed for the plugin or `--check-rules`) |
| API key | `AI_GATEWAY_API_KEY` in the environment or in `$HERMES_HOME/.env` |

## Install

```bash
git clone https://github.com/iosmand/hermes-vercel-ai-gateway-smart-router.git
cd hermes-vercel-ai-gateway-smart-router
export HERMES_HOME="$HOME/.hermes"

# 1) your rules (the example contains no real values)
mkdir -p "$HERMES_HOME/ai-gateway-router"
cp rules.example.json "$HERMES_HOME/ai-gateway-router/rules.json"
$EDITOR "$HERMES_HOME/ai-gateway-router/rules.json"

# 2) plugin + rules
./deploy.sh
```

`deploy.sh` copies `plugin/*` to `$HERMES_HOME/plugins/model-providers/ai-gateway/`,
keeps your existing `rules.json`, backs up every file it overwrites as
`.bak-<timestamp>`, validates the JSON, byte-compiles the plugin and runs a
verification in a throwaway `HERMES_HOME`. It never restarts Hermes; reload the
gateway (or the CLI) to pick up new plugin code.

## Rule reference

Every key is optional except that a model needs a rule (or the `"*"` default
rule) to be routed at all.

| Key | Type | Default | Meaning | Code |
|---|---|---|---|---|
| `role` | string | – | free-form note, never used | `planning.rule_for` |
| `precision.mode` | `auto`\|`strict`\|`exclude`\|`any` | `auto` | precision policy applied to the candidate pool | `planning.resolve_precision` |
| `precision.require` | string | `fp8` | quantization required in `strict` mode | idem |
| `precision.exclude` | list | `["fp4"]` | quantization excluded in `exclude` mode | idem |
| `precision.allow_unknown` | bool | `false` | strict mode also keeps providers that declare nothing | idem |
| `precision_policy.strict_min_pool` | int | 3 | declared-FP8 providers needed for auto→strict | idem |
| `precision_policy.strict_max_unknown_share` | float | 0.30 | max undeclared share for auto→strict | idem |
| `workload.input_tokens` | int | 8000 | expected prompt size for cost scoring | `planning._score` |
| `workload.output_tokens` | int | 1200 | expected completion size | idem |
| `workload.cached_ratio` | float | 0.0 | share of the prompt served from cache | idem |
| `time_value_cents_per_s` | float | 0.15 | what one second of waiting is worth | idem |
| `require_tools` | bool | `true` | drop providers without the `tool-use` tag | `planning.rank` |
| `require_metrics` | bool | `false` | drop providers without latency/throughput metrics | idem |
| `missing_metric_penalty` | float | 2.0 | pessimism for a missing metric (latency ×, throughput ÷) | `planning._score` |
| `min_uptime_15m` | float | – | uptime floor (0–100); ≤0/NaN always fails | `planning.rank` |
| `min_context` | int | – | minimum `context_length` | idem |
| `model_context` | int | max over providers | reference context for the shrink penalty | `planning._quality` |
| `max_candidates` | int | 4 | providers placed in `order`/`only` | `planning.build_gateway` |
| `session_affinity` | bool | `true` | pin one provider per session | idem |
| `stats_ttl_s` | float | 600 | catalog cache TTL (seconds) | `planning.stats` |
| `quality.*` | object | see `DEFAULT_QUALITY` | FP4 / unknown-precision / structured-output / context penalties | `planning._quality` |

A `"*"` entry in `models` is the default rule for every model without an
explicit entry. `python3 plan.py --check-rules` validates the file.

## Scoring

```
score_cents = (work_cost + time_value_cents_per_s * seconds) * risk / quality
work_cost   = (in - cached) * price_in + cached * price_cache + out * price_out
seconds     = ttft_ms / 1000 + out / tps
risk        = 100 / uptime          (unknown uptime = 1.0, measured 0% = 100.0)
quality     = product of the penalties in quality.* (lower = worse)
```

The provider with the lowest score is chosen; `p95` is never blended into
throughput — the catalog's `throughput_last_1h.p50` is the fast tail.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `HERMES_AI_GATEWAY_ROUTING` | on | `off`/`0`/`false` disables injection **and** strike recording |
| `HERMES_AI_GATEWAY_LEDGER` | on | `off` disables `ledger.jsonl` rows |
| `HERMES_AI_GATEWAY_LEDGER_MAX_BYTES` | 5000000 | rotate `ledger.jsonl` to `.1` past this size (0 = never) |
| `HERMES_AI_GATEWAY_RULES` | – | alternative `rules.json` path |
| `HERMES_AI_GATEWAY_RULES_SOURCE` | `<repo>/rules.json` | source file `deploy.sh` copies |
| `HERMES_AI_GATEWAY_STATS_TTL` | 600 | stats TTL used by `infer_provider` |
| `HERMES_AI_GATEWAY_LOG_UTC` | – | `collect_desktop.py`: log timestamps are UTC |
| `AI_GATEWAY_API_KEY` | – | the gateway key (env or `$HERMES_HOME/.env`) |

## State files

All of them live in `$HERMES_HOME/ai-gateway-router/` (mode 0700) and are never
committed.

| File | Written by | Lifetime |
|---|---|---|
| `rules.json` | you / `plan.py` | static |
| `stats/<model>.json` | `planning.stats` | 10 min (background refresh; 3 s cold fetch, 60 s negative cache) |
| `sessions.json` | `planning` | one entry per session, idle TTL 6 h |
| `strikes.json` | `planning.record_strike` | 30 s cooldown, 2 strikes = quarantine |
| `demoted.json` | idem | 10 min quarantine per (model, provider) |
| `strike_pause.json` | idem | 5 min global pause after two providers fail in 60 s |
| `ledger.jsonl` | `transport` | one row per request, rotates at 5 MB |
| `ledger_desktop.jsonl` | `collect_desktop.py` | one row per desktop generation |
| `collect_desktop.offset` | idem | log offset + inode + first-line hash |
| `collect_desktop.pending.jsonl` | idem | ids younger than 20 s or temporarily failing |
| `collect_desktop.missing.jsonl` | idem | ids still 404 after the 60-minute window |
| `.state.lock`, `collect_desktop.lock` | idem | flock files for concurrent runs |

## Desktop collector

Desktop turns bypass the transport wrapper, so they write no `ledger.jsonl`
row. `collect_desktop.py` reads `$HERMES_HOME/logs/agent.log` incrementally and
asks `GET /v1/generation?id=...` for the real numbers:

```bash
python3 collect_desktop.py             # desktop sessions only
python3 collect_desktop.py --all       # every session
python3 collect_desktop.py --coverage  # coverage report, collects nothing
python3 collect_desktop.py --retry-missing
```

Run it every 10 minutes (systemd timer or cron), not as a daemon. Cost: one
`/v1/generation` call per new desktop generation, i.e. roughly one request per
desktop turn — the same order as the traffic itself.

## Tests

```bash
./tests/run_all.sh          # every offline test, one summary line each
```

| Test | Needs |
|---|---|
| `test_policy.py` | nothing (pure scoring math) |
| `test_runtime.py`, `test_strike.py`, `test_strike_limits.py` | a temp `HERMES_HOME`; `test_strike_limits` also starts a local mock HTTP server |
| `test_plugin.py`, `test_transport.py`, `test_review_f1..f6.py` | a temp `HERMES_HOME`; `test_plugin` needs the Hermes package on `PYTHONPATH` |
| `tests/live_smoke.py` | **real network + a real key** — costs money, excluded from `run_all.sh` |

## Cost warning

These commands make **billable** gateway requests: `tests/live_smoke.py`,
`bench.py`, `ledger_probe.py`, `verify_provider.py` and `plan.py --live`.
Everything else in this repository is offline.

## Troubleshooting: routing is silently off

Check, in this order:

1. Is there a rule for the model (or a `"*"` rule)? `python3 plan.py --check-rules`.
2. `HERMES_AI_GATEWAY_ROUTING` must not be `off`.
3. Is `AI_GATEWAY_API_KEY` readable? Without it the catalog cannot be fetched and
   `build_gateway()` returns `{}`.
4. `stats/<model>.json` must exist and contain providers; a failed fetch is
   cached negatively for 60 s.
5. The log tells you why: look for `no provider passes the rule filters`,
   `every provider is quarantined` or `has=... excludes the whole only set`.
6. `no_capable_providers_available` in an error body means `only`/`has` disagreed
   — the fixed code keeps them consistent, an old deployment does not.

## Rollback and uninstall

```bash
rm -rf "$HERMES_HOME/plugins/model-providers/ai-gateway"
rm -rf "$HERMES_HOME/ai-gateway-router"     # deletes rules and history
```

`deploy.sh` keeps a `.bak-<timestamp>` copy of every file it replaced, so a
rollback is `cp <file>.bak-<stamp> <file>`. Nothing outside `$HERMES_HOME` is
modified.

## License

MPL-2.0 — see [LICENSE](LICENSE). Contributions are accepted under the same
license; see [CONTRIBUTING.md](CONTRIBUTING.md).
