# Changelog

## 1.1.0

Review fixes.

### Routing correctness

- Hard filters (precision, `tool-use`, quarantine) are never relaxed; only the
  `min_uptime_15m` / `min_context` floors can be dropped, in one second pass.
- `only` and `has` can no longer contradict each other; strict mode with
  `allow_unknown` sends no `has` filter.
- Routing turning off is logged as a rate-limited WARNING instead of failing
  silently; HTTP 400 `no_capable_providers_available` is reported as a
  routing-config error.
- Missing or non-positive latency/throughput is estimated from the peer median
  with `missing_metric_penalty` (new rule key, default 2.0); `require_metrics`
  (new rule key) excludes providers without metrics.
- A measured 0% or NaN uptime is the worst case, never better than 50%.
- One malformed endpoint no longer disables routing for the whole model.
- Quantization comparison is case-insensitive.

### State and process safety

- Stats cache writes are atomic; a failed fetch is negatively cached for 60 s;
  one thread per model performs the cold fetch (3 s timeout) and the others
  return immediately.
- `rules.json` is validated (unknown keys, types, `precision.mode` enum) with a
  WARNING once per file mtime; `plan.py --check-rules` runs the same check.
- State files are written atomically with mode 0600, the state directory is
  0700, and read-modify-write cycles are serialized with `flock`.
- Session locks use an idle TTL: the timestamp refreshes while the session keeps
  using its provider.
- Strikes and quarantine are keyed by `(model, provider)`; legacy files are
  migrated in memory. A strike context from another model is ignored, and a
  missing context records nothing (the `freshest_session` fallback is gone).
- With `HERMES_AI_GATEWAY_ROUTING=off` no strikes are recorded.
- The "fewer than 2 candidates" quarantine guard recomputes the pool from stats
  after a restart.

### Transport

- The planning module is injected into `wrap_client()`; no `sys.path` mutation
  and no `import planning` inside `transport.py`.
- Stream/non-stream bookkeeping can never break a billed response.
- Abandoned streams write an `outcome="aborted"` row (new ledger column).
- `ledger.jsonl` rotates past `HERMES_AI_GATEWAY_LEDGER_MAX_BYTES` (5 MB).

### Desktop collector

- The offset is not advanced when `state.db` cannot be read.
- Log rotation is detected (inode + first-line hash).
- Only HTTP 404 ages an id out; 401/403 stop the run; 5xx and timeouts are
  transient. Ages come from the log timestamp and are checked after a final
  attempt. `--retry-missing` revives given-up ids.
- `.env` parsing strips `export ` and quotes; a missing file is not an error.
- Non-blocking `flock`, atomic state writes, pending written before the offset,
  idempotent missing rows, partial trailing lines never consumed.
- Log timestamps are local time (`HERMES_AI_GATEWAY_LOG_UTC=1` for UTC logs).

### Owner decisions (review follow-up)

- **F1d:** when every provider of a model is quarantined, only `has` is sent
  (precision keeps being enforced), `order`/`only` are omitted and one WARNING per
  model per minute is logged. Ranking returns when a quarantine expires.
- **F1e:** no relaxed retry — the `no_capable_providers_available` 400 is logged
  as a routing-config warning and nothing else changes.
- **F6g:** the log timezone is detected by comparing the newest `agent.log` line
  with the clock instead of assuming local time; `HERMES_AI_GATEWAY_LOG_UTC`
  remains an explicit override.
- **Session affinity:** the lock's idle TTL is now the rule key `session_ttl_s`
  (default 21600 s) and is documented against the gateway's 5-minute pin.
- **License holder:** "Osman Dağdeviren" in `pyproject.toml` and `plugin.yaml`.

### Packaging

- MPL-2.0 license and headers, `pyproject.toml`, `CONTRIBUTING.md`,
  English `README.md` (Turkish kept as `README.tr.md`).
- `deploy.sh`: backups, argument validation, JSON validation, byte-compilation,
  verification in a throwaway `HERMES_HOME`.
- `tests/run_all.sh` runs every offline test.

### Tests and tooling

- `tests/test_policy.py` now exercises the production path (`planning.rank()`)
  instead of the dev script; 11 cases including missing-price, missing-metrics,
  cache-heavy and `require_tools` behaviour.
- `gwroute.py` is marked dev-only in its header and its divergences from
  `planning._score()` are listed in `README.md`.

## 1.0.0

First public release: provider ranking, session affinity, strikes/quarantine,
ledger, desktop collector.
