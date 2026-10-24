# Contributing

Thanks for taking a look. This project is small on purpose; please keep changes
focused.

## Ground rules

- License: contributions are accepted under **MPL-2.0** (see `LICENSE`). By
  opening a pull request you agree to license your change under the same terms.
- Every `.py` and `.sh` file starts with the MPL-2.0 header notice; keep it.
- No secrets in the repository: no API keys, no `.env`, no ledger rows, no real
  session or generation ids. Tests use placeholder values and a temporary
  `HERMES_HOME`.
- Everything offline must stay offline: the only test allowed to touch the
  network is `tests/live_smoke.py`, and it is excluded from `tests/run_all.sh`
  because it costs money.

## Before you open a pull request

```bash
./tests/run_all.sh            # all offline tests must pass
python3 plan.py --check-rules # if you touched rules.json handling
```

Add a regression test for every behaviour change. A test that cannot fail on the
old code does not prove anything.

## Style

- Python >= 3.10, standard library first; `PyYAML` only for `plan.py`.
- Keep `plugin/planning.py` free of exceptions escaping into a request: every
  public function either returns a value or an empty result and logs.
- Comments and log messages: English for new code (older Turkish comments are
  being migrated).
- Never log prompts, responses, token values or the API key.
