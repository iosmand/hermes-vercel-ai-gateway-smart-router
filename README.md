# AI Gateway Router

Hermes için Vercel AI Gateway sağlayıcı seçim eklentisi. Model başına çalışma
kuralını (`rules.json`) canlı endpoint istatistikleriyle birleştirir; maliyet,
gecikme, uptime, precision ve kalite sinyallerine göre
`providerOptions.gateway` üretir.

## İçerik

| Yol | Amaç |
|---|---|
| `plugin/` | Hermes model-provider eklentisi (`ai-gateway`) |
| `router.yaml` | Genel/örnek kural kaynağı |
| `rules.example.json` | Çalışma zamanında okunacak güvenli örnek kural dosyası |
| `plan.py` | YAML kurallarından yerel plan/kurallar üretir |
| `gwroute.py` | Katalog ve maliyet/hız tabanlı seçim CLI aracı |
| `collect_desktop.py` | Masaüstü turları için isteğe bağlı, log tabanlı ledger kolektörü |
| `tests/` | Ağsız birim testleri ve isteğe bağlı canlı smoke testi |

## Kurulum

1. Bu depoyu klonlayın ve Hermes çalışma dizinini belirtin:

   ```bash
   export HERMES_HOME="$HOME/.hermes"
   cd ai-gateway-router
   ```

2. Kendi model kurallarınızı oluşturun. Örnek dosya gerçek kullanıcı verisi,
oturum kimliği, ledger veya API anahtarı içermez:

   ```bash
   mkdir -p "$HERMES_HOME/ai-gateway-router"
   cp rules.example.json "$HERMES_HOME/ai-gateway-router/rules.json"
   $EDITOR "$HERMES_HOME/ai-gateway-router/rules.json"
   ```

3. Eklentiyi ve kural dosyasını dağıtın:

   ```bash
   ./deploy.sh
   ```

`deploy.sh` bir gateway yeniden başlatması yapmaz. Kaynakta `rules.json`
yoksa `rules.example.json` dosyasını hedefte `rules.json` olarak kurar.

## Kural biçimi

Plugin, `$HERMES_HOME/ai-gateway-router/rules.json` içindeki `models`
nesnesini okur. Her model kuralı şu alanları kullanabilir:

```json
{
  "models": {
    "provider/model-id": {
      "role": "açıklama",
      "precision": {"mode": "auto", "require": "fp8", "exclude": ["fp4", "int4"]},
      "workload": {"input_tokens": 8000, "output_tokens": 1200, "cached_ratio": 0.5},
      "require_tools": true,
      "min_uptime_15m": 95,
      "max_candidates": 4,
      "time_value_cents_per_s": 0.15,
      "session_affinity": true,
      "stats_ttl_s": 600
    }
  }
}
```

`mode:auto`, endpoint katalogundaki precision bildiriminin dağılımına göre
`strict`, `exclude` veya `any` davranışını seçer. Gerçek çalışma kuralı ve
API anahtarı public depoya konmamalıdır.

## Durum dosyaları ve gizlilik

Aşağıdaki dosyalar yalnızca `$HERMES_HOME/ai-gateway-router/` altında tutulur
ve `.gitignore` ile dışlanır: `rules.json`, `plans.json`, `stats/`,
`ledger*.jsonl`, `sessions.json`, `strikes.json`, `demoted.json`,
`strike_pause.json`, `collect_desktop.*`, `generation_truth.jsonl` ve loglar.

Anahtar `AI_GATEWAY_API_KEY` ortam değişkeninden veya
`$HERMES_HOME/.env` dosyasından okunur. Araçlar anahtar değerini yazdırmaz.

## Testler

Ağsız testler:

```bash
export HERMES_HOME="$HOME/.hermes"
export PYTHONPATH=/usr/local/lib/hermes-agent${PYTHONPATH:+:$PYTHONPATH}
python3 tests/test_policy.py
python3 tests/test_plugin.py
python3 tests/test_runtime.py
python3 tests/test_strike.py
python3 tests/test_strike_limits.py
python3 tests/test_transport.py
```

`tests/live_smoke.py` gerçek AI Gateway ağına bağlanır ve geçerli bir
`AI_GATEWAY_API_KEY` ister; varsayılan test dizisine dahil değildir ve maliyet
doğurabilir.

## Lisans

Bu depoda henüz bir lisans dosyası yoktur. Yeniden kullanım veya dış katkı
koşulları için yayıncı bir lisans seçmelidir.
