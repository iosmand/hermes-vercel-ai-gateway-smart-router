# hermes-vercel-ai-gateway-smart-router

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
   cd hermes-vercel-ai-gateway-smart-router
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

`deploy.sh` bir gateway yeniden başlatması yapmaz. Kaynakta operatöre özel
`rules.json` yoksa, hedefteki mevcut `rules.json` korunur; yalnızca ilk
kurulumda `rules.example.json` dosyası örnek olarak kurulur.

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
      "session_ttl_s": 21600,
      "stats_ttl_s": 600
    }
  }
}
```

`mode:auto`, endpoint katalogundaki precision bildiriminin dağılımına göre
`strict`, `exclude` veya `any` davranışını seçer.

`session_ttl_s` oturum kilidinin **boşta** kalma süresidir (varsayılan 6 saat) ve
Vercel'in 5 dakikalık `sessionAffinity` piniyle aynı şey değildir: pin 5 dakika,
bizim kilidimiz yalnız "bu oturum hangi sağlayıcıyı kullandı" bilgisini taşır.
Pin'den kısa seçilirse sıralama hâlâ pinli sağlayıcıdan uzaklaşabilir; bu yüzden
`session_ttl_s` ≥ pin penceresi olmalı (ayrıntı: İngilizce README).

Bir modelin **tüm** sağlayıcıları karantinadaysa plugin yalnız `has` gönderir
(precision korunur), `order`/`only` göndermez ve model başına dakikada bir kez
WARNING yazar; 10 dakikalık karantina bitince sıralama kendiliğinden döner. Gerçek çalışma kuralı ve
API anahtarı public depoya konmamalıdır.

## Durum dosyaları ve gizlilik

Aşağıdaki dosyalar yalnızca `$HERMES_HOME/ai-gateway-router/` altında tutulur
ve `.gitignore` ile dışlanır: `rules.json`, `plans.json`, `stats/`,
`ledger*.jsonl`, `sessions.json`, `strikes.json`, `demoted.json`,
`strike_pause.json`, `collect_desktop.*`, `generation_truth.jsonl` ve loglar.

`collect_desktop.py`, log saat dilimini açılışta son satırı saatle
karşılaştırarak belirler (bu makinede UTC); `HERMES_AI_GATEWAY_LOG_UTC=1|0` ile
elle sabitlenebilir.

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

MPL-2.0 — bkz. [LICENSE](LICENSE). Katkılar aynı lisansla kabul edilir; bkz.
[CONTRIBUTING.md](CONTRIBUTING.md). İngilizce sürüm: [README.md](README.md).
