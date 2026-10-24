"""Vercel AI Gateway provider profile — model bazlı sağlayıcı yönlendirmesi.

Bundled `ai-gateway` profilini gölgeler (registry'de son kayıt kazanır) ve
`providerOptions.gateway` bloğunu enjekte eder:

    order            → plan dosyasından gelen sağlayıcı sırası
    has              → sunucu tarafı filtre (precision / tool-use)
    sessionId        → Hermes oturum kimliğinden türetilmiş opak anahtar
    sessionAffinity  → prompt cache'i aynı sağlayıcıda tut

Plan dosyası `plans.json` (plan.py üretir), sıcak yeniden yüklenir:
mtime değişince bir sonraki istekte okunur — Hermes'i yeniden başlatmak gerekmez.

Ortam değişkenleri
------------------
HERMES_AI_GATEWAY_PLANS   plan dosyası yolu (varsayılan: $HERMES_HOME/ai-gateway-router/plans.json)
HERMES_AI_GATEWAY_ROUTING off → enjeksiyon tamamen kapanır
HERMES_AI_GATEWAY_LEDGER  1  → istek başına maliyet $HERMES_HOME/ai-gateway-router/usage.jsonl'e yazılır
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from providers import get_provider_profile, register_provider
from providers.base import ProviderProfile

try:  # paket olarak yüklendiyse
    from . import planning
except Exception:  # düz modül olarak yüklendiyse
    import sys as _sys

    _here = str(Path(__file__).resolve().parent)
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    import planning

log = logging.getLogger(__name__)

# Gateway'in kabul ettiği anahtarlar. Kanıt: tanınmayan anahtarlar sessizce
# yutuluyor (200 dönüyor), yani yazım hatası fark edilmez → beyaz liste şart.
import contextvars

# 3.1: her istek build_extra_body'de oturum + kilitli sağlayıcı bağlamını yazar;
# classify_api_error aynı istek iş parçacığından okur.
_STRIKE_CTX: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "gw_router_strike_ctx", default=None)

GATEWAY_KEYS = frozenset({
    "order", "only", "has", "sort", "models", "caching",
    "sessionId", "sessionAffinity", "serviceTier", "speed", "providerTimeouts",
})

def _session_key(session_id: str) -> str:
    """Hermes session id → gateway için opak, kararlı anahtar (cache affinity)."""
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
    return f"hermes-{digest}"


def _sanitize(gateway: dict) -> dict:
    clean = {k: v for k, v in gateway.items() if k in GATEWAY_KEYS and v is not None}
    dropped = set(gateway) - set(clean)
    if dropped:
        log.debug("ai-gateway-router: bilinmeyen gateway anahtarları atıldı: %s", sorted(dropped))
    return clean


def _provider_from_body(body: Any) -> str | None:
    """3.2: hata gövdesi başarısız sağlayıcıyı adlandırıyorsa onu döndür."""
    for src in (body, (body or {}).get("error") if isinstance(body, dict) else None):
        if isinstance(src, dict):
            for key in ("provider_name", "providerName", "provider"):
                val = src.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return None


def _full_body(error: Any, body: Any) -> Any:
    """B1: SDK `body` olarak yalnız içteki "error" nesnesini verir; providerMetadata
    kardeş anahtar düşer (canlı doğrulandı). Tam gövde error.response.json() veya
    str(error) metninde (python repr) durur — ikisi de gerçek kanıtla doğrulandı."""
    if isinstance(body, dict) and (body.get("providerMetadata") or body.get("provider_metadata")):
        return body
    if error is not None:
        resp = getattr(error, "response", None)
        if resp is not None:
            try:
                data = resp.json()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        try:
            import ast
            s = str(error)
            i = s.find("{")
            if i >= 0:
                data = ast.literal_eval(s[i:])
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return body


def _providers_from_routing(body: Any) -> list[str]:
    """B1: gerçek gateway hata gövdesindeki başarısız sağlayıcılar.

    Yol: body["providerMetadata"]["gateway"]["routing"]["modelAttempts"][*]
         ["providerAttempts"][*]. Sağlayıcı adının tam anahtarı UNKNOWN (gerçek
         gövdelerde providerAttempts boş) — aday anahtarlar denenir.
    """
    out: list[str] = []
    try:
        meta = body.get("providerMetadata") if isinstance(body, dict) else None
        meta = meta or (body.get("provider_metadata") if isinstance(body, dict) else None)
        routing = ((meta or {}).get("gateway") or {}).get("routing") or {}
        for attempt in routing.get("modelAttempts") or []:
            for pat in (attempt or {}).get("providerAttempts") or []:
                if not isinstance(pat, dict):
                    continue
                for key in ("provider_name", "providerName", "provider", "name"):
                    val = pat.get(key)
                    if isinstance(val, str) and val.strip():
                        name = val.strip()
                        if name not in out:
                            out.append(name)
                        break
    except Exception:
        return out
    return out


def _failed_providers(body: Any) -> list[str]:
    """B1: gövdede adlandırılan başarısız sağlayıcılar (benzersiz, sıralı)."""
    out = _providers_from_routing(body)
    legacy = _provider_from_body(body)
    if legacy and legacy not in out:
        out.append(legacy)
    return out


# ------------------------------------------------------------------- profil
_orig = get_provider_profile("ai-gateway")
_BaseProfile = type(_orig) if _orig is not None else ProviderProfile


class AIGatewayRouterProfile(_BaseProfile):
    """AI Gateway + plan tabanlı sağlayıcı yönlendirmesi."""

    def build_extra_body(
        self,
        *,
        session_id: str | None = None,
        provider_preferences: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> dict[str, Any]:
        body = super().build_extra_body(
            session_id=session_id, provider_preferences=provider_preferences,
            model=model, **context,
        )
        body = dict(body or {})

        if (os.environ.get("HERMES_AI_GATEWAY_ROUTING") or "").strip().lower() in {"off", "0", "false"}:
            return body

        gateway: dict[str, Any] = {}

        # Çalışma zamanı: istatistikler 10 dk cache'li, provider oturum başına kilitlenir.
        session_key = _session_key(session_id) if session_id else None
        gw = planning.build_gateway(model or "", session_key) if model else {}
        if gw.get("_stale"):
            log.debug("ai-gateway-router: %s istatistikleri bayat — arka planda yenileniyor", model)
        gateway.update({
            "order": gw.get("order"),
            "only": gw.get("only"),
            "has": gw.get("has"),
            "sessionAffinity": gw.get("sessionAffinity", True),
        })

        # Kullanıcının açık provider_routing tercihleri planı ezer.
        if provider_preferences:
            for key, value in provider_preferences.items():
                if key in GATEWAY_KEYS and value is not None:
                    gateway[key] = value
            for unsupported in set(provider_preferences) - GATEWAY_KEYS:
                log.debug("ai-gateway-router: gateway'de desteklenmeyen tercih atlandı: %s", unsupported)

        if session_id:
            gateway["sessionId"] = _session_key(str(session_id))

        # 3.1: strike bağlamı — oturum + kilitli sağlayıcı (order[0] = kilitli)
        try:
            _STRIKE_CTX.set({
                "model": model,
                "session_key": gateway.get("sessionId"),
                "locked_provider": (gateway.get("order") or [None])[0],
            })
        except Exception:
            pass

        existing = body.get("providerOptions")
        merged = dict(existing) if isinstance(existing, dict) else {}
        gw_clean = _sanitize(gateway)
        if gw_clean:
            inner = dict(merged.get("gateway") or {})
            inner.update(gw_clean)
            merged["gateway"] = inner
            body["providerOptions"] = merged
        return body

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, supports_reasoning: bool = True, **ctx: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Bundled davranışı korur; profil yüklenme sırası yüzünden base'e
        ulaşılamadıysa reasoning passthrough'u kendimiz uygularız."""
        extra, top = super().build_api_kwargs_extras(
            reasoning_config=reasoning_config, supports_reasoning=supports_reasoning, **ctx
        )
        extra, top = dict(extra or {}), dict(top or {})
        if not extra and supports_reasoning:
            extra["reasoning"] = (
                dict(reasoning_config) if reasoning_config is not None
                else {"enabled": True, "effort": "medium"}
            )
        return extra, top

    def classify_api_error(
        self, error: Any = None, *, status_code: int | None = None,
        error_code: str | None = None, message: str | None = None,
        body: Any = None, model: str | None = None, **ctx: Any,
    ) -> dict | None:
        """Sağlayıcı kaynaklı hatalarda kilitli sağlayıcıya strike yaz.

        None döneriz: Hermes'in kendi hata sınıflandırması ve model fallback zinciri
        değişmez. Yaptığımız tek şey "bu oturumun kilitli sağlayıcısı sorunlu"
        bilgisini biriktirmek; eşik aşılınca kilit bırakılır ve sağlayıcı 10 dk
        karantinaya alınır — sonraki istek sıradaki sağlayıcıya kilitlenir.
        """
        try:
            body = _full_body(error, body)                 # B1: tam gövde (SDK inner'ı tamamlar)
            failed = _failed_providers(body)               # B1: routing + eski anahtarlar
            body_provider = failed[0] if failed else None
            provider_failure = status_code in (500, 502, 503, 504, 529) or (
                error is not None and "timeout" in type(error).__name__.lower()
            )
            # 3.4: 429 yalnızca gövde sağlayıcıyı adlandırıyorsa strike sayılır;
            # gateway seviyesi limit (OpenRouter kredi kontrolü gibi) strike DEĞİLDİR.
            if status_code == 429 and failed:
                provider_failure = True
            if provider_failure and model:
                # B2: tek hata 2+ sağlayıcı adlandırıyorsa gateway seviyesi arıza
                # olasıdır — strike yazılmaz.
                if len(failed) >= 2:
                    log.warning("ai-gateway-router: global failure suspected — tek hatada %d "
                                "sağlayıcı (%s) — strike yok", len(failed), failed)
                    return None
                # 1.4: only kümesinin son sağlayıcısı da düştüyse istek düşer;
                # küme ASLA genişletilmez, hatayı logla.
                last_only = (planning.LAST_GATEWAY.get(model) or {}).get("only") or []
                if last_only:
                    log.error("ai-gateway-router: sağlayıcı hatası model=%s status=%s (%s) — "
                              "only=%s (genişletme yok)",
                              model, status_code, message or type(error).__name__, last_only)

                ctx = _STRIKE_CTX.get()
                target_provider = body_provider                  # 3.2: hata gövdesi sağlayıcıyı adlandırıyorsa
                session_key = (ctx or {}).get("session_key")
                if target_provider is None:
                    target_provider = (ctx or {}).get("locked_provider")  # 3.1: bu isteğin kilidi
                if not ctx or not session_key:
                    # 3.3: bağlam yok → uyarı + eski freshest_session yedeği
                    log.warning("ai-gateway-router: strike bağlamı yok (model=%s) — "
                                "freshest_session yedeği kullanılıyor", model)
                if target_provider is None:
                    hit = planning.freshest_session(model)
                    if hit:
                        session_key, target_provider = hit[0], hit[1]
                if target_provider:
                    planning.record_strike(model, session_key, target_provider,
                                           f"HTTP {status_code or 'timeout'}")
        except Exception as exc:
            log.debug("ai-gateway-router: strike kaydedilemedi: %s", exc)
        return None

    def create_client(self, **client_kwargs: Any) -> Any | None:
        """2.1: yanıtın göründüğü en alt katman bizde — instrumented OpenAI istemcisi.

        Çekirdek (agent/chat_completion_helpers.py:787 ve :2959) yalnız
        client.chat.completions.create() çağırır; biz onu sarmalayıp gerçek
        sağlayıcıyı (resolvedProvider), generationId'yi ve ttft/total sürelerini
        okuyoruz. İstek gövdesi ve yanıt metni ASLA kaydedilmez.
        """
        try:
            import openai
            try:
                client = openai.OpenAI(**client_kwargs)
            except TypeError:
                known = {k: v for k, v in client_kwargs.items()
                         if k in {"api_key", "base_url", "timeout", "max_retries",
                                  "default_headers", "default_query", "http_client"}}
                client = openai.OpenAI(**known)
        except Exception as exc:
            log.warning("ai-gateway-router: OpenAI istemcisi kurulamadı (%s) — standart yol", exc)
            return None
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ai_gateway_router_transport", str(Path(__file__).with_name("transport.py")))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.wrap_client(client)

    def get_usage_cost(self, model: str, usage: Any) -> Any | None:
        """Gateway'in döndürdüğü gerçek maliyeti kullan (tahmin yerine fatura değeri)."""
        raw = getattr(usage, "raw_usage", None)
        cost = None
        if isinstance(raw, dict):
            cost = raw.get("cost")
            if cost is None:
                cost = (raw.get("cost_details") or {}).get("gateway_cost") if isinstance(raw.get("cost_details"), dict) else None
        if cost is None:
            return None
        try:
            amount = Decimal(str(cost))
        except Exception:
            return None
        if amount <= 0:
            return None
        # 2.4: ledger satırını artık transport.py yazıyor (istek başına bir satır);
        # buradaki eski usage.jsonl yazarı kaldırıldı.
        try:
            from agent.usage_pricing import CostResult

            return CostResult(
                amount_usd=amount, status="actual", source="provider_generation_api",
                label="gateway", pricing_version="ai-gateway-usage",
                notes=("AI Gateway response usage.cost",),
            )
        except Exception:
            return None


def _unshadow_hook_methods(prof: Any) -> Any:
    """ProviderProfile dataclass alanı `classify_api_error=None` metodu GÖLGELİYOR
    (instance __dict__ None döndürür). error_classifier getattr() ile okuduğu için
    kanca hiç çalışmazdı — instance özniteliğini kaldırıp sınıf metodunu görünür yap."""
    if prof.__dict__.get("classify_api_error", "yok") is None:
        del prof.__dict__["classify_api_error"]
    return prof


if _orig is not None:
    _fields = {f.name: getattr(_orig, f.name) for f in dataclasses.fields(_orig)}
    _profile = _unshadow_hook_methods(AIGatewayRouterProfile(**_fields))
    # Bundled profil "vercel" adıyla da kayıtlı; alias'ları koruyoruz.
    _profile.display_name = _profile.display_name or "Vercel AI Gateway (router)"
    register_provider(_profile)
else:  # bundled profil yoksa bile temel profili kaydet
    register_provider(_unshadow_hook_methods(AIGatewayRouterProfile(
        name="ai-gateway", aliases=("vercel", "vercel-ai-gateway", "ai_gateway", "aigateway"),
        env_vars=("AI_GATEWAY_API_KEY",), base_url="https://ai-gateway.vercel.sh/v1",
        default_headers={"HTTP-Referer": "https://hermes-agent.nousresearch.com", "X-Title": "Hermes Agent"},
        default_aux_model="google/gemini-3-flash",
    )))
