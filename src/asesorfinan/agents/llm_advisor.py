"""Agent 7 — LLM Advisor.

Supports Anthropic (Claude), OpenAI (GPT), and Google (Gemini).
The active provider is resolved from settings.active_llm_provider —
set LLM_PROVIDER in .env to force a specific one, or let it auto-detect
from whichever API key is present.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from asesorfinan.config import settings
from asesorfinan.models import AdvisorReport, GraphState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Eres un asesor financiero experto y empático. Tu misión es traducir análisis cuantitativos \
complejos en recomendaciones claras, honestas y accionables para el usuario.

Reglas estrictas:
- Nunca exageres los retornos ni minimices los riesgos.
- Usa lenguaje claro, sin jerga innecesaria.
- Si el portafolio tiene riesgos relevantes, menciónalos explícitamente.
- Incluye siempre el disclaimer legal al final.
- Responde en español.
"""

_DISCLAIMER = (
    "\n\n---\n*Disclaimer: Este análisis es generado por un sistema de IA con fines informativos "
    "y no constituye asesoría financiera profesional regulada. Consultá con un asesor certificado "
    "antes de tomar decisiones de inversión.*"
)


# ---------------------------------------------------------------------------
# Provider backends
# ---------------------------------------------------------------------------

class _LLMBackend(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...


class _AnthropicBackend(_LLMBackend):
    def __init__(self) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.llm_max_tokens,
            system=[
                # Prompt caching — reduces latency and cost on repeated calls
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


class _OpenAIBackend(_LLMBackend):
    def __init__(self) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=settings.openai_api_key)

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=settings.openai_model,
            max_tokens=settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content


class _GeminiBackend(_LLMBackend):
    def __init__(self) -> None:
        from google import genai
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def complete(self, system: str, user: str) -> str:
        from google.genai import types
        response = self._client.models.generate_content(
            model=settings.gemini_model,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=settings.llm_max_tokens,
            ),
            contents=user,
        )
        return response.text


def _build_backend() -> _LLMBackend:
    provider = settings.active_llm_provider
    backends = {
        "anthropic": _AnthropicBackend,
        "openai": _OpenAIBackend,
        "gemini": _GeminiBackend,
    }
    if provider not in backends:
        raise ValueError(f"Unknown LLM provider: '{provider}'. Use anthropic | openai | gemini")
    logger.info("LLM backend: %s", provider)
    return backends[provider]()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class LLMAdvisorAgent:
    def __init__(self) -> None:
        self._backend = _build_backend()

    def run(self, state: GraphState) -> GraphState:
        portfolio = state.portfolio
        risk = state.risk_report
        profile = state.user_profile
        predictions = state.predictions

        user_message = self._build_user_message(profile, portfolio, risk, predictions)

        try:
            narrative = self._backend.complete(_SYSTEM_PROMPT, user_message)
            narrative += _DISCLAIMER
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            narrative = f"[Error al generar el informe: {exc}]{_DISCLAIMER}"

        state.advisor_report = AdvisorReport(
            narrative=narrative,
            portfolio=portfolio,
            risk=risk,
        )
        return state

    def _build_user_message(self, profile, portfolio, risk, predictions) -> str:
        alloc_lines = "\n".join(
            f"  - {a.ticker}: {a.weight*100:.1f}% (${a.amount_usd:,.0f}) | "
            f"cluster: {a.cluster_label} | retorno esperado: {a.predicted_return_pct:+.1f}%"
            for a in portfolio.allocations
        )
        pred_lines = "\n".join(
            f"  - {p.ticker}: predicción {p.predicted_label.value} (confianza {p.confidence:.0%})"
            for p in predictions
            if p.ticker in {a.ticker for a in portfolio.allocations}
        )
        risk_emoji = {"verde": "🟢", "amarillo": "🟡", "rojo": "🔴"}.get(risk.signal.value, "⚪")

        return f"""Analizá el siguiente portafolio de inversión y redactá un informe narrativo completo.

## Perfil del usuario
- Capital disponible: ${profile.capital:,.0f} USD
- Horizonte: {profile.horizon_months} meses
- Perfil de riesgo: {profile.risk_profile.value}

## Portafolio propuesto
{alloc_lines}

## Métricas del portafolio
- Retorno anual esperado: {portfolio.expected_annual_return*100:.1f}%
- Volatilidad anual: {portfolio.annual_volatility*100:.1f}%
- Ratio de Sharpe: {portfolio.sharpe_ratio:.2f}

## Predicciones ML (próximos {settings.effective_prediction_horizon} barras / intervalo {settings.data_interval})
{pred_lines}

## Análisis de riesgo {risk_emoji}
{risk.notes}

## Tu tarea
1. Explicá en 2-3 párrafos POR QUÉ esta distribución tiene sentido para el perfil del usuario.
2. Destacá los activos más importantes y su rol en el portafolio.
3. Alertá sobre los riesgos principales que el usuario debe tener en cuenta.
4. Sugerí 2-3 acciones concretas de monitoreo (cuándo revisar, qué indicadores observar).
5. Usá un tono profesional pero accesible.
"""
