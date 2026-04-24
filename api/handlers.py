"""Telegram command handlers for AsesorFinan bot."""

from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from asesorfinan.config import settings
from asesorfinan.models import RiskProfile, UserProfile
from asesorfinan.orchestrator import run_pipeline

logger = logging.getLogger(__name__)

VALID_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"}
VALID_RISKS = {"conservador", "moderado", "agresivo"}

# Serialize pipeline runs: prevents concurrent writes to the global settings
# object and avoids OOM on Render's free-tier instances (512 MB RAM).
_pipeline_lock = asyncio.Lock()

HELP_TEXT = """*AsesorFinan* — Asesor financiero con ML 🤖

*Comando principal:*
`/analizar capital horizonte riesgo [intervalo]`

*Parámetros:*
• `capital` — en USD  _(ej: 10000)_
• `horizonte` — en meses  _(ej: 12)_
• `riesgo` — `conservador` | `moderado` | `agresivo`
• `intervalo` — `1m` `5m` `15m` `1h` `1d` `1wk` `1mo`  _(default: 1d)_

*Ejemplos:*
`/analizar 10000 12 moderado`
`/analizar 5000 3 agresivo 1h`
`/analizar 50000 24 conservador 1mo`

_El análisis tarda entre 30 y 90 segundos._
"""


def build_application(token: str) -> Application:
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("analizar", cmd_analizar))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, msg_unknown))
    return app


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 *Bienvenido a AsesorFinan*\n\n"
        "Generá análisis de portafolio con ML directamente desde el chat\\.\n\n"
        "Usá /analizar para comenzar o /help para ver los comandos\\.",
        parse_mode="MarkdownV2",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown")


async def msg_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "No reconozco ese mensaje. Usá /analizar o /help."
    )


async def cmd_analizar(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []

    if len(args) < 3:
        await update.message.reply_text(
            "❌ Faltan parámetros.\n\n"
            "Uso: `/analizar capital horizonte riesgo [intervalo]`\n"
            "Ej: `/analizar 10000 12 moderado 1d`",
            parse_mode="Markdown",
        )
        return

    # --- Parse & validate ---
    try:
        capital = float(args[0].replace(",", "").replace(".", ""))
        horizon = int(args[1])
    except ValueError:
        await update.message.reply_text(
            "❌ *capital* y *horizonte* deben ser números.\n"
            "Ej: `/analizar 10000 12 moderado`",
            parse_mode="Markdown",
        )
        return

    risk_str = args[2].lower()
    interval = args[3].lower() if len(args) > 3 else "1d"

    if risk_str not in VALID_RISKS:
        await update.message.reply_text(
            f"❌ Riesgo inválido: `{risk_str}`\n"
            "Opciones: `conservador` | `moderado` | `agresivo`",
            parse_mode="Markdown",
        )
        return

    if interval not in VALID_INTERVALS:
        await update.message.reply_text(
            f"❌ Intervalo inválido: `{interval}`\n"
            f"Opciones: {', '.join(sorted(VALID_INTERVALS))}",
            parse_mode="Markdown",
        )
        return

    if capital <= 0 or horizon <= 0:
        await update.message.reply_text("❌ Capital y horizonte deben ser mayores a 0.")
        return

    # --- Confirm receipt ---
    await update.message.reply_text(
        f"⏳ *Analizando...*\n"
        f"Capital: ${capital:,.0f} | {horizon} meses | {risk_str} | {interval}\n\n"
        f"_Esto puede tardar 30–90 segundos._",
        parse_mode="Markdown",
    )

    # --- Run pipeline (serialized) ---
    async with _pipeline_lock:
        settings.data_interval = interval

        profile = UserProfile(
            capital=capital,
            horizon_months=horizon,
            risk_profile=RiskProfile(risk_str),
            excluded_assets=[],
            max_positions=10,
        )

        try:
            loop = asyncio.get_event_loop()
            state = await loop.run_in_executor(None, run_pipeline, profile)
        except Exception as exc:
            logger.exception("Pipeline crashed")
            await update.message.reply_text(
                f"❌ Error en el pipeline:\n`{exc}`",
                parse_mode="Markdown",
            )
            return

    if state.error:
        await update.message.reply_text(f"❌ {state.error}")
        return

    for chunk in _split_message(_format_result(state)):
        await update.message.reply_text(chunk, parse_mode="Markdown")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_result(state) -> str:
    parts: list[str] = []

    portfolio = state.portfolio
    if portfolio:
        parts.append("📊 *Portafolio Recomendado*\n")
        for a in portfolio.allocations:
            parts.append(
                f"• *{a.ticker}* — {a.weight*100:.1f}% · ${a.amount_usd:,.0f} "
                f"| {a.cluster_label} | ret: {a.predicted_return_pct:+.1f}%"
            )
        parts.append(
            f"\n📈 Retorno: *{portfolio.expected_annual_return*100:.1f}%* "
            f"| Vol: *{portfolio.annual_volatility*100:.1f}%* "
            f"| Sharpe: *{portfolio.sharpe_ratio:.2f}*"
        )

    risk = state.risk_report
    if risk:
        emoji = {"verde": "🟢", "amarillo": "🟡", "rojo": "🔴"}.get(risk.signal.value, "⚪")
        parts.append(f"\n\n{emoji} *Riesgo: {risk.signal.value.upper()}*\n{risk.notes}")

    report = state.advisor_report
    if report:
        parts.append(f"\n\n📝 *Informe del Asesor*\n{report.narrative}")

    return "\n".join(parts)


def _split_message(text: str, max_len: int = 4000) -> list[str]:
    """Split at paragraph boundaries to stay under Telegram's 4096-char limit."""
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        pos = text.rfind("\n\n", 0, max_len)
        if pos == -1:
            pos = text.rfind("\n", 0, max_len)
        if pos == -1:
            pos = max_len
        chunks.append(text[:pos])
        text = text[pos:].lstrip()

    return chunks
