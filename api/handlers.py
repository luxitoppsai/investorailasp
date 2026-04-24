"""Telegram handlers — conversational flow with inline buttons."""

from __future__ import annotations

import asyncio
import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from asesorfinan.config import settings
from asesorfinan.models import RiskProfile, UserProfile
from asesorfinan.orchestrator import run_pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conversation states
# ---------------------------------------------------------------------------
ASK_CAPITAL, ASK_HORIZON, ASK_RISK, ASK_INTERVAL, CONFIRM = range(5)

# Serialize runs to avoid concurrent settings writes and OOM on free tier
_pipeline_lock = asyncio.Lock()

VALID_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"}
VALID_RISKS = {"conservador", "moderado", "agresivo"}

# ---------------------------------------------------------------------------
# Keyboards
# ---------------------------------------------------------------------------

_HORIZON_KB = InlineKeyboardMarkup([
    [InlineKeyboardButton("1 mes",   callback_data="horizon_1"),
     InlineKeyboardButton("3 meses", callback_data="horizon_3"),
     InlineKeyboardButton("6 meses", callback_data="horizon_6")],
    [InlineKeyboardButton("12 meses", callback_data="horizon_12"),
     InlineKeyboardButton("24 meses", callback_data="horizon_24"),
     InlineKeyboardButton("36 meses", callback_data="horizon_36")],
])

_RISK_KB = InlineKeyboardMarkup([
    [InlineKeyboardButton("🛡️ Conservador", callback_data="risk_conservador")],
    [InlineKeyboardButton("⚖️ Moderado",    callback_data="risk_moderado")],
    [InlineKeyboardButton("🚀 Agresivo",    callback_data="risk_agresivo")],
])

_INTERVAL_KB = InlineKeyboardMarkup([
    [InlineKeyboardButton("1 hora",  callback_data="interval_1h"),
     InlineKeyboardButton("1 día",   callback_data="interval_1d")],
    [InlineKeyboardButton("1 semana",callback_data="interval_1wk"),
     InlineKeyboardButton("1 mes",   callback_data="interval_1mo")],
])

_CONFIRM_KB = InlineKeyboardMarkup([
    [InlineKeyboardButton("✅ Analizar", callback_data="confirm"),
     InlineKeyboardButton("❌ Cancelar", callback_data="cancel")],
])

# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

def build_application(token: str) -> Application:
    app = Application.builder().token(token).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("analizar", cmd_analizar_start)],
        states={
            ASK_CAPITAL:  [MessageHandler(filters.TEXT & ~filters.COMMAND, recv_capital)],
            ASK_HORIZON:  [CallbackQueryHandler(recv_horizon,   pattern="^horizon_")],
            ASK_RISK:     [CallbackQueryHandler(recv_risk,      pattern="^risk_")],
            ASK_INTERVAL: [CallbackQueryHandler(recv_interval,  pattern="^interval_")],
            CONFIRM:      [CallbackQueryHandler(recv_confirm,   pattern="^(confirm|cancel)$")],
        },
        fallbacks=[CommandHandler("cancelar", cmd_cancel)],
        per_user=True,
        per_chat=True,
    )

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(conv)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, msg_unknown))
    return app

# ---------------------------------------------------------------------------
# Non-conversation commands
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 *Bienvenido a AsesorFinan*\n\n"
        "Usá /analizar para generar un análisis de portafolio con ML\\.\n"
        "Usá /help para ver los comandos\\.",
        parse_mode="MarkdownV2",
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*AsesorFinan* — Asesor financiero con ML 🤖\n\n"
        "• /analizar — Iniciar análisis guiado paso a paso\n"
        "• /cancelar — Cancelar el análisis en curso\n"
        "• /help — Mostrar esta ayuda",
        parse_mode="Markdown",
    )

async def msg_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Usá /analizar para comenzar o /help para ver los comandos.")

async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("❌ Análisis cancelado.")
    return ConversationHandler.END

# ---------------------------------------------------------------------------
# Conversation steps
# ---------------------------------------------------------------------------

async def cmd_analizar_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text(
        "💰 *¿Cuánto capital querés invertir?*\n\n"
        "Escribí el monto en USD \\(ej: `10000`\\)",
        parse_mode="MarkdownV2",
    )
    return ASK_CAPITAL


async def recv_capital(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.replace(",", "").replace("$", "").strip()
    try:
        capital = float(text)
        if capital <= 0:
            raise ValueError
    except ValueError:
        await update.message.reply_text(
            "❌ Monto inválido. Ingresá un número mayor a 0 (ej: `10000`).",
            parse_mode="Markdown",
        )
        return ASK_CAPITAL

    context.user_data["capital"] = capital
    await update.message.reply_text(
        f"✅ Capital: *${capital:,.0f}*\n\n📅 *¿Cuál es tu horizonte de inversión?*",
        parse_mode="Markdown",
        reply_markup=_HORIZON_KB,
    )
    return ASK_HORIZON


async def recv_horizon(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    horizon = int(query.data.split("_")[1])
    context.user_data["horizon"] = horizon

    await query.edit_message_text(
        f"✅ Capital: *${context.user_data['capital']:,.0f}* · Horizonte: *{horizon} meses*\n\n"
        f"⚖️ *¿Cuál es tu perfil de riesgo?*",
        parse_mode="Markdown",
        reply_markup=_RISK_KB,
    )
    return ASK_RISK


async def recv_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    risk = query.data.split("_")[1]
    context.user_data["risk"] = risk

    risk_emoji = {"conservador": "🛡️", "moderado": "⚖️", "agresivo": "🚀"}.get(risk, "")
    await query.edit_message_text(
        f"✅ Capital: *${context.user_data['capital']:,.0f}* · "
        f"Horizonte: *{context.user_data['horizon']} meses* · "
        f"Riesgo: *{risk_emoji} {risk.capitalize()}*\n\n"
        f"📊 *¿Qué intervalo de análisis querés usar?*",
        parse_mode="Markdown",
        reply_markup=_INTERVAL_KB,
    )
    return ASK_INTERVAL


async def recv_interval(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    interval = query.data.split("_")[1]
    context.user_data["interval"] = interval

    d = context.user_data
    risk_emoji = {"conservador": "🛡️", "moderado": "⚖️", "agresivo": "🚀"}.get(d["risk"], "")
    await query.edit_message_text(
        f"*Confirmá tu análisis:*\n\n"
        f"💰 Capital: *${d['capital']:,.0f}*\n"
        f"📅 Horizonte: *{d['horizon']} meses*\n"
        f"⚖️ Riesgo: *{risk_emoji} {d['risk'].capitalize()}*\n"
        f"📊 Intervalo: *{interval}*",
        parse_mode="Markdown",
        reply_markup=_CONFIRM_KB,
    )
    return CONFIRM


async def recv_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    if query.data == "cancel":
        context.user_data.clear()
        await query.edit_message_text("❌ Análisis cancelado.")
        return ConversationHandler.END

    d = context.user_data
    await query.edit_message_text(
        f"⏳ *Analizando...*\n\n"
        f"💰 ${d['capital']:,.0f} · {d['horizon']}m · {d['risk']} · {d['interval']}\n\n"
        f"_Esto puede tardar 30–90 segundos._",
        parse_mode="Markdown",
    )

    async with _pipeline_lock:
        settings.data_interval = d["interval"]
        profile = UserProfile(
            capital=d["capital"],
            horizon_months=d["horizon"],
            risk_profile=RiskProfile(d["risk"]),
            excluded_assets=[],
            max_positions=10,
        )
        try:
            loop = asyncio.get_event_loop()
            state = await loop.run_in_executor(None, run_pipeline, profile)
        except Exception as exc:
            logger.exception("Pipeline crashed")
            await query.message.reply_text(
                f"❌ Error en el pipeline:\n`{exc}`", parse_mode="Markdown"
            )
            context.user_data.clear()
            return ConversationHandler.END

    context.user_data.clear()

    if state.error:
        await query.message.reply_text(f"❌ {state.error}")
        return ConversationHandler.END

    for chunk in _split_message(_format_result(state)):
        await query.message.reply_text(chunk, parse_mode="Markdown")

    return ConversationHandler.END

# ---------------------------------------------------------------------------
# Formatting
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
