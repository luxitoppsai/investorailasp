"""Telegram handlers — guided conversational flow with inline buttons."""

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
ASK_ASSETS, ASK_CUSTOM_TICKERS, ASK_CAPITAL, ASK_INTERVAL, ASK_HORIZON, ASK_RISK, CONFIRM = range(7)

_pipeline_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Asset presets
# ---------------------------------------------------------------------------
ASSET_PRESETS = {
    "etf": {
        "label": "📦 ETFs Diversificados",
        "tickers": ["SPY", "QQQ", "GLD", "TLT", "BND", "VNQ", "EFA", "IWM", "HYG", "SHY"],
        "desc": "SPY · QQQ · GLD · TLT · BND y más",
    },
    "tech": {
        "label": "📱 Acciones Tech",
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
        "desc": "Apple · Microsoft · Google · Amazon · NVIDIA · Tesla",
    },
    "mix": {
        "label": "🌍 Mix ETFs + Acciones",
        "tickers": ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "GLD", "TLT", "TSLA"],
        "desc": "Combina ETFs con las acciones más grandes del mercado",
    },
}

_ASSETS_KB = InlineKeyboardMarkup([
    [InlineKeyboardButton(ASSET_PRESETS["etf"]["label"],  callback_data="assets_etf")],
    [InlineKeyboardButton(ASSET_PRESETS["tech"]["label"], callback_data="assets_tech")],
    [InlineKeyboardButton(ASSET_PRESETS["mix"]["label"],  callback_data="assets_mix")],
    [InlineKeyboardButton("✏️ Elegir mis propios tickers", callback_data="assets_custom")],
])

_INTERVAL_KB = InlineKeyboardMarkup([
    [InlineKeyboardButton("⚡ 1 hora  — intraday",      callback_data="interval_1h"),
     InlineKeyboardButton("📅 1 día   — swing trading", callback_data="interval_1d")],
    [InlineKeyboardButton("📆 1 semana — medio plazo",  callback_data="interval_1wk"),
     InlineKeyboardButton("🗓️ 1 mes   — largo plazo",  callback_data="interval_1mo")],
])

_RISK_KB = InlineKeyboardMarkup([
    [InlineKeyboardButton("🛡️ Conservador — preservar capital, bajo riesgo",  callback_data="risk_conservador")],
    [InlineKeyboardButton("⚖️ Moderado    — balance entre riesgo y retorno",  callback_data="risk_moderado")],
    [InlineKeyboardButton("🚀 Agresivo    — maximizar retorno, mayor riesgo",  callback_data="risk_agresivo")],
])

_CONFIRM_KB = InlineKeyboardMarkup([
    [InlineKeyboardButton("✅ Analizar", callback_data="confirm"),
     InlineKeyboardButton("❌ Cancelar", callback_data="cancel")],
])


def _horizon_kb(interval: str) -> InlineKeyboardMarkup:
    """Return horizon options appropriate for the chosen interval."""
    if interval in ("1m", "2m", "5m", "15m", "30m", "90m", "1h"):
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("1 día",     callback_data="horizon_1d"),
             InlineKeyboardButton("3 días",    callback_data="horizon_3d"),
             InlineKeyboardButton("1 semana",  callback_data="horizon_7d")],
            [InlineKeyboardButton("2 semanas", callback_data="horizon_14d"),
             InlineKeyboardButton("1 mes",     callback_data="horizon_30d")],
        ])
    elif interval in ("1d", "5d"):
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("1 mes",   callback_data="horizon_30d"),
             InlineKeyboardButton("3 meses", callback_data="horizon_90d"),
             InlineKeyboardButton("6 meses", callback_data="horizon_180d")],
            [InlineKeyboardButton("1 año",   callback_data="horizon_365d"),
             InlineKeyboardButton("2 años",  callback_data="horizon_730d")],
        ])
    else:  # 1wk, 1mo, 3mo
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("6 meses", callback_data="horizon_180d"),
             InlineKeyboardButton("1 año",   callback_data="horizon_365d")],
            [InlineKeyboardButton("2 años",  callback_data="horizon_730d"),
             InlineKeyboardButton("3 años",  callback_data="horizon_1095d")],
        ])


def _fmt_horizon(days: int) -> str:
    if days < 30:
        return f"{days} día{'s' if days != 1 else ''}"
    if days < 365:
        m = round(days / 30)
        return f"{m} mes{'es' if m != 1 else ''}"
    y = days / 365
    return f"{y:.1f} años" if y != round(y) else f"{int(y)} año{'s' if int(y) != 1 else ''}"


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

def build_application(token: str) -> Application:
    app = Application.builder().token(token).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("analizar", cmd_analizar_start)],
        states={
            ASK_ASSETS:         [CallbackQueryHandler(recv_assets,          pattern="^assets_")],
            ASK_CUSTOM_TICKERS: [MessageHandler(filters.TEXT & ~filters.COMMAND, recv_custom_tickers)],
            ASK_CAPITAL:        [MessageHandler(filters.TEXT & ~filters.COMMAND, recv_capital)],
            ASK_INTERVAL:       [CallbackQueryHandler(recv_interval,        pattern="^interval_")],
            ASK_HORIZON:        [CallbackQueryHandler(recv_horizon,         pattern="^horizon_")],
            ASK_RISK:           [CallbackQueryHandler(recv_risk,            pattern="^risk_")],
            CONFIRM:            [CallbackQueryHandler(recv_confirm,         pattern="^(confirm|cancel)$")],
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
        "Analizá portafolios con inteligencia artificial y ML\\.\n\n"
        "Usá /analizar para comenzar o /help para ver los comandos\\.",
        parse_mode="MarkdownV2",
    )

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*AsesorFinan* 🤖\n\n"
        "• /analizar — Iniciar análisis guiado\n"
        "• /cancelar — Cancelar el análisis en curso\n"
        "• /help — Esta ayuda\n\n"
        "_El bot te guía paso a paso: elegís activos, capital, "
        "intervalo de análisis, horizonte y perfil de riesgo._",
        parse_mode="Markdown",
    )

async def msg_unknown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Usá /analizar para comenzar o /help para ver los comandos.")

async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("❌ Análisis cancelado.")
    return ConversationHandler.END


# ---------------------------------------------------------------------------
# Step 1 — Asset selection
# ---------------------------------------------------------------------------

async def cmd_analizar_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text(
        "📈 *¿Qué activos querés analizar?*\n\n"
        "• *ETFs Diversificados* — fondos indexados clásicos (SPY, QQQ, GLD…)\n"
        "• *Acciones Tech* — las grandes tecnológicas (Apple, Google, Tesla…)\n"
        "• *Mix* — combina ETFs con las acciones más importantes\n"
        "• *Personalizados* — escribís los tickers que quieras",
        parse_mode="Markdown",
        reply_markup=_ASSETS_KB,
    )
    return ASK_ASSETS


async def recv_assets(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    choice = query.data.split("_")[1]  # etf | tech | mix | custom

    if choice == "custom":
        await query.edit_message_text(
            "✏️ *Escribí los tickers separados por espacios*\n\n"
            "Ejemplos: `TSLA GOOGL AAPL MSFT AMZN NVDA`\n\n"
            "_Podés usar cualquier ticker válido de Yahoo Finance_",
            parse_mode="Markdown",
        )
        return ASK_CUSTOM_TICKERS

    preset = ASSET_PRESETS[choice]
    context.user_data["tickers"] = preset["tickers"]
    context.user_data["assets_label"] = preset["label"]

    await query.edit_message_text(
        f"{preset['label']}\n_{preset['desc']}_\n\n"
        f"💰 *¿Cuánto capital querés invertir?*\n"
        f"Escribí el monto en USD (ej: `10000`)",
        parse_mode="Markdown",
    )
    return ASK_CAPITAL


async def recv_custom_tickers(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    raw = update.message.text.upper().split()
    tickers = [t.strip() for t in raw if t.strip()]

    if len(tickers) < 2:
        await update.message.reply_text(
            "❌ Necesitás al menos 2 tickers para armar un portafolio.\n"
            "Ej: `TSLA GOOGL AAPL MSFT`",
            parse_mode="Markdown",
        )
        return ASK_CUSTOM_TICKERS

    context.user_data["tickers"] = tickers
    context.user_data["assets_label"] = f"✏️ {' · '.join(tickers[:5])}{'…' if len(tickers) > 5 else ''}"

    await update.message.reply_text(
        f"✅ Tickers: *{' · '.join(tickers)}*\n\n"
        f"💰 *¿Cuánto capital querés invertir?*\n"
        f"Escribí el monto en USD (ej: `10000`)",
        parse_mode="Markdown",
    )
    return ASK_CAPITAL


# ---------------------------------------------------------------------------
# Step 2 — Capital
# ---------------------------------------------------------------------------

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
        f"✅ Capital: *${capital:,.0f}*\n\n"
        f"📊 *¿Con qué granularidad querés analizar?*\n\n"
        f"• *1 hora* — ideal para ver movimientos del día\n"
        f"• *1 día* — el estándar para swing trading y portafolios\n"
        f"• *1 semana* — tendencias de mediano plazo\n"
        f"• *1 mes* — visión estratégica de largo plazo",
        parse_mode="Markdown",
        reply_markup=_INTERVAL_KB,
    )
    return ASK_INTERVAL


# ---------------------------------------------------------------------------
# Step 3 — Interval
# ---------------------------------------------------------------------------

async def recv_interval(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    interval = query.data.split("_")[1]
    context.user_data["interval"] = interval

    interval_labels = {
        "1h": "⚡ 1 hora", "1d": "📅 1 día",
        "1wk": "📆 1 semana", "1mo": "🗓️ 1 mes",
    }
    label = interval_labels.get(interval, interval)

    await query.edit_message_text(
        f"✅ Capital: *${context.user_data['capital']:,.0f}* · Intervalo: *{label}*\n\n"
        f"📅 *¿Cuál es tu horizonte?*\n\n"
        f"_¿Por cuánto tiempo planeás mantener esta inversión?_",
        parse_mode="Markdown",
        reply_markup=_horizon_kb(interval),
    )
    return ASK_HORIZON


# ---------------------------------------------------------------------------
# Step 4 — Horizon (adaptive)
# ---------------------------------------------------------------------------

async def recv_horizon(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    days = int(query.data.split("_")[1].replace("d", ""))
    context.user_data["horizon_days"] = days

    interval_labels = {
        "1h": "⚡ 1h", "1d": "📅 1d",
        "1wk": "📆 1sem", "1mo": "🗓️ 1mes",
    }
    iv_label = interval_labels.get(context.user_data["interval"], context.user_data["interval"])

    await query.edit_message_text(
        f"✅ Capital: *${context.user_data['capital']:,.0f}* · "
        f"Intervalo: *{iv_label}* · "
        f"Horizonte: *{_fmt_horizon(days)}*\n\n"
        f"⚖️ *¿Cuál es tu perfil de riesgo?*",
        parse_mode="Markdown",
        reply_markup=_RISK_KB,
    )
    return ASK_RISK


# ---------------------------------------------------------------------------
# Step 5 — Risk profile
# ---------------------------------------------------------------------------

async def recv_risk(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    risk = query.data.split("_")[1]
    context.user_data["risk"] = risk

    d = context.user_data
    risk_emoji = {"conservador": "🛡️", "moderado": "⚖️", "agresivo": "🚀"}.get(risk, "")
    interval_labels = {
        "1h": "⚡ 1h", "1d": "📅 1d", "1wk": "📆 1sem", "1mo": "🗓️ 1mes",
    }
    iv_label = interval_labels.get(d["interval"], d["interval"])

    await query.edit_message_text(
        f"*Confirmá tu análisis:*\n\n"
        f"📦 Activos: {d['assets_label']}\n"
        f"💰 Capital: *${d['capital']:,.0f}*\n"
        f"📊 Intervalo: *{iv_label}*\n"
        f"📅 Horizonte: *{_fmt_horizon(d['horizon_days'])}*\n"
        f"⚖️ Riesgo: *{risk_emoji} {risk.capitalize()}*",
        parse_mode="Markdown",
        reply_markup=_CONFIRM_KB,
    )
    return CONFIRM


# ---------------------------------------------------------------------------
# Step 6 — Confirm & run
# ---------------------------------------------------------------------------

async def recv_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    if query.data == "cancel":
        context.user_data.clear()
        await query.edit_message_text("❌ Análisis cancelado.")
        return ConversationHandler.END

    d = context.user_data
    horizon_months = max(0.03, d["horizon_days"] / 30)

    await query.edit_message_text(
        f"⏳ *Analizando...*\n\n"
        f"{d['assets_label']} · ${d['capital']:,.0f} · "
        f"{_fmt_horizon(d['horizon_days'])} · {d['risk']}\n\n"
        f"_Esto puede tardar 30–90 segundos._",
        parse_mode="Markdown",
    )

    async with _pipeline_lock:
        settings.data_interval = d["interval"]
        profile = UserProfile(
            capital=d["capital"],
            horizon_months=horizon_months,
            risk_profile=RiskProfile(d["risk"]),
            custom_assets=d["tickers"],
            max_positions=min(10, len(d["tickers"])),
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
# Result formatting
# ---------------------------------------------------------------------------

def _format_result(state) -> str:
    parts: list[str] = []

    portfolio = state.portfolio
    if portfolio:
        parts.append("📊 *Portafolio Recomendado*\n")
        for a in portfolio.allocations:
            if a.predicted_return_pct > 0.5:
                pred_icon = "▲"
            elif a.predicted_return_pct < -0.5:
                pred_icon = "▼"
            else:
                pred_icon = "➡"
            parts.append(
                f"• *{a.ticker}* — {a.weight*100:.1f}% · ${a.amount_usd:,.0f} "
                f"| {pred_icon} {a.predicted_return_pct:+.1f}% | {a.cluster_label}"
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
