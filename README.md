# AsesorFinan

Asesor financiero multi-agente con Machine Learning. Analiza activos financieros, construye portafolios optimizados y genera recomendaciones en lenguaje natural. Disponible como CLI y como bot de Telegram desplegado en Render.

---

## Tabla de Contenidos

- [Arquitectura General](#arquitectura-general)
- [Pipeline Multi-Agente](#pipeline-multi-agente)
  - [Agente 1 — Data Collector](#agente-1--data-collector)
  - [Agente 2 — Feature Engineer](#agente-2--feature-engineer)
  - [Agente 3 — Asset Clusterer](#agente-3--asset-clusterer)
  - [Agente 4 — Return Predictor](#agente-4--return-predictor)
  - [Agente 5 — Portfolio Optimizer](#agente-5--portfolio-optimizer)
  - [Agente 6 — Risk Manager](#agente-6--risk-manager)
  - [Agente 7 — LLM Advisor](#agente-7--llm-advisor)
- [Soporte Multi-Timeframe](#soporte-multi-timeframe)
- [Proveedores LLM](#proveedores-llm)
- [Bot de Telegram](#bot-de-telegram)
- [Instalación y Uso Local](#instalación-y-uso-local)
- [Deploy en Render](#deploy-en-render)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Variables de Entorno](#variables-de-entorno)

---

## Arquitectura General

```
Usuario (CLI o Telegram)
        │
        ▼
  Orchestrator (LangGraph StateGraph)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Agent 1      Agent 2       Agent 3      Agent 4      │
│  Data         Feature       Asset        Return        │
│  Collector ──► Engineer ──► Clusterer ──► Predictor   │
│                                                        │
│  yfinance      Returns      HDBSCAN /    XGBoost       │
│  FRED API      Volatility   KMeans       Walk-forward  │
│  Alpha Vantage Momentum     PCA          CV            │
│  Options IV    Técnicos                                │
└───────────────────────────┬───────────────────────────┘
                            │
                            ▼
             ┌──────────────────────────┐
             │  Agent 5                 │
             │  Portfolio Optimizer     │
             │  Black-Litterman +       │
             │  Markowitz / PyPortOpt   │
             └──────────────┬───────────┘
                            │
                            ▼
             ┌──────────────────────────┐
             │  Agent 6 — Risk Manager  │
             │  Monte Carlo VaR/CVaR    │◄─── rechazado (hasta 3 reintentos)
             │  Stress Test histórico   │         con cap más estricto
             │  Semáforo verde/rojo     │
             └──────────────┬───────────┘
                            │ aprobado
                            ▼
             ┌──────────────────────────┐
             │  Agent 7 — LLM Advisor   │
             │  Claude / GPT / Gemini   │
             │  Informe narrativo final │
             └──────────────────────────┘
```

El orquestador implementa un **loop de riesgo con reintentos**: si el Risk Manager rechaza el portafolio, vuelve al optimizador con restricciones más estrictas (el cap por activo se reduce 10% por intento, hasta 3 veces).

---

## Pipeline Multi-Agente

### Agente 1 — Data Collector

**Archivo:** `src/asesorfinan/agents/data_collector.py`

Descarga y consolida datos de múltiples fuentes:

| Fuente | Datos | Uso |
|--------|-------|-----|
| **yfinance** | OHLCV histórico por intervalo | Precios base para todo el pipeline |
| **FRED API** | 14 series macro (Fed Rate, Yield Curve 10y2y, VIX, M2, HY Spread, NFCI…) | Features macroeconómicas |
| **yfinance `.info`** | Beta, P/E, Dividend Yield, 52w Range, Market Cap | Features fundamentales por ticker |
| **yfinance options** | IV at-the-money, Put/Call ratio | Sentimiento de opciones |
| **Alpha Vantage NEWS_SENTIMENT** | Scores pre-computados por ticker (últimos 14 días) | Sentimiento de noticias |
| **yfinance news** (fallback) | Títulos + resúmenes con keyword scoring | Sentimiento si no hay Alpha Vantage key |

**Señales derivadas de precios** (sin APIs externas):
- `credit_proxy_21d` — spread HYG vs TLT (proxy de riesgo crediticio)
- `breadth_1m` — fracción de activos con retorno positivo en el período
- `mkt_vol_21d` — volatilidad realizada del portafolio igual-ponderado

**Comportamiento por intervalo:**
- **Intraday** (1m–1h): omite FRED, fundamentales y opciones — no son relevantes a esa resolución y ralentizan el proceso
- **Daily/Weekly/Monthly**: descarga todas las fuentes

El lookback se calcula automáticamente desde `settings.max_lookback_days` con un buffer de 3 días para no chocar con el límite exclusivo de yfinance.

---

### Agente 2 — Feature Engineer

**Archivo:** `src/asesorfinan/agents/feature_engineer.py`

Construye la matriz de features por ticker. Todas las ventanas se escalan proporcionalmente según `settings.bars_per_day` para que los mismos parámetros funcionen en cualquier timeframe.

**Features base (siempre disponibles):**

| Feature | Descripción |
|---------|-------------|
| `ret_30d` | Retorno de los últimos ~30 días de trading |
| `ret_90d` | Retorno de los últimos ~90 días |
| `ret_1y` | Retorno del último año |
| `annual_vol` | Volatilidad anualizada (log returns × √annualization_factor) |
| `sharpe` | Ratio de Sharpe anualizado |
| `max_drawdown` | Máximo drawdown histórico |

**Features técnicas (via pandas-ta, opcional):**
RSI(14), MACD(12,26,9), Bollinger Bands %B, SMA(50), EMA(20), Price vs SMA50.

**Features macro** (mismo valor para todos los tickers, snapshot del último dato disponible):
Fed Rate, Yield Curve, VIX, HY Spread, NFCI, M2, y señales derivadas de precios.

**Features fundamentales** (por ticker, cuando disponibles):
Beta, P/E trailing/forward, Dividend Yield, Price-to-Book, ROE, Debt/Equity, IV ATM, Put/Call ratio, Sentimiento de noticias (4 features).

---

### Agente 3 — Asset Clusterer

**Archivo:** `src/asesorfinan/agents/asset_clusterer.py`

Agrupa los activos en regímenes de mercado usando clustering no supervisado:

1. **Normalización** de la matriz de features
2. **PCA** para reducción dimensional (umbral de varianza explicada: 95%)
3. **HDBSCAN** (si disponible) o **KMeans** con `n_clusters=4` por defecto
4. Asigna nombres descriptivos a los clusters según sus características dominantes

Los cluster IDs se usan como feature adicional en el predictor ML y como metadato en el portafolio final.

---

### Agente 4 — Return Predictor

**Archivo:** `src/asesorfinan/agents/return_predictor.py`

Entrena un modelo **XGBoost** (fallback: GradientBoostingClassifier de sklearn) por activo para predecir la dirección del retorno en los próximos N barras.

**Labels de clasificación:**
- `sube` — retorno futuro > threshold
- `baja` — retorno futuro < -threshold
- `neutro` — retorno dentro del threshold

El threshold y el horizonte se determinan automáticamente por intervalo (ver [Soporte Multi-Timeframe](#soporte-multi-timeframe)).

**Construcción del dataset — sin data leakage:**
Para cada punto de entrenamiento `i`, todas las features se calculan exclusivamente sobre `[i-window, i]`:
- Media, volatilidad, último retorno
- Momentum corto y medio (fracciones del window)
- Max/min, fracción de días positivos, autocorrelación, skew
- Bollinger %B aproximado, RSI proxy
- Cluster ID del activo

**Validación con walk-forward CV** (no k-fold cruzado):
El dataset se divide en `walk_forward_splits` (default: 5) franjas cronológicas. El modelo se entrena siempre en datos pasados y valida en datos futuros, respetando la naturaleza temporal de las series. El modelo final se entrena sobre todos los datos.

**Output por activo:** label predicho (`sube`/`neutro`/`baja`), confianza (probabilidad máxima del clasificador), retorno esperado (proxy del feature `ret_30d`).

---

### Agente 5 — Portfolio Optimizer

**Archivo:** `src/asesorfinan/agents/portfolio_optimizer.py`

Construye el portafolio óptimo combinando teoría de portafolios con las predicciones ML:

**Black-Litterman** (si PyPortfolioOpt está disponible):
1. Calcula retornos históricos esperados y covarianza con Ledoit-Wolf shrinkage
2. Convierte las predicciones ML en "views" absolutas:
   - `sube` con confianza C → retorno histórico + C × 0.05
   - `baja` con confianza C → retorno histórico − C × 0.05
   - `neutro` → sin view (deja dominar el prior)
3. Aplica el modelo Black-Litterman para combinar prior y views
4. Optimiza según perfil de riesgo:
   - **Conservador** → `min_volatility()`
   - **Moderado** → `max_sharpe(risk_free_rate=0.05)`
   - **Agresivo** → `max_quadratic_utility(risk_aversion=0.5)`

**Fallback** (si PyPortfolioOpt no disponible): igual-ponderado.

Retornos y covarianzas se anualizan con `settings.annualization_factor` = 252 × bars_per_day, garantizando correctitud en cualquier timeframe.

---

### Agente 6 — Risk Manager

**Archivo:** `src/asesorfinan/agents/risk_manager.py`

Valida el portafolio propuesto con dos métodos:

**Monte Carlo VaR/CVaR:**
- 10.000 simulaciones en bar-space (no en días calendario)
- `mu_bar = mu_annual / annualization_factor` — per-bar drift
- `sigma_bar = sigma_annual / √annualization_factor` — per-bar vol
- Horizonte = `effective_prediction_horizon` barras
- Calcula VaR 95% y CVaR (Expected Shortfall)

**Stress Testing:**
Simula el portafolio en 4 escenarios históricos: Crisis 2008, COVID 2020, Dot-com 2000, Subida de tasas 2022.
- Si el período está dentro del historial disponible → usa datos reales
- Si no (común con lookbacks cortos) → **fallback paramétrico** basado en betas del portafolio × shocks de mercado documentados (−55%, −34%, −49%, −25%)

**Semáforo de riesgo:**

| Perfil | Verde si VaR ≤ | Amarillo si VaR ≤ |
|--------|---------------|------------------|
| Conservador | 5% | 15% |
| Moderado | 8% | 25% |
| Agresivo | 12% | 40% |

Si el portafolio es rechazado (rojo), el orquestador vuelve al optimizador con el cap por activo reducido en 10%, hasta 3 reintentos.

---

### Agente 7 — LLM Advisor

**Archivo:** `src/asesorfinan/agents/llm_advisor.py`

Genera el informe narrativo final usando el LLM configurado. El prompt incluye el portafolio completo, las métricas (Sharpe, volatilidad, retorno esperado), las predicciones ML por activo y el análisis de riesgo.

Soporta 3 proveedores intercambiables via `.env`:
- **Anthropic Claude** — con prompt caching (reduce costos en ~80% en el cache hit)
- **OpenAI GPT**
- **Google Gemini**

---

## Soporte Multi-Timeframe

El sistema soporta todos los intervalos de yfinance. Cada intervalo tiene parámetros óptimos pre-configurados:

| Intervalo | Horizonte predicción | Ventana features | Threshold retorno | Max lookback |
|-----------|---------------------|-----------------|-------------------|-------------|
| `1m` | 5 barras (5 min) | 60 barras | 0.1% | 7 días |
| `5m` | 6 barras (30 min) | 78 barras | 0.2% | 60 días |
| `15m` | 8 barras (2h) | 52 barras | 0.3% | 60 días |
| `1h` | 7 barras (1 día) | 35 barras | 0.5% | 730 días |
| `1d` | 30 barras (6 sem) | 20 barras | 3.0% | 10 años |
| `1wk` | 4 barras (1 mes) | 20 barras | 5.0% | 10 años |
| `1mo` | 3 barras (3 meses) | 24 barras | 7.0% | 10 años |

Todos los parámetros son sobreescribibles via `.env` o flags CLI.

---

## Proveedores LLM

Se auto-detecta el proveedor por la key presente en `.env`. Para forzar uno específico:

```env
LLM_PROVIDER=anthropic   # o openai o gemini
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AI...
```

Modelos por defecto (sobreescribibles):
```env
CLAUDE_MODEL=claude-sonnet-4-6
OPENAI_MODEL=gpt-4o
GEMINI_MODEL=gemini-2.0-flash
```

---

## Bot de Telegram

**Archivos:** `api/server.py`, `api/handlers.py`

Implementado con `python-telegram-bot` v21+ en modo polling, integrado en un servidor FastAPI. Un único proceso maneja tanto las actualizaciones de Telegram como el endpoint `/health` para keep-alive.

**Flujo conversacional guiado:**

```
/analizar
  → Selección de activos (presets o custom)
  → Capital (texto libre)
  → Intervalo (botones con descripción)
  → Horizonte (botones adaptativos según intervalo)
  → Perfil de riesgo (botones con descripción)
  → Confirmación
  → Pipeline (~30–90 segundos)
  → Resultados en mensajes formateados
```

**Presets de activos disponibles:**
- **ETFs Diversificados**: SPY, QQQ, GLD, TLT, BND, VNQ, EFA, IWM, HYG, SHY
- **Acciones Tech**: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA
- **Mix ETFs + Acciones**: combina los más representativos de ambos
- **Personalizados**: el usuario escribe cualquier ticker válido de Yahoo Finance

Las ejecuciones del pipeline se serializan con `asyncio.Lock` para evitar condiciones de carrera en el objeto `settings` global y prevenir OOM en instancias con poca RAM.

---

## Instalación y Uso Local

### Requisitos

- Python 3.11+
- Conda (recomendado) o virtualenv
- Al menos una API key de LLM (Claude, GPT o Gemini)

### Setup

```bash
# Crear entorno conda
conda create -n asesorfinan python=3.11
conda activate asesorfinan

# Instalar dependencias
pip install -r requirements.txt
pip install -e .

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys
```

### Uso CLI

```bash
# Análisis diario estándar
python main.py --capital 10000 --horizon 12 --risk moderado

# Con intervalo horario
python main.py --capital 5000 --horizon 3 --risk agresivo --interval 1h

# Con horizonte mensual
python main.py --capital 50000 --horizon 24 --risk conservador --interval 1mo

# Sobreescribir horizonte de predicción en barras
python main.py --capital 10000 --horizon 6 --risk moderado --interval 1h --prediction-bars 14
```

### Uso como bot local (para desarrollo)

```bash
# Exportar token y correr
export TELEGRAM_BOT_TOKEN=tu_token
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

---

## Deploy en Render

El archivo `render.yaml` define el servicio. Para desplegarlo:

1. **Crear bot en Telegram** via `@BotFather` → obtener token
2. **Subir el repo a GitHub**
3. **En Render**: New → Web Service → conectar repo
4. **En Settings**: configurar las variables de entorno (ver abajo)
5. **Start Command**: `uvicorn api.server:app --host 0.0.0.0 --port $PORT`
6. **Build Command**: `pip install -r requirements.txt && pip install -e .`

**Keep-alive** (free tier duerme a los 15 min sin actividad): crear un cron job en [cron-job.org](https://cron-job.org) que haga `GET https://tu-servicio.onrender.com/health` cada 10 minutos.

---

## Estructura del Proyecto

```
asesorfinan/
├── src/asesorfinan/
│   ├── config.py               # Settings, lookup tables por intervalo, propiedades derivadas
│   ├── models.py               # Pydantic models: UserProfile, GraphState, Portfolio, etc.
│   ├── orchestrator.py         # LangGraph StateGraph + loop de riesgo con reintentos
│   └── agents/
│       ├── data_collector.py   # yfinance + FRED + Alpha Vantage + opciones + noticias
│       ├── feature_engineer.py # Matrix de features por ticker (escala con bars_per_day)
│       ├── asset_clusterer.py  # PCA + HDBSCAN/KMeans
│       ├── return_predictor.py # XGBoost + walk-forward CV (sin data leakage)
│       ├── portfolio_optimizer.py # Black-Litterman + PyPortfolioOpt
│       ├── risk_manager.py     # Monte Carlo VaR/CVaR + stress test paramétrico
│       └── llm_advisor.py      # Claude / GPT / Gemini con prompt caching
├── api/
│   ├── server.py               # FastAPI + ciclo de vida del bot Telegram
│   └── handlers.py             # Flujo conversacional con ConversationHandler
├── main.py                     # CLI (Typer + Rich)
├── render.yaml                 # Configuración de deploy en Render
├── .python-version             # Python 3.11.9 (para Render)
├── requirements.txt
└── pyproject.toml
```

---

## Variables de Entorno

```env
# LLM (al menos una requerida)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=

# LLM config (opcional)
LLM_PROVIDER=               # anthropic | openai | gemini (auto-detecta si está vacío)
CLAUDE_MODEL=claude-sonnet-4-6
OPENAI_MODEL=gpt-4o
GEMINI_MODEL=gemini-2.0-flash
LLM_MAX_TOKENS=4096

# Datos (opcionales, mejoran el análisis)
FRED_API_KEY=               # Gratuito en fred.stlouisfed.org
ALPHA_VANTAGE_API_KEY=      # Gratuito en alphavantage.co

# Bot de Telegram (requerido para el bot)
TELEGRAM_BOT_TOKEN=

# Timeframe (opcional, por defecto 1d)
DATA_INTERVAL=1d
PREDICTION_HORIZON_BARS=0   # 0 = auto por intervalo
FEATURE_WINDOW_BARS_OVERRIDE=0

# ML (opcional)
N_CLUSTERS=4
WALK_FORWARD_SPLITS=5
RETURN_THRESHOLD_PCT=0.0    # 0 = auto por intervalo
MAX_WEIGHT_PER_ASSET=0.35
MAX_RISK_RETRIES=3
```
