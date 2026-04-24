"""Agent 1 — Data Collector.

Sources (all free):
  - Yahoo Finance (yfinance): daily OHLCV prices + per-ticker fundamentals + options IV + news
  - Alpha Vantage: NEWS_SENTIMENT endpoint with pre-computed ticker-level scores
  - FRED API: macro series including yield curve spreads, financial stress, credit spreads
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from asesorfinan.config import settings
from asesorfinan.models import GraphState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FRED series
# ---------------------------------------------------------------------------

FRED_SERIES = {
    # Rates
    "fed_rate":    "FEDFUNDS",          # Effective Fed Funds Rate
    "2y_yield":    "DGS2",              # 2-Year Treasury
    "10y_yield":   "DGS10",             # 10-Year Treasury
    # Yield curve spread — single best recession leading indicator
    "curve_10y2y": "T10Y2Y",            # 10Y minus 2Y (inverts before recessions)
    "curve_10y3m": "T10Y3M",            # 10Y minus 3M (alternative slope)
    # Inflation & growth
    "cpi":         "CPIAUCSL",          # CPI All Urban Consumers
    "gdp_growth":  "A191RL1Q225SBEA",   # Real GDP growth (quarterly)
    "unemployment":"UNRATE",
    # Financial stress — composite signal for regime detection
    "nfci":        "NFCI",              # Chicago Fed National Financial Conditions Index
    "nfci_credit": "NFCICREDIT",        # NFCI Credit Subindex
    # Credit risk premia
    "hy_spread":   "BAMLH0A0HYM2",      # ICE BofA High Yield OAS
    "ig_spread":   "BAMLC0A0CM",        # ICE BofA Investment Grade OAS
    # Volatility
    "vix":         "VIXCLS",            # CBOE VIX
    # Liquidity
    "m2":          "M2SL",              # M2 Money Supply
}

# ---------------------------------------------------------------------------
# News sentiment — keyword lists for yfinance fallback scoring
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = {
    "bull", "bullish", "gain", "gains", "growth", "profit", "profits", "revenue",
    "beat", "beats", "exceed", "upgrade", "outperform", "buy", "surge", "surges",
    "rally", "rallies", "rise", "rises", "high", "record", "strong", "strength",
    "positive", "boost", "boosted", "recover", "recovery", "upside",
}
_NEGATIVE_WORDS = {
    "bear", "bearish", "loss", "losses", "decline", "declines", "miss", "misses",
    "disappoint", "disappoint", "downgrade", "underperform", "sell", "crash", "crashes",
    "fall", "falls", "drop", "drops", "low", "weak", "weakness", "negative",
    "cut", "cuts", "recession", "risk", "fear", "warning", "concern", "lawsuit",
}

_AV_NEWS_URL = "https://www.alphavantage.co/query"

# ---------------------------------------------------------------------------
# Fundamental fields to extract from yf.Ticker.info
_FUNDAMENTAL_FIELDS = [
    "beta",
    "trailingPE",
    "forwardPE",
    "dividendYield",
    "priceToBook",
    "returnOnEquity",
    "debtToEquity",
    "marketCap",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "fiftyDayAverage",
    "twoHundredDayAverage",
]


class DataCollectorAgent:
    def __init__(self) -> None:
        self._fred = None

    def _get_fred(self):
        if self._fred is None:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=settings.fred_api_key)
            except Exception as exc:
                logger.warning("FRED client unavailable: %s — macro data will be empty.", exc)
        return self._fred

    def run(self, state: GraphState) -> GraphState:
        profile = state.user_profile
        tickers = [t for t in settings.default_assets if t not in profile.excluded_assets]
        end = date.today()
        # yfinance enforces hard upper limits (e.g. 730 days for 1h) as an exclusive
        # boundary, so requesting exactly max_lookback_days from today will be rejected.
        # Subtract a 3-day buffer so the start always falls safely inside the window.
        lookback = settings.max_lookback_days - 3 if settings.is_intraday else settings.max_lookback_days
        start = end - timedelta(days=max(1, lookback))

        logger.info(
            "Downloading price data: %d tickers, interval=%s, %d-day lookback",
            len(tickers), settings.data_interval, lookback,
        )
        prices = self._fetch_prices(tickers, str(start), str(end))

        if settings.is_intraday:
            # FRED macro series and per-ticker fundamentals are not meaningful at intraday resolution
            logger.info("Intraday interval (%s): skipping FRED macro + fundamentals", settings.data_interval)
            macro = self._enrich_macro_with_price_signals(pd.DataFrame(), prices)
            fundamentals = pd.DataFrame(index=prices.columns)
        else:
            logger.info("Downloading macro data from FRED")
            macro = self._fetch_macro(str(start), str(end))
            macro = self._enrich_macro_with_price_signals(macro, prices)

            logger.info("Fetching per-ticker fundamentals and options IV from yfinance")
            fundamentals = self._fetch_fundamentals(list(prices.columns))

            logger.info("Fetching news sentiment")
            news_features = self._fetch_news_sentiment(list(prices.columns))
            if news_features:
                news_df = pd.DataFrame.from_dict(news_features, orient="index")
                news_df.index.name = "ticker"
                fundamentals = fundamentals.join(news_df, how="left")
                logger.info(
                    "News sentiment merged: %d tickers, %d features",
                    news_df.shape[0], news_df.shape[1],
                )

        state.prices_df = prices
        state.macro_df = macro
        state.fundamentals_df = fundamentals
        return state

    # ------------------------------------------------------------------
    # Price data
    # ------------------------------------------------------------------

    def _fetch_prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        raw = yf.download(
            tickers, start=start, end=end,
            interval=settings.data_interval,
            auto_adjust=True, progress=False,
        )
        if raw.empty:
            raise RuntimeError("yfinance returned empty data — check your internet connection.")

        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if not isinstance(raw.columns, pd.MultiIndex):
            close.columns = tickers

        threshold = int(0.80 * len(close))
        close = close.dropna(axis=1, thresh=threshold).ffill().dropna()
        logger.info("Price data: %s", close.shape)
        return close

    # ------------------------------------------------------------------
    # FRED macro data
    # ------------------------------------------------------------------

    def _fetch_macro(self, start: str, end: str) -> pd.DataFrame:
        fred = self._get_fred()
        if fred is None:
            return pd.DataFrame()

        frames: dict[str, pd.Series] = {}
        for col, series_id in FRED_SERIES.items():
            try:
                s = fred.get_series(series_id, observation_start=start, observation_end=end)
                frames[col] = s
            except Exception as exc:
                logger.warning("FRED %s (%s): %s", col, series_id, exc)

        if not frames:
            return pd.DataFrame()

        date_range = pd.date_range(start, end, freq="D")
        macro = (
            pd.DataFrame(frames)
            .resample("D").ffill()
            .reindex(date_range)
            .ffill()
            .dropna(how="all")
        )

        # Derived features
        if "10y_yield" in macro and "2y_yield" in macro:
            macro["curve_2y10y_computed"] = macro["10y_yield"] - macro["2y_yield"]

        logger.info("Macro data: %s columns, %s rows", macro.shape[1], macro.shape[0])
        return macro

    def _enrich_macro_with_price_signals(
        self, macro: pd.DataFrame, prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute cross-asset signals from price data — available even without FRED."""
        log_ret = np.log(prices / prices.shift(1)).dropna()

        # Scale the 21-trading-day window to bars and cap at 25% of available data
        roll_bars = max(5, min(round(21 * settings.bars_per_day), len(prices) // 4))

        # Credit proxy: HYG vs TLT spread (if both available)
        if "HYG" in prices.columns and "TLT" in prices.columns:
            hyg_ret = log_ret["HYG"].rolling(roll_bars).mean()
            tlt_ret = log_ret["TLT"].rolling(roll_bars).mean()
            credit_proxy = (hyg_ret - tlt_ret).rename("credit_proxy_21d")
            macro = macro.join(credit_proxy, how="left")

        # Equity momentum breadth: fraction of assets with positive 1M return
        monthly_ret = prices.pct_change(roll_bars)
        breadth = (monthly_ret > 0).mean(axis=1).rename("breadth_1m")
        macro = macro.join(breadth, how="left")

        # Market volatility: equal-weight portfolio realized vol
        eq_weight_ret = log_ret.mean(axis=1)
        mkt_vol = eq_weight_ret.rolling(roll_bars).std().rename("mkt_vol_21d")
        macro = macro.join(mkt_vol, how="left")

        return macro.ffill()

    # ------------------------------------------------------------------
    # Fundamentals + options IV from yfinance
    # ------------------------------------------------------------------

    def _fetch_fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        rows = []
        for ticker in tickers:
            row = self._ticker_snapshot(ticker)
            row["ticker"] = ticker
            rows.append(row)

        df = pd.DataFrame(rows).set_index("ticker")
        logger.info("Fundamentals: %d tickers, %d features", df.shape[0], df.shape[1])
        return df

    def _ticker_snapshot(self, ticker: str) -> dict:
        row: dict = {}
        try:
            t = yf.Ticker(ticker)
            info = t.info

            for field in _FUNDAMENTAL_FIELDS:
                val = info.get(field)
                row[field] = float(val) if val is not None else np.nan

            # Price position features derived from fundamentals
            if not np.isnan(row.get("fiftyTwoWeekHigh", np.nan)) and not np.isnan(row.get("fiftyTwoWeekLow", np.nan)):
                hi = row["fiftyTwoWeekHigh"]
                lo = row["fiftyTwoWeekLow"]
                rng = hi - lo
                # Current price vs 52-week range (0 = at low, 1 = at high)
                price_now = info.get("regularMarketPrice") or info.get("previousClose") or hi
                row["pct_52w_range"] = float((price_now - lo) / rng) if rng > 0 else 0.5

        except Exception as exc:
            logger.debug("Fundamentals for %s failed: %s", ticker, exc)

        # Options IV: implied volatility at nearest ATM strike
        row.update(self._fetch_options_iv(ticker))
        return row

    def _fetch_options_iv(self, ticker: str) -> dict:
        """Fetch at-the-money IV and put/call OI ratio from the nearest expiry."""
        result = {"atm_iv": np.nan, "pc_ratio": np.nan}
        try:
            t = yf.Ticker(ticker)
            expirations = t.options
            if not expirations:
                return result

            chain = t.option_chain(expirations[0])
            calls, puts = chain.calls, chain.puts

            if calls.empty or puts.empty:
                return result

            # Identify ATM strike (closest to current price)
            current_price = t.info.get("regularMarketPrice") or t.info.get("previousClose")
            if not current_price:
                return result

            calls = calls.dropna(subset=["impliedVolatility", "strike"])
            atm_call = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:1]]
            if not atm_call.empty:
                result["atm_iv"] = float(atm_call["impliedVolatility"].iloc[0])

            # Put/call open-interest ratio (sentiment indicator)
            total_call_oi = calls["openInterest"].sum()
            total_put_oi = puts["openInterest"].sum() if not puts.empty else 0
            if total_call_oi > 0:
                result["pc_ratio"] = float(total_put_oi / total_call_oi)

        except Exception as exc:
            logger.debug("Options IV for %s failed: %s", ticker, exc)

        return result

    # ------------------------------------------------------------------
    # News sentiment
    # ------------------------------------------------------------------

    def _fetch_news_sentiment(self, tickers: list[str]) -> dict[str, dict]:
        """Returns {ticker: {news_sentiment_score, news_bullish_ratio,
                              news_volume_7d, news_sentiment_trend}}
        Tries Alpha Vantage first (pre-computed scores, single batched call),
        falls back to yfinance news + keyword scoring.
        """
        if settings.alpha_vantage_api_key:
            try:
                result = self._av_news_sentiment(tickers)
                if result:
                    logger.info("News sentiment via Alpha Vantage (%d tickers)", len(result))
                    return result
            except Exception as exc:
                logger.warning("Alpha Vantage news failed: %s — falling back to yfinance", exc)

        logger.info("News sentiment via yfinance keyword scoring")
        return self._yf_news_sentiment(tickers)

    def _av_news_sentiment(self, tickers: list[str]) -> dict[str, dict]:
        """Batch call to Alpha Vantage NEWS_SENTIMENT for all tickers at once."""
        now = datetime.utcnow()
        time_from_14d = (now - timedelta(days=14)).strftime("%Y%m%dT%H%M")

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ",".join(tickers),
            "time_from": time_from_14d,
            "limit": "200",
            "sort": "LATEST",
            "apikey": settings.alpha_vantage_api_key,
        }
        resp = requests.get(_AV_NEWS_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if "feed" not in data:
            logger.debug("Alpha Vantage news: unexpected response — %s", list(data.keys()))
            return {}

        cutoff_7d = now - timedelta(days=7)
        cutoff_14d = now - timedelta(days=14)

        # Accumulate per-ticker sentiment scores split by recency window
        scores: dict[str, list[float]] = {t: [] for t in tickers}
        scores_prev: dict[str, list[float]] = {t: [] for t in tickers}

        for article in data.get("feed", []):
            pub_str = article.get("time_published", "")
            try:
                pub_dt = datetime.strptime(pub_str, "%Y%m%dT%H%M%S")
            except ValueError:
                continue

            for ts in article.get("ticker_sentiment", []):
                tkr = ts.get("ticker", "")
                if tkr not in scores:
                    continue
                score = float(ts.get("ticker_sentiment_score", 0))
                if pub_dt >= cutoff_7d:
                    scores[tkr].append(score)
                elif pub_dt >= cutoff_14d:
                    scores_prev[tkr].append(score)

        result: dict[str, dict] = {}
        for tkr in tickers:
            recent = scores[tkr]
            prev = scores_prev[tkr]
            avg_score = float(np.mean(recent)) if recent else 0.0
            prev_score = float(np.mean(prev)) if prev else avg_score
            bullish_ratio = float(sum(1 for s in recent if s > 0.15) / len(recent)) if recent else 0.5

            result[tkr] = {
                "news_sentiment_score": avg_score,
                "news_bullish_ratio": bullish_ratio,
                "news_volume_7d": len(recent),
                "news_sentiment_trend": avg_score - prev_score,
            }
        return result

    def _yf_news_sentiment(self, tickers: list[str]) -> dict[str, dict]:
        """Per-ticker news from yfinance + simple keyword scoring."""
        result: dict[str, dict] = {}
        for tkr in tickers:
            try:
                articles = yf.Ticker(tkr).news or []
                result[tkr] = self._score_articles(articles)
            except Exception as exc:
                logger.debug("yfinance news for %s failed: %s", tkr, exc)
                result[tkr] = {
                    "news_sentiment_score": 0.0,
                    "news_bullish_ratio": 0.5,
                    "news_volume_7d": 0,
                    "news_sentiment_trend": 0.0,
                }
        return result

    def _score_articles(self, articles: list[dict]) -> dict:
        """Keyword-based sentiment from a list of yfinance news articles."""
        now = datetime.utcnow()
        cutoff_7d = now - timedelta(days=7)
        cutoff_14d = now - timedelta(days=14)

        scores_recent: list[float] = []
        scores_prev: list[float] = []

        for art in articles:
            # yfinance 1.x nests content under 'content' key; older versions use flat dict
            content = art.get("content", art)
            title = content.get("title", "") or art.get("title", "")
            summary = content.get("summary", "") or art.get("summary", "")
            text = f"{title} {summary}".lower()
            words = set(re.findall(r"\b\w+\b", text))

            pos = len(words & _POSITIVE_WORDS)
            neg = len(words & _NEGATIVE_WORDS)
            score = (pos - neg) / (pos + neg + 1)

            # Resolve publish time
            pub_ts = (
                content.get("pubDate")
                or art.get("providerPublishTime")
                or art.get("displayTime")
            )
            try:
                if isinstance(pub_ts, (int, float)):
                    pub_dt = datetime.utcfromtimestamp(pub_ts)
                elif isinstance(pub_ts, str):
                    pub_dt = datetime.fromisoformat(pub_ts.replace("Z", "+00:00")).replace(tzinfo=None)
                else:
                    pub_dt = now  # fallback: treat as recent
            except Exception:
                pub_dt = now

            if pub_dt >= cutoff_7d:
                scores_recent.append(score)
            elif pub_dt >= cutoff_14d:
                scores_prev.append(score)

        avg_recent = float(np.mean(scores_recent)) if scores_recent else 0.0
        avg_prev = float(np.mean(scores_prev)) if scores_prev else avg_recent
        bullish_ratio = float(sum(1 for s in scores_recent if s > 0) / len(scores_recent)) if scores_recent else 0.5

        return {
            "news_sentiment_score": avg_recent,
            "news_bullish_ratio": bullish_ratio,
            "news_volume_7d": len(scores_recent),
            "news_sentiment_trend": avg_recent - avg_prev,
        }
