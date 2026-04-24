"""Agent 2 — Feature Engineer.

Computes technical indicators and risk/return features per ticker,
then merges optional macro features.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from asesorfinan.config import settings
from asesorfinan.models import GraphState

logger = logging.getLogger(__name__)

class FeatureEngineerAgent:
    def run(self, state: GraphState) -> GraphState:
        prices: pd.DataFrame = state.prices_df
        macro: pd.DataFrame = state.macro_df
        fundamentals: pd.DataFrame = state.fundamentals_df

        logger.info("Engineering features for %d assets", prices.shape[1])

        ticker_features: list[dict] = []
        for ticker in prices.columns:
            series = prices[ticker].dropna()
            features = self._compute_ticker_features(ticker, series)
            ticker_features.append(features)

        features_df = pd.DataFrame(ticker_features).set_index("ticker")

        # Merge fundamentals (beta, P/E, IV, etc.) — available per ticker
        if fundamentals is not None and not fundamentals.empty:
            shared_tickers = features_df.index.intersection(fundamentals.index)
            features_df = features_df.join(fundamentals.loc[shared_tickers], how="left")
            logger.info("Merged %d fundamental features", fundamentals.shape[1])

        # Merge macro snapshot — same value for all tickers (market-level signal)
        if macro is not None and not macro.empty:
            macro_snapshot = self._macro_snapshot(macro)
            for col, val in macro_snapshot.items():
                features_df[f"macro_{col}"] = val

        # Drop rows where ALL key clustering features are NaN (not all columns need to be present)
        key_cols = [c for c in ["ret_30d", "annual_vol", "sharpe"] if c in features_df.columns]
        features_df = features_df.dropna(subset=key_cols)
        logger.info("Feature matrix: %d tickers × %d features", *features_df.shape)
        state.features_df = features_df
        return state

    # ------------------------------------------------------------------

    def _compute_ticker_features(self, ticker: str, s: pd.Series) -> dict:
        log_ret = np.log(s / s.shift(1)).dropna()

        # Scale calendar-day lookbacks to bars for the active interval
        bpd = settings.bars_per_day
        d30 = max(2, round(30 * bpd))
        d90 = max(2, round(90 * bpd))
        d252 = max(2, round(252 * bpd))

        ret_30d = float((s.iloc[-1] / s.iloc[-d30 - 1] - 1) * 100) if len(s) > d30 else 0.0
        ret_90d = float((s.iloc[-1] / s.iloc[-d90 - 1] - 1) * 100) if len(s) > d90 else 0.0
        ret_1y = float((s.iloc[-1] / s.iloc[-d252 - 1] - 1) * 100) if len(s) > d252 else 0.0

        ann = settings.annualization_factor
        annual_vol = float(log_ret.std() * np.sqrt(ann) * 100)
        sharpe = float((log_ret.mean() * ann) / (log_ret.std() * np.sqrt(ann) + 1e-9))

        # Max drawdown
        cumret = (1 + log_ret).cumprod()
        rolling_max = cumret.cummax()
        drawdown = (cumret - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min() * 100)

        row: dict = {
            "ticker": ticker,
            "ret_30d": ret_30d,
            "ret_90d": ret_90d,
            "ret_1y": ret_1y,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
        }

        # Technical indicators via pandas_ta (optional, graceful fallback)
        try:
            import pandas_ta as ta  # type: ignore
            df = pd.DataFrame({"close": s})
            df.ta.rsi(length=14, append=True)
            df.ta.macd(append=True)
            df.ta.bbands(length=20, append=True)
            df.ta.sma(length=50, append=True)
            df.ta.ema(length=20, append=True)

            last = df.iloc[-1]
            row["rsi_14"] = float(last.get("RSI_14", np.nan))
            row["macd"] = float(last.get("MACD_12_26_9", np.nan))
            row["macd_signal"] = float(last.get("MACDs_12_26_9", np.nan))
            row["bb_pct"] = float(last.get("BBP_5_2.0", np.nan))  # percent-B
            row["sma_50"] = float(last.get("SMA_50", np.nan))
            row["ema_20"] = float(last.get("EMA_20", np.nan))
            # Price relative to SMA50 (momentum)
            row["price_vs_sma50"] = float((s.iloc[-1] / row["sma_50"] - 1) * 100) if row["sma_50"] else 0.0
        except Exception as exc:
            logger.warning("pandas_ta unavailable for %s: %s", ticker, exc)

        return row

    def _macro_snapshot(self, macro: pd.DataFrame) -> dict[str, float]:
        """Return the most recent non-null value for each macro series."""
        snapshot = {}
        for col in macro.columns:
            series = macro[col].dropna()
            if not series.empty:
                snapshot[col] = float(series.iloc[-1])
        return snapshot
