"""Agent 5 — Portfolio Optimizer.

Builds the optimal portfolio using:
  - Markowitz Mean-Variance as baseline
  - Black-Litterman incorporating ML return predictions as "views"

Library: PyPortfolioOpt
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from asesorfinan.config import settings
from asesorfinan.models import (
    AssetPrediction,
    GraphState,
    Portfolio,
    PortfolioAllocation,
    ReturnLabel,
    RiskProfile,
)

logger = logging.getLogger(__name__)

class PortfolioOptimizerAgent:
    def run(self, state: GraphState) -> GraphState:
        prices: pd.DataFrame = state.prices_df
        predictions: list[AssetPrediction] = state.predictions
        profile = state.user_profile

        # Filter to assets for which we have predictions
        pred_map = {p.ticker: p for p in predictions}
        tickers = [t for t in prices.columns if t in pred_map]

        if not tickers:
            raise RuntimeError("No tickers with valid predictions — cannot build portfolio")

        prices = prices[tickers]

        logger.info("Building portfolio for %d assets | profile: %s", len(tickers), profile.risk_profile)

        mu, S = self._expected_returns_and_cov(prices, pred_map, profile)
        weights = self._optimize(mu, S, profile, pred_map)

        # Limit to top max_positions
        weights = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True)[: profile.max_positions])
        total_w = sum(weights.values())
        weights = {k: v / total_w for k, v in weights.items()}

        portfolio = self._build_portfolio(weights, mu, S, pred_map, profile)
        logger.info(
            "Portfolio built: E[ret]=%.1f%% σ=%.1f%% Sharpe=%.2f",
            portfolio.expected_annual_return * 100,
            portfolio.annual_volatility * 100,
            portfolio.sharpe_ratio,
        )
        state.portfolio = portfolio
        return state

    # ------------------------------------------------------------------

    def _expected_returns_and_cov(
        self,
        prices: pd.DataFrame,
        pred_map: dict[str, AssetPrediction],
        profile,
    ) -> tuple[pd.Series, pd.DataFrame]:
        try:
            from pypfopt import expected_returns, risk_models, black_litterman

            ann = round(settings.annualization_factor)
            # Historical mean returns (CAPM or mean historical)
            mu_hist = expected_returns.mean_historical_return(prices, frequency=ann)
            S = risk_models.CovarianceShrinkage(prices, frequency=ann).ledoit_wolf()

            # Black-Litterman views from ML predictions
            viewdict = self._build_bl_views(pred_map, mu_hist)
            if viewdict:
                bl = black_litterman.BlackLittermanModel(S, absolute_views=viewdict, pi=mu_hist)
                mu = bl.bl_returns()
            else:
                mu = mu_hist

            return mu, S

        except ImportError:
            logger.warning("PyPortfolioOpt not installed — using simple historical returns")
            ann = settings.annualization_factor
            log_ret = np.log(prices / prices.shift(1)).dropna()
            mu = log_ret.mean() * ann
            S = log_ret.cov() * ann
            return mu, S

    def _build_bl_views(
        self,
        pred_map: dict[str, AssetPrediction],
        mu_hist: pd.Series,
    ) -> dict[str, float]:
        """Convert ML predictions to absolute return views for Black-Litterman."""
        views: dict[str, float] = {}
        for ticker, pred in pred_map.items():
            if ticker not in mu_hist.index:
                continue
            base = float(mu_hist[ticker])
            confidence = pred.confidence

            if pred.predicted_label == ReturnLabel.up:
                views[ticker] = base + confidence * 0.15
            elif pred.predicted_label == ReturnLabel.down:
                views[ticker] = base - confidence * 0.15
            # neutral → no view (let prior dominate)

        return views

    def _optimize(
        self,
        mu: pd.Series,
        S: pd.DataFrame,
        profile,
        pred_map: dict | None = None,
    ) -> dict[str, float]:
        try:
            from pypfopt import EfficientFrontier
            from pypfopt.objective_functions import L2_reg

            # Assets with a confident negative prediction get a tighter weight cap
            # so the optimizer can't hide losses behind diversification math.
            per_asset_bounds = []
            for ticker in mu.index:
                pred = pred_map.get(ticker) if pred_map else None
                if pred and pred.predicted_label == ReturnLabel.down and pred.confidence > 0.45:
                    upper = min(0.10, settings.max_weight_per_asset)
                else:
                    upper = settings.max_weight_per_asset
                per_asset_bounds.append((0.0, upper))

            ef = EfficientFrontier(mu, S, weight_bounds=per_asset_bounds)
            ef.add_objective(L2_reg, gamma=0.1)

            if profile.risk_profile == RiskProfile.conservative:
                ef.min_volatility()
            elif profile.risk_profile == RiskProfile.aggressive:
                ef.max_quadratic_utility(risk_aversion=0.5)
            else:
                ef.max_sharpe(risk_free_rate=0.05)

            cleaned = ef.clean_weights()
            return {k: v for k, v in cleaned.items() if v > 0.001}

        except ImportError:
            # Naive equal-weight fallback
            logger.warning("PyPortfolioOpt unavailable — using equal-weight portfolio")
            tickers = list(mu.index)
            w = 1.0 / len(tickers)
            return {t: w for t in tickers}

    def _build_portfolio(
        self,
        weights: dict[str, float],
        mu: pd.Series,
        S: pd.DataFrame,
        pred_map: dict[str, AssetPrediction],
        profile,
    ) -> Portfolio:
        w_arr = np.array([weights[t] for t in weights])
        mu_arr = np.array([float(mu.get(t, 0)) for t in weights])
        S_sub = S.loc[list(weights.keys()), list(weights.keys())].values

        expected_return = float(w_arr @ mu_arr)
        variance = float(w_arr @ S_sub @ w_arr)
        volatility = float(np.sqrt(variance))
        sharpe = float((expected_return - 0.05) / (volatility + 1e-9))

        allocations = []
        for ticker, weight in weights.items():
            pred = pred_map.get(ticker)
            allocations.append(
                PortfolioAllocation(
                    ticker=ticker,
                    weight=round(weight, 4),
                    amount_usd=round(profile.capital * weight, 2),
                    cluster_label=pred.cluster_label if pred else "desconocido",
                    predicted_return_pct=pred.predicted_return_pct if pred else 0.0,
                )
            )

        return Portfolio(
            allocations=sorted(allocations, key=lambda x: x.weight, reverse=True),
            expected_annual_return=expected_return,
            annual_volatility=volatility,
            sharpe_ratio=sharpe,
            total_invested=profile.capital,
        )
