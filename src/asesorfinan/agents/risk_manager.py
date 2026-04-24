"""Agent 6 — Risk Manager.

Validates the proposed portfolio using:
  - Monte Carlo simulation → VaR and CVaR at 95% confidence
  - Historical stress testing (2008 crisis, 2020 COVID crash)
  - Semáforo de riesgo: green / yellow / red
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from asesorfinan.config import settings
from asesorfinan.models import GraphState, Portfolio, RiskReport, RiskSignal

logger = logging.getLogger(__name__)

N_SIMULATIONS = 10_000

# Historical stress periods (start, end)
STRESS_PERIODS = {
    "crisis_2008": ("2008-09-01", "2009-03-31"),
    "covid_2020": ("2020-02-19", "2020-03-23"),
    "dot_com_2000": ("2000-03-01", "2002-10-31"),
    "rate_hike_2022": ("2022-01-01", "2022-10-15"),
}

# Parametric fallback: peak-to-trough drawdown of US equity market (SPY proxy)
# Used when the period is not in the available price history.
# Source: standard finance literature.
PARAMETRIC_MARKET_SHOCKS = {
    "crisis_2008": -0.55,
    "covid_2020": -0.34,
    "dot_com_2000": -0.49,
    "rate_hike_2022": -0.25,
}


class RiskManagerAgent:
    def run(self, state: GraphState) -> GraphState:
        portfolio: Portfolio = state.portfolio
        prices: pd.DataFrame = state.prices_df
        profile = state.user_profile

        weights = {alloc.ticker: alloc.weight for alloc in portfolio.allocations}
        tickers = list(weights.keys())
        prices_sub = prices[[t for t in tickers if t in prices.columns]]

        log_returns = np.log(prices_sub / prices_sub.shift(1)).dropna()
        w = np.array([weights.get(t, 0) for t in prices_sub.columns])
        w /= w.sum()

        portfolio_daily_returns = log_returns.values @ w

        var_95, cvar_95 = self._monte_carlo_var(
            portfolio.expected_annual_return,
            portfolio.annual_volatility,
            profile.capital,
            horizon_bars=settings.effective_prediction_horizon,
        )

        max_stress_drawdown = self._stress_test(prices_sub, w)

        var_pct = var_95 / profile.capital
        signal, approved = self._evaluate(var_pct, max_stress_drawdown, profile)

        notes = self._build_notes(var_95, cvar_95, var_pct, max_stress_drawdown, signal, profile.capital)
        logger.info("Risk: VaR=%.1f%% | stress=%.1f%% | signal=%s | approved=%s",
                    var_pct * 100, max_stress_drawdown * 100, signal, approved)

        state.risk_report = RiskReport(
            var_95=round(var_95, 2),
            cvar_95=round(cvar_95, 2),
            var_pct=round(var_pct, 4),
            max_drawdown_stress=round(max_stress_drawdown, 4),
            signal=signal,
            approved=approved,
            notes=notes,
        )
        return state

    # ------------------------------------------------------------------

    def _monte_carlo_var(
        self,
        mu_annual: float,
        sigma_annual: float,
        capital: float,
        horizon_bars: int,
    ) -> tuple[float, float]:
        # Work in bar-space so the simulation is correct for any interval.
        # mu_annual / annualization_factor converts back to per-bar drift.
        ann = settings.annualization_factor
        mu_bar = mu_annual / ann
        sigma_bar = sigma_annual / np.sqrt(ann)

        rng = np.random.default_rng(42)
        bar_returns = rng.normal(mu_bar, sigma_bar, (N_SIMULATIONS, horizon_bars))
        total_returns = np.exp(bar_returns.sum(axis=1)) - 1
        pnl = total_returns * capital

        var_95 = float(-np.percentile(pnl, (1 - settings.var_confidence) * 100))
        cvar_95 = float(-pnl[pnl < -var_95].mean()) if (pnl < -var_95).any() else var_95

        return var_95, cvar_95

    def _stress_test(self, prices: pd.DataFrame, weights: np.ndarray) -> float:
        """Return the worst portfolio drawdown across all stress periods.

        For periods within the available price history, uses actual data.
        For periods outside the history (common with 3-year lookback), falls
        back to a beta-scaled parametric shock so the result is never silently
        zero — which would give a false green signal.
        """
        # Compute per-asset beta vs. the first column (proxy benchmark)
        betas = self._compute_betas(prices)
        worst_drawdown = 0.0

        for period_name, (start, end) in STRESS_PERIODS.items():
            drawdown = self._historical_drawdown(prices, weights, start, end)
            if drawdown is not None:
                source = "historical"
            else:
                drawdown = self._parametric_drawdown(betas, weights, PARAMETRIC_MARKET_SHOCKS[period_name])
                source = "parametric"

            logger.debug(
                "Stress %s [%s]: drawdown=%.1f%%", period_name, source, drawdown * 100
            )
            if drawdown > worst_drawdown:
                worst_drawdown = drawdown

        return worst_drawdown

    def _historical_drawdown(
        self, prices: pd.DataFrame, weights: np.ndarray, start: str, end: str
    ) -> float | None:
        try:
            period = prices.loc[start:end]
            if len(period) < 5:
                return None
            normalised = period / period.iloc[0]
            portfolio_val = (normalised.values * weights).sum(axis=1)
            series = pd.Series(portfolio_val)
            peak = series.cummax()
            return float(abs((series - peak) / peak).max())
        except Exception:
            return None

    def _compute_betas(self, prices: pd.DataFrame) -> np.ndarray:
        """Beta of each asset vs. the equal-weight portfolio (market proxy)."""
        log_ret = np.log(prices / prices.shift(1)).dropna()
        market = log_ret.mean(axis=1)
        market_var = float(market.var()) + 1e-12
        betas = np.array([
            float(log_ret[col].cov(market) / market_var)
            for col in log_ret.columns
        ])
        return betas

    def _parametric_drawdown(
        self, betas: np.ndarray, weights: np.ndarray, market_shock: float
    ) -> float:
        """Estimate portfolio drawdown as weighted sum of beta-scaled market shock."""
        asset_shocks = betas * market_shock          # negative values = losses
        portfolio_shock = float((weights * asset_shocks).sum())
        return abs(min(portfolio_shock, 0.0))        # return as positive magnitude

    def _evaluate(
        self,
        var_pct: float,
        stress_drawdown: float,
        profile,
    ) -> tuple[RiskSignal, bool]:
        from asesorfinan.models import RiskProfile

        thresholds = {
            RiskProfile.conservative: (0.05, 0.15),
            RiskProfile.moderate:     (0.08, 0.25),
            RiskProfile.aggressive:   (0.12, 0.40),
        }
        green_thr, yellow_thr = thresholds.get(profile.risk_profile, (0.08, 0.25))

        if var_pct <= green_thr and stress_drawdown <= yellow_thr:
            return RiskSignal.green, True
        if var_pct <= yellow_thr:
            return RiskSignal.yellow, True
        return RiskSignal.red, False

    def _build_notes(
        self,
        var_95: float,
        cvar_95: float,
        var_pct: float,
        stress_dd: float,
        signal: RiskSignal,
        capital: float,
    ) -> str:
        horizon = settings.effective_prediction_horizon
        lines = [
            f"VaR 95% (Monte Carlo {horizon} barras/{settings.data_interval}): ${var_95:,.0f} ({var_pct*100:.1f}% del capital)",
            f"CVaR 95%: ${cvar_95:,.0f}",
            f"Peor drawdown en escenarios históricos de estrés: {stress_dd*100:.1f}%",
            f"Semáforo: {signal.value.upper()}",
        ]
        if signal == RiskSignal.red:
            lines.append("RECHAZO: El portafolio supera los umbrales de riesgo para el perfil del usuario.")
        elif signal == RiskSignal.yellow:
            lines.append("ATENCIÓN: El portafolio está dentro de límites pero cerca del umbral máximo.")
        else:
            lines.append("APROBADO: El portafolio cumple los criterios de riesgo.")
        return "\n".join(lines)
