"""Shared Pydantic models and LangGraph state for the multi-agent pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Domain enums
# ---------------------------------------------------------------------------

class RiskProfile(str, Enum):
    conservative = "conservador"
    moderate = "moderado"
    aggressive = "agresivo"


class ReturnLabel(str, Enum):
    up = "sube"
    neutral = "neutro"
    down = "baja"


class RiskSignal(str, Enum):
    green = "verde"
    yellow = "amarillo"
    red = "rojo"


# ---------------------------------------------------------------------------
# Input: user profile
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    capital: float = Field(..., gt=0, description="Capital disponible en USD")
    horizon_months: float = Field(..., gt=0, le=120, description="Horizonte de inversión en meses (puede ser fracción para días)")
    risk_profile: RiskProfile
    excluded_assets: list[str] = Field(default_factory=list)
    max_positions: int = Field(default=10, ge=2, le=20)
    custom_assets: list[str] = Field(default_factory=list, description="Tickers personalizados; si está vacío usa default_assets de config")


# ---------------------------------------------------------------------------
# Agent outputs — kept as plain dicts/dataframes wrapped in TypedDict
# to stay compatible with LangGraph's state reducer.
# ---------------------------------------------------------------------------

class AssetPrediction(BaseModel):
    ticker: str
    cluster_id: int
    cluster_label: str
    predicted_label: ReturnLabel
    predicted_return_pct: float
    confidence: float


class PortfolioAllocation(BaseModel):
    ticker: str
    weight: float
    amount_usd: float
    cluster_label: str
    predicted_return_pct: float


class Portfolio(BaseModel):
    allocations: list[PortfolioAllocation]
    expected_annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    total_invested: float


class RiskReport(BaseModel):
    var_95: float           # absolute USD loss at 95% confidence
    cvar_95: float          # expected loss beyond VaR
    var_pct: float          # VaR as fraction of capital
    max_drawdown_stress: float  # worst drawdown in stress scenarios
    signal: RiskSignal
    approved: bool
    notes: str


class AdvisorReport(BaseModel):
    narrative: str
    portfolio: Portfolio
    risk: RiskReport


# ---------------------------------------------------------------------------
# LangGraph pipeline state
# ---------------------------------------------------------------------------

class GraphState(BaseModel):
    """Mutable state passed between all agents in the LangGraph."""

    # Input
    user_profile: UserProfile | None = None

    # Agent 1
    prices_df: Any = None           # pd.DataFrame — tickers as columns, dates as index
    macro_df: Any = None            # pd.DataFrame — macro series
    fundamentals_df: Any = None     # pd.DataFrame — per-ticker fundamentals snapshot

    # Agent 2
    features_df: Any = None         # pd.DataFrame — per-ticker feature matrix

    # Agent 3
    cluster_labels: dict[str, int] = Field(default_factory=dict)
    cluster_names: dict[int, str] = Field(default_factory=dict)

    # Agent 4
    predictions: list[AssetPrediction] = Field(default_factory=list)

    # Agent 5
    portfolio: Portfolio | None = None

    # Agent 6
    risk_report: RiskReport | None = None
    risk_retry_count: int = 0

    # Agent 7
    advisor_report: AdvisorReport | None = None

    # Control
    error: str | None = None

    class Config:
        arbitrary_types_allowed = True
