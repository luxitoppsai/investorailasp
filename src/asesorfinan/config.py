from pydantic_settings import BaseSettings, SettingsConfigDict

# yfinance hard limits per interval (max calendar days you can request)
_INTERVAL_MAX_DAYS = {
    "1m":  7,
    "2m":  60,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "90m": 60,
    "1h":  730,
    "1d":  365 * 10,
    "5d":  365 * 10,
    "1wk": 365 * 10,
    "1mo": 365 * 10,
    "3mo": 365 * 10,
}

# Approximate trading bars in one full trading day per interval
_BARS_PER_DAY = {
    "1m":  390,    # 6.5h × 60
    "2m":  195,
    "5m":  78,
    "15m": 26,
    "30m": 13,
    "90m": 4,
    "1h":  7,      # 9:30–16:00 = 6.5 bars (use 7 to be safe)
    "1d":  1,
    "5d":  0.2,
    "1wk": 0.2,
    "1mo": 1/21,   # ~21 trading days per month
    "3mo": 1/63,   # ~63 trading days per quarter
}

# Default prediction horizon in bars per interval
_DEFAULT_HORIZON_BARS = {
    "1m":  5,    # 5 minutes
    "2m":  5,
    "5m":  6,    # 30 minutes
    "15m": 8,    # 2 hours
    "30m": 6,
    "90m": 4,
    "1h":  7,    # 1 full trading day
    "1d":  30,   # ~6 trading weeks
    "5d":  4,
    "1wk": 4,
    "1mo": 3,    # 3 months
    "3mo": 2,    # 2 quarters
}

# Default feature window in bars per interval (~1 trading week or meaningful cycle)
_DEFAULT_WINDOW_BARS = {
    "1m":  60,   # ~2.5 hours
    "2m":  60,
    "5m":  78,   # 1 trading day
    "15m": 52,   # 2 trading days
    "30m": 26,
    "90m": 20,
    "1h":  35,   # ~5 trading days
    "1d":  20,   # 4 trading weeks
    "5d":  20,
    "1wk": 20,
    "1mo": 24,   # 2 years of monthly bars
    "3mo": 12,   # 3 years of quarterly bars
}

# Return classification threshold per interval (% move to call UP/DOWN)
_RETURN_THRESHOLD = {
    "1m":  0.1,
    "2m":  0.15,
    "5m":  0.2,
    "15m": 0.3,
    "30m": 0.5,
    "90m": 0.7,
    "1h":  0.5,
    "1d":  3.0,
    "5d":  5.0,
    "1wk": 5.0,
    "1mo": 7.0,
    "3mo": 10.0,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- API Keys ---
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    fred_api_key: str = ""
    alpha_vantage_api_key: str = ""
    log_level: str = "INFO"

    # --- LLM Provider ---
    # Set explicitly ("anthropic" | "openai" | "gemini") or leave empty to
    # auto-detect from whichever key is present (anthropic → openai → gemini).
    llm_provider: str = ""

    # Model names per provider — override in .env if needed
    claude_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-4o"
    gemini_model: str = "gemini-2.0-flash"
    llm_max_tokens: int = 4096

    @property
    def active_llm_provider(self) -> str:
        if self.llm_provider:
            return self.llm_provider.lower()
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        if self.gemini_api_key:
            return "gemini"
        raise ValueError(
            "No LLM API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY in .env"
        )

    # --- Timeframe ---
    # yfinance interval: 1m 2m 5m 15m 30m 90m 1h 1d 5d 1wk 1mo 3mo
    data_interval: str = "1d"
    # Override prediction horizon in bars (0 = auto from interval)
    prediction_horizon_bars: int = 0
    # Override feature window in bars (0 = auto from interval)
    feature_window_bars_override: int = 0

    # --- Data collection ---
    default_assets: list[str] = [
        "SPY", "QQQ", "GLD", "TLT", "BND",
        "VNQ", "EFA", "IWM", "HYG", "SHY",
    ]

    # --- Clustering ---
    n_clusters: int = 4
    pca_variance_threshold: float = 0.95

    # --- ML prediction ---
    return_threshold_pct: float = 0.0   # 0 = auto from interval
    walk_forward_splits: int = 5

    # --- Risk thresholds ---
    var_confidence: float = 0.95
    max_var_pct: float = 0.10
    max_weight_per_asset: float = 0.35

    # --- Retry limit for orchestrator risk loop ---
    max_risk_retries: int = 3

    # ------------------------------------------------------------------
    # Derived / computed properties
    # ------------------------------------------------------------------

    @property
    def effective_prediction_horizon(self) -> int:
        if self.prediction_horizon_bars > 0:
            return self.prediction_horizon_bars
        return _DEFAULT_HORIZON_BARS.get(self.data_interval, 30)

    @property
    def effective_feature_window(self) -> int:
        if self.feature_window_bars_override > 0:
            return self.feature_window_bars_override
        return _DEFAULT_WINDOW_BARS.get(self.data_interval, 20)

    @property
    def effective_return_threshold(self) -> float:
        if self.return_threshold_pct > 0:
            return self.return_threshold_pct
        return _RETURN_THRESHOLD.get(self.data_interval, 3.0)

    @property
    def bars_per_day(self) -> float:
        return _BARS_PER_DAY.get(self.data_interval, 1.0)

    @property
    def annualization_factor(self) -> float:
        """Number of bars in a trading year — used for Sharpe / vol scaling."""
        return 252 * self.bars_per_day

    @property
    def max_lookback_days(self) -> int:
        return _INTERVAL_MAX_DAYS.get(self.data_interval, 365 * 10)

    @property
    def is_intraday(self) -> bool:
        return self.data_interval not in ("1d", "5d", "1wk", "1mo", "3mo")


settings = Settings()
