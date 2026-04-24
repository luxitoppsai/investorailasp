"""Agent 4 — Return Predictor (ML Supervisado).

Predicts the directional return of each asset over the next N bars
using XGBoost with walk-forward (time-series) cross-validation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from asesorfinan.config import settings
from asesorfinan.models import AssetPrediction, GraphState, ReturnLabel

logger = logging.getLogger(__name__)


def _label_return(r: float, threshold: float) -> str:
    # Return plain str values, not enum members — numpy 3.12 converts enum members
    # to 'ReturnLabel.up' etc., which get truncated and fail ReturnLabel() validation.
    if r > threshold:
        return ReturnLabel.up.value
    if r < -threshold:
        return ReturnLabel.down.value
    return ReturnLabel.neutral.value


class ReturnPredictorAgent:
    """Builds one XGBoost model per asset using walk-forward CV."""

    def run(self, state: GraphState) -> GraphState:
        prices: pd.DataFrame = state.prices_df
        features_df: pd.DataFrame = state.features_df
        cluster_labels = state.cluster_labels
        cluster_names = state.cluster_names

        predictions: list[AssetPrediction] = []

        for ticker in features_df.index:
            try:
                pred = self._predict_ticker(
                    ticker, prices[ticker], features_df.loc[ticker],
                    cluster_labels, cluster_names,
                )
                predictions.append(pred)
            except Exception as exc:
                logger.warning("Prediction failed for %s: %s — skipping", ticker, exc)

        logger.info("Generated predictions for %d assets", len(predictions))
        state.predictions = predictions
        return state

    # ------------------------------------------------------------------

    def _predict_ticker(
        self,
        ticker: str,
        price_series: pd.Series,
        ticker_features: pd.Series,
        cluster_labels: dict[str, int],
        cluster_names: dict[int, str],
    ) -> AssetPrediction:
        horizon = settings.effective_prediction_horizon
        window = settings.effective_feature_window
        threshold = settings.effective_return_threshold / 100.0

        X_train, y_train, X_predict = self._build_dataset(
            price_series, cluster_labels.get(ticker, 0), horizon, window, threshold,
        )

        if len(X_train) < 30:
            # Not enough history — return neutral prediction
            return AssetPrediction(
                ticker=ticker,
                cluster_id=cluster_labels.get(ticker, 0),
                cluster_label=cluster_names.get(cluster_labels.get(ticker, 0), "desconocido"),
                predicted_label=ReturnLabel.neutral,
                predicted_return_pct=0.0,
                confidence=0.33,
            )

        model, classes, accuracy = self._walk_forward_train(X_train, y_train)

        proba = model.predict_proba(X_predict.reshape(1, -1))[0]
        pred_class = str(classes[proba.argmax()])   # cast numpy.str_ → str for ReturnLabel()
        confidence = float(proba.max())

        # Forward-looking: direction × confidence × threshold
        # More meaningful than showing a historical lookback as "predicted return"
        thresh = settings.effective_return_threshold
        if pred_class == ReturnLabel.up.value:
            expected_return_pct = confidence * thresh
        elif pred_class == ReturnLabel.down.value:
            expected_return_pct = -confidence * thresh
        else:
            expected_return_pct = 0.0

        cid = cluster_labels.get(ticker, 0)
        return AssetPrediction(
            ticker=ticker,
            cluster_id=cid,
            cluster_label=cluster_names.get(cid, "desconocido"),
            predicted_label=ReturnLabel(pred_class),
            predicted_return_pct=expected_return_pct,
            confidence=confidence,
        )

    def _build_dataset(
        self,
        price_series: pd.Series,
        cluster_id: int,
        horizon: int,
        window: int,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build rolling-window feature rows from price history.

        All features are computed exclusively from data available UP TO each
        training point — no future information leaks into earlier windows.
        """
        log_ret = np.log(price_series / price_series.shift(1)).dropna()
        prices_aligned = price_series.reindex(log_ret.index)

        rows = []
        targets = []

        for i in range(window, len(log_ret) - horizon):
            ret_window = log_ret.iloc[i - window:i]
            price_window = prices_aligned.iloc[i - window:i]
            future_ret = float(prices_aligned.iloc[i + horizon] / prices_aligned.iloc[i] - 1)

            # --- All features computed within [i-window, i] --- no leakage ---
            mean_ret = float(ret_window.mean())
            vol = float(ret_window.std()) + 1e-9
            last_ret = float(ret_window.iloc[-1])
            half = max(1, window // 4)
            mom_short = float(ret_window.iloc[-half:].mean())
            mom_mid = float(ret_window.iloc[-(half * 2):].mean()) if len(ret_window) >= half * 2 else mom_short
            max_ret = float(ret_window.max())
            min_ret = float(ret_window.min())
            up_frac = float((ret_window > 0).mean())
            autocorr = float(ret_window.autocorr(lag=1)) if len(ret_window) > 5 else 0.0
            skew = float(ret_window.skew())

            # Price position relative to window range (proxy for Bollinger %B)
            price_hi = float(price_window.max())
            price_lo = float(price_window.min())
            price_range = price_hi - price_lo
            bb_pct = float((prices_aligned.iloc[i] - price_lo) / price_range) if price_range > 0 else 0.5

            # Simplified RSI from the window (no look-ahead)
            gains = ret_window[ret_window > 0].mean() if (ret_window > 0).any() else 0.0
            losses = -ret_window[ret_window < 0].mean() if (ret_window < 0).any() else 1e-9
            rsi_proxy = float(100 - 100 / (1 + gains / losses))

            feat_row = [
                mean_ret, vol, last_ret, mom_short, mom_mid,
                max_ret, min_ret, up_frac,
                autocorr if not np.isnan(autocorr) else 0.0,
                skew if not np.isnan(skew) else 0.0,
                bb_pct, rsi_proxy,
                cluster_id,
            ]
            rows.append(feat_row)
            targets.append(_label_return(future_ret, threshold))

        X = np.array(rows, dtype=np.float32)
        y = np.array(targets)
        X_predict = np.array(rows[-1], dtype=np.float32)

        return X, y, X_predict

    def _walk_forward_train(self, X: np.ndarray, y: np.ndarray) -> tuple[Any, Any, float]:
        """Train with time-series walk-forward splits; return final model."""
        try:
            from xgboost import XGBClassifier
            # use_label_encoder was removed in XGBoost 2.0 — do not include it
            model_kwargs: dict = {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "eval_metric": "mlogloss",
                "verbosity": 0,
                "random_state": 42,
            }
            use_xgboost = True
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier  # type: ignore
            model_kwargs = {"n_estimators": 100, "max_depth": 3, "random_state": 42}
            use_xgboost = False

        n = len(X)
        split_size = n // (settings.walk_forward_splits + 1)
        accuracies = []

        for fold in range(settings.walk_forward_splits):
            train_end = split_size * (fold + 1)
            val_end = train_end + split_size
            if val_end > n:
                break
            X_tr, y_tr = X[:train_end], y[:train_end]
            X_val, y_val = X[train_end:val_end], y[train_end:val_end]

            le = LabelEncoder()
            y_tr_enc = le.fit_transform(y_tr)
            y_val_enc = le.transform(y_val)

            if use_xgboost:
                from xgboost import XGBClassifier
                m = XGBClassifier(**model_kwargs)
            else:
                from sklearn.ensemble import GradientBoostingClassifier  # type: ignore
                m = GradientBoostingClassifier(**model_kwargs)
            m.fit(X_tr, y_tr_enc)
            preds = m.predict(X_val)
            accuracies.append(accuracy_score(y_val_enc, preds))

        # Final model trained on all data
        le_final = LabelEncoder()
        y_enc = le_final.fit_transform(y)

        if use_xgboost:
            from xgboost import XGBClassifier
            final_model = XGBClassifier(**model_kwargs)
        else:
            from sklearn.ensemble import GradientBoostingClassifier  # type: ignore
            final_model = GradientBoostingClassifier(**model_kwargs)

        final_model.fit(X, y_enc)

        mean_acc = float(np.mean(accuracies)) if accuracies else 0.33
        logger.debug("Walk-forward accuracy: %.3f", mean_acc)
        # Return classes separately — XGBoost 3.x classes_ property has no setter
        return final_model, le_final.classes_, mean_acc
