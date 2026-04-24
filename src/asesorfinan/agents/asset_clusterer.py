"""Agent 3 — Asset Clusterer (ML Unsupervised).

Uses PCA + KMeans (or HDBSCAN) to group assets by their actual risk/return
behaviour rather than their nominal asset class label.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from asesorfinan.config import settings
from asesorfinan.models import GraphState

logger = logging.getLogger(__name__)

# Human-readable cluster names assigned after fitting based on cluster centroid stats
_CLUSTER_NAME_TEMPLATES = [
    "bajo riesgo estable",
    "riesgo moderado diversificado",
    "alta volatilidad crecimiento",
    "alta volatilidad especulativo",
    "defensivo anti-correlacionado",
]

# Features used exclusively for clustering (risk/return profile, not macro)
_CLUSTER_FEATURES = ["ret_30d", "ret_90d", "ret_1y", "annual_vol", "sharpe", "max_drawdown"]


class AssetClustererAgent:
    def run(self, state: GraphState) -> GraphState:
        features_df: pd.DataFrame = state.features_df

        available = [c for c in _CLUSTER_FEATURES if c in features_df.columns]
        X = features_df[available].values

        logger.info("Clustering %d assets on features: %s", len(X), available)

        X_scaled = StandardScaler().fit_transform(X)
        X_reduced = self._apply_pca(X_scaled)

        labels = self._cluster(X_reduced, n_clusters=settings.n_clusters)

        cluster_labels = {ticker: int(lbl) for ticker, lbl in zip(features_df.index, labels)}
        cluster_names = self._name_clusters(features_df[available], labels)

        logger.info("Cluster distribution: %s", pd.Series(labels).value_counts().to_dict())

        state.cluster_labels = cluster_labels
        state.cluster_names = cluster_names
        return state

    # ------------------------------------------------------------------

    def _apply_pca(self, X: np.ndarray) -> np.ndarray:
        n_components = min(X.shape[0], X.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_reduced = pca.fit_transform(X)

        cumvar = np.cumsum(pca.explained_variance_ratio_)
        keep = int(np.searchsorted(cumvar, settings.pca_variance_threshold) + 1)
        keep = max(2, min(keep, X_reduced.shape[1]))
        logger.info("PCA: keeping %d components (%.1f%% variance)", keep, cumvar[keep - 1] * 100)
        return X_reduced[:, :keep]

    def _cluster(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        # Try HDBSCAN first; fall back to KMeans if not installed or too few samples
        if len(X) < n_clusters * 2:
            logger.warning("Too few assets for HDBSCAN — using KMeans with %d clusters", n_clusters)
            return self._kmeans(X, n_clusters)

        try:
            import hdbscan  # type: ignore
            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
            labels = clusterer.fit_predict(X)
            n_found = len(set(labels) - {-1})
            logger.info("HDBSCAN found %d clusters (noise: %d)", n_found, (labels == -1).sum())
            if n_found < 2:
                logger.info("HDBSCAN produced trivial clusters — falling back to KMeans")
                return self._kmeans(X, n_clusters)
            # Assign noise points (-1) to nearest cluster
            if (labels == -1).any():
                labels = self._assign_noise(X, labels)
            return labels
        except ImportError:
            logger.info("hdbscan not installed — using KMeans")
            return self._kmeans(X, n_clusters)

    def _kmeans(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        from sklearn.cluster import KMeans
        k = min(n_clusters, len(X))
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        return km.fit_predict(X)

    def _assign_noise(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Reassign HDBSCAN noise points to the nearest non-noise centroid."""
        from sklearn.metrics import pairwise_distances
        unique = sorted(set(labels) - {-1})
        centroids = np.array([X[labels == c].mean(axis=0) for c in unique])
        noise_idx = np.where(labels == -1)[0]
        dists = pairwise_distances(X[noise_idx], centroids)
        labels[noise_idx] = np.array(unique)[dists.argmin(axis=1)]
        return labels

    def _name_clusters(self, feature_df: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
        """Assign interpretable names based on centroid volatility and return."""
        df = feature_df.copy()
        df["_cluster"] = labels
        centroids = df.groupby("_cluster").mean()

        # Sort clusters by annual_vol ascending to get meaningful ordering
        vol_col = "annual_vol" if "annual_vol" in centroids.columns else centroids.columns[0]
        sorted_ids = centroids[vol_col].sort_values().index.tolist()

        names: dict[int, str] = {}
        for rank, cid in enumerate(sorted_ids):
            names[int(cid)] = _CLUSTER_NAME_TEMPLATES[min(rank, len(_CLUSTER_NAME_TEMPLATES) - 1)]

        return names
