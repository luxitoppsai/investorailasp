"""LangGraph Orchestrator.

Defines the multi-agent pipeline as a StateGraph with a risk-retry loop:

  collect_data → engineer_features → cluster_assets → predict_returns
       → optimize_portfolio → assess_risk
           ├── [approved or max retries] → advise → END
           └── [rejected]  → optimize_portfolio  (tighten constraints)

Node functions return ONLY the fields they mutate — not the full state.
This avoids DataFrame serialization issues and is the idiomatic LangGraph pattern.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, StateGraph

from asesorfinan.agents import (
    AssetClustererAgent,
    DataCollectorAgent,
    FeatureEngineerAgent,
    LLMAdvisorAgent,
    PortfolioOptimizerAgent,
    ReturnPredictorAgent,
    RiskManagerAgent,
)
from asesorfinan.config import settings
from asesorfinan.models import GraphState, UserProfile

logger = logging.getLogger(__name__)

_data_collector = DataCollectorAgent()
_feature_engineer = FeatureEngineerAgent()
_asset_clusterer = AssetClustererAgent()
_return_predictor = ReturnPredictorAgent()
_portfolio_optimizer = PortfolioOptimizerAgent()
_risk_manager = RiskManagerAgent()
_llm_advisor = LLMAdvisorAgent()


# ---------------------------------------------------------------------------
# Node functions — return ONLY the fields changed by each agent
# ---------------------------------------------------------------------------

def node_collect_data(state: GraphState) -> dict:
    logger.info("▶ Agent 1 — Data Collector")
    updated = _data_collector.run(state)
    return {
        "prices_df": updated.prices_df,
        "macro_df": updated.macro_df,
        "fundamentals_df": updated.fundamentals_df,
    }


def node_engineer_features(state: GraphState) -> dict:
    logger.info("▶ Agent 2 — Feature Engineer")
    updated = _feature_engineer.run(state)
    return {"features_df": updated.features_df}


def node_cluster_assets(state: GraphState) -> dict:
    logger.info("▶ Agent 3 — Asset Clusterer")
    updated = _asset_clusterer.run(state)
    return {"cluster_labels": updated.cluster_labels, "cluster_names": updated.cluster_names}


def node_predict_returns(state: GraphState) -> dict:
    logger.info("▶ Agent 4 — Return Predictor")
    updated = _return_predictor.run(state)
    return {"predictions": updated.predictions}


def node_optimize_portfolio(state: GraphState) -> dict:
    logger.info("▶ Agent 5 — Portfolio Optimizer (retry #%d)", state.risk_retry_count)
    if state.risk_retry_count > 0:
        # Tighten per-asset cap on each retry to force more diversification.
        # We pass the tighter cap via a temporary settings override scoped to this call.
        tighter_cap = max(0.10, settings.max_weight_per_asset * (1 - 0.10 * state.risk_retry_count))
        logger.info("Tightening max weight cap to %.2f for retry", tighter_cap)
        original = settings.max_weight_per_asset
        settings.max_weight_per_asset = tighter_cap
        try:
            updated = _portfolio_optimizer.run(state)
        finally:
            settings.max_weight_per_asset = original
    else:
        updated = _portfolio_optimizer.run(state)
    return {"portfolio": updated.portfolio}


def node_assess_risk(state: GraphState) -> dict:
    logger.info("▶ Agent 6 — Risk Manager")
    updated = _risk_manager.run(state)
    new_retry_count = state.risk_retry_count + (0 if updated.risk_report.approved else 1)
    return {"risk_report": updated.risk_report, "risk_retry_count": new_retry_count}


def node_advise(state: GraphState) -> dict:
    logger.info("▶ Agent 7 — LLM Advisor")
    updated = _llm_advisor.run(state)
    return {"advisor_report": updated.advisor_report}


# ---------------------------------------------------------------------------
# Conditional edge: after risk assessment
# ---------------------------------------------------------------------------

def route_after_risk(state: GraphState) -> Literal["optimize_portfolio", "advise"]:
    risk = state.risk_report
    if risk.approved or state.risk_retry_count >= settings.max_risk_retries:
        if not risk.approved:
            logger.warning(
                "Max risk retries (%d) reached — proceeding with unapproved portfolio",
                settings.max_risk_retries,
            )
        return "advise"
    logger.info("Portfolio rejected by risk manager — retrying optimization")
    return "optimize_portfolio"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("collect_data", node_collect_data)
    graph.add_node("engineer_features", node_engineer_features)
    graph.add_node("cluster_assets", node_cluster_assets)
    graph.add_node("predict_returns", node_predict_returns)
    graph.add_node("optimize_portfolio", node_optimize_portfolio)
    graph.add_node("assess_risk", node_assess_risk)
    graph.add_node("advise", node_advise)

    graph.set_entry_point("collect_data")
    graph.add_edge("collect_data", "engineer_features")
    graph.add_edge("engineer_features", "cluster_assets")
    graph.add_edge("cluster_assets", "predict_returns")
    graph.add_edge("predict_returns", "optimize_portfolio")
    graph.add_edge("optimize_portfolio", "assess_risk")
    graph.add_conditional_edges(
        "assess_risk",
        route_after_risk,
        {"optimize_portfolio": "optimize_portfolio", "advise": "advise"},
    )
    graph.add_edge("advise", END)

    return graph.compile()


def run_pipeline(user_profile: UserProfile) -> GraphState:
    """Run the full multi-agent pipeline and return the final state."""
    compiled = build_graph()
    initial_state = GraphState(user_profile=user_profile)
    final = compiled.invoke(initial_state)
    return GraphState(**final)
