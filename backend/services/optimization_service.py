"""Optimization service - wraps BayesianOptimizer"""

import logging
import traceback
from typing import Any, Dict, List, Optional

from backend.services.analysis_service import _make_serializable

logger = logging.getLogger(__name__)


def initialize_optimizer(
    session: dict,
    response_columns: List[str],
    directions: Dict[str, str],
    constraints: Optional[Dict[str, Dict[str, float]]] = None,
    n_suggestions: int = 5,
) -> Dict[str, Any]:
    """Initialize Bayesian Optimizer and get suggestions"""
    from core.optimizer import BayesianOptimizer

    handler = session.get("data_handler")
    if handler is None or handler.clean_data is None:
        raise ValueError("No analysis data. Run analysis first.")

    try:
        logger.info(f"[OPTIMIZE.SVC] responses={response_columns}, directions={directions}, n={n_suggestions}")
        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=handler.clean_data,
            factor_columns=handler.factor_columns,
            categorical_factors=handler.categorical_factors,
            numeric_factors=handler.numeric_factors,
            response_columns=response_columns,
            response_directions=directions,
            response_constraints=constraints,
        )
        logger.info("[OPTIMIZE.SVC] Initializing optimizer...")
        optimizer.initialize_optimizer()

        logger.info("[OPTIMIZE.SVC] Getting suggestions...")
        suggestions = optimizer.get_next_suggestions(n=n_suggestions)
        optimizer.last_suggestions = suggestions
        session["optimizer"] = optimizer

        serialized_suggestions = [_make_serializable(s) for s in suggestions]
        has_pareto = optimizer.is_multi_objective
        logger.info(f"[OPTIMIZE.SVC] Success: {len(suggestions)} suggestions, pareto={has_pareto}")

        return {
            "suggestions": serialized_suggestions,
            "has_pareto": has_pareto,
        }
    except Exception as e:
        logger.error(f"[OPTIMIZE.SVC] ERROR: {e}\n{traceback.format_exc()}")
        raise
