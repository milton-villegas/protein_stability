"""Optimization service - wraps BayesianOptimizer"""

import logging
import traceback
from typing import Any, Dict, List, Optional

from backend.services.analysis_service import _make_serializable
from gui.tabs.analysis.validation import validate_constraint

logger = logging.getLogger(__name__)


def initialize_optimizer(
    session: dict,
    response_columns: List[str],
    directions: Dict[str, str],
    constraints: Optional[Dict[str, Dict[str, float]]] = None,
    n_suggestions: int = 5,
    exploration_mode: bool = False,
) -> Dict[str, Any]:
    """Initialize Bayesian Optimizer and get suggestions"""
    from core.optimizer import BayesianOptimizer

    handler = session.get("data_handler")
    if handler is None or handler.clean_data is None:
        raise ValueError("No analysis data. Run analysis first.")

    try:
        logger.info(f"[OPTIMIZE.SVC] responses={response_columns}, directions={directions}, "
                     f"n={n_suggestions}, exploration_mode={exploration_mode}")

        # Validate constraints (matching Tkinter behavior)
        validation_warnings = []
        if constraints:
            clean_data = handler.clean_data
            total_experiments = len(clean_data)
            for resp_name, constraint in constraints.items():
                if resp_name not in clean_data.columns:
                    continue
                direction = directions.get(resp_name, 'maximize')
                data_min = float(clean_data[resp_name].min())
                data_max = float(clean_data[resp_name].max())
                results = validate_constraint(
                    resp_name, direction, constraint,
                    data_min, data_max, total_experiments,
                )
                for r in results:
                    if r.get('should_stop'):
                        raise ValueError(r['message'])
                    validation_warnings.append(r)

        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=handler.clean_data,
            factor_columns=handler.factor_columns,
            categorical_factors=handler.categorical_factors,
            numeric_factors=handler.numeric_factors,
            response_columns=response_columns,
            response_directions=directions,
            response_constraints=constraints,
            exploration_mode=exploration_mode,
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

        result = {
            "suggestions": serialized_suggestions,
            "has_pareto": has_pareto,
        }
        if validation_warnings:
            result["validation_warnings"] = [
                {"severity": w["severity"], "message": w["message"]}
                for w in validation_warnings
            ]
        return result
    except Exception as e:
        logger.error(f"[OPTIMIZE.SVC] ERROR: {e}\n{traceback.format_exc()}")
        raise
