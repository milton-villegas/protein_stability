"""Analysis service - wraps DataHandler and DoEAnalyzer"""

import logging
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from core.data_handler import DataHandler
from core.doe_analyzer import DoEAnalyzer

logger = logging.getLogger(__name__)


def load_and_detect(filepath: str) -> Dict[str, Any]:
    """Load an Excel file and detect columns"""
    handler = DataHandler()
    handler.load_excel(filepath)

    potential_responses = handler.get_potential_response_columns()

    # Preview first rows - convert to plain Python types for JSON serialization
    preview_df = handler.data.head(10).fillna("")
    preview = []
    for _, row in preview_df.iterrows():
        clean_row = {}
        for col in preview_df.columns:
            val = row[col]
            if isinstance(val, (np.integer,)):
                clean_row[str(col)] = int(val)
            elif isinstance(val, (np.floating,)):
                clean_row[str(col)] = float(val) if not np.isnan(val) else ""
            elif isinstance(val, np.bool_):
                clean_row[str(col)] = bool(val)
            elif hasattr(val, 'isoformat'):
                clean_row[str(col)] = str(val)
            else:
                clean_row[str(col)] = str(val) if val != "" else ""
        preview.append(clean_row)

    return {
        "handler": handler,
        "columns": [str(c) for c in handler.data.columns],
        "potential_responses": [str(r) for r in potential_responses],
        "preview_rows": preview,
        "total_rows": len(handler.data),
    }


def configure_analysis(
    handler: DataHandler,
    response_columns: List[str],
) -> Dict[str, Any]:
    """Configure analysis by detecting column types"""
    handler.detect_columns(response_columns=response_columns)
    clean_data = handler.preprocess_data()

    return {
        "factor_columns": handler.factor_columns,
        "categorical_factors": handler.categorical_factors,
        "numeric_factors": handler.numeric_factors,
        "data_shape": list(clean_data.shape),
    }


def run_analysis(
    analyzer: DoEAnalyzer,
    model_type: str,
) -> Dict[str, Any]:
    """Run statistical analysis and return serializable results"""
    logger.info(f"[SERVICE.RUN] model_type={model_type}, "
                f"response_columns={analyzer.response_columns}, "
                f"response_column={getattr(analyzer, 'response_column', None)}")

    try:
        # Auto model selection: compare all models, pick best per response
        if model_type == "auto":
            logger.info("[SERVICE.RUN] Auto model selection: comparing all models")
            raw_results = {}
            responses = analyzer.response_columns or [analyzer.response_column]
            for resp_name in responses:
                comparison = analyzer.compare_all_models(response_name=resp_name)
                best = analyzer.select_best_model(comparison)
                chosen = best.get("recommended_model", "linear")
                logger.info(f"[SERVICE.RUN] Auto-selected '{chosen}' for '{resp_name}'")
                raw_results[resp_name] = analyzer.fit_model(chosen, response_name=resp_name)
        elif analyzer.response_columns:
            logger.info(f"[SERVICE.RUN] Fitting model for all responses: {analyzer.response_columns}")
            raw_results = analyzer.fit_model_all_responses(model_type=model_type)
        else:
            logger.info(f"[SERVICE.RUN] Fitting single response: {analyzer.response_column}")
            raw_results = {analyzer.response_column: analyzer.fit_model(model_type=model_type)}

        logger.info(f"[SERVICE.RUN] Raw results keys: {list(raw_results.keys())}")

        # Convert results to JSON-serializable format
        results = {}
        for response_name, res in raw_results.items():
            logger.info(f"[SERVICE.RUN] Serializing '{response_name}', "
                        f"result type={type(res).__name__}, "
                        f"keys={list(res.keys()) if isinstance(res, dict) else 'N/A'}")
            for k, v in (res.items() if isinstance(res, dict) else []):
                logger.info(f"[SERVICE.RUN]   '{response_name}'.'{k}' -> type={type(v).__name__}")
            serialized = _serialize_results(res)
            results[response_name] = serialized

        logger.info("[SERVICE.RUN] Serialization complete")
        return results
    except Exception as e:
        logger.error(f"[SERVICE.RUN] ERROR: {e}\n{traceback.format_exc()}")
        raise


def compare_models(analyzer: DoEAnalyzer) -> Dict[str, Any]:
    """Compare all model types and return serializable results"""
    if analyzer.response_columns:
        raw = analyzer.compare_all_models_all_responses()
    else:
        raw = {analyzer.response_column: analyzer.compare_all_models()}

    comparisons = {}
    recommendations = {}

    for response_name, comparison in raw.items():
        # comparison has keys: 'models', 'fitted_models', 'comparison_table', 'errors'
        # Normalize model stats keys and skip fitted_models (statsmodels objects)
        models = comparison.get("models", {})
        normalized_models = {}
        for model_name, stats in models.items():
            normalized_models[model_name] = _normalize_model_stats(stats)
        comparisons[response_name] = normalized_models

        best = analyzer.select_best_model(comparison)
        recommendations[response_name] = {
            "best_model": best.get("recommended_model", "linear"),
            "reason": best.get("reason", ""),
            "scores": _make_serializable(best.get("scores", {})),
        }

    return {
        "comparisons": comparisons,
        "recommendations": recommendations,
    }


def get_main_effects(
    analyzer: DoEAnalyzer,
    response_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Get main effects data"""
    if response_name:
        effects = analyzer.calculate_main_effects(response_name=response_name)
    else:
        effects = analyzer.calculate_main_effects()

    serialized = {}
    for factor, df in effects.items():
        serialized[factor] = df.where(df.notna(), other=None).to_dict(orient="records")

    return serialized


def get_analysis_summary(
    analyzer: DoEAnalyzer,
    handler: DataHandler,
    directions: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Generate analysis summary with warnings, significant factors, recommendations.
    Mirrors gui/tabs/analysis_tab.py display_statistics + display_recommendations."""
    directions = directions or {}
    summaries = {}

    # Get results per response
    all_results = getattr(analyzer, "all_results", None) or {}
    single_results = getattr(analyzer, "results", None)
    if not all_results and single_results:
        resp_name = analyzer.response_column
        all_results = {resp_name: single_results}

    for response_name, results in all_results.items():
        stats = results.get("model_stats", {})
        coefficients = results.get("coefficients")
        r_squared = float(stats.get("R-squared", 0))
        n_obs = int(stats.get("Observations", 0))
        direction = directions.get(response_name, "maximize")

        # Significant factors
        sig_factors = []
        interactions = []
        optimal_directions = []
        if coefficients is not None and isinstance(coefficients, pd.DataFrame):
            sig_df = coefficients[coefficients["p-value"] < 0.05]
            for factor_name in sig_df.index:
                if factor_name == "Intercept":
                    continue
                coef = float(sig_df.loc[factor_name, "Coefficient"])
                pval = float(sig_df.loc[factor_name, "p-value"])
                entry = {"factor": str(factor_name), "coefficient": coef, "p_value": pval, "abs_effect": abs(coef)}
                if ":" in str(factor_name):
                    interactions.append(entry)
                else:
                    sig_factors.append(entry)
                    # Optimal direction
                    clean_name = str(factor_name).replace("C(Q('", "").replace("'))", "").replace("Q('", "").replace("')", "")
                    optimal_directions.append({
                        "factor": clean_name,
                        "direction": "INCREASE" if coef > 0 else "DECREASE",
                        "effect": coef,
                    })

            sig_factors.sort(key=lambda x: x["abs_effect"], reverse=True)
            interactions.sort(key=lambda x: x["abs_effect"], reverse=True)

        # Warnings
        warnings = []
        if r_squared < 0.5:
            warnings.append(f"LOW R-squared ({r_squared:.3f}): Model explains only {r_squared*100:.1f}% of variance")
        elif r_squared < 0.7:
            warnings.append(f"MODERATE R-squared ({r_squared:.3f}): Model is acceptable but could be improved")
        if len(sig_factors) == 0:
            warnings.append("NO SIGNIFICANT FACTORS: No factors with p < 0.05 found")
        if n_obs < 20:
            warnings.append(f"SMALL SAMPLE SIZE: Only {n_obs} observations")

        # Confidence
        if r_squared >= 0.8 and len(sig_factors) > 0 and n_obs >= 20:
            confidence = "HIGH"
        elif r_squared >= 0.6 and len(sig_factors) > 0:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Best observed experiment
        best_experiment = None
        if handler.clean_data is not None and response_name in handler.clean_data.columns:
            clean_data = handler.clean_data
            if direction == "minimize":
                best_idx = clean_data[response_name].idxmin()
            else:
                best_idx = clean_data[response_name].idxmax()
            best_val = float(clean_data.loc[best_idx, response_name])
            conditions = {}
            for factor in handler.factor_columns:
                val = clean_data.loc[best_idx, factor]
                conditions[factor] = float(val) if isinstance(val, (int, float, np.integer, np.floating)) else str(val)
            best_experiment = {
                "value": best_val,
                "direction": direction,
                "conditions": conditions,
            }
            if "ID" in clean_data.columns:
                best_experiment["id"] = str(clean_data.loc[best_idx, "ID"])

        # Next steps
        if confidence == "HIGH":
            next_steps = [
                "Run 3-5 confirmation experiments at the predicted optimal condition",
                "Compare results to model prediction to validate",
                "If confirmed, implement optimized condition in production",
            ]
        elif confidence == "MEDIUM":
            next_steps = [
                "Run confirmation experiments at predicted optimal condition",
                "Consider additional replicates to improve model confidence",
                "May need to refine factor ranges or add more data",
            ]
        else:
            next_steps = [
                "Results may not be reliable enough for immediate use",
                "Consider running more experiments",
                "Check for experimental errors or measurement issues",
                "May need to reconsider factors or expand factor ranges",
            ]

        summaries[response_name] = {
            "r_squared": r_squared,
            "n_observations": n_obs,
            "n_significant": len(sig_factors),
            "warnings": warnings,
            "confidence": confidence,
            "significant_factors": sig_factors,
            "interactions": interactions,
            "optimal_directions": optimal_directions,
            "best_experiment": best_experiment,
            "next_steps": next_steps,
        }

    return _make_serializable(summaries)


def _normalize_key(key: str) -> str:
    """Convert display keys like 'R-squared' to frontend keys like 'r_squared'"""
    return (
        key.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("²", "_squared")
        .replace("adj_r_squared", "adj_r_squared")
        .replace("adjusted_r_squared", "adj_r_squared")
        .replace("f_p_value", "f_pvalue")
        .replace("df_residuals", "df_residuals")
    )


def _normalize_model_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize model_stats keys to match frontend expectations"""
    return {_normalize_key(k): _make_serializable(v) for k, v in stats.items()}


def _serialize_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Convert analysis results to JSON-serializable format"""
    serialized = _make_serializable(results)
    # Normalize model_stats keys
    if isinstance(serialized, dict) and "model_stats" in serialized:
        serialized["model_stats"] = _normalize_model_stats(results["model_stats"])
    return serialized


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to Python natives"""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        records = obj.where(obj.notna(), other=None).to_dict(orient="records")
        return [_make_serializable(r) for r in records]
    elif isinstance(obj, pd.Series):
        return _make_serializable(obj.where(obj.notna(), other=None).to_dict())
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, np.ndarray):
        converted = obj.tolist()
        if isinstance(converted, list):
            return [_make_serializable(x) for x in converted]
        else:
            # 0-d array: tolist() returns a scalar
            return _make_serializable(converted)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    else:
        # Skip non-serializable objects (e.g. statsmodels RegressionResultsWrapper)
        try:
            import json
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            logger.info(f"[SERIALIZE] Skipping non-serializable type: {type(obj).__name__}")
            return None
