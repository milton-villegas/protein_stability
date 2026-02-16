"""Analysis service - wraps DataHandler and DoEAnalyzer"""

from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from core.data_handler import DataHandler
from core.doe_analyzer import DoEAnalyzer


def load_and_detect(filepath: str) -> Dict[str, Any]:
    """Load an Excel file and detect columns"""
    handler = DataHandler()
    handler.load_excel(filepath)

    potential_responses = handler.get_potential_response_columns()

    # Preview first rows
    preview = handler.data.head(10).fillna("").to_dict(orient="records")

    return {
        "handler": handler,
        "columns": list(handler.data.columns),
        "potential_responses": potential_responses,
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
    if analyzer.response_columns:
        raw_results = analyzer.fit_model_all_responses(model_type=model_type)
    else:
        raw_results = {analyzer.response_column: analyzer.fit_model(model_type=model_type)}

    # Convert results to JSON-serializable format
    results = {}
    for response_name, res in raw_results.items():
        serialized = _serialize_results(res)
        results[response_name] = serialized

    return results


def compare_models(analyzer: DoEAnalyzer) -> Dict[str, Any]:
    """Compare all model types and return serializable results"""
    if analyzer.response_columns:
        raw = analyzer.compare_all_models_all_responses()
    else:
        raw = {analyzer.response_column: analyzer.compare_all_models()}

    comparisons = {}
    recommendations = {}

    for response_name, comparison in raw.items():
        serialized = {}
        for model_name, model_data in comparison.items():
            serialized[model_name] = _make_serializable(model_data)

        comparisons[response_name] = serialized

        best = analyzer.select_best_model(comparison)
        recommendations[response_name] = best.get("best_model", "linear")

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
        serialized[factor] = df.fillna(0).to_dict(orient="records")

    return serialized


def _serialize_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Convert analysis results to JSON-serializable format"""
    return _make_serializable(results)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to Python natives"""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.fillna(0).to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.fillna(0).to_dict()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else 0
    elif isinstance(obj, np.ndarray):
        return [_make_serializable(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return 0
    else:
        return obj
