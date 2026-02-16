"""Analysis-related Pydantic schemas"""

from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class AnalysisConfigureRequest(BaseModel):
    response_columns: List[str]
    directions: Optional[Dict[str, str]] = None
    constraints: Optional[Dict[str, Dict[str, float]]] = None


class AnalysisRunRequest(BaseModel):
    model_type: str = "linear"


class AnalysisRunResponse(BaseModel):
    results: Dict[str, Any]


class ModelComparisonResponse(BaseModel):
    comparisons: Dict[str, Any]
    recommendations: Dict[str, str] = {}


class OptimizeRequest(BaseModel):
    response_columns: List[str]
    directions: Dict[str, str]
    constraints: Optional[Dict[str, Dict[str, float]]] = None
    n_suggestions: int = 5


class OptimizeResponse(BaseModel):
    suggestions: List[Dict[str, Any]]
    has_pareto: bool = False


class UploadResponse(BaseModel):
    columns: List[str]
    potential_responses: List[str]
    factor_columns: List[str]
    preview_rows: List[Dict[str, Any]]
    total_rows: int
