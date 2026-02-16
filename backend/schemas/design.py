"""Design-related Pydantic schemas"""

from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class DesignGenerateRequest(BaseModel):
    design_type: str
    final_volume: float = 200.0
    protein_stock: Optional[float] = None
    protein_final: Optional[float] = None
    params: Dict[str, Any] = {}


class BuildFactorialRequest(BaseModel):
    final_volume: float = 200.0
    protein_stock: Optional[float] = None
    protein_final: Optional[float] = None


class DesignGenerateResponse(BaseModel):
    design_points: List[Dict[str, Any]]
    total_runs: int
    plates_required: int
    warnings: List[str] = []


class ExportRequest(BaseModel):
    final_volume: float = 200.0
    protein_stock: Optional[float] = None
    protein_final: Optional[float] = None
