"""Factor-related Pydantic schemas"""

from pydantic import BaseModel
from typing import Dict, List, Optional


class FactorAddRequest(BaseModel):
    name: str
    levels: List[str]
    stock_conc: Optional[float] = None
    per_level_concs: Optional[Dict[str, Dict[str, float]]] = None


class FactorUpdateRequest(BaseModel):
    levels: List[str]
    stock_conc: Optional[float] = None
    per_level_concs: Optional[Dict[str, Dict[str, float]]] = None


class FactorFromAvailableRequest(BaseModel):
    internal_name: str
    levels: List[str]
    stock_conc: Optional[float] = None
    per_level_concs: Optional[Dict[str, Dict[str, float]]] = None


class FactorsResponse(BaseModel):
    factors: Dict[str, List[str]]
    stock_concs: Dict[str, float]
    per_level_concs: Dict[str, Dict[str, Dict[str, float]]]
    total_combinations: int
    plates_required: int
