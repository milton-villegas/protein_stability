"""Project-related Pydantic schemas"""

from pydantic import BaseModel
from typing import Optional


class ProjectCreateRequest(BaseModel):
    name: str = "Untitled Project"


class ProjectInfoResponse(BaseModel):
    name: str
    has_design: bool
    has_results: bool
    factors_count: int
    design_runs: Optional[int] = None
