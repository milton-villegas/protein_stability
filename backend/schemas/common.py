"""Shared Pydantic models"""

from pydantic import BaseModel
from typing import Optional, List


class SuccessResponse(BaseModel):
    success: bool = True
    message: str = ""


class ErrorResponse(BaseModel):
    success: bool = False
    detail: str
    errors: List[str] = []
