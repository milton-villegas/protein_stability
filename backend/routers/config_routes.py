"""Configuration routes - serves static config data"""

from fastapi import APIRouter

from utils.constants import AVAILABLE_FACTORS, METADATA_COLUMNS
from config.design_config import (
    DESIGN_TYPES, FACTOR_CONSTRAINTS, UNIT_OPTIONS,
    CATEGORICAL_FACTORS, WELLS_PER_PLATE, MAX_PLATES,
    MAX_TOTAL_WELLS, FRACTIONAL_RESOLUTION_OPTIONS,
    CCD_TYPE_OPTIONS, D_OPTIMAL_MODEL_OPTIONS,
    DEFAULT_CENTER_POINTS, MIN_SAMPLE_SIZE, MAX_SAMPLE_SIZE,
    DEFAULT_FINAL_VOLUME,
)

router = APIRouter()


FACTOR_CATEGORIES = [
    {"name": "Buffer System", "factors": ["buffer pH", "buffer_concentration"]},
    {"name": "Detergents", "factors": ["detergent", "detergent_concentration"]},
    {"name": "Reducing Agents", "factors": ["reducing_agent", "reducing_agent_concentration"]},
    {"name": "Salts", "factors": ["nacl", "kcl"]},
    {"name": "Metals", "factors": ["zinc", "magnesium", "calcium"]},
    {"name": "Additives", "factors": ["glycerol", "dmso"]},
]


@router.get("/factors")
async def get_available_factors():
    """Get all available factors with display names"""
    return {
        "factors": AVAILABLE_FACTORS,
        "categorical_factors": list(CATEGORICAL_FACTORS),
        "factor_categories": FACTOR_CATEGORIES,
    }


@router.get("/design-types")
async def get_design_types():
    """Get all supported design types with metadata"""
    return {
        "design_types": DESIGN_TYPES,
        "resolution_options": FRACTIONAL_RESOLUTION_OPTIONS,
        "ccd_type_options": CCD_TYPE_OPTIONS,
        "d_optimal_model_options": D_OPTIMAL_MODEL_OPTIONS,
        "default_center_points": DEFAULT_CENTER_POINTS,
        "min_sample_size": MIN_SAMPLE_SIZE,
        "max_sample_size": MAX_SAMPLE_SIZE,
    }


@router.get("/constraints")
async def get_factor_constraints():
    """Get factor validation constraints"""
    return {
        "constraints": FACTOR_CONSTRAINTS,
        "unit_options": UNIT_OPTIONS,
    }


@router.get("/constants")
async def get_constants():
    """Get well plate and other constants"""
    return {
        "wells_per_plate": WELLS_PER_PLATE,
        "max_plates": MAX_PLATES,
        "max_total_wells": MAX_TOTAL_WELLS,
        "default_final_volume": DEFAULT_FINAL_VOLUME,
        "metadata_columns": METADATA_COLUMNS,
    }
