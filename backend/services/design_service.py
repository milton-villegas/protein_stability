"""Design service - wraps core design modules"""

import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.doe_designer import DoEDesigner
from core.design_factory import DesignFactory
from core.design_validator import DesignValidator
from core.volume_calculator import VolumeCalculator
from core.well_mapper import WellMapper
from config.design_config import WELLS_PER_PLATE, CATEGORICAL_FACTORS


def get_combinations_count(factors: Dict[str, List[str]]) -> int:
    """Calculate total combinations for full factorial"""
    if not factors:
        return 0
    count = 1
    for levels in factors.values():
        count *= len(levels)
    return count


def get_plates_required(num_samples: int) -> int:
    """Calculate number of 96-well plates needed"""
    if num_samples <= 0:
        return 0
    return math.ceil(num_samples / WELLS_PER_PLATE)


def build_factorial_design(
    designer: DoEDesigner,
    factors: Dict[str, List[str]],
    stock_concs: Dict[str, float],
    final_volume: float,
    per_level_concs: Optional[Dict] = None,
) -> Tuple[List[Dict], List[Dict], List[str]]:
    """
    Build full factorial design and return serializable data.

    Returns:
        (excel_data, volume_data, warnings)
    """
    excel_df, volume_df = designer.build_factorial_design(
        factors, stock_concs, final_volume
    )

    excel_data = excel_df.to_dict(orient="records")
    volume_data = volume_df.to_dict(orient="records")
    warnings = []

    return excel_data, volume_data, warnings


def generate_design(
    factory: DesignFactory,
    design_type: str,
    factors: Dict[str, List[str]],
    params: Dict[str, Any],
) -> Tuple[List[Dict], List[str]]:
    """
    Generate design using DesignFactory.

    Returns:
        (design_points, warnings)
    """
    warnings = []

    # Validate design type requirements
    is_valid, msg = DesignValidator.validate_design_type_requirements(
        design_type,
        len(factors),
        has_pydoe3=True,
        has_smt=False,
    )
    if not is_valid:
        raise ValueError(msg)

    design_points = factory.create_design(design_type, factors, **params)
    return design_points, warnings


def validate_design_params(
    design_type: str,
    factors: Dict[str, List[str]],
    has_pydoe3: bool = False,
    has_smt: bool = False,
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate design parameters.

    Returns:
        (valid, errors, warnings)
    """
    errors = []
    warnings = []

    if not factors:
        errors.append("No factors defined")
        return False, errors, warnings

    is_valid, msg = DesignValidator.validate_design_type_requirements(
        design_type, len(factors), has_pydoe3, has_smt
    )
    if not is_valid:
        errors.append(msg)

    return len(errors) == 0, errors, warnings
