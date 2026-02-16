"""Design service - wraps core design modules"""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.doe_designer import DoEDesigner
from core.design_factory import DesignFactory
from core.design_validator import DesignValidator
from core.volume_calculator import VolumeCalculator
from core.well_mapper import WellMapper
from config.design_config import WELLS_PER_PLATE, MAX_TOTAL_WELLS, CATEGORICAL_FACTORS

logger = logging.getLogger(__name__)


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
    protein_stock: Optional[float] = None,
    protein_final: Optional[float] = None,
) -> Tuple[List[Dict], List[Dict], List[str]]:
    """
    Build full factorial design and return serializable data.

    Returns:
        (excel_data, volume_data, warnings)
    """
    excel_df, volume_df = designer.build_factorial_design(
        factors, stock_concs, final_volume
    )

    # Add Source and Batch columns (matching original SCOUT format)
    if "Source" not in excel_df.columns:
        resp_idx = list(excel_df.columns).index("Response") if "Response" in excel_df.columns else len(excel_df.columns)
        excel_df.insert(resp_idx, "Batch", 0)
        excel_df.insert(resp_idx, "Source", "FULL_FACTORIAL")

    # Convert numeric strings to numbers for display
    for col in excel_df.columns:
        if col in ("ID", "Plate_96", "Well_96", "Well_384", "Source", "Batch", "Response"):
            continue
        excel_df[col] = excel_df[col].apply(
            lambda v: float(v) if _is_numeric(v) else v
        )

    # Add protein volume row if protein params provided
    if protein_stock and protein_final and protein_stock > 0:
        protein_vol = (protein_final / protein_stock) * final_volume
        volume_df["protein"] = protein_vol
        # Adjust water column
        if "water" in volume_df.columns:
            volume_df["water"] = volume_df["water"] - protein_vol

    excel_data = _serialize_df(excel_df)
    volume_data = _serialize_df(volume_df)
    warnings = []

    return excel_data, volume_data, warnings


def generate_design(
    factory: DesignFactory,
    design_type: str,
    factors: Dict[str, List[str]],
    params: Dict[str, Any],
    stock_concs: Optional[Dict[str, float]] = None,
    final_volume: float = 200.0,
) -> Tuple[List[Dict], List[str]]:
    """
    Generate design using DesignFactory, add well positions and volumes.

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

    raw_points = factory.create_design(design_type, factors, **params)

    if len(raw_points) > MAX_TOTAL_WELLS:
        raise ValueError(
            f"Design has {len(raw_points)} runs, exceeding {MAX_TOTAL_WELLS} well limit"
        )

    # Add well positions
    mapper = WellMapper()
    for i, point in enumerate(raw_points):
        plate_96, well_96, well_384 = mapper.generate_well_position_384_order(i)
        point["ID"] = i + 1
        point["Plate_96"] = plate_96
        point["Well_96"] = well_96
        point["Well_384"] = well_384

    # Calculate volumes if stock concentrations available
    if stock_concs:
        try:
            calculator = VolumeCalculator()
            for point in raw_points:
                factor_values = {}
                for fname in factors:
                    val = point.get(fname)
                    if val is not None:
                        try:
                            factor_values[fname] = float(val)
                        except (ValueError, TypeError):
                            pass
                if factor_values:
                    volumes = calculator.calculate_volumes(
                        factor_values, stock_concs, final_volume
                    )
                    point["_volumes"] = volumes
        except Exception as e:
            logger.warning(f"[GENERATE] Volume calc failed: {e}")
            warnings.append(f"Volume calculation skipped: {str(e)}")

    # Serialize numpy types
    serialized = []
    for point in raw_points:
        clean = {}
        for k, v in point.items():
            if k == "_volumes":
                continue
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, np.bool_):
                clean[k] = bool(v)
            else:
                clean[k] = v
        serialized.append(clean)

    return serialized, warnings


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


def _is_numeric(v) -> bool:
    """Check if a value can be converted to float"""
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        try:
            float(v)
            return True
        except ValueError:
            return False
    return False


def _serialize_df(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to list of dicts with plain Python types"""
    records = df.to_dict(orient="records")
    serialized = []
    for row in records:
        clean = {}
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                clean[str(k)] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[str(k)] = float(v) if not np.isnan(v) else 0
            elif isinstance(v, np.bool_):
                clean[str(k)] = bool(v)
            elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                clean[str(k)] = 0
            else:
                clean[str(k)] = v
        serialized.append(clean)
    return serialized
