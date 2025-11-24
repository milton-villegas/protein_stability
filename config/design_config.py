"""
Design Configuration Constants
Centralized configuration for experiment designer
"""

from typing import Dict, List, Tuple, Any

# ============================================================================
# WELL PLATE CONFIGURATION
# ============================================================================

WELLS_PER_PLATE = 96
"""Number of wells in a standard 96-well plate"""

MAX_PLATES = 4
"""Maximum number of plates supported"""

MAX_TOTAL_WELLS = WELLS_PER_PLATE * MAX_PLATES
"""Maximum total wells across all plates (384)"""

ROWS_PER_PLATE = 8
"""Number of rows in a 96-well plate (A-H)"""

COLS_PER_PLATE = 12
"""Number of columns in a 96-well plate (1-12)"""

PLATE_384_ROWS = 16
"""Number of rows in a 384-well plate (A-P)"""

PLATE_384_COLS = 24
"""Number of columns in a 384-well plate (1-24)"""


# ============================================================================
# CATEGORICAL FACTORS
# ============================================================================

CATEGORICAL_FACTORS = ("buffer pH", "detergent", "reducing_agent")
"""Factors that are categorical (non-numeric) or have special handling"""

NONE_VALUES = ('none', '0', '', 'nan')
"""String values that represent None/empty/null"""


# ============================================================================
# FACTOR CONSTRAINTS
# ============================================================================

FACTOR_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
    "buffer pH": {
        "min": 1.0,
        "max": 14.0,
        "type": "numeric",
        "description": "pH value from 1 to 14"
    },
    "glycerol": {
        "min": 0.0,
        "max": 100.0,
        "type": "percentage",
        "description": "Percentage (0-100%)"
    },
    "dmso": {
        "min": 0.0,
        "max": 100.0,
        "type": "percentage",
        "description": "Percentage (0-100%)"
    },
    "detergent_concentration": {
        "min": 0.0,
        "max": 100.0,
        "type": "percentage",
        "description": "Percentage (0-100%)"
    },
    "buffer_concentration": {
        "min": 0.0,
        "max": 10000.0,
        "type": "concentration",
        "description": "Concentration in mM"
    },
    "nacl": {
        "min": 0.0,
        "max": 10000.0,
        "type": "concentration",
        "description": "Concentration in mM"
    },
    "mgcl2": {
        "min": 0.0,
        "max": 10000.0,
        "type": "concentration",
        "description": "Concentration in mM"
    },
    "cacl2": {
        "min": 0.0,
        "max": 10000.0,
        "type": "concentration",
        "description": "Concentration in mM"
    },
    "edta": {
        "min": 0.0,
        "max": 10000.0,
        "type": "concentration",
        "description": "Concentration in mM"
    },
    "reducing_agent_concentration": {
        "min": 0.0,
        "max": 10000.0,
        "type": "concentration",
        "description": "Concentration in mM"
    }
}


# ============================================================================
# UNIT OPTIONS
# ============================================================================

UNIT_OPTIONS: Dict[str, List[str]] = {
    "buffer_concentration": ["mM", "M", "µM"],
    "nacl": ["mM", "M", "µM"],
    "mgcl2": ["mM", "M", "µM"],
    "cacl2": ["mM", "M", "µM"],
    "edta": ["mM", "M", "µM"],
    "reducing_agent_concentration": ["mM", "M", "µM"],
    "glycerol": ["%", "v/v"],
    "dmso": ["%", "v/v"],
    "detergent_concentration": ["%", "w/v", "v/v"],
}


# ============================================================================
# DESIGN TYPE CONFIGURATION
# ============================================================================

DESIGN_TYPES: Dict[str, Dict[str, Any]] = {
    "full_factorial": {
        "display_name": "Full Factorial (all combinations)",
        "min_factors": 1,
        "max_factors": None,
        "supports_categorical": True,
        "requires_pydoe3": False,
        "requires_smt": False,
        "description": "Tests all possible combinations of factor levels",
        "parameters": []
    },
    "lhs": {
        "display_name": "Latin Hypercube (space-filling)",
        "min_factors": 2,
        "max_factors": None,
        "supports_categorical": True,
        "requires_pydoe3": True,
        "requires_smt": True,
        "description": "Space-filling design with configurable sample size",
        "parameters": ["sample_size", "use_smt"]
    },
    "d_optimal": {
        "display_name": "D-Optimal (model-optimized)",
        "min_factors": 2,
        "max_factors": None,
        "supports_categorical": True,
        "requires_pydoe3": False,
        "requires_smt": False,
        "description": "Optimal design for specified model with exact run count",
        "parameters": ["n_samples", "model_type"]
    },
    "fractional": {
        "display_name": "2-Level Fractional Factorial (screening)",
        "min_factors": 3,
        "max_factors": None,
        "supports_categorical": False,
        "requires_pydoe3": True,
        "requires_smt": False,
        "description": "Efficient screening for many factors with 2 levels each",
        "parameters": ["resolution"]
    },
    "plackett_burman": {
        "display_name": "Plackett-Burman (efficient screening)",
        "min_factors": 2,
        "max_factors": None,
        "supports_categorical": False,
        "requires_pydoe3": True,
        "requires_smt": False,
        "description": "Ultra-efficient screening for many factors",
        "parameters": []
    },
    "central_composite": {
        "display_name": "Central Composite (optimization)",
        "min_factors": 2,
        "max_factors": None,
        "supports_categorical": True,
        "requires_pydoe3": True,
        "requires_smt": False,
        "description": "Response surface optimization with star points",
        "parameters": ["ccd_type", "center_points"]
    },
    "box_behnken": {
        "display_name": "Box-Behnken (optimization)",
        "min_factors": 3,
        "max_factors": None,
        "supports_categorical": True,
        "requires_pydoe3": True,
        "requires_smt": False,
        "description": "Response surface optimization without extreme corners",
        "parameters": ["center_points"]
    }
}


# ============================================================================
# DESIGN PARAMETERS
# ============================================================================

FRACTIONAL_RESOLUTION_OPTIONS = ["III", "IV", "V"]
"""Resolution options for fractional factorial designs"""

CCD_TYPE_OPTIONS = ["circumscribed", "inscribed", "faced"]
"""Central Composite Design type options"""

D_OPTIMAL_MODEL_OPTIONS = ["linear", "interactions", "quadratic"]
"""Model type options for D-Optimal designs"""

DEFAULT_CENTER_POINTS = 3
"""Default number of center points for response surface designs"""

MIN_SAMPLE_SIZE = 8
"""Minimum sample size for LHS designs"""

MAX_SAMPLE_SIZE = MAX_TOTAL_WELLS
"""Maximum sample size for LHS designs"""


# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

EXCEL_COLORS = {
    "header_blue": "2196F3",
    "header_green": "4CAF50",
    "header_orange": "FF9800",
    "header_purple": "9C27B0"
}
"""Color codes for Excel export headers"""

DEFAULT_FINAL_VOLUME = 200.0
"""Default final volume in microliters (µL)"""


# ============================================================================
# VALIDATION MESSAGES
# ============================================================================

ERROR_MESSAGES = {
    "no_factors": "No factors defined. Please add at least one factor.",
    "max_wells_exceeded": f"Sample size cannot exceed {MAX_TOTAL_WELLS} ({MAX_PLATES} plates of {WELLS_PER_PLATE} wells each).",
    "negative_water": "Impossible design detected: negative water volumes required",
    "invalid_ph": "Invalid pH value: must be between 1 and 14",
    "invalid_percentage": "Invalid percentage: must be between 0 and 100",
    "invalid_concentration": "Invalid concentration: cannot be negative",
    "missing_stock": "Stock concentration is required for this factor",
    "invalid_resolution": "Resolution must be one of: III, IV, or V",
    "min_factors_not_met": "This design type requires at least {min_factors} factors",
    "missing_pydoe3": "This design type requires pyDOE3 library",
    "missing_smt": "Optimized LHS requires SMT library"
}


# ============================================================================
# FILE NAMING
# ============================================================================

DEFAULT_FILENAME_PREFIX = "experimental_design"
"""Default prefix for export filenames"""

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
"""Format for timestamps in filenames"""
