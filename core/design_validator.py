"""
Design Validation Service
Handles validation for factors, designs, and input constraints
"""

from typing import Dict, List, Optional, Tuple, Any
from config.design_config import (
    FACTOR_CONSTRAINTS,
    MAX_TOTAL_WELLS,
    MAX_PLATES,
    WELLS_PER_PLATE,
    MIN_SAMPLE_SIZE,
    MAX_SAMPLE_SIZE,
    FRACTIONAL_RESOLUTION_OPTIONS,
    ERROR_MESSAGES,
    DESIGN_TYPES,
    CATEGORICAL_FACTORS
)


class DesignValidator:
    """Validator for experimental design parameters and constraints"""

    @staticmethod
    def validate_factor_value(factor_name: str, value: float) -> Tuple[bool, str]:
        """
        Validate that a factor value is within acceptable range.

        Args:
            factor_name: Name of the factor
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if valid, False otherwise
            - error_message: Description of error if invalid, empty string otherwise

        Examples:
            >>> DesignValidator.validate_factor_value("buffer pH", 7.0)
            (True, '')
            >>> DesignValidator.validate_factor_value("buffer pH", 15.0)
            (False, 'Invalid pH value: must be between 1 and 14')
        """
        if factor_name not in FACTOR_CONSTRAINTS:
            # No constraints defined, allow any value
            return True, ""

        constraint = FACTOR_CONSTRAINTS[factor_name]
        min_val = constraint.get("min", float('-inf'))
        max_val = constraint.get("max", float('inf'))

        if not (min_val <= value <= max_val):
            factor_type = constraint.get("type", "value")

            if factor_type == "numeric" and factor_name == "buffer pH":
                return False, ERROR_MESSAGES["invalid_ph"]
            elif factor_type == "percentage":
                return False, ERROR_MESSAGES["invalid_percentage"]
            elif factor_type == "concentration":
                if value < 0:
                    return False, ERROR_MESSAGES["invalid_concentration"]
                else:
                    return False, f"Value {value} exceeds maximum allowed ({max_val})"
            else:
                return False, f"Value {value} must be between {min_val} and {max_val}"

        return True, ""

    @staticmethod
    def validate_factor_levels(
        factor_name: str,
        levels: List[str]
    ) -> Tuple[bool, str]:
        """
        Validate that factor levels are acceptable.

        Args:
            factor_name: Name of the factor
            levels: List of level values (as strings)

        Returns:
            Tuple of (is_valid, error_message)

        Examples:
            >>> DesignValidator.validate_factor_levels("nacl", ["100", "200"])
            (True, '')
            >>> DesignValidator.validate_factor_levels("buffer pH", [])
            (False, 'At least one level is required')
        """
        if not levels:
            return False, "At least one level is required"

        # For categorical factors, allow any non-empty strings
        if factor_name in CATEGORICAL_FACTORS:
            return True, ""

        # For numeric factors, validate each level
        for level_str in levels:
            try:
                value = float(level_str)
                is_valid, error_msg = DesignValidator.validate_factor_value(factor_name, value)
                if not is_valid:
                    return False, f"Level '{level_str}': {error_msg}"
            except ValueError:
                return False, f"Level '{level_str}' is not a valid number"

        return True, ""

    @staticmethod
    def validate_sample_size(num_samples: int) -> Tuple[bool, str]:
        """
        Validate that sample size is within acceptable range.

        Args:
            num_samples: Number of samples

        Returns:
            Tuple of (is_valid, error_message)

        Examples:
            >>> DesignValidator.validate_sample_size(100)
            (True, '')
            >>> DesignValidator.validate_sample_size(500)
            (False, 'Sample size cannot exceed 384 (4 plates of 96 wells each).')
        """
        if num_samples <= 0:
            return False, "Sample size must be positive"

        if num_samples > MAX_TOTAL_WELLS:
            return False, ERROR_MESSAGES["max_wells_exceeded"]

        return True, ""

    @staticmethod
    def validate_lhs_parameters(
        num_samples: int,
        num_factors: int
    ) -> Tuple[bool, str]:
        """
        Validate Latin Hypercube Sampling parameters.

        Args:
            num_samples: Number of samples
            num_factors: Number of factors

        Returns:
            Tuple of (is_valid, error_message)

        Examples:
            >>> DesignValidator.validate_lhs_parameters(50, 3)
            (True, '')
            >>> DesignValidator.validate_lhs_parameters(5, 3)
            (False, 'LHS sample size must be at least 8')
        """
        if num_samples < MIN_SAMPLE_SIZE:
            return False, f"LHS sample size must be at least {MIN_SAMPLE_SIZE}"

        if num_samples > MAX_SAMPLE_SIZE:
            return False, ERROR_MESSAGES["max_wells_exceeded"]

        if num_factors < 2:
            return False, "LHS requires at least 2 factors"

        return True, ""

    @staticmethod
    def validate_fractional_factorial(
        num_factors: int,
        resolution: str
    ) -> Tuple[bool, str]:
        """
        Validate fractional factorial parameters.

        Args:
            num_factors: Number of factors
            resolution: Design resolution ("III", "IV", or "V")

        Returns:
            Tuple of (is_valid, error_message)

        Examples:
            >>> DesignValidator.validate_fractional_factorial(5, "IV")
            (True, '')
            >>> DesignValidator.validate_fractional_factorial(2, "IV")
            (False, 'Fractional factorial requires at least 3 factors')
        """
        if num_factors < 3:
            return False, "Fractional factorial requires at least 3 factors"

        if resolution not in FRACTIONAL_RESOLUTION_OPTIONS:
            return False, ERROR_MESSAGES["invalid_resolution"]

        return True, ""

    @staticmethod
    def validate_design_type_requirements(
        design_type: str,
        num_factors: int,
        has_pydoe3: bool = False,
        has_smt: bool = False
    ) -> Tuple[bool, str]:
        """
        Validate that requirements for a design type are met.

        Args:
            design_type: Type of design (e.g., "lhs", "fractional")
            num_factors: Number of factors
            has_pydoe3: Whether pyDOE3 is available
            has_smt: Whether SMT is available

        Returns:
            Tuple of (is_valid, error_message)

        Examples:
            >>> DesignValidator.validate_design_type_requirements("lhs", 3, True, True)
            (True, '')
            >>> DesignValidator.validate_design_type_requirements("lhs", 3, False, True)
            (False, 'This design type requires pyDOE3 library')
        """
        if design_type not in DESIGN_TYPES:
            return False, f"Unknown design type: {design_type}"

        design_config = DESIGN_TYPES[design_type]

        # Check minimum factors
        min_factors = design_config.get("min_factors")
        if min_factors and num_factors < min_factors:
            msg = ERROR_MESSAGES["min_factors_not_met"].format(min_factors=min_factors)
            return False, msg

        # Check library requirements
        if design_config.get("requires_pydoe3") and not has_pydoe3:
            return False, ERROR_MESSAGES["missing_pydoe3"]

        if design_type == "lhs" and design_config.get("requires_smt") and not has_smt:
            # SMT is optional for LHS, just warn
            pass

        return True, ""

    @staticmethod
    def validate_stock_concentration(
        factor_name: str,
        stock_conc: Optional[float]
    ) -> Tuple[bool, str]:
        """
        Validate stock concentration for a factor.

        Args:
            factor_name: Name of the factor
            stock_conc: Stock concentration value (can be None)

        Returns:
            Tuple of (is_valid, error_message)

        Examples:
            >>> DesignValidator.validate_stock_concentration("nacl", 5000.0)
            (True, '')
            >>> DesignValidator.validate_stock_concentration("nacl", -10.0)
            (False, 'Stock concentration cannot be negative')
        """
        # Categorical factors don't need stock concentration
        if factor_name in CATEGORICAL_FACTORS:
            return True, ""

        if stock_conc is None:
            return False, ERROR_MESSAGES["missing_stock"]

        if stock_conc < 0:
            return False, "Stock concentration cannot be negative"

        if stock_conc == 0:
            return False, "Stock concentration cannot be zero"

        return True, ""

    @staticmethod
    def check_categorical_concentration_pairing(
        factors: Dict[str, List[str]]
    ) -> Tuple[bool, List[str]]:
        """
        Check that categorical factors have corresponding concentration factors.

        Args:
            factors: Dictionary of factor_name → levels

        Returns:
            Tuple of (is_valid, warnings)
            - is_valid: Always True (warnings only)
            - warnings: List of warning messages

        Examples:
            >>> factors = {"detergent": ["Tween-20"], "detergent_concentration": ["0.1"]}
            >>> valid, warnings = DesignValidator.check_categorical_concentration_pairing(factors)
            >>> valid
            True
            >>> warnings
            []
        """
        warnings = []

        if "detergent" in factors and "detergent_concentration" not in factors:
            warnings.append(
                "Detergent factor found without detergent_concentration. "
                "Consider adding detergent_concentration factor."
            )

        if "reducing_agent" in factors and "reducing_agent_concentration" not in factors:
            warnings.append(
                "Reducing agent factor found without reducing_agent_concentration. "
                "Consider adding reducing_agent_concentration factor."
            )

        if "buffer pH" in factors and "buffer_concentration" not in factors:
            warnings.append(
                "Buffer pH factor found without buffer_concentration. "
                "Consider adding buffer_concentration factor."
            )

        return True, warnings


class CategoricalValidator:
    """Validator for categorical factors and their combinations"""

    @staticmethod
    def filter_invalid_categorical_combinations(
        combinations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter out invalid categorical factor combinations.

        Rules:
        1. If detergent is "None", detergent_concentration must be 0
        2. If reducing_agent is "None", reducing_agent_concentration must be 0
        3. Vice versa: if concentration is 0, categorical should be "None"

        Args:
            combinations: List of factor combination dictionaries

        Returns:
            Filtered list of valid combinations

        Examples:
            >>> combos = [
            ...     {"detergent": "Tween-20", "detergent_concentration": 0.1},
            ...     {"detergent": "None", "detergent_concentration": 0.1},  # Invalid
            ... ]
            >>> valid = CategoricalValidator.filter_invalid_categorical_combinations(combos)
            >>> len(valid)
            1
        """
        valid_combinations = []

        for combo in combinations:
            if CategoricalValidator._is_valid_combination(combo):
                valid_combinations.append(combo)

        return valid_combinations

    @staticmethod
    def _is_valid_combination(combination: Dict[str, Any]) -> bool:
        """
        Check if a single combination is valid.

        Args:
            combination: Factor combination dictionary

        Returns:
            True if valid, False otherwise
        """
        # Check detergent pairing
        if "detergent" in combination and "detergent_concentration" in combination:
            det = str(combination["detergent"]).strip()
            det_conc = float(combination["detergent_concentration"])

            det_is_none = det.lower() in ('none', '0', '', 'nan')

            # If detergent is None but concentration is not 0, invalid
            if det_is_none and det_conc != 0:
                return False

            # If detergent is not None but concentration is 0, invalid
            if not det_is_none and det_conc == 0:
                return False

        # Check reducing agent pairing
        if "reducing_agent" in combination and "reducing_agent_concentration" in combination:
            agent = str(combination["reducing_agent"]).strip()
            agent_conc = float(combination["reducing_agent_concentration"])

            agent_is_none = agent.lower() in ('none', '0', '', 'nan')

            # If agent is None but concentration is not 0, invalid
            if agent_is_none and agent_conc != 0:
                return False

            # If agent is not None but concentration is 0, invalid
            if not agent_is_none and agent_conc == 0:
                return False

        return True
