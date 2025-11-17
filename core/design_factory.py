"""
Design Factory Service
Orchestrates experimental design generation for various design types
"""

import itertools
import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from config.design_config import (
    CATEGORICAL_FACTORS,
    DESIGN_TYPES,
    DEFAULT_CENTER_POINTS
)


class DesignFactory:
    """
    Factory for creating experimental designs.

    Supports:
    - Full Factorial
    - Latin Hypercube Sampling (LHS)
    - Fractional Factorial (2-level)
    - Plackett-Burman
    - Central Composite Design (CCD)
    - Box-Behnken
    """

    def __init__(self, has_pydoe3: bool = False, has_smt: bool = False):
        """
        Initialize design factory.

        Args:
            has_pydoe3: Whether pyDOE3 library is available
            has_smt: Whether SMT library is available
        """
        self.has_pydoe3 = has_pydoe3
        self.has_smt = has_smt

    def create_design(
        self,
        design_type: str,
        factors: Dict[str, List[str]],
        **params
    ) -> List[Dict[str, Any]]:
        """
        Create an experimental design.

        Args:
            design_type: Type of design ("full_factorial", "lhs", etc.)
            factors: Dictionary of factor_name → list of levels
            **params: Design-specific parameters:
                - n_samples: Number of samples (for LHS)
                - resolution: Resolution level (for fractional factorial)
                - ccd_type: CCD type (for central composite)
                - center_points: Number of center points (for CCD, Box-Behnken)
                - use_smt: Use SMT for optimized LHS (for LHS)

        Returns:
            List of dictionaries, each representing a design point

        Raises:
            ValueError: If design type is unknown or parameters are invalid

        Examples:
            >>> factory = DesignFactory(has_pydoe3=True)
            >>> factors = {"temp": ["20", "30"], "pH": ["6", "7"]}
            >>> design = factory.create_design("full_factorial", factors)
            >>> len(design)
            4
        """
        if design_type == "full_factorial":
            return self._generate_full_factorial(factors)
        elif design_type == "lhs":
            n_samples = params.get("n_samples", 50)
            use_smt = params.get("use_smt", False)
            return self._generate_lhs(factors, n_samples, use_smt)
        elif design_type == "fractional":
            resolution = params.get("resolution", "IV")
            return self._generate_fractional_factorial(factors, resolution)
        elif design_type == "plackett_burman":
            return self._generate_plackett_burman(factors)
        elif design_type == "central_composite":
            ccd_type = params.get("ccd_type", "circumscribed")
            center_points = params.get("center_points", DEFAULT_CENTER_POINTS)
            return self._generate_central_composite(factors, ccd_type, center_points)
        elif design_type == "box_behnken":
            center_points = params.get("center_points", DEFAULT_CENTER_POINTS)
            return self._generate_box_behnken(factors, center_points)
        else:
            raise ValueError(f"Unknown design type: {design_type}")

    def _generate_full_factorial(
        self,
        factors: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Generate full factorial design (all combinations).

        Args:
            factors: Dictionary of factor_name → list of levels

        Returns:
            List of design points

        Examples:
            >>> factory = DesignFactory()
            >>> factors = {"A": ["1", "2"], "B": ["x", "y"]}
            >>> design = factory._generate_full_factorial(factors)
            >>> len(design)
            4
        """
        factor_names = list(factors.keys())
        level_lists = [factors[f] for f in factor_names]
        combinations = list(itertools.product(*level_lists))

        design_points = []
        for combo in combinations:
            point = {factor_names[i]: combo[i] for i in range(len(factor_names))}
            design_points.append(point)

        return design_points

    def _generate_lhs(
        self,
        factors: Dict[str, List[str]],
        n_samples: int,
        use_smt: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate Latin Hypercube Sampling design.

        Args:
            factors: Dictionary of factor_name → list of levels
            n_samples: Number of samples to generate
            use_smt: Use SMT library for optimized LHS

        Returns:
            List of design points

        Raises:
            ImportError: If required libraries are not available
        """
        factor_names = list(factors.keys())

        # Separate numeric and categorical factors
        numeric_factor_names = []
        categorical_factor_names = []
        for fn in factor_names:
            if fn in CATEGORICAL_FACTORS:
                categorical_factor_names.append(fn)
            else:
                numeric_factor_names.append(fn)

        n_numeric = len(numeric_factor_names)

        # Generate numeric combinations
        if n_numeric > 0:
            if use_smt and self.has_smt:
                numeric_combinations = self._generate_lhs_smt(
                    numeric_factor_names, factors, n_samples
                )
            elif self.has_pydoe3:
                numeric_combinations = self._generate_lhs_pydoe3(
                    numeric_factor_names, factors, n_samples
                )
            else:
                raise ImportError(
                    "pyDOE3 is required for Latin Hypercube Sampling. "
                    "Install with: pip install pyDOE3"
                )
        else:
            # No numeric factors
            numeric_combinations = [[] for _ in range(n_samples)]

        # Generate categorical combinations
        if categorical_factor_names:
            categorical_combinations = self._distribute_categorical_factors(
                categorical_factor_names, factors, n_samples
            )
        else:
            categorical_combinations = [[] for _ in range(n_samples)]

        # Combine numeric and categorical
        design_points = []
        for i in range(n_samples):
            point = {}
            for j, fn in enumerate(numeric_factor_names):
                point[fn] = numeric_combinations[i][j]
            for j, fn in enumerate(categorical_factor_names):
                point[fn] = categorical_combinations[i][j]
            design_points.append(point)

        return design_points

    def _generate_lhs_pydoe3(
        self,
        factor_names: List[str],
        factors: Dict[str, List[str]],
        n_samples: int
    ) -> List[List[str]]:
        """Generate LHS using pyDOE3."""
        import pyDOE3

        n_factors = len(factor_names)
        lhs_design = pyDOE3.lhs(n=n_factors, samples=n_samples, criterion='center')

        combinations = []
        for sample in lhs_design:
            combo = []
            for i, factor_name in enumerate(factor_names):
                levels = factors[factor_name]
                # Map [0,1] to level index
                level_idx = int(sample[i] * len(levels))
                level_idx = min(level_idx, len(levels) - 1)
                combo.append(levels[level_idx])
            combinations.append(combo)

        return combinations

    def _generate_lhs_smt(
        self,
        factor_names: List[str],
        factors: Dict[str, List[str]],
        n_samples: int
    ) -> List[List[str]]:
        """Generate optimized LHS using SMT."""
        from smt.sampling_methods import LHS

        # Build xlimits for SMT
        xlimits = []
        for factor_name in factor_names:
            levels = factors[factor_name]
            numeric_levels = [float(lv) for lv in levels]
            xlimits.append([min(numeric_levels), max(numeric_levels)])

        xlimits = np.array(xlimits)

        # Generate optimized LHS
        sampling = LHS(xlimits=xlimits, criterion='maximin')
        lhs_design = sampling(n_samples)

        # Map continuous values to discrete levels
        combinations = []
        for sample in lhs_design:
            combo = []
            for i, factor_name in enumerate(factor_names):
                levels = factors[factor_name]
                numeric_levels = [float(lv) for lv in levels]
                # Find closest level
                value = sample[i]
                closest_idx = min(
                    range(len(numeric_levels)),
                    key=lambda j: abs(numeric_levels[j] - value)
                )
                combo.append(levels[closest_idx])
            combinations.append(combo)

        return combinations

    def _distribute_categorical_factors(
        self,
        factor_names: List[str],
        factors: Dict[str, List[str]],
        n_samples: int
    ) -> List[List[str]]:
        """
        Distribute categorical factors evenly across samples.

        Args:
            factor_names: List of categorical factor names
            factors: Factor dictionary
            n_samples: Number of samples

        Returns:
            List of categorical combinations
        """
        level_lists = [factors[fn] for fn in factor_names]
        all_combos = list(itertools.product(*level_lists))

        # Cycle through combinations to match n_samples
        combinations = []
        for i in range(n_samples):
            combo_idx = i % len(all_combos)
            combinations.append(list(all_combos[combo_idx]))

        return combinations

    def _generate_fractional_factorial(
        self,
        factors: Dict[str, List[str]],
        resolution: str
    ) -> List[Dict[str, Any]]:
        """
        Generate 2-level fractional factorial design.

        Args:
            factors: Dictionary of factor_name → list of levels (must have exactly 2 levels each)
            resolution: Resolution level ("III", "IV", or "V")

        Returns:
            List of design points

        Raises:
            ImportError: If pyDOE3 is not available
            ValueError: If factors don't have exactly 2 levels
        """
        if not self.has_pydoe3:
            raise ImportError(
                "pyDOE3 is required for fractional factorial designs. "
                "Install with: pip install pyDOE3"
            )

        import pyDOE3

        factor_names = list(factors.keys())
        n_factors = len(factor_names)

        # Validate all factors have 2 levels
        for fn in factor_names:
            if len(factors[fn]) != 2:
                raise ValueError(
                    f"Fractional factorial requires exactly 2 levels per factor. "
                    f"Factor '{fn}' has {len(factors[fn])} levels."
                )

        # Build generator string
        gen_string = f"{n_factors}-{resolution}"
        design = pyDOE3.fracfact(gen_string)

        # Map from [-1, 1] to actual levels
        design_points = []
        for row in design:
            point = {}
            for i, factor_name in enumerate(factor_names):
                # -1 → level 0, +1 → level 1
                level_idx = 0 if row[i] < 0 else 1
                point[factor_name] = factors[factor_name][level_idx]
            design_points.append(point)

        return design_points

    def _generate_plackett_burman(
        self,
        factors: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Generate Plackett-Burman design.

        Args:
            factors: Dictionary of factor_name → list of levels (must have exactly 2 levels each)

        Returns:
            List of design points

        Raises:
            ImportError: If pyDOE3 is not available
            ValueError: If factors don't have exactly 2 levels
        """
        if not self.has_pydoe3:
            raise ImportError(
                "pyDOE3 is required for Plackett-Burman designs. "
                "Install with: pip install pyDOE3"
            )

        import pyDOE3

        factor_names = list(factors.keys())
        n_factors = len(factor_names)

        # Validate all factors have 2 levels
        for fn in factor_names:
            if len(factors[fn]) != 2:
                raise ValueError(
                    f"Plackett-Burman requires exactly 2 levels per factor. "
                    f"Factor '{fn}' has {len(factors[fn])} levels."
                )

        design = pyDOE3.pbdesign(n_factors)

        # Map from [-1, 1] to actual levels
        design_points = []
        for row in design:
            point = {}
            for i, factor_name in enumerate(factor_names):
                level_idx = 0 if row[i] < 0 else 1
                point[factor_name] = factors[factor_name][level_idx]
            design_points.append(point)

        return design_points

    def _generate_central_composite(
        self,
        factors: Dict[str, List[str]],
        ccd_type: str = "circumscribed",
        center_points: int = DEFAULT_CENTER_POINTS
    ) -> List[Dict[str, Any]]:
        """
        Generate Central Composite Design.

        Args:
            factors: Dictionary of factor_name → list of levels
            ccd_type: Type of CCD ("circumscribed", "inscribed", or "faced")
            center_points: Number of center point replicates

        Returns:
            List of design points

        Raises:
            ImportError: If pyDOE3 is not available
        """
        if not self.has_pydoe3:
            raise ImportError(
                "pyDOE3 is required for Central Composite designs. "
                "Install with: pip install pyDOE3"
            )

        import pyDOE3

        factor_names = list(factors.keys())

        # Separate numeric and categorical
        numeric_factor_names = [fn for fn in factor_names if fn not in CATEGORICAL_FACTORS]
        categorical_factor_names = [fn for fn in factor_names if fn in CATEGORICAL_FACTORS]

        n_numeric = len(numeric_factor_names)

        if n_numeric < 2:
            raise ValueError("Central Composite Design requires at least 2 numeric factors")

        # Generate CCD for numeric factors
        design = pyDOE3.ccdesign(
            n=n_numeric,
            center=(0, center_points),
            face=ccd_type
        )

        # Map coded values to actual levels
        numeric_combinations = []
        for row in design:
            combo = []
            for i, factor_name in enumerate(numeric_factor_names):
                levels = factors[factor_name]
                numeric_levels = [float(lv) for lv in levels]
                min_val = min(numeric_levels)
                max_val = max(numeric_levels)
                mid_val = (min_val + max_val) / 2
                range_val = (max_val - min_val) / 2

                # Decode from [-1, 1] or beyond to actual value
                actual_value = mid_val + row[i] * range_val

                # Find closest level
                closest_idx = min(
                    range(len(numeric_levels)),
                    key=lambda j: abs(numeric_levels[j] - actual_value)
                )
                combo.append(levels[closest_idx])
            numeric_combinations.append(combo)

        # Handle categorical factors
        if categorical_factor_names:
            n_samples = len(numeric_combinations)
            categorical_combinations = self._distribute_categorical_factors(
                categorical_factor_names, factors, n_samples
            )
        else:
            categorical_combinations = [[] for _ in range(len(numeric_combinations))]

        # Combine
        design_points = []
        for i in range(len(numeric_combinations)):
            point = {}
            for j, fn in enumerate(numeric_factor_names):
                point[fn] = numeric_combinations[i][j]
            for j, fn in enumerate(categorical_factor_names):
                point[fn] = categorical_combinations[i][j]
            design_points.append(point)

        return design_points

    def _generate_box_behnken(
        self,
        factors: Dict[str, List[str]],
        center_points: int = DEFAULT_CENTER_POINTS
    ) -> List[Dict[str, Any]]:
        """
        Generate Box-Behnken design.

        Args:
            factors: Dictionary of factor_name → list of levels (need at least 3 levels each)
            center_points: Number of center point replicates

        Returns:
            List of design points

        Raises:
            ImportError: If pyDOE3 is not available
            ValueError: If insufficient factors or levels
        """
        if not self.has_pydoe3:
            raise ImportError(
                "pyDOE3 is required for Box-Behnken designs. "
                "Install with: pip install pyDOE3"
            )

        import pyDOE3

        factor_names = list(factors.keys())

        # Separate numeric and categorical
        numeric_factor_names = [fn for fn in factor_names if fn not in CATEGORICAL_FACTORS]
        categorical_factor_names = [fn for fn in factor_names if fn in CATEGORICAL_FACTORS]

        n_numeric = len(numeric_factor_names)

        if n_numeric < 3:
            raise ValueError("Box-Behnken design requires at least 3 numeric factors")

        # Generate Box-Behnken design
        design = pyDOE3.bbdesign(n=n_numeric, center=center_points)

        # Map coded values [-1, 0, 1] to actual levels
        numeric_combinations = []
        for row in design:
            combo = []
            for i, factor_name in enumerate(numeric_factor_names):
                levels = factors[factor_name]
                numeric_levels = [float(lv) for lv in levels]

                if len(numeric_levels) < 3:
                    raise ValueError(
                        f"Box-Behnken requires at least 3 levels. "
                        f"Factor '{factor_name}' has {len(numeric_levels)} levels."
                    )

                # Map [-1, 0, 1] to [min, mid, max]
                min_val = min(numeric_levels)
                max_val = max(numeric_levels)
                mid_val = (min_val + max_val) / 2

                coded_val = row[i]
                if coded_val < -0.5:
                    actual_value = min_val
                elif coded_val > 0.5:
                    actual_value = max_val
                else:
                    actual_value = mid_val

                # Find closest level
                closest_idx = min(
                    range(len(numeric_levels)),
                    key=lambda j: abs(numeric_levels[j] - actual_value)
                )
                combo.append(levels[closest_idx])
            numeric_combinations.append(combo)

        # Handle categorical factors
        if categorical_factor_names:
            n_samples = len(numeric_combinations)
            categorical_combinations = self._distribute_categorical_factors(
                categorical_factor_names, factors, n_samples
            )
        else:
            categorical_combinations = [[] for _ in range(len(numeric_combinations))]

        # Combine
        design_points = []
        for i in range(len(numeric_combinations)):
            point = {}
            for j, fn in enumerate(numeric_factor_names):
                point[fn] = numeric_combinations[i][j]
            for j, fn in enumerate(categorical_factor_names):
                point[fn] = categorical_combinations[i][j]
            design_points.append(point)

        return design_points
