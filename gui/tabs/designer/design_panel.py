#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Design Panel Mixin for Experimental Modeling Suite

This module provides methods for generating various experimental designs
including Latin Hypercube Sampling, Fractional Factorial, Plackett-Burman,
Central Composite, and Box-Behnken designs.

The mixin requires:
- self.model: FactorModel instance for factor management
- self.optimize_lhs_var: BooleanVar for SMT optimization preference
"""

import itertools
import numpy as np
from typing import Dict, List, Tuple

from utils.constants import AVAILABLE_FACTORS

# Optional pyDOE3 for advanced designs
try:
    import pyDOE3
    HAS_PYDOE3 = True
except Exception:
    HAS_PYDOE3 = False

# Optional SMT for optimized LHS
try:
    from smt.sampling_methods import LHS
    HAS_SMT = True
except Exception:
    HAS_SMT = False


class DesignPanelMixin:
    """Mixin class providing experimental design generation methods.

    This mixin provides methods for generating various experimental designs
    including well position generation, design type generation, and filtering
    of categorical combinations.

    Attributes:
        model: FactorModel instance containing factors and their levels
        optimize_lhs_var: tkinter BooleanVar indicating whether to use SMT optimization

    Design Types:
        - Latin Hypercube Sampling (LHS)
        - 2-Level Fractional Factorial
        - Plackett-Burman
        - Central Composite Design (CCD)
        - Box-Behnken
    """

    def _generate_well_position(self, idx: int) -> Tuple[int, str]:
        """Generate 96-well plate and well position from index.

        Args:
            idx: Zero-based index of the well

        Returns:
            Tuple of (plate_number, well_position) where plate_number is 1-based
            and well_position is in format like 'A1', 'B12', etc.
        """
        plate_num = (idx // 96) + 1
        well_idx = idx % 96

        row = chr(ord('A') + (well_idx // 12))
        col = (well_idx % 12) + 1
        well_pos = f"{row}{col}"

        return plate_num, well_pos

    def _convert_96_to_384_well(self, plate_num: int, well_96: str) -> str:
        """Convert 96-well to 384-well position.

        Mapping rules:
        - Plate 1 -> cols 1-6, Plate 2 -> cols 7-12, etc.
        - Odd cols (1,3,5) use first row of pair (A->A, B->C, C->E)
        - Even cols (2,4,6) use second row (A->B, B->D, C->F)

        Args:
            plate_num: 1-based plate number
            well_96: Well position in 96-well format (e.g., 'B3')

        Returns:
            Well position in 384-well format (e.g., 'C2')
        """
        import math

        # Parse 96-well position (e.g., "B3" -> row=B, col=3)
        row_96 = well_96[0]  # Letter (A-H)
        col_96 = int(well_96[1:])  # Number (1-12)

        # Convert row letter to index (A=0, B=1, ..., H=7)
        row_96_index = ord(row_96) - ord('A')

        # Map to 384-well column: (plate-1)*6 + ceil(col/2)
        col_384 = (plate_num - 1) * 6 + math.ceil(col_96 / 2)

        # Map 96-well row to 384-well row based on column parity
        # Each 96 row -> 2 consecutive 384 rows (odd col->first, even col->second)
        if col_96 % 2 == 1:  # Odd column
            row_384_index = row_96_index * 2
        else:  # Even column
            row_384_index = row_96_index * 2 + 1

        # Convert back to letter (A=0, B=1, ..., P=15)
        row_384 = chr(ord('A') + row_384_index)

        return f"{row_384}{col_384}"

    def _filter_categorical_combinations(self, combinations: List[Tuple], factor_names: List[str]) -> List[Tuple]:
        """Filter out illogical categorical-concentration pairings.

        Rules:
        - If detergent is None/0, detergent_concentration must be 0
        - If detergent has a value (Triton, etc.), detergent_concentration must be > 0
        - Same logic for reducing_agent and buffer pH

        Args:
            combinations: List of tuples representing factor combinations
            factor_names: List of factor names corresponding to tuple positions

        Returns:
            Filtered list of combinations with illogical pairings removed
        """
        filtered = []

        for combo in combinations:
            row_dict = {factor_names[i]: combo[i] for i in range(len(factor_names))}
            valid = True

            # Check detergent-concentration pairing
            if "detergent" in row_dict and "detergent_concentration" in row_dict:
                det = str(row_dict["detergent"]).strip()
                det_conc = float(row_dict["detergent_concentration"])

                # If detergent is None/empty, concentration must be 0
                if det.lower() in ['none', '0', '', 'nan']:
                    if det_conc != 0:
                        valid = False
                # If detergent has a value, concentration must be > 0
                else:
                    if det_conc == 0:
                        valid = False

            # Check reducing_agent-concentration pairing
            if "reducing_agent" in row_dict and "reducing_agent_concentration" in row_dict:
                agent = str(row_dict["reducing_agent"]).strip()
                agent_conc = float(row_dict["reducing_agent_concentration"])

                # If agent is None/empty, concentration must be 0
                if agent.lower() in ['none', '0', '', 'nan']:
                    if agent_conc != 0:
                        valid = False
                # If agent has a value, concentration must be > 0
                else:
                    if agent_conc == 0:
                        valid = False

            # Check buffer pH-concentration pairing
            if "buffer pH" in row_dict and "buffer_concentration" in row_dict:
                # pH is always defined if present, so concentration should always be > 0
                buffer_conc = float(row_dict["buffer_concentration"])
                if buffer_conc == 0:
                    valid = False

            if valid:
                filtered.append(combo)

        return filtered

    def _generate_lhs_design(self, factors: Dict[str, List[str]], n_samples: int) -> List[Tuple]:
        """Generate Latin Hypercube Sampling design using pyDOE3 or SMT.

        Args:
            factors: Dictionary of factor names to level lists
            n_samples: Number of samples to generate

        Returns:
            List of tuples representing combinations

        Raises:
            ImportError: If pyDOE3 is not available and SMT is not used
        """
        factor_names = list(factors.keys())
        n_factors = len(factor_names)

        # Identify categorical factors (non-numeric)
        categorical_factors = ["buffer pH", "detergent", "reducing_agent"]

        # Separate numeric and categorical factors
        numeric_factor_names = []
        categorical_factor_names = []
        for fn in factor_names:
            if fn in categorical_factors:
                categorical_factor_names.append(fn)
            else:
                numeric_factor_names.append(fn)

        n_numeric = len(numeric_factor_names)

        # Check if user wants optimized LHS with SMT
        use_smt = self.optimize_lhs_var.get() and HAS_SMT

        if n_numeric > 0:
            if use_smt:
                # Use SMT for optimized LHS with maximin criterion (numeric factors only)
                xlimits = []
                for factor_name in numeric_factor_names:
                    levels = factors[factor_name]
                    # Convert to float and get min/max range
                    numeric_levels = [float(lv) for lv in levels]
                    xlimits.append([min(numeric_levels), max(numeric_levels)])

                xlimits = np.array(xlimits)

                # Generate optimized LHS
                sampling = LHS(xlimits=xlimits, criterion='maximin')
                lhs_design = sampling(n_samples)

                # Map continuous values to discrete levels
                numeric_combinations = []
                for sample in lhs_design:
                    combo = []
                    for i, factor_name in enumerate(numeric_factor_names):
                        levels = factors[factor_name]
                        numeric_levels = [float(lv) for lv in levels]
                        # Find closest level to the continuous value
                        value = sample[i]
                        closest_idx = min(range(len(numeric_levels)),
                                        key=lambda j: abs(numeric_levels[j] - value))
                        combo.append(levels[closest_idx])
                    numeric_combinations.append(combo)

            else:
                # Use standard pyDOE3 LHS (numeric factors only)
                if not HAS_PYDOE3:
                    raise ImportError("pyDOE3 is required for Latin Hypercube Sampling. "
                                    "Install with: pip install pyDOE3")

                # Generate LHS design in [0,1] hypercube
                lhs_design = pyDOE3.lhs(n=n_numeric, samples=n_samples, criterion='center')

                # Map to actual factor levels
                numeric_combinations = []
                for sample in lhs_design:
                    combo = []
                    for i, factor_name in enumerate(numeric_factor_names):
                        levels = factors[factor_name]
                        # Map [0,1] to level index
                        level_idx = int(sample[i] * len(levels))
                        level_idx = min(level_idx, len(levels) - 1)  # Ensure within bounds
                        combo.append(levels[level_idx])
                    numeric_combinations.append(combo)
        else:
            # No numeric factors, create empty combinations
            numeric_combinations = [[] for _ in range(n_samples)]

        # Handle categorical factors - cycle through combinations evenly
        if categorical_factor_names:
            # Get all combinations of categorical factors
            categorical_level_lists = [factors[fn] for fn in categorical_factor_names]
            all_cat_combos = list(itertools.product(*categorical_level_lists))

            # Distribute evenly across samples
            categorical_combinations = []
            for i in range(n_samples):
                # Cycle through categorical combinations
                cat_idx = i % len(all_cat_combos)
                categorical_combinations.append(list(all_cat_combos[cat_idx]))
        else:
            categorical_combinations = [[] for _ in range(n_samples)]

        # Combine numeric and categorical factors in original order
        combinations = []
        for i in range(n_samples):
            combo = []
            numeric_idx = 0
            categorical_idx = 0
            for fn in factor_names:
                if fn in categorical_factors:
                    combo.append(categorical_combinations[i][categorical_idx])
                    categorical_idx += 1
                else:
                    combo.append(numeric_combinations[i][numeric_idx])
                    numeric_idx += 1
            combinations.append(tuple(combo))

        return combinations

    def _generate_fractional_factorial(self, factors: Dict[str, List[str]], resolution: str) -> List[Tuple]:
        """Generate 2-level fractional factorial design using pyDOE3.

        Args:
            factors: Dictionary of factor names to level lists (must have exactly 2 levels each)
            resolution: Resolution level ('III', 'IV', or 'V')

        Returns:
            List of tuples representing combinations

        Raises:
            ImportError: If pyDOE3 is not available
            ValueError: If factors don't have exactly 2 levels or invalid resolution
        """
        if not HAS_PYDOE3:
            raise ImportError("pyDOE3 is required for Fractional Factorial designs. "
                            "Install with: pip install pyDOE3")

        factor_names = list(factors.keys())
        n_factors = len(factor_names)

        # Validate that all factors have exactly 2 levels
        for fn in factor_names:
            if len(factors[fn]) != 2:
                raise ValueError(f"Fractional Factorial requires exactly 2 levels per factor. "
                               f"Factor '{AVAILABLE_FACTORS.get(fn, fn)}' has {len(factors[fn])} levels.")

        # Build generator string based on resolution
        # For pyDOE3.fracfact, we use notation like "a b c ab" for 4 factors
        if resolution == "III":
            # Resolution III: Main effects aliased with 2-factor interactions
            gen_string = " ".join([chr(97 + i) for i in range(n_factors)])  # "a b c d..."
        elif resolution == "IV":
            # Resolution IV: Main effects clear, 2-factor interactions may be aliased
            if n_factors <= 4:
                gen_string = " ".join([chr(97 + i) for i in range(n_factors)])
            else:
                # Add some interaction terms
                gen_string = " ".join([chr(97 + i) for i in range(min(n_factors, 6))])
        elif resolution == "V":
            # Resolution V: Main effects and 2-factor interactions clear
            gen_string = " ".join([chr(97 + i) for i in range(n_factors)])
        else:
            raise ValueError(f"Invalid resolution: {resolution}. Must be III, IV, or V.")

        # Generate fractional factorial design
        design = pyDOE3.fracfact(gen_string)

        # Convert from -1/+1 coding to actual factor levels
        combinations = []
        for row in design:
            combo = []
            for i, factor_name in enumerate(factor_names):
                levels = factors[factor_name]
                # Map -1 to low level (index 0), +1 to high level (index 1)
                level_idx = 0 if row[i] < 0 else 1
                combo.append(levels[level_idx])
            combinations.append(tuple(combo))

        return combinations

    def _generate_plackett_burman(self, factors: Dict[str, List[str]]) -> List[Tuple]:
        """Generate Plackett-Burman screening design using pyDOE3.

        Plackett-Burman designs are highly efficient screening designs that
        require only N+1 runs for N factors (where N+1 is a multiple of 4).

        Args:
            factors: Dictionary of factor names to level lists (must have exactly 2 levels each)

        Returns:
            List of tuples representing combinations

        Raises:
            ImportError: If pyDOE3 is not available
            ValueError: If factors don't have exactly 2 levels
        """
        if not HAS_PYDOE3:
            raise ImportError("pyDOE3 is required for Plackett-Burman designs. "
                            "Install with: pip install pyDOE3")

        factor_names = list(factors.keys())
        n_factors = len(factor_names)

        # Validate that all factors have exactly 2 levels
        for fn in factor_names:
            if len(factors[fn]) != 2:
                raise ValueError(f"Plackett-Burman requires exactly 2 levels per factor. "
                               f"Factor '{AVAILABLE_FACTORS.get(fn, fn)}' has {len(factors[fn])} levels.")

        # Generate Plackett-Burman design
        design = pyDOE3.pbdesign(n_factors)

        # Convert from -1/+1 coding to actual factor levels
        combinations = []
        for row in design:
            combo = []
            for i, factor_name in enumerate(factor_names):
                levels = factors[factor_name]
                # Map -1 to low level (index 0), +1 to high level (index 1)
                level_idx = 0 if row[i] < 0 else 1
                combo.append(levels[level_idx])
            combinations.append(tuple(combo))

        return combinations

    def _generate_central_composite(self, factors: Dict[str, List[str]], ccd_type: str) -> List[Tuple]:
        """Generate Central Composite Design using pyDOE3.

        Central Composite Designs are response surface designs that include
        factorial points, center points, and axial (star) points.

        Args:
            factors: Dictionary of factor names to level lists
            ccd_type: Type of CCD ('faced', 'inscribed', or 'circumscribed')

        Returns:
            List of tuples representing combinations

        Raises:
            ImportError: If pyDOE3 is not available
            ValueError: If no numeric factors are provided
        """
        if not HAS_PYDOE3:
            raise ImportError("pyDOE3 is required for Central Composite designs. "
                            "Install with: pip install pyDOE3")

        factor_names = list(factors.keys())

        # Identify categorical factors
        categorical_factors = ["buffer pH", "detergent", "reducing_agent"]

        # Separate numeric and categorical factors
        numeric_factor_names = []
        categorical_factor_names = []
        for fn in factor_names:
            if fn in categorical_factors:
                categorical_factor_names.append(fn)
            else:
                numeric_factor_names.append(fn)

        n_numeric = len(numeric_factor_names)

        if n_numeric == 0:
            raise ValueError("Central Composite Design requires at least one numeric factor.")

        # Determine face parameter based on type
        if ccd_type == "faced":
            face = "faced"
        elif ccd_type == "inscribed":
            face = "inscribed"
        elif ccd_type == "circumscribed":
            face = "circumscribed"
        else:
            face = "faced"  # Default

        # Generate CCD design for numeric factors only
        design = pyDOE3.ccdesign(n_numeric, center=(4, 4), alpha='orthogonal', face=face)

        # Map coded values to actual factor levels (numeric only)
        numeric_combinations = []
        for row in design:
            combo = []
            for i, factor_name in enumerate(numeric_factor_names):
                levels = factors[factor_name]
                numeric_levels = sorted([float(lv) for lv in levels])

                # Map coded value to actual level
                if len(numeric_levels) >= 3:
                    min_val = numeric_levels[0]
                    max_val = numeric_levels[-1]
                    center_val = numeric_levels[len(numeric_levels)//2]
                else:
                    min_val = numeric_levels[0]
                    max_val = numeric_levels[-1]
                    center_val = (min_val + max_val) / 2

                coded_val = row[i]

                # Map coded value to actual value
                if abs(coded_val) < 0.1:  # Center point (0)
                    actual_val = center_val
                elif coded_val < -0.5:  # Low level (-1)
                    actual_val = min_val
                elif coded_val > 0.5:  # High level (+1)
                    actual_val = max_val
                else:
                    # Axial point (alpha)
                    actual_val = center_val + coded_val * (max_val - min_val) / 2

                # Find closest existing level or use computed value
                if len(levels) > 2:
                    closest = min(levels, key=lambda x: abs(float(x) - actual_val))
                    combo.append(closest)
                else:
                    combo.append(str(round(actual_val, 2)))

            numeric_combinations.append(combo)

        # Handle categorical factors - use all combinations
        if categorical_factor_names:
            categorical_level_lists = [factors[fn] for fn in categorical_factor_names]
            categorical_combos = list(itertools.product(*categorical_level_lists))

            # Combine numeric CCD with all categorical combinations
            all_combinations = []
            for num_combo in numeric_combinations:
                for cat_combo in categorical_combos:
                    # Merge in original factor order
                    combo = []
                    num_idx = 0
                    cat_idx = 0
                    for fn in factor_names:
                        if fn in categorical_factors:
                            combo.append(cat_combo[cat_idx])
                            cat_idx += 1
                        else:
                            combo.append(num_combo[num_idx])
                            num_idx += 1
                    all_combinations.append(tuple(combo))
            return all_combinations
        else:
            # No categorical factors, return numeric combinations in order
            return [tuple(combo) for combo in numeric_combinations]

    def _generate_box_behnken(self, factors: Dict[str, List[str]]) -> List[Tuple]:
        """Generate Box-Behnken design using pyDOE3.

        Box-Behnken designs are response surface designs that do not include
        extreme corner points (all factors at their extreme levels simultaneously).
        This makes them useful when extreme conditions should be avoided.

        Args:
            factors: Dictionary of factor names to level lists (requires 3+ factors)

        Returns:
            List of tuples representing combinations

        Raises:
            ImportError: If pyDOE3 is not available
            ValueError: If fewer than 3 factors or fewer than 3 numeric factors
        """
        if not HAS_PYDOE3:
            raise ImportError("pyDOE3 is required for Box-Behnken designs. "
                            "Install with: pip install pyDOE3")

        factor_names = list(factors.keys())

        # Identify categorical factors
        categorical_factors = ["buffer pH", "detergent", "reducing_agent"]

        # Separate numeric and categorical factors
        numeric_factor_names = []
        categorical_factor_names = []
        for fn in factor_names:
            if fn in categorical_factors:
                categorical_factor_names.append(fn)
            else:
                numeric_factor_names.append(fn)

        n_numeric = len(numeric_factor_names)
        n_total = len(factor_names)

        if n_total < 3:
            raise ValueError("Box-Behnken design requires at least 3 factors. "
                           f"You have {n_total} factor(s).")

        if n_numeric < 3:
            raise ValueError("Box-Behnken design requires at least 3 numeric factors. "
                           f"You have {n_numeric} numeric factor(s).")

        # Generate Box-Behnken design for numeric factors only
        design = pyDOE3.bbdesign(n_numeric, center=3)

        # Map coded values to actual factor levels (numeric only)
        numeric_combinations = []
        for row in design:
            combo = []
            for i, factor_name in enumerate(numeric_factor_names):
                levels = factors[factor_name]
                numeric_levels = sorted([float(lv) for lv in levels])

                # Map coded value to actual level
                if len(numeric_levels) >= 3:
                    min_val = numeric_levels[0]
                    max_val = numeric_levels[-1]
                    center_val = numeric_levels[len(numeric_levels)//2]
                else:
                    min_val = numeric_levels[0]
                    max_val = numeric_levels[-1]
                    center_val = (min_val + max_val) / 2

                coded_val = row[i]

                # Map to closest level
                if abs(coded_val) < 0.1:  # Center (0)
                    actual_val = center_val
                elif coded_val < -0.5:  # Low (-1)
                    actual_val = min_val
                else:  # High (+1)
                    actual_val = max_val

                # Find closest existing level or use computed value
                if len(levels) >= 3:
                    closest = min(levels, key=lambda x: abs(float(x) - actual_val))
                    combo.append(closest)
                else:
                    combo.append(str(round(actual_val, 2)))

            numeric_combinations.append(combo)

        # Handle categorical factors - use all combinations
        if categorical_factor_names:
            categorical_level_lists = [factors[fn] for fn in categorical_factor_names]
            categorical_combos = list(itertools.product(*categorical_level_lists))

            # Combine numeric BB with all categorical combinations
            all_combinations = []
            for num_combo in numeric_combinations:
                for cat_combo in categorical_combos:
                    # Merge in original factor order
                    combo = []
                    num_idx = 0
                    cat_idx = 0
                    for fn in factor_names:
                        if fn in categorical_factors:
                            combo.append(cat_combo[cat_idx])
                            cat_idx += 1
                        else:
                            combo.append(num_combo[num_idx])
                            num_idx += 1
                    all_combinations.append(tuple(combo))
            return all_combinations
        else:
            # No categorical factors, return numeric combinations in order
            return [tuple(combo) for combo in numeric_combinations]

    def _generate_d_optimal_design(self, factors: Dict[str, List[str]], n_samples: int, model_type: str) -> List[Tuple]:
        """Generate D-optimal design using Fedorov's exchange algorithm.

        D-optimal designs minimize the covariance of parameter estimates by
        maximizing the determinant of the information matrix (X'X).

        Args:
            factors: Dictionary of factor names to level lists
            n_samples: Exact number of experimental runs desired
            model_type: Type of model to optimize for:
                - "linear": Main effects only
                - "interactions": Main effects + 2-way interactions
                - "quadratic": Full second-order model

        Returns:
            List of tuples representing combinations

        Raises:
            ValueError: If n_samples is less than minimum required for model
        """
        from core.design_factory import DesignFactory

        # Create factory instance
        factory = DesignFactory(has_pydoe3=HAS_PYDOE3, has_smt=HAS_SMT)

        # Generate D-optimal design using factory
        design_points = factory.create_design(
            "d_optimal",
            factors,
            n_samples=n_samples,
            model_type=model_type
        )

        # Convert list of dicts to list of tuples (maintaining factor order)
        factor_names = list(factors.keys())
        combinations = []
        for point in design_points:
            combo = tuple(point[fn] for fn in factor_names)
            combinations.append(combo)

        return combinations
