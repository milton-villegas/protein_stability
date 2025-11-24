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
    - D-Optimal
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
                - n_samples: Number of samples (for LHS, D-Optimal)
                - resolution: Resolution level (for fractional factorial)
                - ccd_type: CCD type (for central composite)
                - center_points: Number of center points (for CCD, Box-Behnken)
                - use_smt: Use SMT for optimized LHS (for LHS)
                - model_type: Model type for D-Optimal ("linear", "interactions", "quadratic")

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
        elif design_type == "d_optimal":
            n_samples = params.get("n_samples", 20)
            model_type = params.get("model_type", "quadratic")
            return self._generate_d_optimal(factors, n_samples, model_type)
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

    def _generate_d_optimal(
        self,
        factors: Dict[str, List[str]],
        n_samples: int,
        model_type: str = "quadratic"
    ) -> List[Dict[str, Any]]:
        """
        Generate D-optimal design using Fedorov's exchange algorithm.

        D-optimal designs minimize the covariance of parameter estimates by
        maximizing the determinant of the information matrix (X'X).

        Args:
            factors: Dictionary of factor_name → list of levels
            n_samples: Exact number of experimental runs desired
            model_type: Type of model to optimize for:
                - "linear": Main effects only
                - "interactions": Main effects + 2-way interactions
                - "quadratic": Full second-order model

        Returns:
            List of design points (dictionaries)

        Raises:
            ValueError: If n_samples is less than minimum required for model
        """
        factor_names = list(factors.keys())
        n_factors = len(factor_names)

        if n_factors < 2:
            raise ValueError("D-optimal design requires at least 2 factors")

        # Calculate minimum runs needed for the model
        min_runs = self._calculate_min_runs(n_factors, model_type)
        if n_samples < min_runs:
            raise ValueError(
                f"D-optimal {model_type} model with {n_factors} factors "
                f"requires at least {min_runs} runs. You requested {n_samples}."
            )

        # Generate candidate set
        candidates = self._generate_candidate_set(factors)

        if len(candidates) < n_samples:
            raise ValueError(
                f"Not enough candidate points ({len(candidates)}) "
                f"for requested sample size ({n_samples})"
            )

        # Convert candidates to numeric matrix for optimization
        factor_levels = {fn: sorted(set(factors[fn])) for fn in factor_names}
        candidate_matrix = self._encode_candidates(candidates, factor_names, factor_levels)

        # Build model matrix function
        model_matrix_func = self._get_model_matrix_func(model_type, n_factors)

        # Run Fedorov exchange algorithm
        selected_indices = self._fedorov_exchange(
            candidate_matrix, n_samples, model_matrix_func
        )

        # Convert selected indices back to design points
        design_points = [candidates[i] for i in selected_indices]

        return design_points

    def _calculate_min_runs(self, n_factors: int, model_type: str) -> int:
        """Calculate minimum runs required for a model type."""
        if model_type == "linear":
            # Intercept + main effects
            return n_factors + 1
        elif model_type == "interactions":
            # Intercept + main effects + 2-way interactions
            n_interactions = n_factors * (n_factors - 1) // 2
            return n_factors + n_interactions + 1
        elif model_type == "quadratic":
            # Intercept + main effects + interactions + squared terms
            n_interactions = n_factors * (n_factors - 1) // 2
            return n_factors + n_interactions + n_factors + 1
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _generate_candidate_set(
        self,
        factors: Dict[str, List[str]],
        max_candidates: int = 5000
    ) -> List[Dict[str, Any]]:
        """
        Generate candidate set for D-optimal selection.

        Uses full factorial if small enough, otherwise uses LHS sampling.
        """
        # Calculate full factorial size
        total_combinations = 1
        for levels in factors.values():
            total_combinations *= len(levels)

        if total_combinations <= max_candidates:
            # Use full factorial as candidates
            return self._generate_full_factorial(factors)
        else:
            # Use LHS to generate representative candidates
            if self.has_pydoe3:
                return self._generate_lhs(factors, max_candidates, use_smt=False)
            else:
                # Fallback: random sampling from full factorial
                full_design = self._generate_full_factorial(factors)
                indices = np.random.choice(
                    len(full_design), size=min(max_candidates, len(full_design)),
                    replace=False
                )
                return [full_design[i] for i in indices]

    def _encode_candidates(
        self,
        candidates: List[Dict[str, Any]],
        factor_names: List[str],
        factor_levels: Dict[str, List]
    ) -> np.ndarray:
        """
        Encode candidates as numeric matrix with values scaled to [-1, 1].
        """
        n_candidates = len(candidates)
        n_factors = len(factor_names)
        matrix = np.zeros((n_candidates, n_factors))

        for i, candidate in enumerate(candidates):
            for j, fn in enumerate(factor_names):
                levels = factor_levels[fn]
                value = candidate[fn]

                # Try to convert to numeric
                try:
                    num_value = float(value)
                    num_levels = [float(lv) for lv in levels]
                    min_val = min(num_levels)
                    max_val = max(num_levels)

                    if max_val > min_val:
                        # Scale to [-1, 1]
                        matrix[i, j] = 2 * (num_value - min_val) / (max_val - min_val) - 1
                    else:
                        matrix[i, j] = 0
                except (ValueError, TypeError):
                    # Categorical: encode as index scaled to [-1, 1]
                    try:
                        idx = levels.index(value)
                    except ValueError:
                        idx = 0
                    if len(levels) > 1:
                        matrix[i, j] = 2 * idx / (len(levels) - 1) - 1
                    else:
                        matrix[i, j] = 0

        return matrix

    def _get_model_matrix_func(self, model_type: str, n_factors: int):
        """
        Return a function that builds the model matrix for given encoded points.
        """
        def build_linear(X: np.ndarray) -> np.ndarray:
            """Build linear model matrix: [1, x1, x2, ..., xk]"""
            n = X.shape[0]
            return np.column_stack([np.ones(n), X])

        def build_interactions(X: np.ndarray) -> np.ndarray:
            """Build interaction model: [1, x1, ..., xk, x1*x2, x1*x3, ...]"""
            n = X.shape[0]
            cols = [np.ones(n)]
            # Main effects
            for j in range(n_factors):
                cols.append(X[:, j])
            # 2-way interactions
            for j1 in range(n_factors):
                for j2 in range(j1 + 1, n_factors):
                    cols.append(X[:, j1] * X[:, j2])
            return np.column_stack(cols)

        def build_quadratic(X: np.ndarray) -> np.ndarray:
            """Build quadratic model: [1, x1, ..., x1*x2, ..., x1^2, ...]"""
            n = X.shape[0]
            cols = [np.ones(n)]
            # Main effects
            for j in range(n_factors):
                cols.append(X[:, j])
            # 2-way interactions
            for j1 in range(n_factors):
                for j2 in range(j1 + 1, n_factors):
                    cols.append(X[:, j1] * X[:, j2])
            # Squared terms
            for j in range(n_factors):
                cols.append(X[:, j] ** 2)
            return np.column_stack(cols)

        if model_type == "linear":
            return build_linear
        elif model_type == "interactions":
            return build_interactions
        elif model_type == "quadratic":
            return build_quadratic
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _fedorov_exchange(
        self,
        candidate_matrix: np.ndarray,
        n_samples: int,
        model_matrix_func,
        max_iterations: int = 100
    ) -> List[int]:
        """
        Fedorov's exchange algorithm for D-optimal design selection.

        Args:
            candidate_matrix: Encoded candidate points (n_candidates x n_factors)
            n_samples: Number of points to select
            model_matrix_func: Function to build model matrix
            max_iterations: Maximum exchange iterations

        Returns:
            List of selected candidate indices
        """
        n_candidates = candidate_matrix.shape[0]

        # Initialize with random selection
        all_indices = list(range(n_candidates))
        selected = list(np.random.choice(all_indices, size=n_samples, replace=False))
        not_selected = [i for i in all_indices if i not in selected]

        def calculate_d_value(indices):
            """Calculate log determinant of X'X for numerical stability."""
            X = model_matrix_func(candidate_matrix[indices])
            try:
                XtX = X.T @ X
                # Use log determinant for numerical stability
                sign, logdet = np.linalg.slogdet(XtX)
                if sign <= 0:
                    return -np.inf
                return logdet
            except np.linalg.LinAlgError:
                return -np.inf

        current_d = calculate_d_value(selected)

        for iteration in range(max_iterations):
            improved = False

            for i, sel_idx in enumerate(selected):
                best_swap = None
                best_d = current_d

                for not_sel_idx in not_selected:
                    # Try swapping
                    test_selected = selected.copy()
                    test_selected[i] = not_sel_idx

                    test_d = calculate_d_value(test_selected)

                    if test_d > best_d:
                        best_d = test_d
                        best_swap = (i, not_sel_idx, sel_idx)

                if best_swap is not None:
                    i, new_idx, old_idx = best_swap
                    selected[i] = new_idx
                    not_selected.remove(new_idx)
                    not_selected.append(old_idx)
                    current_d = best_d
                    improved = True

            if not improved:
                break

        return selected
