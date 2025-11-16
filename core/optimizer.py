"""
Bayesian Optimization Logic
Extracted from doe_analysis_gui.pyw
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Check if Ax is available
try:
    from ax.service.ax_client import AxClient, ObjectiveProperties
    AX_AVAILABLE = True
except ImportError:
    AX_AVAILABLE = False


class BayesianOptimizer:
    """Bayesian Optimization for intelligent experiment suggestions"""

    def __init__(self):
        self.ax_client = None
        self.data = None
        self.factor_columns = []
        self.numeric_factors = []
        self.categorical_factors = []
        self.response_column = None
        self.factor_bounds = {}
        self.is_initialized = False
        self.name_mapping = {}  # sanitized → original
        self.reverse_mapping = {}  # original → sanitized

    def _sanitize_name(self, name: str) -> str:
        """Replace spaces and special characters with underscores for Ax compatibility"""
        sanitized = name.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        return sanitized

    def set_data(self, data: pd.DataFrame, factor_columns: List[str],
                 categorical_factors: List[str], numeric_factors: List[str],
                 response_column: str):
        """Set data and factor information"""
        self.data = data.copy()
        self.factor_columns = factor_columns
        self.categorical_factors = categorical_factors
        self.numeric_factors = numeric_factors
        self.response_column = response_column

        # Create name mappings
        self.reverse_mapping = {name: self._sanitize_name(name) for name in factor_columns}
        self.name_mapping = {self._sanitize_name(name): name for name in factor_columns}

        self._calculate_bounds()

    def _calculate_bounds(self):
        """Calculate bounds for each factor from the data"""
        self.factor_bounds = {}

        for factor in self.numeric_factors:
            min_val = float(self.data[factor].min())
            max_val = float(self.data[factor].max())
            self.factor_bounds[factor] = (min_val, max_val)

        for factor in self.categorical_factors:
            unique_vals = self.data[factor].unique().tolist()
            self.factor_bounds[factor] = unique_vals

    def initialize_optimizer(self, minimize: bool = False):
        """
        Initialize Ax client with parameters

        Args:
            minimize: If True, minimize response; if False, maximize

        Raises:
            ImportError: If Ax platform not installed
        """
        if not AX_AVAILABLE:
            raise ImportError("Ax platform not available. Install with: pip install ax-platform")

        # Build parameters list with sanitized names
        parameters = []
        for factor in self.factor_columns:
            sanitized_name = self.reverse_mapping[factor]

            if factor in self.numeric_factors:
                # Special handling for pH - treat as ordered categorical
                if 'ph' in factor.lower() and 'buffer' in factor.lower():
                    tested_ph_values = sorted(self.data[factor].unique().tolist())

                    print(f"ℹ️  Treating '{factor}' as ordered categorical parameter")
                    print(f"   Tested pH values: {tested_ph_values}")
                    print(f"   BO will only suggest from these tested values")

                    parameters.append({
                        "name": sanitized_name,
                        "type": "choice",
                        "values": tested_ph_values,
                        "is_ordered": True,
                        "value_type": "float"
                    })
                else:
                    # Normal continuous numeric factors
                    min_val, max_val = self.factor_bounds[factor]
                    parameters.append({
                        "name": sanitized_name,
                        "type": "range",
                        "bounds": [min_val, max_val],
                        "value_type": "float"
                    })
            elif factor in self.categorical_factors:
                parameters.append({
                    "name": sanitized_name,
                    "type": "choice",
                    "values": self.factor_bounds[factor],
                    "value_type": "str"
                })

        # Create Ax client
        self.ax_client = AxClient()
        self.ax_client.create_experiment(
            name="doe_optimization",
            parameters=parameters,
            objectives={self.response_column: ObjectiveProperties(minimize=minimize)},
            choose_generation_strategy_kwargs={"num_initialization_trials": 0}
        )

        # Add existing data as completed trials
        for idx, row in self.data.iterrows():
            params = {}
            for factor in self.factor_columns:
                sanitized_name = self.reverse_mapping[factor]
                val = row[factor]

                if factor in self.numeric_factors:
                    params[sanitized_name] = float(val)
                else:
                    params[sanitized_name] = str(val)

            # Add trial
            _, trial_index = self.ax_client.attach_trial(parameters=params)
            self.ax_client.complete_trial(
                trial_index=trial_index,
                raw_data=float(row[self.response_column])
            )

        self.is_initialized = True

    def get_next_suggestions(self, n: int = 5) -> List[Dict[str, float]]:
        """
        Get next experiment suggestions

        Args:
            n: Number of suggestions to generate

        Returns:
            List of dicts mapping factor_name → suggested value

        Raises:
            ValueError: If optimizer not initialized
        """
        if not self.is_initialized:
            raise ValueError("Optimizer not initialized. Call initialize_optimizer() first")

        suggestions = []
        for _ in range(n):
            params, trial_index = self.ax_client.get_next_trial()

            # Convert sanitized names back to original names
            original_params = {}
            for sanitized_name, value in params.items():
                original_name = self.name_mapping[sanitized_name]

                # Apply rounding
                if isinstance(value, (int, float)):
                    if 'ph' in original_name.lower() and 'buffer' in original_name.lower():
                        # pH is categorical - no rounding
                        original_params[original_name] = float(value)
                    else:
                        # Round to 2 decimals for other numeric factors
                        original_params[original_name] = round(value, 2)
                else:
                    # Categorical factors
                    original_params[original_name] = value

            suggestions.append(original_params)
            # Abandon trial to generate more suggestions
            self.ax_client.abandon_trial(trial_index)

        return suggestions

    def get_best_parameters(self) -> Tuple[Dict[str, float], float]:
        """
        Get best parameters found so far

        Returns:
            (best_params, best_value): Best parameters and corresponding response value
        """
        if not self.is_initialized:
            raise ValueError("Optimizer not initialized")

        best_params, values = self.ax_client.get_best_parameters()

        # Convert sanitized names to original
        original_params = {}
        for sanitized_name, value in best_params.items():
            original_name = self.name_mapping[sanitized_name]
            original_params[original_name] = value

        best_value = values[0][self.response_column]

        return original_params, best_value
