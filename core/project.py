"""
Shared project data model
Combines functionality from FactorModel (Designer) and DataHandler (Analysis)
"""
import pickle
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from utils.constants import AVAILABLE_FACTORS, METADATA_COLUMNS


class DoEProject:
    """
    Container for all DoE project data
    Shared between Designer and Analysis tabs
    """

    def __init__(self):
        self.name = "Untitled Project"
        self.created_date = datetime.now()
        self.modified_date = datetime.now()

        # Design parameters (from FactorModel)
        self._factors: Dict[str, List[str]] = {}  # factor_name → levels
        self._stock_concs: Dict[str, float] = {}  # factor_name → stock concentration
        # Per-level concentrations: factor_name → level → {"stock": float, "final": float}
        self._per_level_concs: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.design_matrix: Optional[pd.DataFrame] = None  # Generated design

        # Analysis data (from DataHandler)
        self.results_data: Optional[pd.DataFrame] = None  # Experimental results
        self.clean_data: Optional[pd.DataFrame] = None  # Preprocessed data
        self.response_column: Optional[str] = None  # Selected response
        self.factor_columns: List[str] = []
        self.categorical_factors: List[str] = []
        self.numeric_factors: List[str] = []

        # Bayesian Optimization state
        self.ax_client = None  # Ax client instance
        self.optimization_history: List[dict] = []

    # ========== Factor Management (from FactorModel) ==========

    def add_factor(self, name: str, levels: List[str], stock_conc: Optional[float] = None):
        """Add a new factor with levels"""
        name = name.strip()
        if not name:
            raise ValueError("Factor name cannot be empty")
        if not levels:
            raise ValueError("At least one level is required")
        self._factors[name] = list(levels)
        if stock_conc is not None:
            self._stock_concs[name] = stock_conc

    def update_factor(self, name: str, levels: List[str], stock_conc: Optional[float] = None):
        """Update existing factor"""
        if name not in self._factors:
            raise ValueError(f"Factor '{name}' does not exist")
        if not levels:
            raise ValueError("At least one level is required")
        self._factors[name] = list(levels)
        if stock_conc is not None:
            self._stock_concs[name] = stock_conc

    def remove_factor(self, name: str):
        """Remove a factor"""
        if name in self._factors:
            del self._factors[name]
        if name in self._stock_concs:
            del self._stock_concs[name]
        if name in self._per_level_concs:
            del self._per_level_concs[name]

    def get_factors(self) -> Dict[str, List[str]]:
        """Get all factors"""
        return {k: list(v) for k, v in self._factors.items()}

    def get_stock_conc(self, name: str) -> Optional[float]:
        """Get stock concentration for a factor"""
        return self._stock_concs.get(name)

    def get_all_stock_concs(self) -> Dict[str, float]:
        """Get all stock concentrations"""
        return dict(self._stock_concs)

    # ========== Per-Level Concentration Management ==========

    def set_per_level_concs(self, factor_name: str, level_concs: Dict[str, Dict[str, float]]):
        """
        Set per-level concentrations for a categorical factor.

        Args:
            factor_name: Name of the categorical factor (e.g., "detergent")
            level_concs: Dict mapping level → {"stock": float, "final": float}
                        Example: {"DDM": {"stock": 0.2, "final": 0.00609}}
        """
        self._per_level_concs[factor_name] = level_concs

    def get_per_level_concs(self, factor_name: str) -> Optional[Dict[str, Dict[str, float]]]:
        """Get per-level concentrations for a factor, or None if not set."""
        return self._per_level_concs.get(factor_name)

    def has_per_level_concs(self, factor_name: str) -> bool:
        """Check if factor has per-level concentrations defined."""
        return factor_name in self._per_level_concs and bool(self._per_level_concs[factor_name])

    def get_level_conc(self, factor_name: str, level: str, conc_type: str = "stock") -> Optional[float]:
        """
        Get stock or final concentration for a specific level.

        Args:
            factor_name: Name of the factor
            level: The level value (e.g., "DDM")
            conc_type: "stock" or "final"

        Returns:
            Concentration value or None if not found
        """
        if factor_name in self._per_level_concs:
            level_data = self._per_level_concs[factor_name].get(str(level))
            if level_data:
                return level_data.get(conc_type)
        return None

    def clear_per_level_concs(self, factor_name: str):
        """Clear per-level concentrations for a factor."""
        if factor_name in self._per_level_concs:
            del self._per_level_concs[factor_name]

    def get_all_per_level_concs(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get all per-level concentrations."""
        return dict(self._per_level_concs)

    def clear_factors(self):
        """Clear all factors"""
        self._factors.clear()
        self._stock_concs.clear()
        self._per_level_concs.clear()
        self.design_matrix = None

    # ========== Analysis Data Management (from DataHandler) ==========

    def load_results(self, filepath: str):
        """Load experimental results from Excel"""
        self.results_data = pd.read_excel(filepath)
        self.modified_date = datetime.now()

    def load_stock_concentrations_from_sheet(self, filepath: str):
        """Load stock concentrations from Stock_Concentrations sheet"""
        try:
            from utils.sanitization import smart_factor_match

            stock_df = pd.read_excel(filepath, sheet_name="Stock_Concentrations")

            for _, row in stock_df.iterrows():
                factor_name = str(row['Factor Name']).strip()
                stock_value_raw = row['Stock Value']

                if pd.isna(stock_value_raw):
                    continue

                stock_value = float(stock_value_raw)
                internal_name = smart_factor_match(factor_name)

                if internal_name:
                    self._stock_concs[internal_name] = stock_value

        except Exception:
            pass  # Sheet doesn't exist or error reading

    def detect_columns(self, response_column: str):
        """Detect factor types from loaded results data"""
        if self.results_data is None:
            raise ValueError("No results data loaded")

        self.response_column = response_column

        # All columns except response and metadata are factors
        self.factor_columns = [
            col for col in self.results_data.columns
            if col != response_column and col not in METADATA_COLUMNS
        ]

        # Detect categorical vs numeric factors
        self.categorical_factors = []
        self.numeric_factors = []

        for col in self.factor_columns:
            if self.results_data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(self.results_data[col]):
                self.categorical_factors.append(col)
            else:
                self.numeric_factors.append(col)

    def preprocess_data(self):
        """Clean and prepare data for analysis"""
        if self.results_data is None:
            raise ValueError("No results data loaded")

        # Create copy
        self.clean_data = self.results_data.copy()

        # Drop metadata columns
        columns_to_drop = [col for col in METADATA_COLUMNS if col in self.clean_data.columns]
        if columns_to_drop:
            self.clean_data = self.clean_data.drop(columns=columns_to_drop)

        # Drop rows with missing response
        if self.response_column:
            self.clean_data = self.clean_data.dropna(subset=[self.response_column])

        # Handle categorical variables
        for col in self.categorical_factors:
            self.clean_data[col] = self.clean_data[col].fillna('None')
            self.clean_data[col] = self.clean_data[col].astype(str)

        # Handle numeric variables
        for col in self.numeric_factors:
            self.clean_data = self.clean_data.dropna(subset=[col])

        return self.clean_data

    # ========== Project Persistence ==========

    def save(self, filepath: str):
        """Save project to file"""
        self.modified_date = datetime.now()
        with open(filepath, 'wb') as f:
            # Don't save ax_client (not picklable)
            ax_client_backup = self.ax_client
            self.ax_client = None
            pickle.dump(self, f)
            self.ax_client = ax_client_backup

    @classmethod
    def load(cls, filepath: str) -> 'DoEProject':
        """Load project from file"""
        with open(filepath, 'rb') as f:
            project = pickle.load(f)
        return project

    def __repr__(self):
        return f"DoEProject(name='{self.name}', factors={len(self._factors)}, has_results={self.results_data is not None})"
