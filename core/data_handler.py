"""
Data loading and preprocessing
Extracted from analysis_tab.py
"""
import pandas as pd
from utils.constants import METADATA_COLUMNS, AVAILABLE_FACTORS
from utils.sanitization import smart_factor_match


class DataHandler:
    """Data loading and preprocessing"""

    def __init__(self):
        self.data = None
        self.clean_data = None
        self.factor_columns = []
        self.categorical_factors = []
        self.numeric_factors = []
        self.response_column = None  # Kept for backward compatibility
        self.response_columns = []  # New: support multiple responses
        self.stock_concentrations = {}  # Store stock concentrations from metadata

    def load_excel(self, filepath: str):
        """Load data from Excel file"""
        self.data = pd.read_excel(filepath)

        # Try to load stock concentrations from metadata sheet
        self._load_stock_concentrations(filepath)

    def _load_stock_concentrations(self, filepath: str):
        """Load stock concentrations from Stock_Concentrations sheet if it exists"""
        try:
            # Try to read the Stock_Concentrations sheet
            stock_df = pd.read_excel(filepath, sheet_name="Stock_Concentrations")

            # Parse the stock concentrations with intelligent matching
            self.stock_concentrations = {}
            for _, row in stock_df.iterrows():
                factor_name = str(row['Factor Name']).strip()
                stock_value_raw = row['Stock Value']

                # Skip rows with None or NaN values
                if pd.isna(stock_value_raw):
                    continue

                stock_value = float(stock_value_raw)

                # Smart matching algorithm - convert display name to internal key
                internal_name = smart_factor_match(factor_name)

                if internal_name:
                    self.stock_concentrations[internal_name] = stock_value

            print(f"✓ Loaded stock concentrations from metadata: {self.stock_concentrations}")

        except Exception as e:
            # Sheet doesn't exist or error reading - that's okay, will use dialog
            print(f"ℹ️  Note: Stock concentrations sheet not found or error reading ({e})")
            self.stock_concentrations = {}

    def get_stock_concentrations(self) -> dict:
        """Get stock concentrations (either from metadata or empty dict)"""
        return self.stock_concentrations.copy()

    def get_potential_response_columns(self):
        """Get list of numeric columns that could be responses (excluding metadata and detected factors)

        Uses heuristics to distinguish factors from responses:
        - Factors typically have repeating values (designed combinations)
        - Responses typically have more unique values (measured outcomes)
        - Factors appear in known factor list (smart default)
        """
        if self.data is None:
            return []

        # Get all numeric columns first
        all_numeric = []
        for col in self.data.columns:
            if col not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(self.data[col]):
                all_numeric.append(col)

        if not all_numeric:
            return []

        # Identify likely factors using multiple heuristics
        likely_factors = set()

        # Heuristic 1: Check against known factor names (smart default)
        for col in all_numeric:
            col_lower = col.lower()
            col_base = col.split('(')[0].strip().lower() if '(' in col else col_lower

            for internal_name, display_name in AVAILABLE_FACTORS.items():
                if (col_lower == internal_name.lower() or
                    col_lower == display_name.lower() or
                    col_base == display_name.split('(')[0].strip().lower()):
                    likely_factors.add(col)
                    break

        # Heuristic 2: Check value patterns (factors have more repeating values)
        for col in all_numeric:
            if col not in likely_factors:
                unique_ratio = len(self.data[col].unique()) / len(self.data[col])
                # If less than 50% unique values, likely a factor (designed combinations repeat)
                if unique_ratio < 0.5:
                    likely_factors.add(col)

        # Return numeric columns that are NOT likely factors
        potential_responses = [col for col in all_numeric if col not in likely_factors]

        return potential_responses

    def detect_columns(self, response_column=None, response_columns=None):
        """Detect factor types, exclude metadata

        Args:
            response_column: Single response (backward compatibility)
            response_columns: List of response columns (new multi-response support)
        """
        if self.data is None:
            raise ValueError("No data loaded")

        # Support both old (single) and new (multiple) response specification
        if response_columns is not None:
            self.response_columns = response_columns if isinstance(response_columns, list) else [response_columns]
            self.response_column = self.response_columns[0] if self.response_columns else None
        elif response_column is not None:
            self.response_column = response_column
            self.response_columns = [response_column]
        else:
            raise ValueError("Must specify either response_column or response_columns")

        # All columns except responses and metadata are factors
        self.factor_columns = [
            col for col in self.data.columns
            if col not in self.response_columns and col not in METADATA_COLUMNS
        ]

        # Detect categorical vs numeric factors
        self.categorical_factors = []
        self.numeric_factors = []

        for col in self.factor_columns:
            if self.data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(self.data[col]):
                self.categorical_factors.append(col)
            else:
                self.numeric_factors.append(col)

    def preprocess_data(self):
        """Clean and prepare data for analysis"""
        if self.data is None:
            raise ValueError("No data loaded")

        # Create copy
        self.clean_data = self.data.copy()

        # Drop metadata columns
        columns_to_drop = [col for col in METADATA_COLUMNS if col in self.clean_data.columns]
        if columns_to_drop:
            self.clean_data = self.clean_data.drop(columns=columns_to_drop)

        # Drop rows with missing values in ANY response column
        if self.response_columns:
            self.clean_data = self.clean_data.dropna(subset=self.response_columns)

        # Handle categorical variables - fill NaN with 'None'
        for col in self.categorical_factors:
            self.clean_data[col] = self.clean_data[col].fillna('None')
            self.clean_data[col] = self.clean_data[col].astype(str)

        # Handle numeric variables - drop rows with NaN in factors
        for col in self.numeric_factors:
            self.clean_data = self.clean_data.dropna(subset=[col])

        return self.clean_data
