"""
Data loading and preprocessing
Extracted from analysis_tab.py
"""
import pandas as pd
from utils.constants import METADATA_COLUMNS
from utils.sanitization import smart_factor_match


class DataHandler:
    """Data loading and preprocessing"""

    def __init__(self):
        self.data = None
        self.clean_data = None
        self.factor_columns = []
        self.categorical_factors = []
        self.numeric_factors = []
        self.response_column = None
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
            print(f"Note: Stock concentrations sheet not found or error reading ({e})")
            self.stock_concentrations = {}

    def get_stock_concentrations(self) -> dict:
        """Get stock concentrations (either from metadata or empty dict)"""
        return self.stock_concentrations.copy()

    def detect_columns(self, response_column: str):
        """Detect factor types, exclude metadata"""
        if self.data is None:
            raise ValueError("No data loaded")

        self.response_column = response_column

        # All columns except response and metadata are factors
        self.factor_columns = [
            col for col in self.data.columns
            if col != response_column and col not in METADATA_COLUMNS
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

        # Drop rows with missing response
        self.clean_data = self.clean_data.dropna(subset=[self.response_column])

        # Handle categorical variables - fill NaN with 'None'
        for col in self.categorical_factors:
            self.clean_data[col] = self.clean_data[col].fillna('None')
            self.clean_data[col] = self.clean_data[col].astype(str)

        # Handle numeric variables - drop rows with NaN in factors
        for col in self.numeric_factors:
            self.clean_data = self.clean_data.dropna(subset=[col])

        return self.clean_data
