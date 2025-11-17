"""Tests for DataHandler class"""
import pytest
import pandas as pd
import numpy as np
from core.data_handler import DataHandler


class TestDataHandlerInit:
    """Test DataHandler initialization"""

    def test_init_creates_empty_attributes(self):
        """Test that initialization creates empty attributes"""
        handler = DataHandler()

        assert handler.data is None
        assert handler.clean_data is None
        assert handler.factor_columns == []
        assert handler.categorical_factors == []
        assert handler.numeric_factors == []
        assert handler.response_column is None
        assert handler.response_columns == []
        assert handler.stock_concentrations == {}


class TestDataHandlerLoadExcel:
    """Test Excel file loading"""

    def test_load_excel_success(self, temp_excel_file):
        """Test successful loading of Excel file"""
        handler = DataHandler()
        handler.load_excel(str(temp_excel_file))

        assert handler.data is not None
        assert isinstance(handler.data, pd.DataFrame)
        assert len(handler.data) == 5
        assert 'Response' in handler.data.columns

    def test_load_excel_with_stock_concentrations(self, temp_excel_with_stock_concs):
        """Test loading Excel with stock concentrations metadata"""
        handler = DataHandler()
        handler.load_excel(str(temp_excel_with_stock_concs))

        assert handler.data is not None
        assert handler.stock_concentrations is not None
        # Stock concentrations should be loaded from metadata sheet
        assert len(handler.stock_concentrations) > 0

    def test_load_excel_nonexistent_file(self):
        """Test loading non-existent file raises error"""
        handler = DataHandler()

        with pytest.raises(FileNotFoundError):
            handler.load_excel("/nonexistent/path/file.xlsx")


class TestDataHandlerStockConcentrations:
    """Test stock concentration handling"""

    def test_get_stock_concentrations_empty(self):
        """Test getting stock concentrations when none loaded"""
        handler = DataHandler()
        stocks = handler.get_stock_concentrations()

        assert isinstance(stocks, dict)
        assert len(stocks) == 0

    def test_get_stock_concentrations_returns_copy(self, temp_excel_with_stock_concs):
        """Test that get_stock_concentrations returns a copy, not reference"""
        handler = DataHandler()
        handler.load_excel(str(temp_excel_with_stock_concs))

        stocks1 = handler.get_stock_concentrations()
        stocks2 = handler.get_stock_concentrations()

        # Should be equal but different objects
        assert stocks1 == stocks2
        assert stocks1 is not stocks2


class TestDataHandlerDetectColumns:
    """Test column detection"""

    def test_detect_columns_identifies_factors(self, temp_excel_file):
        """Test that detect_columns correctly identifies factor columns"""
        handler = DataHandler()
        handler.load_excel(str(temp_excel_file))
        handler.detect_columns('Response')

        assert handler.response_column == 'Response'
        assert 'Response' not in handler.factor_columns
        # Metadata columns should be excluded
        assert 'ID' not in handler.factor_columns
        assert 'Plate_96' not in handler.factor_columns
        assert 'Well_96' not in handler.factor_columns

    def test_detect_columns_separates_categorical_numeric(self, temp_excel_file):
        """Test that detect_columns separates categorical and numeric factors"""
        handler = DataHandler()
        handler.load_excel(str(temp_excel_file))
        handler.detect_columns('Response')

        # NaCl and pH should be numeric
        assert 'NaCl' in handler.numeric_factors
        assert 'pH' in handler.numeric_factors

        # buffer should be categorical
        assert 'buffer' in handler.categorical_factors

    def test_detect_columns_no_data_raises_error(self):
        """Test that detect_columns raises error when no data loaded"""
        handler = DataHandler()

        with pytest.raises(ValueError, match="No data loaded"):
            handler.detect_columns('Response')


class TestDataHandlerPreprocessData:
    """Test data preprocessing"""

    def test_preprocess_removes_metadata_columns(self, temp_excel_file):
        """Test that preprocessing removes metadata columns"""
        handler = DataHandler()
        handler.load_excel(str(temp_excel_file))
        handler.detect_columns('Response')
        clean_data = handler.preprocess_data()

        # Metadata columns should be removed
        assert 'ID' not in clean_data.columns
        assert 'Plate_96' not in clean_data.columns
        assert 'Well_96' not in clean_data.columns

        # Factor and response columns should remain
        assert 'NaCl' in clean_data.columns
        assert 'pH' in clean_data.columns
        assert 'buffer' in clean_data.columns
        assert 'Response' in clean_data.columns

    def test_preprocess_handles_missing_response(self, sample_doe_data_with_missing, tmp_path):
        """Test that rows with missing response are dropped"""
        # Save data with missing response
        filepath = tmp_path / "missing_response.xlsx"
        sample_doe_data_with_missing.to_excel(filepath, index=False)

        handler = DataHandler()
        handler.load_excel(str(filepath))
        handler.detect_columns('Response')
        clean_data = handler.preprocess_data()

        # Rows 2 and 3 should be dropped (missing numeric factors and/or response)
        assert len(clean_data) == 3
        assert not clean_data['Response'].isna().any()

    def test_preprocess_handles_missing_numeric_factors(self, sample_doe_data_with_missing, tmp_path):
        """Test that rows with missing numeric factors are dropped"""
        filepath = tmp_path / "missing_numeric.xlsx"
        sample_doe_data_with_missing.to_excel(filepath, index=False)

        handler = DataHandler()
        handler.load_excel(str(filepath))
        handler.detect_columns('Response')
        clean_data = handler.preprocess_data()

        # Rows with missing numeric factors should be dropped
        assert not clean_data['NaCl'].isna().any()
        assert not clean_data['pH'].isna().any()

    def test_preprocess_handles_missing_categorical_factors(self, sample_doe_data_with_missing, tmp_path):
        """Test that missing categorical values are handled"""
        filepath = tmp_path / "missing_categorical.xlsx"
        sample_doe_data_with_missing.to_excel(filepath, index=False)

        handler = DataHandler()
        handler.load_excel(str(filepath))
        handler.detect_columns('Response')
        clean_data = handler.preprocess_data()

        # After preprocessing, categorical columns should have no NaN
        # (rows with missing numeric factors are dropped, which removes the categorical NaN)
        assert not clean_data['buffer'].isna().any()

    def test_preprocess_no_data_raises_error(self):
        """Test that preprocess raises error when no data loaded"""
        handler = DataHandler()

        with pytest.raises(ValueError, match="No data loaded"):
            handler.preprocess_data()

    def test_preprocess_returns_clean_data(self, temp_excel_file):
        """Test that preprocessing returns clean data"""
        handler = DataHandler()
        handler.load_excel(str(temp_excel_file))
        handler.detect_columns('Response')
        clean_data = handler.preprocess_data()

        assert isinstance(clean_data, pd.DataFrame)
        assert handler.clean_data is not None
        assert clean_data is handler.clean_data


class TestDataHandlerIntegration:
    """Integration tests for full workflow"""

    def test_full_workflow(self, temp_excel_file):
        """Test complete workflow from loading to preprocessing"""
        handler = DataHandler()

        # Load
        handler.load_excel(str(temp_excel_file))
        assert handler.data is not None

        # Detect columns
        handler.detect_columns('Response')
        assert len(handler.factor_columns) > 0
        assert handler.response_column == 'Response'

        # Preprocess
        clean_data = handler.preprocess_data()
        assert len(clean_data) > 0
        assert clean_data is handler.clean_data

    def test_full_workflow_with_stocks(self, temp_excel_with_stock_concs):
        """Test complete workflow with stock concentrations"""
        handler = DataHandler()

        # Load (should load stock concentrations automatically)
        handler.load_excel(str(temp_excel_with_stock_concs))
        assert handler.data is not None
        stocks = handler.get_stock_concentrations()
        assert len(stocks) > 0

        # Continue with normal workflow
        handler.detect_columns('Response')
        clean_data = handler.preprocess_data()
        assert len(clean_data) > 0


class TestDataHandlerMultiResponse:
    """Test multi-response functionality"""

    def test_get_potential_response_columns(self, temp_excel_file):
        """Test getting list of potential response columns"""
        handler = DataHandler()
        handler.load_excel(str(temp_excel_file))

        potential = handler.get_potential_response_columns()

        # Should include response columns, exclude factors, metadata, and categorical
        assert 'Response' in potential
        assert 'NaCl' not in potential  # Factor excluded
        assert 'pH' not in potential  # Factor excluded (buffer pH)
        assert 'ID' not in potential  # Metadata excluded
        assert 'buffer' not in potential  # Categorical excluded

    def test_detect_columns_with_response_columns_list(self, temp_excel_file):
        """Test detect_columns with response_columns parameter"""
        handler = DataHandler()
        handler.load_excel(str(temp_excel_file))
        handler.detect_columns(response_columns=['Response'])

        assert handler.response_columns == ['Response']
        assert handler.response_column == 'Response'
        assert 'Response' not in handler.factor_columns

    def test_detect_columns_with_multiple_responses(self, tmp_path):
        """Test detect_columns with multiple response columns"""
        # Create data with multiple response columns
        data = pd.DataFrame({
            'NaCl': [50, 100, 150],
            'pH': [6.0, 7.0, 8.0],
            'Thermal_Stability': [0.5, 0.7, 0.9],
            'Aggregation': [0.2, 0.3, 0.1]
        })
        filepath = tmp_path / "multi_response.xlsx"
        data.to_excel(filepath, index=False)

        handler = DataHandler()
        handler.load_excel(str(filepath))
        handler.detect_columns(response_columns=['Thermal_Stability', 'Aggregation'])

        assert handler.response_columns == ['Thermal_Stability', 'Aggregation']
        assert handler.response_column == 'Thermal_Stability'  # First one
        assert 'Thermal_Stability' not in handler.factor_columns
        assert 'Aggregation' not in handler.factor_columns
        assert 'NaCl' in handler.factor_columns
        assert 'pH' in handler.factor_columns

    def test_detect_columns_backward_compatibility(self, temp_excel_file):
        """Test that old single response_column parameter still works"""
        handler = DataHandler()
        handler.load_excel(str(temp_excel_file))
        handler.detect_columns(response_column='Response')

        assert handler.response_column == 'Response'
        assert handler.response_columns == ['Response']

    def test_preprocess_drops_rows_with_any_missing_response(self, tmp_path):
        """Test preprocessing drops rows with missing values in any response"""
        data = pd.DataFrame({
            'NaCl': [50, 100, 150, 200],
            'Response1': [0.5, np.nan, 0.9, 0.7],
            'Response2': [0.2, 0.3, np.nan, 0.4]
        })
        filepath = tmp_path / "missing_multi_response.xlsx"
        data.to_excel(filepath, index=False)

        handler = DataHandler()
        handler.load_excel(str(filepath))
        handler.detect_columns(response_columns=['Response1', 'Response2'])
        clean_data = handler.preprocess_data()

        # Should only keep rows with no NaN in either response
        assert len(clean_data) == 2  # Rows 0 and 3
        assert not clean_data['Response1'].isna().any()
        assert not clean_data['Response2'].isna().any()
