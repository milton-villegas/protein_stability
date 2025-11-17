"""Edge case tests for BayesianOptimizer class"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from core.optimizer import BayesianOptimizer, AX_AVAILABLE


class TestBayesianOptimizerEdgeCases:
    """Test edge cases and error handling for BayesianOptimizer"""

    def test_empty_dataset(self):
        """Test handling of empty dataset"""
        optimizer = BayesianOptimizer()
        empty_data = pd.DataFrame()

        optimizer.set_data(
            data=empty_data,
            factor_columns=[],
            categorical_factors=[],
            numeric_factors=[],
            response_column='Response'
        )

        assert len(optimizer.data) == 0
        assert optimizer.factor_bounds == {}

    def test_single_unique_value_factor(self):
        """Test factor with only one unique value"""
        data = pd.DataFrame({
            'pH': [7.0, 7.0, 7.0],  # All same!
            'NaCl': [100, 150, 200],
            'Response': [0.5, 0.6, 0.7]
        })

        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=data,
            factor_columns=['pH', 'NaCl'],
            categorical_factors=[],
            numeric_factors=['pH', 'NaCl'],
            response_column='Response'
        )

        # pH bounds should be degenerate (7.0, 7.0)
        ph_bounds = optimizer.factor_bounds['pH']
        assert ph_bounds[0] == 7.0
        assert ph_bounds[1] == 7.0

    def test_single_observation(self):
        """Test with only one data point"""
        data = pd.DataFrame({
            'pH': [7.0],
            'NaCl': [100],
            'Response': [0.5]
        })

        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=data,
            factor_columns=['pH', 'NaCl'],
            categorical_factors=[],
            numeric_factors=['pH', 'NaCl'],
            response_column='Response'
        )

        # Should set bounds based on single point
        assert len(optimizer.data) == 1
        assert 'pH' in optimizer.factor_bounds
        assert 'NaCl' in optimizer.factor_bounds

    def test_mixed_data_types(self):
        """Test with mixed numeric and categorical factors"""
        data = pd.DataFrame({
            'pH': [7.0, 7.5, 8.0],
            'Buffer': ['Tris', 'HEPES', 'Tris'],
            'NaCl': [100, 150, 200],
            'Response': [0.5, 0.6, 0.7]
        })

        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=data,
            factor_columns=['pH', 'Buffer', 'NaCl'],
            categorical_factors=['Buffer'],
            numeric_factors=['pH', 'NaCl'],
            response_column='Response'
        )

        # Numeric factors should have tuple bounds
        assert isinstance(optimizer.factor_bounds['pH'], tuple)
        assert isinstance(optimizer.factor_bounds['NaCl'], tuple)

        # Categorical should have list of values
        assert isinstance(optimizer.factor_bounds['Buffer'], list)
        assert set(optimizer.factor_bounds['Buffer']) == {'Tris', 'HEPES'}

    def test_extreme_numeric_range(self):
        """Test with very wide numeric ranges"""
        data = pd.DataFrame({
            'SmallFactor': [1e-10, 1e-9, 1e-8],
            'LargeFactor': [1e10, 1e11, 1e12],
            'Response': [0.5, 0.6, 0.7]
        })

        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=data,
            factor_columns=['SmallFactor', 'LargeFactor'],
            categorical_factors=[],
            numeric_factors=['SmallFactor', 'LargeFactor'],
            response_column='Response'
        )

        # Should handle extreme values
        small_bounds = optimizer.factor_bounds['SmallFactor']
        large_bounds = optimizer.factor_bounds['LargeFactor']

        assert small_bounds[0] == pytest.approx(1e-10)
        assert large_bounds[1] == pytest.approx(1e12)

    def test_name_sanitization_edge_cases(self):
        """Test name sanitization with various special characters"""
        optimizer = BayesianOptimizer()

        # Test various special characters
        assert optimizer._sanitize_name("Factor-1") == "Factor_1"
        assert optimizer._sanitize_name("Factor (mM)") == "Factor_mM"
        assert optimizer._sanitize_name("pH Buffer") == "pH_Buffer"
        assert optimizer._sanitize_name("Test-Factor (%)") == "Test_Factor_%"

    def test_categorical_only_factors(self):
        """Test optimization with only categorical factors"""
        data = pd.DataFrame({
            'Buffer': ['Tris', 'HEPES', 'PBS'],
            'Salt': ['NaCl', 'KCl', 'NaCl'],
            'Response': [0.5, 0.7, 0.6]
        })

        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=data,
            factor_columns=['Buffer', 'Salt'],
            categorical_factors=['Buffer', 'Salt'],
            numeric_factors=[],
            response_column='Response'
        )

        # Should handle all categorical
        assert len(optimizer.categorical_factors) == 2
        assert len(optimizer.numeric_factors) == 0

    def test_single_numeric_factor(self):
        """Test with only one numeric factor"""
        data = pd.DataFrame({
            'pH': [7.0, 7.5, 8.0],
            'Response': [0.5, 0.8, 0.6]
        })

        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=data,
            factor_columns=['pH'],
            categorical_factors=[],
            numeric_factors=['pH'],
            response_column='Response'
        )

        # Should work with single factor
        assert len(optimizer.numeric_factors) == 1
        assert 'pH' in optimizer.factor_bounds

    def test_duplicate_data_points(self):
        """Test with duplicate factor combinations"""
        data = pd.DataFrame({
            'pH': [7.0, 7.0, 8.0, 8.0],
            'NaCl': [100, 100, 200, 200],
            'Response': [0.5, 0.52, 0.7, 0.68]  # Different responses for same factors
        })

        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=data,
            factor_columns=['pH', 'NaCl'],
            categorical_factors=[],
            numeric_factors=['pH', 'NaCl'],
            response_column='Response'
        )

        # Should accept all data including duplicates
        assert len(optimizer.data) == 4

    def test_negative_response_values(self):
        """Test with negative response values"""
        data = pd.DataFrame({
            'pH': [7.0, 7.5, 8.0],
            'Response': [-0.5, -0.3, -0.1]  # Negative values
        })

        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=data,
            factor_columns=['pH'],
            categorical_factors=[],
            numeric_factors=['pH'],
            response_column='Response'
        )

        # Should handle negative responses
        assert optimizer.data['Response'].min() < 0

    def test_missing_values_in_factors(self):
        """Test data with NaN values in factors"""
        data = pd.DataFrame({
            'pH': [7.0, None, 8.0],
            'NaCl': [100, 150, 200],
            'Response': [0.5, 0.6, 0.7]
        })

        optimizer = BayesianOptimizer()

        # Set data should work (preprocessing happens elsewhere)
        optimizer.set_data(
            data=data,
            factor_columns=['pH', 'NaCl'],
            categorical_factors=[],
            numeric_factors=['pH', 'NaCl'],
            response_column='Response'
        )

        # Bounds calculation might skip NaN
        assert optimizer.data is not None

    @pytest.mark.skipif(not AX_AVAILABLE, reason="Requires ax-platform")
    def test_initialization_without_data(self):
        """Test initialization fails without data"""
        optimizer = BayesianOptimizer()

        # Should fail if no data set
        with pytest.raises((AttributeError, ValueError, Exception)):
            optimizer.initialize_optimizer()

    def test_bounds_with_identical_min_max(self):
        """Test when min and max are identical for a factor"""
        data = pd.DataFrame({
            'pH': [7.5, 7.5, 7.5],  # All identical
            'Response': [0.5, 0.6, 0.7]
        })

        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=data,
            factor_columns=['pH'],
            categorical_factors=[],
            numeric_factors=['pH'],
            response_column='Response'
        )

        bounds = optimizer.factor_bounds['pH']
        assert bounds[0] == bounds[1] == 7.5
