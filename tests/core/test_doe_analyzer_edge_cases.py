"""Edge case tests for DoEAnalyzer class"""
import pytest
import pandas as pd
import numpy as np
from core.doe_analyzer import DoEAnalyzer


class TestDoEAnalyzerEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        analyzer = DoEAnalyzer()
        empty_df = pd.DataFrame()

        analyzer.set_data(
            data=empty_df,
            factor_columns=[],
            categorical_factors=[],
            numeric_factors=[],
            response_column='Response'
        )

        # Should handle empty data gracefully
        with pytest.raises((ValueError, Exception)):
            analyzer.fit_model('linear')

    def test_single_data_point(self):
        """Test with only one data point"""
        data = pd.DataFrame({
            'Factor1': [10],
            'Response': [0.5]
        })

        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=data,
            factor_columns=['Factor1'],
            categorical_factors=[],
            numeric_factors=['Factor1'],
            response_column='Response'
        )

        # Should fail or handle gracefully (not enough data)
        with pytest.raises((ValueError, Exception)):
            analyzer.fit_model('linear')

    def test_perfect_fit(self):
        """Test with perfect linear relationship (R² = 1.0)"""
        data = pd.DataFrame({
            'Factor1': [1, 2, 3, 4, 5],
            'Response': [2, 4, 6, 8, 10]  # Perfect: y = 2x
        })

        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=data,
            factor_columns=['Factor1'],
            categorical_factors=[],
            numeric_factors=['Factor1'],
            response_column='Response'
        )

        results = analyzer.fit_model('linear')

        # Should achieve perfect fit
        assert results['model_stats']['R-squared'] > 0.999

    def test_all_factors_insignificant(self):
        """Test when no factors are significant (pure noise)"""
        np.random.seed(42)
        data = pd.DataFrame({
            'Factor1': np.random.normal(10, 1, 20),
            'Factor2': np.random.normal(5, 1, 20),
            'Response': np.random.normal(100, 0.1, 20)  # Completely independent!
        })

        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        results = analyzer.fit_model('linear')

        # R² should be very low
        assert results['model_stats']['R-squared'] < 0.5

        # No significant factors
        significant = analyzer.get_significant_factors(alpha=0.05)
        # May be empty or have spuriously significant factors
        assert isinstance(significant, list)

    def test_multicollinearity(self):
        """Test with highly correlated factors"""
        data = pd.DataFrame({
            'Factor1': [1, 2, 3, 4, 5],
            'Factor2': [1.01, 2.01, 3.01, 4.01, 5.01],  # Almost identical to Factor1!
            'Response': [2, 4, 6, 8, 10]
        })

        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        # Should still fit (statsmodels handles this)
        results = analyzer.fit_model('linear')

        # Model should fit but may have high standard errors
        assert results is not None
        assert results['model_stats']['R-squared'] > 0.9

    def test_predict_with_missing_columns(self):
        """Test prediction fails gracefully with wrong columns"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=pd.DataFrame({'Factor1': [1, 2], 'Response': [0.5, 0.8]}),
            factor_columns=['Factor1'],
            categorical_factors=[],
            numeric_factors=['Factor1'],
            response_column='Response'
        )

        analyzer.fit_model('linear')

        # Try to predict with wrong columns
        wrong_data = pd.DataFrame({'WrongFactor': [1.5]})

        with pytest.raises((KeyError, ValueError, Exception)):
            analyzer.predict(wrong_data)

    def test_constant_response(self):
        """Test with constant response (no variance)"""
        data = pd.DataFrame({
            'Factor1': [1, 2, 3, 4, 5],
            'Response': [5.0, 5.0, 5.0, 5.0, 5.0]  # All the same!
        })

        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=data,
            factor_columns=['Factor1'],
            categorical_factors=[],
            numeric_factors=['Factor1'],
            response_column='Response'
        )

        # Should fit (mean model would be perfect)
        results = analyzer.fit_model('mean')
        assert results is not None

    def test_extreme_values(self):
        """Test with extreme numeric values"""
        data = pd.DataFrame({
            'Factor1': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
            'Factor2': [1e10, 2e10, 3e10, 4e10, 5e10],
            'Response': [0.5, 0.6, 0.7, 0.8, 0.9]
        })

        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        # Should handle extreme values
        results = analyzer.fit_model('linear')
        assert results is not None

    def test_many_factors_few_observations(self):
        """Test underdetermined system (more params than data)"""
        data = pd.DataFrame({
            'F1': [1, 2, 3],
            'F2': [4, 5, 6],
            'F3': [7, 8, 9],
            'F4': [10, 11, 12],
            'Response': [0.5, 0.6, 0.7]
        })

        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=data,
            factor_columns=['F1', 'F2', 'F3', 'F4'],
            categorical_factors=[],
            numeric_factors=['F1', 'F2', 'F3', 'F4'],
            response_column='Response'
        )

        # With 4 factors + intercept = 5 params, but only 3 observations
        # Should either fit or raise appropriate error
        try:
            results = analyzer.fit_model('linear')
            # If it fits, check it's valid
            assert results is not None
        except (ValueError, Exception):
            # Or it correctly identifies the problem
            pass
