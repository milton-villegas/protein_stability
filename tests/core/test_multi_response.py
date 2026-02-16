"""
Tests for multi-response Design of Experiments analysis

This module tests the multi-response functionality added to support
analyzing multiple response variables simultaneously (e.g., Tm, Aggregation, Activity).
"""

import pytest
import pandas as pd
import numpy as np
from core.doe_analyzer import DoEAnalyzer


class TestMultiResponseDoE:
    """Test multi-response DoE analysis functionality"""

    @pytest.fixture
    def multi_response_data(self):
        """Create synthetic data with 3 responses that have different factor dependencies"""
        np.random.seed(42)
        n = 20

        # Factors
        ph = np.random.choice([6, 7, 8, 9], n)
        nacl = np.random.uniform(0, 200, n)
        glycerol = np.random.uniform(0, 20, n)

        # Response 1 (Tm): Depends on pH (optimal 7-8) and Glycerol (positive)
        tm = 45 + 5 * np.sin((ph - 6.5) * np.pi / 3) + 0.3 * glycerol + np.random.normal(0, 1, n)

        # Response 2 (Aggregation): Depends on NaCl (positive) and pH (negative)
        aggregation = 10 + 0.05 * nacl - 2 * ph + np.random.normal(0, 0.5, n)

        # Response 3 (Activity): Depends on pH (optimal 8) and Glycerol (optimal 10%)
        activity = 50 + 10 * np.exp(-((ph - 8)**2) / 2) - 0.1 * (glycerol - 10)**2 + np.random.normal(0, 2, n)

        data = pd.DataFrame({
            'Buffer pH': ph,
            'NaCl (mM)': nacl,
            'Glycerol (%)': glycerol,
            'Tm': tm,
            'Aggregation': aggregation,
            'Activity': activity
        })

        return data

    def test_set_data_with_multiple_responses(self, multi_response_data):
        """Test setting data with multiple response columns"""
        analyzer = DoEAnalyzer()

        analyzer.set_data(
            data=multi_response_data,
            factor_columns=['Buffer pH', 'NaCl (mM)', 'Glycerol (%)'],
            categorical_factors=['Buffer pH'],
            numeric_factors=['NaCl (mM)', 'Glycerol (%)'],
            response_columns=['Tm', 'Aggregation', 'Activity']
        )

        assert analyzer.response_columns == ['Tm', 'Aggregation', 'Activity']
        assert analyzer.response_column == 'Tm'  # First response for backward compatibility
        assert len(analyzer.data) == 20

    def test_fit_model_for_each_response(self, multi_response_data):
        """Test fitting separate models for each response"""
        analyzer = DoEAnalyzer()

        analyzer.set_data(
            data=multi_response_data,
            factor_columns=['Buffer pH', 'NaCl (mM)', 'Glycerol (%)'],
            categorical_factors=['Buffer pH'],
            numeric_factors=['NaCl (mM)', 'Glycerol (%)'],
            response_columns=['Tm', 'Aggregation', 'Activity']
        )

        # Fit linear model for each response
        results_tm = analyzer.fit_model('linear', response_name='Tm')
        results_agg = analyzer.fit_model('linear', response_name='Aggregation')
        results_act = analyzer.fit_model('linear', response_name='Activity')

        # Each should have different R² values
        assert 'model_stats' in results_tm
        assert 'model_stats' in results_agg
        assert 'model_stats' in results_act

        # R² should be different for each response (different dependencies)
        r2_tm = results_tm['model_stats']['R-squared']
        r2_agg = results_agg['model_stats']['R-squared']
        r2_act = results_act['model_stats']['R-squared']

        assert 0 <= r2_tm <= 1
        assert 0 <= r2_agg <= 1
        assert 0 <= r2_act <= 1

    def test_compare_all_models_all_responses(self, multi_response_data):
        """Test comparing all models for all responses simultaneously"""
        analyzer = DoEAnalyzer()

        analyzer.set_data(
            data=multi_response_data,
            factor_columns=['Buffer pH', 'NaCl (mM)', 'Glycerol (%)'],
            categorical_factors=['Buffer pH'],
            numeric_factors=['NaCl (mM)', 'Glycerol (%)'],
            response_columns=['Tm', 'Aggregation', 'Activity']
        )

        all_comparisons = analyzer.compare_all_models_all_responses()

        # Should have results for all 3 responses
        assert 'Tm' in all_comparisons
        assert 'Aggregation' in all_comparisons
        assert 'Activity' in all_comparisons

        # Each response should have comparison data
        for response_name, comparison_data in all_comparisons.items():
            assert 'models' in comparison_data
            assert 'comparison_table' in comparison_data
            assert len(comparison_data['models']) > 0

    def test_calculate_main_effects_for_each_response(self, multi_response_data):
        """Test calculating main effects for each response separately"""
        analyzer = DoEAnalyzer()

        analyzer.set_data(
            data=multi_response_data,
            factor_columns=['Buffer pH', 'NaCl (mM)', 'Glycerol (%)'],
            categorical_factors=['Buffer pH'],
            numeric_factors=['NaCl (mM)', 'Glycerol (%)'],
            response_columns=['Tm', 'Aggregation', 'Activity']
        )

        # Fit models first
        analyzer.fit_model('linear', response_name='Tm')
        main_effects_tm = analyzer.calculate_main_effects(response_name='Tm')

        analyzer.fit_model('linear', response_name='Aggregation')
        main_effects_agg = analyzer.calculate_main_effects(response_name='Aggregation')

        # Main effects should be different for different responses
        assert len(main_effects_tm) > 0
        assert len(main_effects_agg) > 0

        # Tm should have main effects for Glycerol (designed that way)
        if 'Glycerol (%)' in main_effects_tm:
            assert 'Mean Response' in main_effects_tm['Glycerol (%)'].columns

        # Aggregation should have main effects for pH (designed that way)
        if 'Buffer pH' in main_effects_agg:
            assert 'Mean Response' in main_effects_agg['Buffer pH'].columns

    def test_single_response_backward_compatibility(self, multi_response_data):
        """Test that single response still works (backward compatibility)"""
        analyzer = DoEAnalyzer()

        # Old API: response_column instead of response_columns
        analyzer.set_data(
            data=multi_response_data,
            factor_columns=['Buffer pH', 'NaCl (mM)', 'Glycerol (%)'],
            categorical_factors=['Buffer pH'],
            numeric_factors=['NaCl (mM)', 'Glycerol (%)'],
            response_column='Tm'  # Old API
        )

        assert analyzer.response_column == 'Tm'
        assert analyzer.response_columns == ['Tm']  # Should be converted to list

        # Should still work
        results = analyzer.fit_model('linear')
        assert 'model_stats' in results

    def test_different_models_for_different_responses(self, multi_response_data):
        """Test that different responses can use different model types"""
        analyzer = DoEAnalyzer()

        analyzer.set_data(
            data=multi_response_data,
            factor_columns=['Buffer pH', 'NaCl (mM)', 'Glycerol (%)'],
            categorical_factors=['Buffer pH'],
            numeric_factors=['NaCl (mM)', 'Glycerol (%)'],
            response_columns=['Tm', 'Aggregation', 'Activity']
        )

        # Use linear for Tm
        results_tm = analyzer.fit_model('linear', response_name='Tm')

        # Use quadratic for Activity (has optimal value)
        results_act = analyzer.fit_model('quadratic', response_name='Activity')

        # Both should succeed but may have different R²
        r2_tm = results_tm['model_stats']['R-squared']
        r2_act = results_act['model_stats']['R-squared']

        assert 0 <= r2_tm <= 1
        assert 0 <= r2_act <= 1

    def test_formula_building_with_response_name(self, multi_response_data):
        """Test that formulas are built correctly for each response"""
        analyzer = DoEAnalyzer()

        analyzer.set_data(
            data=multi_response_data,
            factor_columns=['Buffer pH', 'NaCl (mM)', 'Glycerol (%)'],
            categorical_factors=['Buffer pH'],
            numeric_factors=['NaCl (mM)', 'Glycerol (%)'],
            response_columns=['Tm', 'Aggregation']
        )

        formula_tm = analyzer.build_formula('linear', response_name='Tm')
        formula_agg = analyzer.build_formula('linear', response_name='Aggregation')

        # Formulas should use different response names
        assert "Q('Tm')" in formula_tm
        assert "Q('Aggregation')" in formula_agg

        # But same factors
        assert "Q('Buffer pH')" in formula_tm
        assert "Q('Buffer pH')" in formula_agg

    def test_error_handling_invalid_response_name(self, multi_response_data):
        """Test error handling when invalid response name is provided"""
        analyzer = DoEAnalyzer()

        analyzer.set_data(
            data=multi_response_data,
            factor_columns=['Buffer pH', 'NaCl (mM)', 'Glycerol (%)'],
            categorical_factors=['Buffer pH'],
            numeric_factors=['NaCl (mM)', 'Glycerol (%)'],
            response_columns=['Tm', 'Aggregation']
        )

        # Should raise error for invalid response name
        with pytest.raises(Exception):
            analyzer.fit_model('linear', response_name='NonExistent')

    def test_two_responses_minimum(self):
        """Test multi-response analysis with minimum 2 responses"""
        data = pd.DataFrame({
            'Factor1': [1, 2, 3, 4, 5],
            'Factor2': [10, 20, 30, 40, 50],
            'Response1': [5, 10, 15, 20, 25],
            'Response2': [50, 40, 30, 20, 10]
        })

        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_columns=['Response1', 'Response2']
        )

        all_comparisons = analyzer.compare_all_models_all_responses()

        assert len(all_comparisons) == 2
        assert 'Response1' in all_comparisons
        assert 'Response2' in all_comparisons


class TestMultiResponseValidation:
    """Test validation and edge cases for multi-response analysis"""

    def test_must_specify_response(self):
        """Test that either response_column or response_columns must be specified"""
        analyzer = DoEAnalyzer()
        data = pd.DataFrame({'Factor1': [1, 2, 3], 'Response': [4, 5, 6]})

        with pytest.raises(ValueError, match="Must specify either response_column or response_columns"):
            analyzer.set_data(
                data=data,
                factor_columns=['Factor1'],
                categorical_factors=[],
                numeric_factors=['Factor1']
                # Missing both response_column and response_columns
            )

    def test_response_columns_converted_to_list(self):
        """Test that single response_columns value is converted to list"""
        analyzer = DoEAnalyzer()
        data = pd.DataFrame({'Factor1': [1, 2, 3], 'Response': [4, 5, 6]})

        # Pass string instead of list
        analyzer.set_data(
            data=data,
            factor_columns=['Factor1'],
            categorical_factors=[],
            numeric_factors=['Factor1'],
            response_columns='Response'  # String, not list
        )

        assert analyzer.response_columns == ['Response']  # Should be list
        assert analyzer.response_column == 'Response'
