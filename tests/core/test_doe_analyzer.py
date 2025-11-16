"""Tests for DoEAnalyzer class"""
import pytest
import pandas as pd
import numpy as np
from core.doe_analyzer import DoEAnalyzer


@pytest.fixture
def simple_2factor_data():
    """Create simple 2-factor dataset with known linear relationship"""
    return pd.DataFrame({
        'Factor1': [10, 10, 20, 20, 10, 10, 20, 20],
        'Factor2': [100, 200, 100, 200, 100, 200, 100, 200],
        'Response': [0.5, 0.7, 0.6, 0.9, 0.52, 0.68, 0.62, 0.88]
    })


@pytest.fixture
def categorical_data():
    """Create dataset with categorical and numeric factors"""
    return pd.DataFrame({
        'Temperature': [20, 20, 30, 30, 20, 20, 30, 30],
        'Buffer': ['Tris', 'HEPES', 'Tris', 'HEPES', 'Tris', 'HEPES', 'Tris', 'HEPES'],
        'Response': [0.4, 0.6, 0.5, 0.8, 0.42, 0.58, 0.52, 0.78]
    })


@pytest.fixture
def quadratic_data():
    """Create dataset with quadratic relationship"""
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = 0.1 * x**2 + 0.2 * x + 0.3 + np.random.normal(0, 0.01, len(x))
    return pd.DataFrame({
        'Factor1': x,
        'Response': y
    })


class TestDoEAnalyzerInit:
    """Test DoEAnalyzer initialization"""

    def test_init_creates_attributes(self):
        """Test that initialization creates expected attributes"""
        analyzer = DoEAnalyzer()

        assert analyzer.data is None
        assert analyzer.model is None
        assert analyzer.model_type == 'linear'
        assert analyzer.factor_columns == []
        assert analyzer.categorical_factors == []
        assert analyzer.numeric_factors == []
        assert analyzer.response_column is None
        assert analyzer.results is None

    def test_model_types_defined(self):
        """Test that MODEL_TYPES dictionary is defined"""
        analyzer = DoEAnalyzer()

        assert 'mean' in analyzer.MODEL_TYPES
        assert 'linear' in analyzer.MODEL_TYPES
        assert 'interactions' in analyzer.MODEL_TYPES
        assert 'quadratic' in analyzer.MODEL_TYPES


class TestDoEAnalyzerSetData:
    """Test setting data"""

    def test_set_data_stores_correctly(self, simple_2factor_data):
        """Test that set_data stores data correctly"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=simple_2factor_data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        assert analyzer.data is not None
        assert len(analyzer.data) == 8
        assert analyzer.factor_columns == ['Factor1', 'Factor2']
        assert analyzer.numeric_factors == ['Factor1', 'Factor2']
        assert analyzer.response_column == 'Response'

    def test_set_data_makes_copy(self, simple_2factor_data):
        """Test that set_data makes a copy of the data"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=simple_2factor_data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        # Modify original data
        simple_2factor_data.loc[0, 'Response'] = 999.0

        # Analyzer's data should not be affected
        assert analyzer.data.loc[0, 'Response'] != 999.0


class TestDoEAnalyzerBuildFormula:
    """Test formula building"""

    def test_build_formula_mean_model(self, simple_2factor_data):
        """Test mean model formula"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=simple_2factor_data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        formula = analyzer.build_formula('mean')
        assert "Q('Response') ~ 1" in formula

    def test_build_formula_linear_model(self, simple_2factor_data):
        """Test linear model formula"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=simple_2factor_data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        formula = analyzer.build_formula('linear')
        assert "Q('Response')" in formula
        assert "Q('Factor1')" in formula
        assert "Q('Factor2')" in formula
        assert ":" not in formula  # No interactions

    def test_build_formula_interactions_model(self, simple_2factor_data):
        """Test interaction model formula"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=simple_2factor_data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        formula = analyzer.build_formula('interactions')
        assert "Q('Factor1')" in formula
        assert "Q('Factor2')" in formula
        assert ":" in formula  # Has interactions

    def test_build_formula_categorical_factors(self, categorical_data):
        """Test formula with categorical factors"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=categorical_data,
            factor_columns=['Temperature', 'Buffer'],
            categorical_factors=['Buffer'],
            numeric_factors=['Temperature'],
            response_column='Response'
        )

        formula = analyzer.build_formula('linear')
        assert "C(Q('Buffer'))" in formula  # Categorical
        assert "Q('Temperature')" in formula  # Numeric

    def test_build_formula_quadratic_model(self, simple_2factor_data):
        """Test quadratic model formula"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=simple_2factor_data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        formula = analyzer.build_formula('quadratic')
        assert "Q('Factor1')" in formula
        assert "Q('Factor2')" in formula
        assert "**2" in formula  # Has squared terms
        assert ":" in formula  # Has interactions

    def test_build_formula_invalid_type_raises_error(self, simple_2factor_data):
        """Test that invalid model type raises error"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=simple_2factor_data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        with pytest.raises(ValueError, match="Unknown model type"):
            analyzer.build_formula('invalid_model')


class TestDoEAnalyzerFitModel:
    """Test model fitting"""

    def test_fit_model_linear(self, simple_2factor_data):
        """Test fitting linear model"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=simple_2factor_data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        results = analyzer.fit_model('linear')

        assert results is not None
        assert 'coefficients' in results
        assert 'model_stats' in results
        assert 'predictions' in results
        assert 'residuals' in results

    def test_fit_model_creates_good_fit(self, simple_2factor_data):
        """Test that model achieves good R-squared"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=simple_2factor_data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        results = analyzer.fit_model('linear')

        # With this simple linear data, R² should be high
        assert results['model_stats']['R-squared'] > 0.8

    def test_fit_model_coefficients_present(self, simple_2factor_data):
        """Test that coefficients are extracted"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=simple_2factor_data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        results = analyzer.fit_model('linear')
        coef_df = results['coefficients']

        assert 'Intercept' in coef_df.index
        assert 'Coefficient' in coef_df.columns
        assert 'p-value' in coef_df.columns
        assert 't-statistic' in coef_df.columns

    def test_fit_model_predictions_correct_length(self, simple_2factor_data):
        """Test that predictions match data length"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=simple_2factor_data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        results = analyzer.fit_model('linear')

        assert len(results['predictions']) == len(simple_2factor_data)
        assert len(results['residuals']) == len(simple_2factor_data)

    def test_fit_model_no_data_raises_error(self):
        """Test that fitting without data raises error"""
        analyzer = DoEAnalyzer()

        with pytest.raises(ValueError, match="No data set"):
            analyzer.fit_model('linear')

    def test_fit_model_with_categorical(self, categorical_data):
        """Test fitting model with categorical factors"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=categorical_data,
            factor_columns=['Temperature', 'Buffer'],
            categorical_factors=['Buffer'],
            numeric_factors=['Temperature'],
            response_column='Response'
        )

        results = analyzer.fit_model('linear')

        assert results is not None
        assert results['model_stats']['R-squared'] > 0.5


class TestDoEAnalyzerSignificantFactors:
    """Test significant factor detection"""

    def test_get_significant_factors(self, simple_2factor_data):
        """Test getting significant factors"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=simple_2factor_data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        analyzer.fit_model('linear')
        significant = analyzer.get_significant_factors(alpha=0.05)

        # Should be a list
        assert isinstance(significant, list)
        # Intercept should not be in the list
        assert 'Intercept' not in significant

    def test_get_significant_factors_no_results_raises_error(self):
        """Test that getting significant factors without results raises error"""
        analyzer = DoEAnalyzer()

        with pytest.raises(ValueError, match="No results available"):
            analyzer.get_significant_factors()


class TestDoEAnalyzerMainEffects:
    """Test main effects calculation"""

    def test_calculate_main_effects(self, simple_2factor_data):
        """Test calculating main effects"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=simple_2factor_data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        main_effects = analyzer.calculate_main_effects()

        assert 'Factor1' in main_effects
        assert 'Factor2' in main_effects
        assert isinstance(main_effects['Factor1'], pd.DataFrame)
        assert 'Mean Response' in main_effects['Factor1'].columns
        assert 'Std Dev' in main_effects['Factor1'].columns
        assert 'Count' in main_effects['Factor1'].columns

    def test_calculate_main_effects_values(self, simple_2factor_data):
        """Test that main effects values are correct"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=simple_2factor_data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        main_effects = analyzer.calculate_main_effects()

        # Check Factor1 effects
        factor1_effects = main_effects['Factor1']
        assert len(factor1_effects) == 2  # Two levels: 10 and 20

        # Each level should have 4 observations
        assert factor1_effects.loc[10, 'Count'] == 4
        assert factor1_effects.loc[20, 'Count'] == 4

    def test_calculate_main_effects_no_data_raises_error(self):
        """Test that calculating main effects without data raises error"""
        analyzer = DoEAnalyzer()

        with pytest.raises(ValueError, match="No data available"):
            analyzer.calculate_main_effects()


class TestDoEAnalyzerPredict:
    """Test prediction functionality"""

    def test_predict_new_data(self, simple_2factor_data):
        """Test predicting response for new data"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=simple_2factor_data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        analyzer.fit_model('linear')

        # New data point
        new_data = pd.DataFrame({
            'Factor1': [15],
            'Factor2': [150]
        })

        predictions = analyzer.predict(new_data)

        assert len(predictions) == 1
        # Can be Series or ndarray depending on statsmodels version
        assert isinstance(predictions, (np.ndarray, pd.Series))
        # Prediction should be in reasonable range
        prediction_value = predictions.iloc[0] if isinstance(predictions, pd.Series) else predictions[0]
        assert 0 < prediction_value < 1

    def test_predict_no_model_raises_error(self, simple_2factor_data):
        """Test that predicting without a fitted model raises error"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=simple_2factor_data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        new_data = pd.DataFrame({
            'Factor1': [15],
            'Factor2': [150]
        })

        with pytest.raises(ValueError, match="No model fitted"):
            analyzer.predict(new_data)


class TestDoEAnalyzerIntegration:
    """Integration tests for complete workflow"""

    def test_full_analysis_workflow(self, simple_2factor_data):
        """Test complete analysis workflow"""
        analyzer = DoEAnalyzer()

        # Set data
        analyzer.set_data(
            data=simple_2factor_data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        # Fit model
        results = analyzer.fit_model('linear')
        assert results is not None

        # Get significant factors
        significant = analyzer.get_significant_factors()
        assert len(significant) >= 0

        # Calculate main effects
        main_effects = analyzer.calculate_main_effects()
        assert len(main_effects) == 2

        # Make predictions
        new_data = pd.DataFrame({'Factor1': [15], 'Factor2': [150]})
        predictions = analyzer.predict(new_data)
        assert len(predictions) == 1

    def test_multiple_model_types(self, simple_2factor_data):
        """Test fitting different model types"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=simple_2factor_data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_column='Response'
        )

        # Try different model types
        for model_type in ['mean', 'linear', 'interactions']:
            results = analyzer.fit_model(model_type)
            assert results is not None
            assert results['model_type'] == model_type

    def test_quadratic_fit(self, quadratic_data):
        """Test fitting quadratic model to quadratic data"""
        analyzer = DoEAnalyzer()
        analyzer.set_data(
            data=quadratic_data,
            factor_columns=['Factor1'],
            categorical_factors=[],
            numeric_factors=['Factor1'],
            response_column='Response'
        )

        # Linear fit should be worse
        linear_results = analyzer.fit_model('linear')
        linear_r2 = linear_results['model_stats']['R-squared']

        # Quadratic fit should be better
        quad_results = analyzer.fit_model('quadratic')
        quad_r2 = quad_results['model_stats']['R-squared']

        assert quad_r2 > linear_r2
        assert quad_r2 > 0.95  # Should be very good fit
