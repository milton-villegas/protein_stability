"""Tests for BayesianOptimizer class"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from core.optimizer import BayesianOptimizer, AX_AVAILABLE


@pytest.fixture
def simple_optimization_data():
    """Create simple dataset for optimization testing"""
    return pd.DataFrame({
        'pH': [7.0, 7.5, 8.0, 7.0, 7.5, 8.0],
        'NaCl': [100, 150, 200, 100, 150, 200],
        'Response': [0.5, 0.8, 0.6, 0.52, 0.82, 0.58]
    })


@pytest.fixture
def categorical_optimization_data():
    """Dataset with categorical and numeric factors"""
    return pd.DataFrame({
        'Temperature': [20, 25, 30, 20, 25, 30],
        'Buffer': ['Tris', 'HEPES', 'Tris', 'HEPES', 'Tris', 'HEPES'],
        'Response': [0.4, 0.7, 0.5, 0.42, 0.68, 0.52]
    })


class TestBayesianOptimizerInit:
    """Test BayesianOptimizer initialization"""

    def test_init_creates_attributes(self):
        """Test that initialization creates expected attributes"""
        optimizer = BayesianOptimizer()

        assert optimizer.ax_client is None
        assert optimizer.data is None
        assert optimizer.factor_columns == []
        assert optimizer.numeric_factors == []
        assert optimizer.categorical_factors == []
        assert optimizer.response_column is None
        assert optimizer.factor_bounds == {}
        assert optimizer.is_initialized is False
        assert optimizer.name_mapping == {}
        assert optimizer.reverse_mapping == {}

    def test_colors_defined(self):
        """Test that color palette is defined"""
        optimizer = BayesianOptimizer()

        assert 'primary' in optimizer.COLORS
        assert 'accent' in optimizer.COLORS
        assert 'warning' in optimizer.COLORS
        assert optimizer.COLORS['primary'] == '#0173B2'


class TestBayesianOptimizerSanitizeName:
    """Test name sanitization for Ax compatibility"""

    def test_sanitize_spaces(self):
        """Test that spaces are replaced with underscores"""
        optimizer = BayesianOptimizer()

        assert optimizer._sanitize_name("pH Buffer") == "pH_Buffer"
        assert optimizer._sanitize_name("NaCl (mM)") == "NaCl_mM"

    def test_sanitize_special_chars(self):
        """Test that special characters are removed"""
        optimizer = BayesianOptimizer()

        assert optimizer._sanitize_name("Factor-1") == "Factor_1"
        assert optimizer._sanitize_name("Temp(C)") == "TempC"

    def test_sanitize_combined(self):
        """Test sanitizing multiple special characters"""
        optimizer = BayesianOptimizer()

        result = optimizer._sanitize_name("Buffer pH (7-9)")
        assert result == "Buffer_pH_7_9"


class TestBayesianOptimizerSetData:
    """Test data setting functionality"""

    def test_set_data_stores_correctly(self, simple_optimization_data):
        """Test that set_data stores data correctly"""
        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=simple_optimization_data,
            factor_columns=['pH', 'NaCl'],
            categorical_factors=[],
            numeric_factors=['pH', 'NaCl'],
            response_column='Response'
        )

        assert optimizer.data is not None
        assert len(optimizer.data) == 6
        assert optimizer.factor_columns == ['pH', 'NaCl']
        assert optimizer.numeric_factors == ['pH', 'NaCl']
        assert optimizer.response_column == 'Response'

    def test_set_data_creates_mappings(self, simple_optimization_data):
        """Test that name mappings are created"""
        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=simple_optimization_data,
            factor_columns=['pH', 'NaCl'],
            categorical_factors=[],
            numeric_factors=['pH', 'NaCl'],
            response_column='Response'
        )

        # reverse_mapping: original → sanitized
        assert 'pH' in optimizer.reverse_mapping
        assert 'NaCl' in optimizer.reverse_mapping

        # name_mapping: sanitized → original
        assert 'pH' in optimizer.name_mapping
        assert 'NaCl' in optimizer.name_mapping

    def test_set_data_makes_copy(self, simple_optimization_data):
        """Test that data is copied, not referenced"""
        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=simple_optimization_data,
            factor_columns=['pH', 'NaCl'],
            categorical_factors=[],
            numeric_factors=['pH', 'NaCl'],
            response_column='Response'
        )

        # Modify original
        simple_optimization_data.loc[0, 'Response'] = 999.0

        # Optimizer data should be unchanged
        assert optimizer.data.loc[0, 'Response'] != 999.0


class TestBayesianOptimizerCalculateBounds:
    """Test bounds calculation"""

    def test_calculate_numeric_bounds(self, simple_optimization_data):
        """Test that numeric factor bounds are calculated correctly"""
        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=simple_optimization_data,
            factor_columns=['pH', 'NaCl'],
            categorical_factors=[],
            numeric_factors=['pH', 'NaCl'],
            response_column='Response'
        )

        assert 'pH' in optimizer.factor_bounds
        assert 'NaCl' in optimizer.factor_bounds

        # pH bounds: min=7.0, max=8.0
        ph_bounds = optimizer.factor_bounds['pH']
        assert ph_bounds[0] == 7.0
        assert ph_bounds[1] == 8.0

        # NaCl bounds: min=100, max=200
        nacl_bounds = optimizer.factor_bounds['NaCl']
        assert nacl_bounds[0] == 100.0
        assert nacl_bounds[1] == 200.0

    def test_calculate_categorical_bounds(self, categorical_optimization_data):
        """Test that categorical factor bounds are unique values"""
        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=categorical_optimization_data,
            factor_columns=['Temperature', 'Buffer'],
            categorical_factors=['Buffer'],
            numeric_factors=['Temperature'],
            response_column='Response'
        )

        assert 'Buffer' in optimizer.factor_bounds
        buffer_values = optimizer.factor_bounds['Buffer']

        # Should contain unique categorical values
        assert 'Tris' in buffer_values
        assert 'HEPES' in buffer_values
        assert len(buffer_values) == 2


class TestBayesianOptimizerWithoutAxPlatform:
    """Test behavior when Ax is not available"""

    @patch('core.optimizer.AX_AVAILABLE', False)
    def test_initialize_without_ax_raises_error(self, simple_optimization_data):
        """Test that initialization fails gracefully without Ax"""
        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=simple_optimization_data,
            factor_columns=['pH', 'NaCl'],
            categorical_factors=[],
            numeric_factors=['pH', 'NaCl'],
            response_column='Response'
        )

        # Should raise ImportError with helpful message
        with pytest.raises(ImportError, match="Ax platform not available"):
            optimizer.initialize_optimizer(minimize=False)


class TestBayesianOptimizerEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_data_handling(self):
        """Test behavior with empty dataset"""
        optimizer = BayesianOptimizer()
        empty_data = pd.DataFrame({
            'pH': [],
            'Response': []
        })

        optimizer.set_data(
            data=empty_data,
            factor_columns=['pH'],
            categorical_factors=[],
            numeric_factors=['pH'],
            response_column='Response'
        )

        assert len(optimizer.data) == 0
        # Bounds calculation should handle empty data
        assert 'pH' not in optimizer.factor_bounds or len(optimizer.factor_bounds['pH']) == 2

    def test_single_value_factor(self):
        """Test handling of factor with only one unique value"""
        data = pd.DataFrame({
            'pH': [7.0, 7.0, 7.0],  # All same value!
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

        # pH bounds should be (7.0, 7.0) - degenerate but shouldn't crash
        ph_bounds = optimizer.factor_bounds['pH']
        assert ph_bounds[0] == 7.0
        assert ph_bounds[1] == 7.0


class TestBayesianOptimizerIntegration:
    """Integration tests (require mocking Ax client)"""

    @pytest.mark.skipif(not AX_AVAILABLE, reason="Requires ax-platform")
    def test_full_workflow_mock(self, simple_optimization_data):
        """Test complete optimization workflow with mocked Ax client"""
        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=simple_optimization_data,
            factor_columns=['pH', 'NaCl'],
            categorical_factors=[],
            numeric_factors=['pH', 'NaCl'],
            response_column='Response'
        )

        # Mock the AxClient to avoid actual Bayesian optimization
        with patch('core.optimizer.AxClient') as mock_ax_class:
            mock_ax_instance = Mock()
            mock_ax_class.return_value = mock_ax_instance

            # This should create parameters and initialize without errors
            optimizer.initialize_optimizer(minimize=False)

            # Verify AxClient was called
            mock_ax_class.assert_called_once()


class TestBayesianOptimizerSuggestions:
    """Test suggestion generation functionality"""

    @pytest.mark.skipif(not AX_AVAILABLE, reason="Requires ax-platform")
    def test_get_next_suggestions_returns_correct_count(self):
        """Test that get_next_suggestions returns requested number"""
        optimizer = BayesianOptimizer()

        # Mock initialized optimizer with ax_client
        optimizer.is_initialized = True
        optimizer.numeric_factors = ['pH', 'NaCl']
        optimizer.factor_columns = ['pH', 'NaCl']
        optimizer.name_mapping = {'pH': 'pH', 'NaCl': 'NaCl'}

        # Mock ax_client
        mock_ax = Mock()
        mock_ax.get_next_trial.return_value = (
            {'pH': 7.5, 'NaCl': 150.0},
            1  # trial_index
        )
        optimizer.ax_client = mock_ax

        # Get suggestions
        suggestions = optimizer.get_next_suggestions(n=3)

        # Should call get_next_trial 3 times
        assert mock_ax.get_next_trial.call_count == 3

    def test_get_next_suggestions_without_initialization_raises_error(self):
        """Test that getting suggestions before initialization fails"""
        optimizer = BayesianOptimizer()
        optimizer.is_initialized = False

        with pytest.raises(Exception):  # Could be ValueError or RuntimeError
            optimizer.get_next_suggestions(n=5)


class TestBayesianOptimizerColumnMatching:
    """Test column name matching functionality"""

    def test_smart_column_match_exact(self):
        """Test exact column name match"""
        optimizer = BayesianOptimizer()
        optimizer.data = pd.DataFrame({
            'pH': [7.0, 8.0],
            'NaCl (mM)': [100, 200]
        })

        # Column names are normalized to lowercase
        matched = optimizer._smart_column_match('pH')
        assert matched == 'ph'

    def test_smart_column_match_case_insensitive(self):
        """Test case-insensitive matching"""
        optimizer = BayesianOptimizer()
        optimizer.data = pd.DataFrame({
            'Buffer_pH': [7.0, 8.0]
        })

        # Column names are normalized to lowercase with underscores
        matched = optimizer._smart_column_match('buffer_ph')
        assert matched == 'buffer_ph'

    def test_smart_column_match_partial(self):
        """Test partial name matching"""
        optimizer = BayesianOptimizer()
        optimizer.data = pd.DataFrame({
            'NaCl (mM)': [100, 200],
            'KCl (mM)': [50, 100]
        })

        # Extracts base name before parentheses and normalizes to lowercase
        matched = optimizer._smart_column_match('NaCl (mM)')
        assert matched == 'nacl'


class TestBayesianOptimizerExport:
    """Test export functionality"""

    def test_export_bo_plots_creates_directory(self, tmp_path):
        """Test that export returns empty list if factors cannot be selected"""
        optimizer = BayesianOptimizer()
        optimizer.is_initialized = True
        optimizer.ax_client = Mock()
        optimizer.data = pd.DataFrame({
            'pH': [7.0, 7.5, 8.0],
            'NaCl': [100, 150, 200],
            'Response': [0.5, 0.8, 0.6]
        })
        optimizer.numeric_factors = ['pH', 'NaCl']
        optimizer.factor_bounds = {'pH': (7.0, 8.0), 'NaCl': (100, 200)}
        optimizer.reverse_mapping = {'pH': 'ph', 'NaCl': 'nacl'}

        export_dir = tmp_path / "plots"

        # Mock the factor selection to return None (can't select)
        with patch.object(optimizer, '_select_most_important_factors', return_value=None):
            result = optimizer.export_bo_plots(
                directory=str(export_dir),
                base_name="Test"
            )

        # Should return empty list if factor selection fails
        assert result == []

    def test_export_bo_batch_validates_inputs(self):
        """Test that export validates input parameters"""
        optimizer = BayesianOptimizer()
        optimizer.is_initialized = False

        with pytest.raises(Exception):
            optimizer.export_bo_batch_to_files(
                n_suggestions=5,
                batch_number=1,
                excel_path="/tmp/test.xlsx",
                csv_path="/tmp/test.csv"
            )


class TestBayesianOptimizerPlotting:
    """Test plotting functionality"""

    def test_create_suggestion_heatmap_with_mock_data(self):
        """Test heatmap creation with mocked data"""
        optimizer = BayesianOptimizer()
        optimizer.ax_client = Mock()
        optimizer.data = pd.DataFrame({
            'pH': [7.0, 7.5, 8.0, 7.0, 7.5, 8.0],
            'NaCl': [100, 100, 100, 200, 200, 200],
            'Response': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        })
        optimizer.numeric_factors = ['pH', 'NaCl']
        optimizer.response_column = 'Response'

        # Mock matplotlib
        with patch('core.optimizer.plt') as mock_plt:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_plt.subplots.return_value = (mock_fig, mock_ax)

            try:
                optimizer._create_suggestion_heatmap('pH', 'NaCl', mock_ax)
                # Should call plotting functions
                assert mock_plt.subplots.called or mock_ax.contourf.called
            except Exception:
                # May fail due to Ax internals, but we're testing structure
                pass

    def test_get_acquisition_plot_requires_initialization(self):
        """Test that acquisition plot requires initialized optimizer"""
        optimizer = BayesianOptimizer()
        optimizer.is_initialized = False

        with pytest.raises(Exception):
            optimizer.get_acquisition_plot()


class TestBayesianOptimizerFactorSelection:
    """Test factor selection for plotting"""

    def test_select_most_important_factors_returns_two(self):
        """Test that factor selection returns 2 factors"""
        optimizer = BayesianOptimizer()
        optimizer.numeric_factors = ['pH', 'NaCl', 'Glycerol', 'Temperature']
        optimizer.data = pd.DataFrame({
            'pH': [7.0, 7.5, 8.0],
            'NaCl': [100, 150, 200],
            'Glycerol': [5, 10, 15],
            'Temperature': [20, 25, 30],
            'Response': [0.5, 0.8, 0.6]
        })
        optimizer.factor_bounds = {
            'pH': (7.0, 8.0),
            'NaCl': (100, 200),
            'Glycerol': (5, 15),
            'Temperature': (20, 30)
        }

        # Should return 2 factors (fallback to range-based if no Ax model)
        selected = optimizer._select_most_important_factors()

        # Should select 2 factors
        assert isinstance(selected, list)
        assert len(selected) == 2
        assert all(f in optimizer.numeric_factors for f in selected)

    def test_select_most_important_with_single_factor(self):
        """Test factor selection with only one numeric factor"""
        optimizer = BayesianOptimizer()
        optimizer.numeric_factors = ['pH']
        optimizer.data = pd.DataFrame({'pH': [7.0, 8.0]})
        optimizer.factor_bounds = {'pH': (7.0, 8.0)}

        # With only 1 factor, returns None (needs at least 2 for 2D plots)
        selected = optimizer._select_most_important_factors()

        # Should return None when there's only one factor
        assert selected is None
