"""Tests for DoEPlotter class"""
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from core.plotter import DoEPlotter


@pytest.fixture
def sample_plot_data():
    """Create sample data for plotting tests"""
    return pd.DataFrame({
        'Factor1': [10, 10, 20, 20, 10, 10, 20, 20],
        'Factor2': [100, 200, 100, 200, 100, 200, 100, 200],
        'Response': [0.5, 0.7, 0.6, 0.9, 0.52, 0.68, 0.62, 0.88]
    })


@pytest.fixture
def single_factor_data():
    """Create sample data with single factor"""
    return pd.DataFrame({
        'Factor1': [10, 10, 20, 20, 10, 20],
        'Response': [0.5, 0.52, 0.6, 0.62, 0.51, 0.61]
    })


class TestDoEPlotterInit:
    """Test DoEPlotter initialization"""

    def test_init_creates_none_attributes(self):
        """Test that initialization creates None attributes"""
        plotter = DoEPlotter()

        assert plotter.data is None
        assert plotter.factor_columns == []
        assert plotter.response_column is None

    def test_colors_defined(self):
        """Test that color palette is defined"""
        plotter = DoEPlotter()

        assert 'primary' in plotter.COLORS
        assert 'palette' in plotter.COLORS
        assert isinstance(plotter.COLORS['palette'], list)
        assert len(plotter.COLORS['palette']) > 0


class TestDoEPlotterSetData:
    """Test setting data for plotting"""

    def test_set_data_stores_correctly(self, sample_plot_data):
        """Test that set_data stores data correctly"""
        plotter = DoEPlotter()
        plotter.set_data(sample_plot_data, ['Factor1', 'Factor2'], 'Response')

        assert plotter.data is not None
        assert plotter.factor_columns == ['Factor1', 'Factor2']
        assert plotter.response_column == 'Response'
        pd.testing.assert_frame_equal(plotter.data, sample_plot_data)


class TestDoEPlotterMainEffects:
    """Test main effects plotting"""

    def test_plot_main_effects_creates_figure(self, sample_plot_data):
        """Test that plot_main_effects creates a figure"""
        plotter = DoEPlotter()
        plotter.set_data(sample_plot_data, ['Factor1', 'Factor2'], 'Response')

        fig = plotter.plot_main_effects()

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_main_effects_correct_number_of_subplots(self, sample_plot_data):
        """Test that correct number of subplots are created"""
        plotter = DoEPlotter()
        plotter.set_data(sample_plot_data, ['Factor1', 'Factor2'], 'Response')

        fig = plotter.plot_main_effects()

        # Should have 2 axes for 2 factors
        axes = fig.get_axes()
        assert len(axes) == 2
        plt.close(fig)

    def test_plot_main_effects_single_factor(self, single_factor_data):
        """Test main effects plot with single factor"""
        plotter = DoEPlotter()
        plotter.set_data(single_factor_data, ['Factor1'], 'Response')

        fig = plotter.plot_main_effects()

        assert fig is not None
        axes = fig.get_axes()
        assert len(axes) == 1
        plt.close(fig)

    def test_plot_main_effects_saves_to_file(self, sample_plot_data, tmp_path):
        """Test that plot can be saved to file"""
        plotter = DoEPlotter()
        plotter.set_data(sample_plot_data, ['Factor1', 'Factor2'], 'Response')

        save_path = tmp_path / "main_effects.png"
        fig = plotter.plot_main_effects(save_path=str(save_path))

        assert save_path.exists()
        plt.close(fig)


class TestDoEPlotterInteractionEffects:
    """Test interaction effects plotting"""

    def test_plot_interaction_creates_figure(self, sample_plot_data):
        """Test that plot_interaction_effects creates a figure"""
        plotter = DoEPlotter()
        plotter.set_data(sample_plot_data, ['Factor1', 'Factor2'], 'Response')

        fig = plotter.plot_interaction_effects()

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_interaction_returns_none_single_factor(self, single_factor_data):
        """Test that interaction plot returns None for single factor"""
        plotter = DoEPlotter()
        plotter.set_data(single_factor_data, ['Factor1'], 'Response')

        fig = plotter.plot_interaction_effects()

        assert fig is None

    def test_plot_interaction_creates_matrix(self, sample_plot_data):
        """Test that interaction plot creates matrix of subplots"""
        plotter = DoEPlotter()
        plotter.set_data(sample_plot_data, ['Factor1', 'Factor2'], 'Response')

        fig = plotter.plot_interaction_effects()

        # Should have 2x2 = 4 axes for 2 factors
        axes = fig.get_axes()
        assert len(axes) == 4
        plt.close(fig)

    def test_plot_interaction_saves_to_file(self, sample_plot_data, tmp_path):
        """Test that interaction plot can be saved"""
        plotter = DoEPlotter()
        plotter.set_data(sample_plot_data, ['Factor1', 'Factor2'], 'Response')

        save_path = tmp_path / "interactions.png"
        fig = plotter.plot_interaction_effects(save_path=str(save_path))

        assert save_path.exists()
        plt.close(fig)


class TestDoEPlotterResiduals:
    """Test residuals plotting"""

    def test_plot_residuals_creates_figure(self):
        """Test that plot_residuals creates a figure"""
        plotter = DoEPlotter()

        predictions = np.array([0.5, 0.7, 0.6, 0.9])
        residuals = np.array([-0.01, 0.01, -0.02, 0.02])

        fig = plotter.plot_residuals(predictions, residuals)

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_residuals_has_correct_subplots(self):
        """Test that residuals plot has 4 subplots"""
        plotter = DoEPlotter()

        predictions = np.array([0.5, 0.7, 0.6, 0.9])
        residuals = np.array([-0.01, 0.01, -0.02, 0.02])

        fig = plotter.plot_residuals(predictions, residuals)

        # Should have 4 axes (2x2 grid: residuals vs fitted, Q-Q plot, scale-location, histogram)
        axes = fig.get_axes()
        assert len(axes) == 4
        plt.close(fig)

    def test_plot_residuals_saves_to_file(self, tmp_path):
        """Test that residuals plot can be saved"""
        plotter = DoEPlotter()

        predictions = np.array([0.5, 0.7, 0.6, 0.9])
        residuals = np.array([-0.01, 0.01, -0.02, 0.02])

        save_path = tmp_path / "residuals.png"
        fig = plotter.plot_residuals(predictions, residuals, save_path=str(save_path))

        assert save_path.exists()
        plt.close(fig)


class TestDoEPlotterIntegration:
    """Integration tests for plotting workflow"""

    def test_full_plotting_workflow(self, sample_plot_data, tmp_path):
        """Test complete plotting workflow"""
        plotter = DoEPlotter()

        # Set data
        plotter.set_data(sample_plot_data, ['Factor1', 'Factor2'], 'Response')

        # Create all plots
        main_fig = plotter.plot_main_effects()
        interaction_fig = plotter.plot_interaction_effects()

        predictions = np.array([0.5, 0.7, 0.6, 0.9, 0.52, 0.68, 0.62, 0.88])
        residuals = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        residuals_fig = plotter.plot_residuals(predictions, residuals)

        # Verify all created
        assert main_fig is not None
        assert interaction_fig is not None
        assert residuals_fig is not None

        # Clean up
        plt.close(main_fig)
        plt.close(interaction_fig)
        plt.close(residuals_fig)

    def test_multiple_plots_same_plotter(self, sample_plot_data, single_factor_data):
        """Test that plotter can be reused with different data"""
        plotter = DoEPlotter()

        # First plot
        plotter.set_data(sample_plot_data, ['Factor1', 'Factor2'], 'Response')
        fig1 = plotter.plot_main_effects()
        assert len(fig1.get_axes()) == 2

        # Second plot with different data
        plotter.set_data(single_factor_data, ['Factor1'], 'Response')
        fig2 = plotter.plot_main_effects()
        assert len(fig2.get_axes()) == 1

        plt.close(fig1)
        plt.close(fig2)
