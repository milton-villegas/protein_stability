"""
Tests for multi-objective Bayesian Optimization

This module tests the multi-objective BO functionality for optimizing
multiple conflicting objectives simultaneously (e.g., maximize Tm while minimizing Aggregation).
"""

import pytest
import pandas as pd
import numpy as np

try:
    from core.optimizer import BayesianOptimizer
    from ax import Experiment
    AX_AVAILABLE = True
except ImportError:
    AX_AVAILABLE = False


@pytest.mark.skipif(not AX_AVAILABLE, reason="Ax platform not installed")
class TestMultiObjectiveBO:
    """Test multi-objective Bayesian Optimization functionality"""

    @pytest.fixture
    def multi_objective_data(self):
        """Create synthetic data with 2 conflicting objectives"""
        np.random.seed(42)
        n = 15

        ph = np.random.choice([6, 7, 8, 9], n)
        nacl = np.random.uniform(0, 200, n)
        glycerol = np.random.uniform(0, 20, n)

        # Tm: Higher is better, increases with glycerol
        tm = 45 + 0.3 * glycerol + 2 * (ph - 6.5) + np.random.normal(0, 1, n)

        # Aggregation: Lower is better, increases with NaCl
        aggregation = 5 + 0.05 * nacl + np.random.normal(0, 0.5, n)

        data = pd.DataFrame({
            'Buffer pH': ph,
            'NaCl (mM)': nacl,
            'Glycerol (%)': glycerol,
            'Tm': tm,
            'Aggregation': aggregation
        })

        return data

    def test_set_data_multi_objective(self, multi_objective_data):
        """Test setting data for multi-objective optimization"""
        optimizer = BayesianOptimizer()

        optimizer.set_data(
            data=multi_objective_data,
            factor_columns=['Buffer pH', 'NaCl (mM)', 'Glycerol (%)'],
            categorical_factors=['Buffer pH'],
            numeric_factors=['NaCl (mM)', 'Glycerol (%)'],
            response_columns=['Tm', 'Aggregation'],
            response_directions={'Tm': 'maximize', 'Aggregation': 'minimize'}
        )

        assert optimizer.is_multi_objective == True
        assert len(optimizer.response_columns) == 2
        assert optimizer.response_directions['Tm'] == 'maximize'
        assert optimizer.response_directions['Aggregation'] == 'minimize'

    def test_initialize_multi_objective_optimizer(self, multi_objective_data):
        """Test initializing multi-objective optimizer with Ax"""
        optimizer = BayesianOptimizer()

        optimizer.set_data(
            data=multi_objective_data,
            factor_columns=['Buffer pH', 'NaCl (mM)', 'Glycerol (%)'],
            categorical_factors=['Buffer pH'],
            numeric_factors=['NaCl (mM)', 'Glycerol (%)'],
            response_columns=['Tm', 'Aggregation'],
            response_directions={'Tm': 'maximize', 'Aggregation': 'minimize'}
        )

        optimizer.initialize_optimizer()

        assert optimizer.is_initialized == True
        assert optimizer.ax_client is not None

    def test_get_pareto_frontier_two_objectives(self, multi_objective_data):
        """Test extracting Pareto frontier for 2 objectives"""
        optimizer = BayesianOptimizer()

        optimizer.set_data(
            data=multi_objective_data,
            factor_columns=['Buffer pH', 'NaCl (mM)', 'Glycerol (%)'],
            categorical_factors=['Buffer pH'],
            numeric_factors=['NaCl (mM)', 'Glycerol (%)'],
            response_columns=['Tm', 'Aggregation'],
            response_directions={'Tm': 'maximize', 'Aggregation': 'minimize'}
        )

        optimizer.initialize_optimizer()

        pareto_points = optimizer.get_pareto_frontier()

        # Should return some Pareto points
        assert pareto_points is not None
        assert len(pareto_points) > 0

        # Each point should have parameters and objectives
        for point in pareto_points:
            assert 'parameters' in point
            assert 'objectives' in point
            assert 'Tm' in point['objectives']
            assert 'Aggregation' in point['objectives']

    def test_plot_pareto_frontier_2d(self, multi_objective_data):
        """Test creating 2D Pareto frontier plot"""
        optimizer = BayesianOptimizer()

        optimizer.set_data(
            data=multi_objective_data,
            factor_columns=['Buffer pH', 'NaCl (mM)', 'Glycerol (%)'],
            categorical_factors=['Buffer pH'],
            numeric_factors=['NaCl (mM)', 'Glycerol (%)'],
            response_columns=['Tm', 'Aggregation'],
            response_directions={'Tm': 'maximize', 'Aggregation': 'minimize'}
        )

        optimizer.initialize_optimizer()

        fig = optimizer.plot_pareto_frontier()

        # Should return a matplotlib figure
        assert fig is not None
        assert hasattr(fig, 'axes')

    def test_three_objectives(self):
        """Test multi-objective BO with 3 objectives"""
        np.random.seed(42)
        n = 15

        data = pd.DataFrame({
            'Factor1': np.random.uniform(0, 10, n),
            'Factor2': np.random.uniform(0, 10, n),
            'Response1': np.random.uniform(40, 60, n),
            'Response2': np.random.uniform(5, 15, n),
            'Response3': np.random.uniform(50, 70, n)
        })

        optimizer = BayesianOptimizer()

        optimizer.set_data(
            data=data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_columns=['Response1', 'Response2', 'Response3'],
            response_directions={
                'Response1': 'maximize',
                'Response2': 'minimize',
                'Response3': 'maximize'
            }
        )

        optimizer.initialize_optimizer()

        assert optimizer.is_multi_objective == True
        assert len(optimizer.response_columns) == 3

        # Should be able to get Pareto frontier
        pareto_points = optimizer.get_pareto_frontier()
        assert pareto_points is not None

        # Should create 3D plot
        fig = optimizer.plot_pareto_frontier()
        assert fig is not None

    def test_single_objective_backward_compatibility(self, multi_objective_data):
        """Test that single-objective mode still works"""
        optimizer = BayesianOptimizer()

        # Old API: single response
        optimizer.set_data(
            data=multi_objective_data,
            factor_columns=['Buffer pH', 'NaCl (mM)', 'Glycerol (%)'],
            categorical_factors=['Buffer pH'],
            numeric_factors=['NaCl (mM)', 'Glycerol (%)'],
            response_column='Tm'  # Single response
        )

        assert optimizer.is_multi_objective == False
        assert optimizer.response_column == 'Tm'
        assert optimizer.response_columns == ['Tm']

        optimizer.initialize_optimizer(minimize=False)

        assert optimizer.is_initialized == True

        # Pareto frontier should return None for single objective
        pareto = optimizer.get_pareto_frontier()
        assert pareto is None

    def test_default_directions_to_maximize(self):
        """Test that default optimization direction is maximize"""
        data = pd.DataFrame({
            'Factor1': [1, 2, 3, 4, 5],
            'Response1': [5, 6, 7, 8, 9],
            'Response2': [10, 9, 8, 7, 6]
        })

        optimizer = BayesianOptimizer()

        optimizer.set_data(
            data=data,
            factor_columns=['Factor1'],
            categorical_factors=[],
            numeric_factors=['Factor1'],
            response_columns=['Response1', 'Response2']
            # No response_directions specified
        )

        # Should default to maximize for both
        assert optimizer.response_directions['Response1'] == 'maximize'
        assert optimizer.response_directions['Response2'] == 'maximize'

    def test_calculate_hypervolume(self, multi_objective_data):
        """Test hypervolume calculation for multi-objective optimization"""
        optimizer = BayesianOptimizer()

        optimizer.set_data(
            data=multi_objective_data,
            factor_columns=['Buffer pH', 'NaCl (mM)', 'Glycerol (%)'],
            categorical_factors=['Buffer pH'],
            numeric_factors=['NaCl (mM)', 'Glycerol (%)'],
            response_columns=['Tm', 'Aggregation'],
            response_directions={'Tm': 'maximize', 'Aggregation': 'minimize'}
        )

        optimizer.initialize_optimizer()

        hypervolume = optimizer.calculate_hypervolume()

        # Should return a numeric value or None if calculation fails
        if hypervolume is not None:
            assert isinstance(hypervolume, (int, float))
            assert hypervolume >= 0

    def test_pareto_values_extraction_format(self, multi_objective_data):
        """Test that Pareto frontier values are extracted correctly as dict (not tuple)"""
        optimizer = BayesianOptimizer()

        optimizer.set_data(
            data=multi_objective_data,
            factor_columns=['Buffer pH', 'NaCl (mM)', 'Glycerol (%)'],
            categorical_factors=['Buffer pH'],
            numeric_factors=['NaCl (mM)', 'Glycerol (%)'],
            response_columns=['Tm', 'Aggregation'],
            response_directions={'Tm': 'maximize', 'Aggregation': 'minimize'}
        )

        optimizer.initialize_optimizer()
        pareto_points = optimizer.get_pareto_frontier()

        assert pareto_points is not None
        assert len(pareto_points) > 0

        # Check that objectives is a dict with mean values (not tuple)
        first_point = pareto_points[0]
        objectives = first_point['objectives']

        assert isinstance(objectives, dict)
        assert 'Tm' in objectives
        assert 'Aggregation' in objectives

        # Values should be floats (means), not tuples
        assert isinstance(objectives['Tm'], (int, float, np.number))
        assert isinstance(objectives['Aggregation'], (int, float, np.number))

    def test_pareto_frontier_with_id_column(self):
        """Test that Pareto frontier includes ID and row_index from original data"""
        np.random.seed(42)
        n = 15

        # Create data with ID column
        data = pd.DataFrame({
            'ID': range(1, n + 1),
            'Buffer pH': np.random.choice([6, 7, 8, 9], n),
            'NaCl (mM)': np.random.uniform(0, 200, n),
            'Glycerol (%)': np.random.uniform(0, 20, n),
            'Tm': np.random.uniform(45, 55, n),
            'Aggregation': np.random.uniform(5, 15, n)
        })

        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=data,
            factor_columns=['Buffer pH', 'NaCl (mM)', 'Glycerol (%)'],
            categorical_factors=['Buffer pH'],
            numeric_factors=['NaCl (mM)', 'Glycerol (%)'],
            response_columns=['Tm', 'Aggregation'],
            response_directions={'Tm': 'maximize', 'Aggregation': 'minimize'}
        )

        optimizer.initialize_optimizer()
        pareto_points = optimizer.get_pareto_frontier()

        assert pareto_points is not None
        assert len(pareto_points) > 0

        # Check that each Pareto point has id and row_index
        for point in pareto_points:
            assert 'id' in point
            assert 'row_index' in point
            assert 'parameters' in point
            assert 'objectives' in point

            # ID should be an integer from 1 to n
            if point['id'] is not None:
                assert isinstance(point['id'], (int, np.integer))
                assert 1 <= point['id'] <= n

            # row_index should be valid
            if point['row_index'] is not None:
                assert point['row_index'] in data.index


class TestMultiObjectivePlotting:
    """Test plotting functionality for multi-objective optimization"""

    @pytest.fixture
    def two_objective_optimizer(self):
        """Create optimizer with 2 objectives"""
        np.random.seed(42)
        data = pd.DataFrame({
            'Factor1': np.random.uniform(0, 10, 15),
            'Factor2': np.random.uniform(0, 10, 15),
            'Objective1': np.random.uniform(40, 60, 15),
            'Objective2': np.random.uniform(5, 15, 15)
        })

        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=data,
            factor_columns=['Factor1', 'Factor2'],
            categorical_factors=[],
            numeric_factors=['Factor1', 'Factor2'],
            response_columns=['Objective1', 'Objective2'],
            response_directions={'Objective1': 'maximize', 'Objective2': 'minimize'}
        )
        optimizer.initialize_optimizer()

        return optimizer

    @pytest.mark.skipif(not AX_AVAILABLE, reason="Ax platform not installed")
    def test_pareto_plot_returns_figure(self, two_objective_optimizer):
        """Test that Pareto frontier plot returns matplotlib figure"""
        fig = two_objective_optimizer.plot_pareto_frontier()

        assert fig is not None
        assert hasattr(fig, 'get_axes')

    @pytest.mark.skipif(not AX_AVAILABLE, reason="Ax platform not installed")
    def test_pareto_plot_none_for_single_objective(self):
        """Test that plot_pareto_frontier returns None for single objective"""
        data = pd.DataFrame({
            'Factor1': [1, 2, 3, 4, 5],
            'Response': [5, 6, 7, 8, 9]
        })

        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=data,
            factor_columns=['Factor1'],
            categorical_factors=[],
            numeric_factors=['Factor1'],
            response_column='Response'
        )
        optimizer.initialize_optimizer()

        fig = optimizer.plot_pareto_frontier()

        assert fig is None

    @pytest.mark.skipif(not AX_AVAILABLE, reason="Ax platform not installed")
    def test_more_than_three_objectives_returns_none(self):
        """Test that Pareto plot returns None for >3 objectives"""
        data = pd.DataFrame({
            'Factor1': np.random.uniform(0, 10, 15),
            'R1': np.random.uniform(0, 10, 15),
            'R2': np.random.uniform(0, 10, 15),
            'R3': np.random.uniform(0, 10, 15),
            'R4': np.random.uniform(0, 10, 15)
        })

        optimizer = BayesianOptimizer()
        optimizer.set_data(
            data=data,
            factor_columns=['Factor1'],
            categorical_factors=[],
            numeric_factors=['Factor1'],
            response_columns=['R1', 'R2', 'R3', 'R4'],
            response_directions={'R1': 'maximize', 'R2': 'maximize', 'R3': 'maximize', 'R4': 'maximize'}
        )
        optimizer.initialize_optimizer()

        fig = optimizer.plot_pareto_frontier()

        # Can't visualize 4D Pareto frontier
        assert fig is None
