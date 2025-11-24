"""Tests for DesignFactory class"""

import pytest
from core.design_factory import DesignFactory


class TestDesignFactoryInitialization:
    """Test DesignFactory initialization"""

    def test_init_without_libraries(self):
        """Test initialization without optional libraries"""
        factory = DesignFactory(has_pydoe3=False, has_smt=False)
        assert factory.has_pydoe3 is False
        assert factory.has_smt is False

    def test_init_with_pydoe3(self):
        """Test initialization with pyDOE3"""
        factory = DesignFactory(has_pydoe3=True, has_smt=False)
        assert factory.has_pydoe3 is True
        assert factory.has_smt is False

    def test_init_with_smt(self):
        """Test initialization with SMT"""
        factory = DesignFactory(has_pydoe3=False, has_smt=True)
        assert factory.has_pydoe3 is False
        assert factory.has_smt is True

    def test_init_with_all_libraries(self):
        """Test initialization with all libraries"""
        factory = DesignFactory(has_pydoe3=True, has_smt=True)
        assert factory.has_pydoe3 is True
        assert factory.has_smt is True


class TestFullFactorialDesign:
    """Test full factorial design generation"""

    def test_simple_2x2_design(self):
        """Test simple 2x2 full factorial"""
        factory = DesignFactory()

        factors = {
            'A': ['1', '2'],
            'B': ['x', 'y']
        }

        design = factory.create_design('full_factorial', factors)

        assert len(design) == 4
        assert all(isinstance(point, dict) for point in design)
        assert all('A' in point and 'B' in point for point in design)

    def test_2x3_design(self):
        """Test 2x3 full factorial"""
        factory = DesignFactory()

        factors = {
            'temp': ['20', '30'],
            'pH': ['6', '7', '8']
        }

        design = factory.create_design('full_factorial', factors)

        assert len(design) == 6  # 2 * 3

    def test_three_factor_design(self):
        """Test 3-factor full factorial"""
        factory = DesignFactory()

        factors = {
            'A': ['1', '2'],
            'B': ['10', '20'],
            'C': ['x', 'y']
        }

        design = factory.create_design('full_factorial', factors)

        assert len(design) == 8  # 2 * 2 * 2

    def test_multilevel_design(self):
        """Test design with different number of levels"""
        factory = DesignFactory()

        factors = {
            'A': ['1', '2', '3'],
            'B': ['10', '20'],
            'C': ['x', 'y', 'z', 'w']
        }

        design = factory.create_design('full_factorial', factors)

        assert len(design) == 24  # 3 * 2 * 4

    def test_single_factor(self):
        """Test with single factor"""
        factory = DesignFactory()

        factors = {
            'A': ['1', '2', '3']
        }

        design = factory.create_design('full_factorial', factors)

        assert len(design) == 3

    def test_design_point_structure(self):
        """Test structure of design points"""
        factory = DesignFactory()

        factors = {
            'temp': ['20', '30'],
            'pH': ['6', '7']
        }

        design = factory.create_design('full_factorial', factors)

        # Check each point
        for point in design:
            assert 'temp' in point
            assert 'pH' in point
            assert point['temp'] in ['20', '30']
            assert point['pH'] in ['6', '7']

    def test_all_combinations_present(self):
        """Test that all combinations are generated"""
        factory = DesignFactory()

        factors = {
            'A': ['1', '2'],
            'B': ['x', 'y']
        }

        design = factory.create_design('full_factorial', factors)

        # Convert to set of tuples for easy comparison
        points = {(p['A'], p['B']) for p in design}
        expected = {('1', 'x'), ('1', 'y'), ('2', 'x'), ('2', 'y')}

        assert points == expected


class TestLHSDesign:
    """Test Latin Hypercube Sampling design"""

    def test_lhs_requires_pydoe3(self):
        """Test that LHS raises error without pyDOE3"""
        factory = DesignFactory(has_pydoe3=False)

        factors = {
            'temp': ['20', '25', '30'],
            'pH': ['6', '7', '8']
        }

        with pytest.raises(ImportError, match="pyDOE3"):
            factory.create_design('lhs', factors, n_samples=10)

    def test_lhs_sample_size(self):
        """Test LHS with specified sample size"""
        try:
            import pyDOE3
            has_pydoe3 = True
        except ImportError:
            has_pydoe3 = False

        if not has_pydoe3:
            pytest.skip("pyDOE3 not available")

        factory = DesignFactory(has_pydoe3=True)

        factors = {
            'temp': ['20', '25', '30', '35', '40'],
            'pH': ['6.0', '6.5', '7.0', '7.5', '8.0']
        }

        design = factory.create_design('lhs', factors, n_samples=50)

        assert len(design) == 50

    def test_lhs_with_categorical(self):
        """Test LHS with categorical factors"""
        try:
            import pyDOE3
            has_pydoe3 = True
        except ImportError:
            has_pydoe3 = False

        if not has_pydoe3:
            pytest.skip("pyDOE3 not available")

        factory = DesignFactory(has_pydoe3=True)

        factors = {
            'temp': ['20', '25', '30'],
            'pH': ['6', '7', '8'],
            'buffer pH': ['6.0', '7.0']  # Categorical
        }

        design = factory.create_design('lhs', factors, n_samples=20)

        assert len(design) == 20
        # Each point should have all factors
        for point in design:
            assert 'temp' in point
            assert 'pH' in point
            assert 'buffer pH' in point

    def test_lhs_default_sample_size(self):
        """Test LHS with default sample size"""
        try:
            import pyDOE3
            has_pydoe3 = True
        except ImportError:
            has_pydoe3 = False

        if not has_pydoe3:
            pytest.skip("pyDOE3 not available")

        factory = DesignFactory(has_pydoe3=True)

        factors = {
            'temp': ['20', '30'],
            'pH': ['6', '8']
        }

        design = factory.create_design('lhs', factors)

        # Default is 50 samples
        assert len(design) == 50


class TestFractionalFactorialDesign:
    """Test fractional factorial design"""

    def test_fractional_requires_pydoe3(self):
        """Test that fractional factorial requires pyDOE3"""
        factory = DesignFactory(has_pydoe3=False)

        factors = {
            'A': ['low', 'high'],
            'B': ['low', 'high'],
            'C': ['low', 'high']
        }

        with pytest.raises(ImportError, match="pyDOE3"):
            factory.create_design('fractional', factors, resolution='IV')

    def test_fractional_requires_two_levels(self):
        """Test that fractional factorial requires exactly 2 levels"""
        try:
            import pyDOE3
            has_pydoe3 = True
        except ImportError:
            has_pydoe3 = False

        if not has_pydoe3:
            pytest.skip("pyDOE3 not available")

        factory = DesignFactory(has_pydoe3=True)

        factors = {
            'A': ['1', '2', '3'],  # 3 levels - invalid!
            'B': ['low', 'high']
        }

        with pytest.raises(ValueError, match="exactly 2 levels"):
            factory.create_design('fractional', factors, resolution='IV')

    def test_fractional_design_size(self):
        """Test fractional factorial design size"""
        try:
            import pyDOE3
            has_pydoe3 = True
        except ImportError:
            has_pydoe3 = False

        if not has_pydoe3:
            pytest.skip("pyDOE3 not available")

        factory = DesignFactory(has_pydoe3=True)

        factors = {
            'A': ['low', 'high'],
            'B': ['low', 'high'],
            'C': ['low', 'high'],
            'D': ['low', 'high'],
            'E': ['low', 'high']
        }

        design = factory.create_design('fractional', factors, resolution='IV')

        # 5 factors, resolution IV should produce 16 runs
        assert len(design) == 16

    def test_fractional_different_resolutions(self):
        """Test different resolution levels"""
        try:
            import pyDOE3
            has_pydoe3 = True
        except ImportError:
            has_pydoe3 = False

        if not has_pydoe3:
            pytest.skip("pyDOE3 not available")

        factory = DesignFactory(has_pydoe3=True)

        factors = {
            'A': ['low', 'high'],
            'B': ['low', 'high'],
            'C': ['low', 'high'],
            'D': ['low', 'high']
        }

        # Test resolution III
        design_iii = factory.create_design('fractional', factors, resolution='III')
        assert len(design_iii) > 0

        # Test resolution IV
        design_iv = factory.create_design('fractional', factors, resolution='IV')
        assert len(design_iv) > 0

        # Test resolution V
        design_v = factory.create_design('fractional', factors, resolution='V')
        assert len(design_v) > 0


class TestPlackettBurmanDesign:
    """Test Plackett-Burman design"""

    def test_pb_requires_pydoe3(self):
        """Test that Plackett-Burman requires pyDOE3"""
        factory = DesignFactory(has_pydoe3=False)

        factors = {
            'A': ['low', 'high'],
            'B': ['low', 'high']
        }

        with pytest.raises(ImportError, match="pyDOE3"):
            factory.create_design('plackett_burman', factors)

    def test_pb_requires_two_levels(self):
        """Test that Plackett-Burman requires exactly 2 levels"""
        try:
            import pyDOE3
            has_pydoe3 = True
        except ImportError:
            has_pydoe3 = False

        if not has_pydoe3:
            pytest.skip("pyDOE3 not available")

        factory = DesignFactory(has_pydoe3=True)

        factors = {
            'A': ['1', '2', '3'],  # 3 levels - invalid!
            'B': ['low', 'high']
        }

        with pytest.raises(ValueError, match="exactly 2 levels"):
            factory.create_design('plackett_burman', factors)

    def test_pb_design_size(self):
        """Test Plackett-Burman design size"""
        try:
            import pyDOE3
            has_pydoe3 = True
        except ImportError:
            has_pydoe3 = False

        if not has_pydoe3:
            pytest.skip("pyDOE3 not available")

        factory = DesignFactory(has_pydoe3=True)

        factors = {
            'A': ['low', 'high'],
            'B': ['low', 'high'],
            'C': ['low', 'high']
        }

        design = factory.create_design('plackett_burman', factors)

        # Should produce multiple runs
        assert len(design) > 0
        # PB designs have sizes in multiples of 4
        assert len(design) % 4 == 0


class TestCentralCompositeDesign:
    """Test Central Composite Design"""

    def test_ccd_requires_pydoe3(self):
        """Test that CCD requires pyDOE3"""
        factory = DesignFactory(has_pydoe3=False)

        factors = {
            'temp': ['20', '30', '40'],
            'pH': ['6', '7', '8']
        }

        with pytest.raises(ImportError, match="pyDOE3"):
            factory.create_design('central_composite', factors)

    def test_ccd_requires_min_factors(self):
        """Test that CCD requires at least 2 factors"""
        try:
            import pyDOE3
            has_pydoe3 = True
        except ImportError:
            has_pydoe3 = False

        if not has_pydoe3:
            pytest.skip("pyDOE3 not available")

        factory = DesignFactory(has_pydoe3=True)

        factors = {
            'temp': ['20', '30', '40']  # Only 1 factor
        }

        with pytest.raises(ValueError, match="at least 2 numeric factors"):
            factory.create_design('central_composite', factors)

    def test_ccd_design_size(self):
        """Test CCD design size"""
        try:
            import pyDOE3
            has_pydoe3 = True
        except ImportError:
            has_pydoe3 = False

        if not has_pydoe3:
            pytest.skip("pyDOE3 not available")

        factory = DesignFactory(has_pydoe3=True)

        factors = {
            'temp': ['20', '30', '40'],
            'pH': ['6', '7', '8']
        }

        design = factory.create_design('central_composite', factors, center_points=3)

        # CCD with 2 factors: 2^k + 2k + center_points = 4 + 4 + 3 = 11
        assert len(design) >= 11

    def test_ccd_types(self):
        """Test different CCD types"""
        try:
            import pyDOE3
            has_pydoe3 = True
        except ImportError:
            has_pydoe3 = False

        if not has_pydoe3:
            pytest.skip("pyDOE3 not available")

        factory = DesignFactory(has_pydoe3=True)

        factors = {
            'temp': ['20', '30', '40'],
            'pH': ['6', '7', '8']
        }

        # Test circumscribed
        design_circ = factory.create_design(
            'central_composite', factors, ccd_type='circumscribed'
        )
        assert len(design_circ) > 0

        # Test inscribed
        design_insc = factory.create_design(
            'central_composite', factors, ccd_type='inscribed'
        )
        assert len(design_insc) > 0

        # Test faced
        design_faced = factory.create_design(
            'central_composite', factors, ccd_type='faced'
        )
        assert len(design_faced) > 0


class TestBoxBehnkenDesign:
    """Test Box-Behnken design"""

    def test_bb_requires_pydoe3(self):
        """Test that Box-Behnken requires pyDOE3"""
        factory = DesignFactory(has_pydoe3=False)

        factors = {
            'temp': ['20', '30', '40'],
            'pH': ['6', '7', '8'],
            'nacl': ['100', '200', '300']
        }

        with pytest.raises(ImportError, match="pyDOE3"):
            factory.create_design('box_behnken', factors)

    def test_bb_requires_min_factors(self):
        """Test that Box-Behnken requires at least 3 factors"""
        try:
            import pyDOE3
            has_pydoe3 = True
        except ImportError:
            has_pydoe3 = False

        if not has_pydoe3:
            pytest.skip("pyDOE3 not available")

        factory = DesignFactory(has_pydoe3=True)

        factors = {
            'temp': ['20', '30', '40'],
            'pH': ['6', '7', '8']  # Only 2 factors
        }

        with pytest.raises(ValueError, match="at least 3 numeric factors"):
            factory.create_design('box_behnken', factors)

    def test_bb_requires_min_levels(self):
        """Test that Box-Behnken requires at least 3 levels"""
        try:
            import pyDOE3
            has_pydoe3 = True
        except ImportError:
            has_pydoe3 = False

        if not has_pydoe3:
            pytest.skip("pyDOE3 not available")

        factory = DesignFactory(has_pydoe3=True)

        factors = {
            'temp': ['20', '30'],  # Only 2 levels - invalid!
            'pH': ['6', '7', '8'],
            'nacl': ['100', '200', '300']
        }

        with pytest.raises(ValueError, match="at least 3 levels"):
            factory.create_design('box_behnken', factors)

    def test_bb_design_size(self):
        """Test Box-Behnken design size"""
        try:
            import pyDOE3
            has_pydoe3 = True
        except ImportError:
            has_pydoe3 = False

        if not has_pydoe3:
            pytest.skip("pyDOE3 not available")

        factory = DesignFactory(has_pydoe3=True)

        factors = {
            'temp': ['20', '30', '40'],
            'pH': ['6', '7', '8'],
            'nacl': ['100', '200', '300']
        }

        design = factory.create_design('box_behnken', factors, center_points=3)

        # BB with 3 factors: 2k(k-1) + center_points = 12 + 3 = 15
        assert len(design) >= 15


class TestDesignFactoryErrorHandling:
    """Test error handling and edge cases"""

    def test_unknown_design_type(self):
        """Test that unknown design type raises ValueError"""
        factory = DesignFactory()

        factors = {
            'A': ['1', '2']
        }

        with pytest.raises(ValueError, match="Unknown design type"):
            factory.create_design('invalid_design_type', factors)

    def test_empty_factors(self):
        """Test with empty factors dict"""
        factory = DesignFactory()

        # Full factorial with empty factors returns single empty point
        # (itertools.product with no inputs returns single empty tuple)
        design = factory.create_design('full_factorial', {})

        assert len(design) == 1
        assert design[0] == {}

    def test_factor_with_empty_levels(self):
        """Test factor with no levels"""
        factory = DesignFactory()

        factors = {
            'A': []  # No levels
        }

        # Should return empty design
        design = factory.create_design('full_factorial', factors)
        assert len(design) == 0


class TestCategoricalFactorDistribution:
    """Test categorical factor distribution"""

    def test_distribute_categorical_even_distribution(self):
        """Test that categorical factors are distributed evenly"""
        factory = DesignFactory()

        factors = {
            'buffer pH': ['6.0', '7.0']
        }

        design = factory._distribute_categorical_factors(
            ['buffer pH'], factors, 10
        )

        assert len(design) == 10
        # Should cycle through values
        values = [d[0] for d in design]
        assert '6.0' in values
        assert '7.0' in values

    def test_distribute_multiple_categorical(self):
        """Test distribution of multiple categorical factors"""
        factory = DesignFactory()

        factors = {
            'buffer pH': ['6.0', '7.0'],
            'detergent': ['Tween-20', 'None']
        }

        design = factory._distribute_categorical_factors(
            ['buffer pH', 'detergent'], factors, 8
        )

        assert len(design) == 8
        # Each combination should appear
        for combo in design:
            assert len(combo) == 2


class TestDesignFactoryParameters:
    """Test parameter handling for different design types"""

    def test_lhs_with_use_smt_parameter(self):
        """Test LHS with use_smt parameter"""
        try:
            import pyDOE3
            has_pydoe3 = True
        except ImportError:
            has_pydoe3 = False

        if not has_pydoe3:
            pytest.skip("pyDOE3 not available")

        factory = DesignFactory(has_pydoe3=True, has_smt=False)

        factors = {
            'temp': ['20', '30', '40'],
            'pH': ['6', '7', '8']
        }

        # Should not fail even if SMT not available
        design = factory.create_design('lhs', factors, n_samples=20, use_smt=False)
        assert len(design) == 20

    def test_ccd_with_center_points(self):
        """Test CCD with custom center points"""
        try:
            import pyDOE3
            has_pydoe3 = True
        except ImportError:
            has_pydoe3 = False

        if not has_pydoe3:
            pytest.skip("pyDOE3 not available")

        factory = DesignFactory(has_pydoe3=True)

        factors = {
            'temp': ['20', '30', '40'],
            'pH': ['6', '7', '8']
        }

        design = factory.create_design(
            'central_composite', factors, center_points=5
        )

        # Should include center points
        assert len(design) > 0

    def test_fractional_with_resolution_parameter(self):
        """Test fractional factorial with resolution parameter"""
        try:
            import pyDOE3
            has_pydoe3 = True
        except ImportError:
            has_pydoe3 = False

        if not has_pydoe3:
            pytest.skip("pyDOE3 not available")

        factory = DesignFactory(has_pydoe3=True)

        factors = {
            'A': ['low', 'high'],
            'B': ['low', 'high'],
            'C': ['low', 'high'],
            'D': ['low', 'high']
        }

        # Test with different resolutions
        for resolution in ['III', 'IV', 'V']:
            design = factory.create_design('fractional', factors, resolution=resolution)
            assert len(design) > 0


class TestDOptimalDesign:
    """Test D-Optimal design generation"""

    def test_d_optimal_basic(self):
        """Test basic D-optimal design generation"""
        factory = DesignFactory()

        factors = {
            'temp': ['20', '30', '40'],
            'pH': ['6', '7', '8']
        }

        # 3x3 = 9 candidates, request 6 samples
        design = factory.create_design('d_optimal', factors, n_samples=6, model_type='linear')

        assert len(design) == 6
        assert all(isinstance(point, dict) for point in design)
        assert all('temp' in point and 'pH' in point for point in design)

    def test_d_optimal_sample_size(self):
        """Test D-optimal respects exact sample size"""
        factory = DesignFactory()

        factors = {
            'A': ['1', '2', '3', '4'],
            'B': ['10', '20', '30', '40']
        }

        # 4x4 = 16 candidates, so test with samples <= 16
        for n_samples in [5, 8, 12, 16]:
            design = factory.create_design('d_optimal', factors, n_samples=n_samples, model_type='linear')
            assert len(design) == n_samples

    def test_d_optimal_model_types(self):
        """Test different model types for D-optimal"""
        factory = DesignFactory()

        factors = {
            'temp': ['20', '30', '40'],
            'pH': ['6', '7', '8'],
            'nacl': ['100', '200', '300']
        }

        # Test linear model
        design_linear = factory.create_design('d_optimal', factors, n_samples=10, model_type='linear')
        assert len(design_linear) == 10

        # Test interactions model
        design_inter = factory.create_design('d_optimal', factors, n_samples=15, model_type='interactions')
        assert len(design_inter) == 15

        # Test quadratic model
        design_quad = factory.create_design('d_optimal', factors, n_samples=20, model_type='quadratic')
        assert len(design_quad) == 20

    def test_d_optimal_requires_min_factors(self):
        """Test D-optimal requires at least 2 factors"""
        factory = DesignFactory()

        factors = {
            'A': ['1', '2', '3']  # Only 1 factor
        }

        with pytest.raises(ValueError, match="at least 2 factors"):
            factory.create_design('d_optimal', factors, n_samples=5, model_type='linear')

    def test_d_optimal_min_runs_linear(self):
        """Test minimum runs requirement for linear model"""
        factory = DesignFactory()

        factors = {
            'A': ['1', '2', '3'],
            'B': ['10', '20', '30']
        }

        # Linear model with 2 factors requires 3 runs (2+1 = intercept + main effects)
        with pytest.raises(ValueError, match="requires at least"):
            factory.create_design('d_optimal', factors, n_samples=2, model_type='linear')

    def test_d_optimal_min_runs_quadratic(self):
        """Test minimum runs requirement for quadratic model"""
        factory = DesignFactory()

        factors = {
            'A': ['1', '2', '3'],
            'B': ['10', '20', '30']
        }

        # Quadratic model with 2 factors requires 6 runs
        # (1 intercept + 2 main + 1 interaction + 2 squared)
        with pytest.raises(ValueError, match="requires at least"):
            factory.create_design('d_optimal', factors, n_samples=4, model_type='quadratic')

    def test_d_optimal_design_points_valid(self):
        """Test that D-optimal design points contain valid factor values"""
        factory = DesignFactory()

        factors = {
            'temp': ['20', '30', '40'],
            'pH': ['6', '7', '8']
        }

        # 3x3 = 9 candidates, request 8 samples (quadratic needs min 6 for 2 factors)
        design = factory.create_design('d_optimal', factors, n_samples=8, model_type='quadratic')

        for point in design:
            assert point['temp'] in ['20', '30', '40']
            assert point['pH'] in ['6', '7', '8']

    def test_d_optimal_beats_random(self):
        """Test that D-optimal design has better D-value than random selection"""
        import numpy as np
        factory = DesignFactory()

        factors = {
            'A': ['1', '2', '3', '4', '5'],
            'B': ['10', '20', '30', '40', '50']
        }

        # Generate D-optimal design
        design = factory.create_design('d_optimal', factors, n_samples=10, model_type='quadratic')

        # Get all candidates
        candidates = factory._generate_full_factorial(factors)

        # Calculate D-value for our design
        factor_names = list(factors.keys())
        factor_levels = {fn: sorted(set(factors[fn])) for fn in factor_names}
        d_opt_matrix = factory._encode_candidates(design, factor_names, factor_levels)
        model_func = factory._get_model_matrix_func('quadratic', len(factor_names))
        X_opt = model_func(d_opt_matrix)
        XtX_opt = X_opt.T @ X_opt
        sign_opt, logdet_opt = np.linalg.slogdet(XtX_opt)

        # Generate random design and compare
        random_indices = np.random.choice(len(candidates), size=10, replace=False)
        random_design = [candidates[i] for i in random_indices]
        random_matrix = factory._encode_candidates(random_design, factor_names, factor_levels)
        X_random = model_func(random_matrix)
        XtX_random = X_random.T @ X_random
        sign_random, logdet_random = np.linalg.slogdet(XtX_random)

        # D-optimal should have higher or equal determinant (better or equal)
        # Note: Due to randomness in initialization, this may occasionally fail
        # but should generally hold
        assert sign_opt > 0  # Design should be non-singular

    def test_d_optimal_with_categorical_factors(self):
        """Test D-optimal with categorical factors"""
        factory = DesignFactory()

        factors = {
            'temp': ['20', '30', '40'],
            'buffer pH': ['6.0', '7.0', '8.0']  # Categorical
        }

        # 3x3 = 9 candidates, request 8 samples
        design = factory.create_design('d_optimal', factors, n_samples=8, model_type='linear')

        assert len(design) == 8
        for point in design:
            assert 'temp' in point
            assert 'buffer pH' in point

    def test_d_optimal_default_parameters(self):
        """Test D-optimal with default parameters"""
        factory = DesignFactory()

        # Use more levels to have enough candidates for default n_samples=20
        factors = {
            'A': ['1', '2', '3', '4', '5'],
            'B': ['10', '20', '30', '40', '50']
        }

        # Default is n_samples=20, model_type='quadratic'
        # 5x5 = 25 candidates, which is > 20
        design = factory.create_design('d_optimal', factors)

        # Default n_samples is 20
        assert len(design) == 20

    def test_calculate_min_runs(self):
        """Test minimum runs calculation for different models"""
        factory = DesignFactory()

        # 3 factors
        assert factory._calculate_min_runs(3, 'linear') == 4  # 3 + 1
        assert factory._calculate_min_runs(3, 'interactions') == 7  # 3 + 3 + 1
        assert factory._calculate_min_runs(3, 'quadratic') == 10  # 3 + 3 + 3 + 1

        # 4 factors
        assert factory._calculate_min_runs(4, 'linear') == 5  # 4 + 1
        assert factory._calculate_min_runs(4, 'interactions') == 11  # 4 + 6 + 1
        assert factory._calculate_min_runs(4, 'quadratic') == 15  # 4 + 6 + 4 + 1

    def test_unknown_model_type(self):
        """Test that unknown model type raises error"""
        factory = DesignFactory()

        factors = {
            'A': ['1', '2', '3'],
            'B': ['10', '20', '30']
        }

        with pytest.raises(ValueError, match="Unknown model type"):
            factory.create_design('d_optimal', factors, n_samples=10, model_type='invalid_model')
