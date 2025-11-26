"""
Integration test: Verify that per-level concentration handling works for all design types.
Tests that when concentration factors are excluded, all design types generate the correct
number of samples and maintain their design properties.
"""
import pytest
from core.design_factory import DesignFactory


class TestPerLevelWithAllDesignTypes:
    """Test all design types with per-level concentration scenario"""

    @pytest.fixture
    def factors_without_concentration(self):
        """Factors dict AFTER per-level fix (concentration excluded)"""
        return {
            "detergent": ["None", "DDM", "LMNG", "OG"],  # 4 levels
            "nacl": ["0", "100", "200", "300"],  # 4 levels
            "glycerol": ["0", "5", "10", "15"]  # 4 levels
        }

    @pytest.fixture
    def factors_with_concentration(self):
        """Factors dict BEFORE per-level fix (concentration incorrectly included)"""
        return {
            "detergent": ["None", "DDM", "LMNG", "OG"],
            "detergent_concentration": ["0", "0.5", "1.0", "2.0"],
            "nacl": ["0", "100", "200"],
            "glycerol": ["0", "5", "10"]
        }

    def test_full_factorial_count(self, factors_without_concentration):
        """Full factorial should generate all combinations"""
        factory = DesignFactory()
        design = factory.generate_design(
            factors=factors_without_concentration,
            design_type="full_factorial"
        )

        # 4 * 4 * 4 = 64 combinations
        assert len(design) == 64
        assert "detergent" in design.columns
        assert "detergent_concentration" not in design.columns

    def test_lhs_generates_requested_samples(self, factors_without_concentration):
        """LHS should generate exactly the requested number of samples"""
        factory = DesignFactory()

        for n_samples in [48, 96, 192]:
            design = factory.generate_design(
                factors=factors_without_concentration,
                design_type="lhs",
                n_samples=n_samples
            )

            assert len(design) == n_samples, \
                f"LHS should generate {n_samples} samples, got {len(design)}"
            assert "detergent_concentration" not in design.columns

    def test_d_optimal_generates_requested_samples(self, factors_without_concentration):
        """D-Optimal should generate requested number of samples"""
        factory = DesignFactory()

        try:
            design = factory.generate_design(
                factors=factors_without_concentration,
                design_type="d_optimal",
                n_samples=48,
                model_type="linear"
            )

            # D-Optimal may generate close to requested (within reason)
            assert 40 <= len(design) <= 60, \
                f"D-Optimal should generate ~48 samples, got {len(design)}"
            assert "detergent_concentration" not in design.columns
        except ImportError:
            pytest.skip("pyDOE3 not available")

    def test_fractional_factorial_with_two_levels(self):
        """Fractional factorial should work with 2-level factors"""
        factory = DesignFactory()

        # 2-level factors only
        factors = {
            "detergent": ["None", "DDM"],  # Would normally pair with concentration
            "nacl": ["0", "200"],
            "glycerol": ["0", "10"]
        }

        try:
            design = factory.generate_design(
                factors=factors,
                design_type="fractional",
                resolution="III"
            )

            # Should generate 2^(k-p) runs where k=3, typically 4 runs for resolution III
            assert len(design) >= 4
            assert "detergent_concentration" not in design.columns
        except ImportError:
            pytest.skip("pyDOE3 not available")

    def test_plackett_burman_with_two_levels(self):
        """Plackett-Burman should work with 2-level factors"""
        factory = DesignFactory()

        factors = {
            "detergent": ["None", "DDM"],
            "nacl": ["0", "200"],
            "glycerol": ["0", "10"]
        }

        try:
            design = factory.generate_design(
                factors=factors,
                design_type="plackett_burman"
            )

            # PB generates N+1 runs for N factors, rounded to multiple of 4
            # For 3 factors, typically 4 runs
            assert len(design) >= 4
            assert "detergent_concentration" not in design.columns
        except ImportError:
            pytest.skip("pyDOE3 not available")

    def test_central_composite_design(self):
        """Central Composite Design should work without concentration factors"""
        factory = DesignFactory()

        # CCD requires numeric factors with 3+ levels
        factors = {
            "nacl": ["0", "100", "200", "300"],
            "glycerol": ["0", "5", "10", "15"]
        }

        try:
            design = factory.generate_design(
                factors=factors,
                design_type="central_composite",
                ccd_type="faced"
            )

            # CCD generates 2^k + 2k + center points
            # For 2 factors: 4 + 4 + center = 9+ runs
            assert len(design) >= 9
        except ImportError:
            pytest.skip("pyDOE3 not available")

    def test_box_behnken_design(self):
        """Box-Behnken should work without concentration factors"""
        factory = DesignFactory()

        # BB requires 3+ factors with 3+ levels each
        factors = {
            "nacl": ["0", "100", "200"],
            "glycerol": ["0", "5", "10"],
            "kcl": ["0", "50", "100"]
        }

        try:
            design = factory.generate_design(
                factors=factors,
                design_type="box_behnken"
            )

            # BB for 3 factors typically generates 13-15 runs
            assert len(design) >= 12
        except ImportError:
            pytest.skip("pyDOE3 not available")

    def test_lhs_space_filling_quality(self, factors_without_concentration):
        """LHS should maintain space-filling quality without concentration factor"""
        factory = DesignFactory()

        design = factory.generate_design(
            factors=factors_without_concentration,
            design_type="lhs",
            n_samples=96
        )

        # Check that all factor levels are well-represented
        for factor in factors_without_concentration.keys():
            unique_count = design[factor].nunique()
            expected_levels = len(factors_without_concentration[factor])

            # Should use most or all levels
            assert unique_count >= expected_levels * 0.75, \
                f"Factor {factor} should use most levels, got {unique_count}/{expected_levels}"

    def test_categorical_factors_distributed_evenly(self, factors_without_concentration):
        """Categorical factors should be distributed evenly in LHS"""
        factory = DesignFactory()

        design = factory.generate_design(
            factors=factors_without_concentration,
            design_type="lhs",
            n_samples=80  # Divisible by 4 (number of detergent levels)
        )

        # Detergent should appear roughly equally (80/4 = 20 times each)
        detergent_counts = design["detergent"].value_counts()
        for level, count in detergent_counts.items():
            assert 15 <= count <= 25, \
                f"Detergent level {level} should appear ~20 times, got {count}"


class TestBackwardCompatibility:
    """Ensure fix doesn't break normal mode (without per-level concentrations)"""

    def test_concentration_factors_work_normally(self):
        """When per-level is NOT active, concentration factors should work normally"""
        factory = DesignFactory()

        # Normal mode: concentration factor is explicitly included
        factors = {
            "detergent": ["DDM", "LMNG"],
            "detergent_concentration": ["0.5", "1.0", "2.0"],
            "nacl": ["0", "100"]
        }

        design = factory.generate_design(
            factors=factors,
            design_type="full_factorial"
        )

        # Should generate 2 * 3 * 2 = 12 combinations
        assert len(design) == 12
        assert "detergent_concentration" in design.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
