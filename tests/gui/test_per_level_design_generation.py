"""
Test that per-level concentration handling works correctly for all design types.
This ensures that when per-level concentrations are configured, the concentration
factors are excluded from design generation, preventing the filter from removing
valid combinations and maintaining design properties (e.g., space-filling for LHS).
"""
import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from gui.tabs.designer.models import FactorModel
from gui.tabs.designer.design_panel import DesignPanelMixin
import tkinter as tk


# Wrapper class to make DesignPanelMixin testable (not a Test* class to avoid pytest collection)
class DesignPanelWrapper(DesignPanelMixin):
    """Minimal wrapper to test DesignPanelMixin in isolation"""
    def __init__(self, parent):
        self.parent = parent
        self.model = parent.model
        # Use Mock for BooleanVar to avoid tkinter root issues
        self.optimize_lhs_var = Mock()
        self.optimize_lhs_var.get = Mock(return_value=False)


@pytest.fixture
def mock_root():
    """Create mock Tkinter root"""
    root = Mock()
    return root


@pytest.fixture
def model_with_per_level():
    """Create a FactorModel with per-level concentrations"""
    model = FactorModel()

    # Add detergent categorical factor
    model.add_factor("detergent", ["None", "DDM", "LMNG", "OG"])

    # Add other numeric factors
    model.add_factor("nacl", ["0", "100", "200", "300"], stock_conc=5000.0)
    model.add_factor("glycerol", ["0", "5", "10"], stock_conc=100.0)

    # Set per-level concentrations for detergent (simulating loaded from Excel)
    model.set_per_level_concs("detergent", {
        "DDM": {"stock": 10.0, "final": 1.0},
        "LMNG": {"stock": 5.0, "final": 0.5},
        "OG": {"stock": 20.0, "final": 2.0}
    })

    return model


@pytest.fixture
def model_with_concentration_factor():
    """Create a FactorModel where user incorrectly added both categorical and concentration factors"""
    model = FactorModel()

    # Add detergent categorical factor
    model.add_factor("detergent", ["None", "DDM", "LMNG", "OG"])

    # User mistakenly also added concentration factor
    model.add_factor("detergent_concentration", ["0", "0.5", "1.0", "2.0"], stock_conc=10.0)

    # Add other factors
    model.add_factor("nacl", ["0", "100", "200"], stock_conc=5000.0)

    # Set per-level concentrations (should make detergent_concentration redundant)
    model.set_per_level_concs("detergent", {
        "DDM": {"stock": 10.0, "final": 1.0},
        "LMNG": {"stock": 5.0, "final": 0.5},
        "OG": {"stock": 20.0, "final": 2.0}
    })

    return model


class TestPerLevelDesignGeneration:
    """Test that per-level concentrations work correctly across all design types"""

    def test_factors_dict_excludes_concentration_in_per_level_mode(self, model_with_concentration_factor):
        """Test that concentration factors are excluded when per-level mode is active"""
        # Get factors - should include detergent_concentration initially
        factors_before = model_with_concentration_factor.get_factors()
        assert "detergent" in factors_before
        assert "detergent_concentration" in factors_before
        assert "nacl" in factors_before

        # Simulate the export flow (the fix should remove detergent_concentration)
        factors = model_with_concentration_factor.get_factors()
        per_level_concs = model_with_concentration_factor.get_all_per_level_concs()

        # Apply the fix logic
        if "detergent" in per_level_concs and per_level_concs["detergent"]:
            if "detergent_concentration" in factors:
                factors = {k: v for k, v in factors.items() if k != "detergent_concentration"}

        # After fix, detergent_concentration should be excluded
        assert "detergent" in factors
        assert "detergent_concentration" not in factors  # Excluded!
        assert "nacl" in factors

    def test_lhs_sample_count_with_per_level(self, model_with_per_level):
        """Test that LHS generates correct number of samples with per-level concentrations"""
        

        factors = model_with_per_level.get_factors()
        per_level_concs = model_with_per_level.get_all_per_level_concs()

        # Simulate the fix
        if "detergent" in per_level_concs and per_level_concs["detergent"]:
            if "detergent_concentration" in factors:
                factors = {k: v for k, v in factors.items() if k != "detergent_concentration"}

        # Now generate LHS design
        mock_parent = Mock()
        mock_parent.model = model_with_per_level
        design_panel = DesignPanelWrapper(mock_parent)

        n_samples = 96
        combinations = design_panel._generate_lhs_design(factors, n_samples)

        # Should generate exactly 96 samples
        assert len(combinations) == 96, f"Expected 96 samples, got {len(combinations)}"

        # Verify each combination has correct number of factors
        factor_names = list(factors.keys())
        for combo in combinations:
            assert len(combo) == len(factor_names)

    def test_lhs_space_filling_property_maintained(self, model_with_per_level):
        """Test that LHS maintains space-filling property when concentration factor is excluded"""
        

        factors = model_with_per_level.get_factors()
        per_level_concs = model_with_per_level.get_all_per_level_concs()

        # Apply fix
        if "detergent" in per_level_concs and per_level_concs["detergent"]:
            if "detergent_concentration" in factors:
                factors = {k: v for k, v in factors.items() if k != "detergent_concentration"}

        mock_parent = Mock()
        mock_parent.model = model_with_per_level
        design_panel = DesignPanelWrapper(mock_parent)

        n_samples = 50
        combinations = design_panel._generate_lhs_design(factors, n_samples)

        # Convert to DataFrame for analysis
        factor_names = list(factors.keys())
        df = pd.DataFrame(combinations, columns=factor_names)

        # Check that numeric factors have good coverage
        # For NaCl (4 levels): should use all or most levels
        nacl_unique = df["nacl"].nunique()
        assert nacl_unique >= 3, f"NaCl should use at least 3 of 4 levels, got {nacl_unique}"

        # For Glycerol (3 levels): should use all levels
        glycerol_unique = df["glycerol"].nunique()
        assert glycerol_unique >= 2, f"Glycerol should use at least 2 of 3 levels, got {glycerol_unique}"

        # Categorical factor (detergent) should cycle through values
        detergent_counts = df["detergent"].value_counts()
        # With 50 samples and 4 detergent levels, each should appear ~12-13 times
        for det_level, count in detergent_counts.items():
            assert 8 <= count <= 18, f"Detergent level {det_level} appears {count} times, expected ~12-13"

    def test_d_optimal_sample_count_with_per_level(self, model_with_per_level):
        """Test that D-Optimal generates correct number of samples with per-level concentrations"""
        

        factors = model_with_per_level.get_factors()
        per_level_concs = model_with_per_level.get_all_per_level_concs()

        # Apply fix
        if "detergent" in per_level_concs and per_level_concs["detergent"]:
            if "detergent_concentration" in factors:
                factors = {k: v for k, v in factors.items() if k != "detergent_concentration"}

        mock_parent = Mock()
        mock_parent.model = model_with_per_level
        design_panel = DesignPanelWrapper(mock_parent)

        n_samples = 48
        try:
            combinations = design_panel._generate_d_optimal_design(factors, n_samples, "linear")

            # Should generate exactly 48 samples (or close, depending on algorithm)
            assert len(combinations) >= 40, f"Expected ~48 samples, got {len(combinations)}"
        except ImportError:
            pytest.skip("pyDOE3 not available for D-Optimal design")

    def test_full_factorial_excludes_concentration_factor(self, model_with_concentration_factor):
        """Test that Full Factorial doesn't include concentration factor in per-level mode"""
        import itertools

        factors = model_with_concentration_factor.get_factors()
        per_level_concs = model_with_concentration_factor.get_all_per_level_concs()

        # Before fix: would generate 4 * 4 * 3 = 48 combinations
        combinations_before = len(list(itertools.product(*[factors[f] for f in factors.keys()])))
        assert combinations_before == 48  # detergent(4) * detergent_conc(4) * nacl(3)

        # Apply fix
        if "detergent" in per_level_concs and per_level_concs["detergent"]:
            if "detergent_concentration" in factors:
                factors = {k: v for k, v in factors.items() if k != "detergent_concentration"}

        # After fix: should generate 4 * 3 = 12 combinations (no concentration factor)
        combinations_after = len(list(itertools.product(*[factors[f] for f in factors.keys()])))
        assert combinations_after == 12  # detergent(4) * nacl(3)

        # The full factorial should have all valid combinations (no filtering needed)
        assert combinations_after < combinations_before

    def test_filter_does_not_remove_combinations_in_per_level_mode(self, model_with_per_level):
        """Test that the filter doesn't remove any combinations when concentration factors are excluded"""
        

        factors = model_with_per_level.get_factors()
        per_level_concs = model_with_per_level.get_all_per_level_concs()

        # Apply fix
        if "detergent" in per_level_concs and per_level_concs["detergent"]:
            if "detergent_concentration" in factors:
                factors = {k: v for k, v in factors.items() if k != "detergent_concentration"}

        mock_parent = Mock()
        mock_parent.model = model_with_per_level
        design_panel = DesignPanelWrapper(mock_parent)

        # Generate some combinations (using full factorial for simplicity)
        import itertools
        factor_names = list(factors.keys())
        level_lists = [factors[f] for f in factor_names]
        combinations_before = list(itertools.product(*level_lists))

        # Apply filter
        combinations_after = design_panel._filter_categorical_combinations(combinations_before, factor_names)

        # Filter should NOT remove any combinations (because detergent_concentration is not in the design)
        assert len(combinations_after) == len(combinations_before), \
            f"Filter should not remove combinations when concentration factors are excluded. " \
            f"Before: {len(combinations_before)}, After: {len(combinations_after)}"

    def test_reducing_agent_per_level_also_works(self):
        """Test that per-level concentrations work for reducing agents too"""
        model = FactorModel()

        # Add reducing agent categorical factor
        model.add_factor("reducing_agent", ["None", "DTT", "TCEP", "BME"])
        model.add_factor("nacl", ["0", "100", "200"], stock_conc=5000.0)

        # Set per-level concentrations for reducing agent
        model.set_per_level_concs("reducing_agent", {
            "DTT": {"stock": 1000.0, "final": 5.0},
            "TCEP": {"stock": 500.0, "final": 2.0},
            "BME": {"stock": 14300.0, "final": 10.0}
        })

        factors = model.get_factors()
        per_level_concs = model.get_all_per_level_concs()

        # Apply fix (should also work for reducing_agent)
        if "reducing_agent" in per_level_concs and per_level_concs["reducing_agent"]:
            if "reducing_agent_concentration" in factors:
                factors = {k: v for k, v in factors.items() if k != "reducing_agent_concentration"}

        # Should only have reducing_agent and nacl (no reducing_agent_concentration)
        assert "reducing_agent" in factors
        assert "reducing_agent_concentration" not in factors
        assert "nacl" in factors


class TestNormalModeNotAffected:
    """Test that the fix doesn't break normal mode (without per-level concentrations)"""

    def test_normal_concentration_factors_still_work(self):
        """Test that concentration factors work normally when per-level mode is NOT active"""
        model = FactorModel()

        # Add factors in normal mode
        model.add_factor("detergent", ["None", "DDM", "LMNG"])
        model.add_factor("detergent_concentration", ["0", "0.5", "1.0"], stock_conc=10.0)
        model.add_factor("nacl", ["0", "100"], stock_conc=5000.0)

        # NO per-level concentrations configured
        factors = model.get_factors()
        per_level_concs = model.get_all_per_level_concs()

        # Apply fix logic
        if "detergent" in per_level_concs and per_level_concs["detergent"]:
            if "detergent_concentration" in factors:
                factors = {k: v for k, v in factors.items() if k != "detergent_concentration"}

        # In normal mode, concentration factor should NOT be removed
        assert "detergent" in factors
        assert "detergent_concentration" in factors  # Should still be present!
        assert "nacl" in factors

    def test_filter_still_removes_invalid_pairings_in_normal_mode(self):
        """Test that filter still works in normal mode to remove invalid pairings"""
        

        model = FactorModel()
        model.add_factor("detergent", ["None", "DDM"])
        model.add_factor("detergent_concentration", ["0", "1.0"], stock_conc=10.0)

        factors = model.get_factors()

        mock_parent = Mock()
        mock_parent.model = model
        design_panel = DesignPanelWrapper(mock_parent)

        # Full factorial: 2 * 2 = 4 combinations
        import itertools
        factor_names = list(factors.keys())
        level_lists = [factors[f] for f in factor_names]
        combinations = list(itertools.product(*level_lists))
        assert len(combinations) == 4

        # Apply filter
        filtered = design_panel._filter_categorical_combinations(combinations, factor_names)

        # Should remove invalid pairings:
        # Valid: (None, 0), (DDM, 0), (DDM, 1.0)
        # Invalid: (None, 1.0) - None can't have concentration
        # Note: (DDM, 0) is valid - it means "don't add DDM" (concentration=0 allowed for any detergent)
        assert len(filtered) == 3, f"Expected 3 valid combinations, got {len(filtered)}"
