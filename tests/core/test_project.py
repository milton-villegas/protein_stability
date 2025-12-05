"""Tests for DoEProject class"""
import pytest
import pandas as pd
import pickle
from pathlib import Path
from core.project import DoEProject


class TestDoEProjectInit:
    """Test DoEProject initialization"""

    def test_init_creates_attributes(self):
        """Test that initialization creates expected attributes"""
        project = DoEProject()

        assert project.name == "Untitled Project"
        assert project.created_date is not None
        assert project.modified_date is not None
        assert project._factors == {}
        assert project._stock_concs == {}
        assert project._per_level_concs == {}
        assert project.design_matrix is None
        assert project.results_data is None
        assert project.clean_data is None
        assert project.response_column is None
        assert project.factor_columns == []
        assert project.categorical_factors == []
        assert project.numeric_factors == []
        assert project.ax_client is None
        assert project.optimization_history == []

    def test_init_sets_timestamps(self):
        """Test that creation and modification dates are set"""
        project = DoEProject()

        assert project.created_date is not None
        assert project.modified_date is not None
        # Timestamps should be very close (within 1 second)
        time_diff = (project.modified_date - project.created_date).total_seconds()
        assert abs(time_diff) < 1.0


class TestDoEProjectFactorManagement:
    """Test factor management functionality"""

    def test_add_factor_simple(self):
        """Test adding a basic factor"""
        project = DoEProject()
        project.add_factor("pH", ["7.0", "7.5", "8.0"])

        factors = project.get_factors()
        assert "pH" in factors
        assert factors["pH"] == ["7.0", "7.5", "8.0"]

    def test_add_factor_with_stock_concentration(self):
        """Test adding factor with stock concentration"""
        project = DoEProject()
        project.add_factor("NaCl", ["100", "200"], stock_conc=5000.0)

        assert "NaCl" in project.get_factors()
        assert project.get_stock_conc("NaCl") == 5000.0

    def test_add_factor_strips_whitespace(self):
        """Test that factor names are stripped of whitespace"""
        project = DoEProject()
        project.add_factor("  pH  ", ["7.0"])

        # Name should be stripped
        factors = project.get_factors()
        assert "pH" in factors
        assert "  pH  " not in factors

    def test_add_factor_empty_name_raises_error(self):
        """Test that empty factor name raises ValueError"""
        project = DoEProject()

        with pytest.raises(ValueError, match="cannot be empty"):
            project.add_factor("", ["7.0"])

    def test_add_factor_whitespace_only_name_raises_error(self):
        """Test that whitespace-only name raises ValueError"""
        project = DoEProject()

        with pytest.raises(ValueError, match="cannot be empty"):
            project.add_factor("   ", ["7.0"])

    def test_add_factor_no_levels_raises_error(self):
        """Test that factor with no levels raises ValueError"""
        project = DoEProject()

        with pytest.raises(ValueError, match="At least one level"):
            project.add_factor("pH", [])

    def test_add_multiple_factors(self):
        """Test adding multiple factors"""
        project = DoEProject()
        project.add_factor("pH", ["7.0", "8.0"])
        project.add_factor("NaCl", ["100", "200", "300"])
        project.add_factor("Glycerol", ["5", "10"])

        factors = project.get_factors()
        assert len(factors) == 3
        assert "pH" in factors
        assert "NaCl" in factors
        assert "Glycerol" in factors

    def test_update_factor(self):
        """Test updating existing factor"""
        project = DoEProject()
        project.add_factor("pH", ["7.0", "8.0"])

        # Update levels
        project.update_factor("pH", ["6.5", "7.0", "7.5", "8.0"])

        factors = project.get_factors()
        assert len(factors["pH"]) == 4
        assert "6.5" in factors["pH"]

    def test_update_factor_with_stock_conc(self):
        """Test updating factor with new stock concentration"""
        project = DoEProject()
        project.add_factor("NaCl", ["100", "200"], stock_conc=1000.0)

        # Update with new stock concentration
        project.update_factor("NaCl", ["100", "200", "300"], stock_conc=5000.0)

        assert project.get_stock_conc("NaCl") == 5000.0
        assert len(project.get_factors()["NaCl"]) == 3

    def test_update_nonexistent_factor_raises_error(self):
        """Test that updating non-existent factor raises error"""
        project = DoEProject()

        with pytest.raises(ValueError, match="does not exist"):
            project.update_factor("NonExistent", ["1", "2"])

    def test_update_factor_empty_levels_raises_error(self):
        """Test that updating with empty levels raises error"""
        project = DoEProject()
        project.add_factor("pH", ["7.0"])

        with pytest.raises(ValueError, match="At least one level"):
            project.update_factor("pH", [])

    def test_remove_factor(self):
        """Test removing a factor"""
        project = DoEProject()
        project.add_factor("pH", ["7.0", "8.0"])
        project.add_factor("NaCl", ["100", "200"])

        project.remove_factor("pH")

        factors = project.get_factors()
        assert "pH" not in factors
        assert "NaCl" in factors

    def test_remove_factor_with_stock_conc(self):
        """Test removing factor also removes stock concentration"""
        project = DoEProject()
        project.add_factor("NaCl", ["100"], stock_conc=5000.0)

        project.remove_factor("NaCl")

        assert "NaCl" not in project.get_factors()
        assert project.get_stock_conc("NaCl") is None

    def test_remove_nonexistent_factor_no_error(self):
        """Test that removing non-existent factor doesn't raise error"""
        project = DoEProject()

        # Should not raise error
        project.remove_factor("NonExistent")

    def test_get_factors_returns_copy(self):
        """Test that get_factors returns a copy, not reference"""
        project = DoEProject()
        project.add_factor("pH", ["7.0", "8.0"])

        factors = project.get_factors()
        factors["pH"].append("9.0")  # Modify returned dict

        # Original should be unchanged
        original = project.get_factors()
        assert len(original["pH"]) == 2
        assert "9.0" not in original["pH"]

    def test_get_stock_conc_nonexistent_returns_none(self):
        """Test getting stock conc for non-existent factor returns None"""
        project = DoEProject()

        assert project.get_stock_conc("NonExistent") is None

    def test_get_all_stock_concs(self):
        """Test getting all stock concentrations"""
        project = DoEProject()
        project.add_factor("NaCl", ["100"], stock_conc=5000.0)
        project.add_factor("KCl", ["50"], stock_conc=3000.0)
        project.add_factor("pH", ["7.0"])  # No stock conc

        stock_concs = project.get_all_stock_concs()
        assert stock_concs["NaCl"] == 5000.0
        assert stock_concs["KCl"] == 3000.0
        assert "pH" not in stock_concs

    def test_clear_factors(self):
        """Test clearing all factors"""
        project = DoEProject()
        project.add_factor("pH", ["7.0"], stock_conc=100.0)
        project.add_factor("NaCl", ["100"])
        project.design_matrix = pd.DataFrame({"pH": [7.0]})

        project.clear_factors()

        assert project.get_factors() == {}
        assert project.get_all_stock_concs() == {}
        assert project.design_matrix is None


class TestDoEProjectDataManagement:
    """Test data loading and preprocessing"""

    def test_detect_columns_numeric_and_categorical(self, tmp_path):
        """Test detecting numeric and categorical columns"""
        # Create test data
        data = pd.DataFrame({
            'pH': [7.0, 7.5, 8.0, 7.0],
            'NaCl': [100, 150, 200, 100],
            'Buffer': ['Tris', 'HEPES', 'Tris', 'HEPES'],
            'Response': [0.5, 0.8, 0.6, 0.52]
        })

        # Save to Excel
        excel_path = tmp_path / "test_data.xlsx"
        data.to_excel(excel_path, index=False)

        # Load and detect
        project = DoEProject()
        project.load_results(str(excel_path))
        project.detect_columns('Response')

        assert project.response_column == 'Response'
        assert 'pH' in project.numeric_factors
        assert 'NaCl' in project.numeric_factors
        assert 'Buffer' in project.categorical_factors
        assert 'Response' not in project.factor_columns

    def test_detect_columns_no_data_raises_error(self):
        """Test that detecting columns without data raises error"""
        project = DoEProject()

        with pytest.raises(ValueError, match="No results data"):
            project.detect_columns('Response')

    def test_preprocess_data_drops_missing_response(self, tmp_path):
        """Test that preprocessing drops rows with missing response"""
        data = pd.DataFrame({
            'pH': [7.0, 7.5, 8.0, 7.0],
            'Response': [0.5, None, 0.6, 0.52]  # One missing
        })

        excel_path = tmp_path / "test_data.xlsx"
        data.to_excel(excel_path, index=False)

        project = DoEProject()
        project.load_results(str(excel_path))
        project.detect_columns('Response')

        clean = project.preprocess_data()

        # Should drop the row with missing response
        assert len(clean) == 3
        assert clean['Response'].notna().all()

    def test_preprocess_data_no_data_raises_error(self):
        """Test that preprocessing without data raises error"""
        project = DoEProject()

        with pytest.raises(ValueError, match="No results data"):
            project.preprocess_data()

    def test_preprocess_data_fills_categorical_na(self, tmp_path):
        """Test that preprocessing fills categorical NaN with 'None'"""
        data = pd.DataFrame({
            'Buffer': ['Tris', None, 'HEPES'],
            'Response': [0.5, 0.6, 0.7]
        })

        excel_path = tmp_path / "test_data.xlsx"
        data.to_excel(excel_path, index=False)

        project = DoEProject()
        project.load_results(str(excel_path))
        project.categorical_factors = ['Buffer']
        project.numeric_factors = []
        project.factor_columns = ['Buffer']
        project.response_column = 'Response'

        clean = project.preprocess_data()

        # Missing categorical should be 'None' string
        assert clean['Buffer'].iloc[1] == 'None'

    def test_preprocess_data_drops_numeric_na(self, tmp_path):
        """Test that preprocessing drops rows with missing numeric factors"""
        data = pd.DataFrame({
            'pH': [7.0, None, 8.0],
            'Response': [0.5, 0.6, 0.7]
        })

        excel_path = tmp_path / "test_data.xlsx"
        data.to_excel(excel_path, index=False)

        project = DoEProject()
        project.load_results(str(excel_path))
        project.numeric_factors = ['pH']
        project.categorical_factors = []
        project.factor_columns = ['pH']
        project.response_column = 'Response'

        clean = project.preprocess_data()

        # Should drop row with missing pH
        assert len(clean) == 2


class TestDoEProjectPersistence:
    """Test save/load functionality"""

    def test_save_and_load_basic(self, tmp_path):
        """Test basic save and load round-trip"""
        project = DoEProject()
        project.name = "Test Project"
        project.add_factor("pH", ["7.0", "8.0"])
        project.add_factor("NaCl", ["100", "200"], stock_conc=5000.0)

        # Save
        save_path = tmp_path / "project.pkl"
        project.save(str(save_path))

        # Load
        loaded = DoEProject.load(str(save_path))

        assert loaded.name == "Test Project"
        assert "pH" in loaded.get_factors()
        assert "NaCl" in loaded.get_factors()
        assert loaded.get_stock_conc("NaCl") == 5000.0

    def test_save_and_load_with_data(self, tmp_path):
        """Test save/load with experimental data"""
        project = DoEProject()
        project.name = "Data Project"
        project.results_data = pd.DataFrame({
            'pH': [7.0, 8.0],
            'Response': [0.5, 0.8]
        })

        # Save
        save_path = tmp_path / "project.pkl"
        project.save(str(save_path))

        # Load
        loaded = DoEProject.load(str(save_path))

        assert loaded.name == "Data Project"
        assert loaded.results_data is not None
        assert len(loaded.results_data) == 2

    def test_save_updates_modified_date(self, tmp_path):
        """Test that saving updates modified date"""
        import time
        project = DoEProject()
        original_modified = project.modified_date

        time.sleep(0.01)  # Small delay

        save_path = tmp_path / "project.pkl"
        project.save(str(save_path))

        # Modified date should be updated
        assert project.modified_date > original_modified

    def test_save_excludes_ax_client(self, tmp_path):
        """Test that ax_client is not saved (not picklable)"""
        project = DoEProject()
        project.ax_client = "MockAxClient"  # Not actually Ax, just testing

        save_path = tmp_path / "project.pkl"
        project.save(str(save_path))

        # ax_client should still be present in original
        assert project.ax_client == "MockAxClient"

        # But not in loaded project
        loaded = DoEProject.load(str(save_path))
        assert loaded.ax_client is None

    def test_load_from_file(self, tmp_path):
        """Test loading from file path"""
        project = DoEProject()
        project.name = "Saved Project"

        save_path = tmp_path / "test.json"
        project.save(str(save_path))

        # Load using classmethod
        loaded = DoEProject.load(str(save_path))
        assert loaded.name == "Saved Project"


class TestDoEProjectRepr:
    """Test string representation"""

    def test_repr_basic(self):
        """Test __repr__ output"""
        project = DoEProject()
        project.name = "My Project"
        project.add_factor("pH", ["7.0"])
        project.add_factor("NaCl", ["100"])

        repr_str = repr(project)

        assert "My Project" in repr_str
        assert "factors=2" in repr_str
        assert "has_results=False" in repr_str

    def test_repr_with_results(self, tmp_path):
        """Test __repr__ with results data"""
        project = DoEProject()
        project.results_data = pd.DataFrame({'pH': [7.0]})

        repr_str = repr(project)

        assert "has_results=True" in repr_str


class TestDoEProjectIntegration:
    """Integration tests for complete workflows"""

    def test_full_design_workflow(self):
        """Test complete design workflow"""
        project = DoEProject()
        project.name = "Full Factorial Design"

        # Add factors
        project.add_factor("pH", ["7.0", "8.0"])
        project.add_factor("NaCl", ["100", "200"], stock_conc=5000.0)

        # Verify
        factors = project.get_factors()
        assert len(factors) == 2
        assert project.get_stock_conc("NaCl") == 5000.0

    def test_full_analysis_workflow(self, tmp_path):
        """Test complete analysis workflow"""
        # Create test data
        data = pd.DataFrame({
            'pH': [7.0, 7.5, 8.0, 7.0, 7.5, 8.0],
            'NaCl': [100, 100, 100, 200, 200, 200],
            'Response': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        })

        excel_path = tmp_path / "results.xlsx"
        data.to_excel(excel_path, index=False)

        # Create project
        project = DoEProject()
        project.name = "Analysis Project"

        # Load data
        project.load_results(str(excel_path))
        assert project.results_data is not None

        # Detect columns
        project.detect_columns('Response')
        assert len(project.numeric_factors) == 2

        # Preprocess
        clean = project.preprocess_data()
        assert len(clean) == 6

    def test_save_load_roundtrip_preserves_all_data(self, tmp_path):
        """Test that save/load preserves all project data"""
        # Create complex project
        project = DoEProject()
        project.name = "Complex Project"
        project.add_factor("pH", ["7.0", "8.0"], stock_conc=1000.0)
        project.add_factor("Buffer", ["Tris", "HEPES"])

        data = pd.DataFrame({
            'pH': [7.0, 8.0],
            'Buffer': ['Tris', 'HEPES'],
            'Response': [0.5, 0.8]
        })

        data_path = tmp_path / "data.xlsx"
        data.to_excel(data_path, index=False)

        project.load_results(str(data_path))
        project.detect_columns('Response')
        project.preprocess_data()

        # Save
        save_path = tmp_path / "complex.pkl"
        project.save(str(save_path))

        # Load
        loaded = DoEProject.load(str(save_path))

        # Verify everything
        assert loaded.name == "Complex Project"
        assert len(loaded.get_factors()) == 2
        assert loaded.get_stock_conc("pH") == 1000.0
        assert loaded.results_data is not None
        assert loaded.clean_data is not None
        assert loaded.response_column == 'Response'
        assert 'pH' in loaded.numeric_factors
        assert 'Buffer' in loaded.categorical_factors


class TestDoEProjectPerLevelConcentrations:
    """Test per-level concentration functionality for categorical factors"""

    def test_set_per_level_concs(self):
        """Test setting per-level concentrations for a factor"""
        project = DoEProject()
        project.add_factor("detergent", ["DDM", "LMNG", "OG"])

        per_level = {
            "DDM": {"stock": 0.2, "final": 0.006},
            "LMNG": {"stock": 0.01, "final": 0.0007},
            "OG": {"stock": 10.0, "final": 0.37}
        }
        project.set_per_level_concs("detergent", per_level)

        result = project.get_per_level_concs("detergent")
        assert result is not None
        assert "DDM" in result
        assert result["DDM"]["stock"] == 0.2
        assert result["DDM"]["final"] == 0.006

    def test_has_per_level_concs_true(self):
        """Test has_per_level_concs returns True when configured"""
        project = DoEProject()
        project.set_per_level_concs("detergent", {"DDM": {"stock": 0.2, "final": 0.006}})

        assert project.has_per_level_concs("detergent") is True

    def test_has_per_level_concs_false(self):
        """Test has_per_level_concs returns False when not configured"""
        project = DoEProject()

        assert project.has_per_level_concs("detergent") is False

    def test_has_per_level_concs_empty(self):
        """Test has_per_level_concs returns False for empty dict"""
        project = DoEProject()
        project.set_per_level_concs("detergent", {})

        assert project.has_per_level_concs("detergent") is False

    def test_get_level_conc_stock(self):
        """Test getting stock concentration for specific level"""
        project = DoEProject()
        project.set_per_level_concs("detergent", {
            "DDM": {"stock": 0.2, "final": 0.006}
        })

        stock = project.get_level_conc("detergent", "DDM", "stock")
        assert stock == 0.2

    def test_get_level_conc_final(self):
        """Test getting final concentration for specific level"""
        project = DoEProject()
        project.set_per_level_concs("detergent", {
            "DDM": {"stock": 0.2, "final": 0.006}
        })

        final = project.get_level_conc("detergent", "DDM", "final")
        assert final == 0.006

    def test_get_level_conc_nonexistent_factor(self):
        """Test get_level_conc returns None for non-existent factor"""
        project = DoEProject()

        result = project.get_level_conc("nonexistent", "DDM", "stock")
        assert result is None

    def test_get_level_conc_nonexistent_level(self):
        """Test get_level_conc returns None for non-existent level"""
        project = DoEProject()
        project.set_per_level_concs("detergent", {
            "DDM": {"stock": 0.2, "final": 0.006}
        })

        result = project.get_level_conc("detergent", "LMNG", "stock")
        assert result is None

    def test_clear_per_level_concs(self):
        """Test clearing per-level concentrations for a factor"""
        project = DoEProject()
        project.set_per_level_concs("detergent", {
            "DDM": {"stock": 0.2, "final": 0.006}
        })

        project.clear_per_level_concs("detergent")

        assert project.has_per_level_concs("detergent") is False

    def test_clear_per_level_concs_nonexistent(self):
        """Test clearing non-existent per-level concentrations doesn't error"""
        project = DoEProject()

        # Should not raise error
        project.clear_per_level_concs("nonexistent")

    def test_get_all_per_level_concs(self):
        """Test getting all per-level concentrations"""
        project = DoEProject()
        project.set_per_level_concs("detergent", {
            "DDM": {"stock": 0.2, "final": 0.006}
        })
        project.set_per_level_concs("reducing_agent", {
            "DTT": {"stock": 1000, "final": 5}
        })

        all_concs = project.get_all_per_level_concs()

        assert "detergent" in all_concs
        assert "reducing_agent" in all_concs
        assert all_concs["detergent"]["DDM"]["stock"] == 0.2
        assert all_concs["reducing_agent"]["DTT"]["stock"] == 1000

    def test_remove_factor_clears_per_level_concs(self):
        """Test that removing a factor also clears its per-level concentrations"""
        project = DoEProject()
        project.add_factor("detergent", ["DDM", "LMNG"])
        project.set_per_level_concs("detergent", {
            "DDM": {"stock": 0.2, "final": 0.006}
        })

        project.remove_factor("detergent")

        assert project.has_per_level_concs("detergent") is False

    def test_clear_factors_clears_per_level_concs(self):
        """Test that clear_factors also clears per-level concentrations"""
        project = DoEProject()
        project.add_factor("detergent", ["DDM", "LMNG"])
        project.set_per_level_concs("detergent", {
            "DDM": {"stock": 0.2, "final": 0.006}
        })

        project.clear_factors()

        assert project.get_all_per_level_concs() == {}

    def test_save_load_preserves_per_level_concs(self, tmp_path):
        """Test that save/load preserves per-level concentrations"""
        project = DoEProject()
        project.add_factor("detergent", ["DDM", "LMNG", "OG"])
        project.set_per_level_concs("detergent", {
            "DDM": {"stock": 0.2, "final": 0.006},
            "LMNG": {"stock": 0.01, "final": 0.0007},
            "OG": {"stock": 10.0, "final": 0.37}
        })

        # Save
        save_path = tmp_path / "project.pkl"
        project.save(str(save_path))

        # Load
        loaded = DoEProject.load(str(save_path))

        # Verify per-level concs preserved
        assert loaded.has_per_level_concs("detergent") is True
        assert loaded.get_level_conc("detergent", "DDM", "stock") == 0.2
        assert loaded.get_level_conc("detergent", "LMNG", "final") == 0.0007
        assert loaded.get_level_conc("detergent", "OG", "stock") == 10.0
