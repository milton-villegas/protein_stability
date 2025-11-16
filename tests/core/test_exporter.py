"""Tests for ResultsExporter class"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from core.exporter import ResultsExporter


@pytest.fixture
def sample_results():
    """Create sample analysis results for export testing"""
    return {
        'model_stats': {
            'AIC': 45.2,
            'BIC': 48.7,
            'R-squared': 0.85,
            'Adj. R-squared': 0.82,
            'F-statistic': 125.5,
            'Prob (F-statistic)': 0.001
        },
        'coefficients': pd.DataFrame({
            'coef': [0.5, 0.3, -0.2, 0.15],
            'std err': [0.05, 0.04, 0.03, 0.05],
            't': [10.0, 7.5, -6.67, 3.0],
            'p-value': [0.00001, 0.0001, 0.0005, 0.045]
        }, index=['Intercept', 'Factor1', 'Factor2', 'Factor1:Factor2'])
    }


@pytest.fixture
def sample_main_effects():
    """Create sample main effects for export testing"""
    return {
        'Factor1': pd.DataFrame({
            'Mean Response': [0.5, 0.7],
            'Std Dev': [0.05, 0.06]
        }, index=[10, 20]),
        'Factor2': pd.DataFrame({
            'Mean Response': [0.6, 0.8],
            'Std Dev': [0.04, 0.07]
        }, index=[100, 200])
    }


class TestResultsExporterInit:
    """Test ResultsExporter initialization"""

    def test_init_creates_none_attributes(self):
        """Test that initialization creates None attributes"""
        exporter = ResultsExporter()

        assert exporter.results is None
        assert exporter.main_effects is None


class TestResultsExporterSetResults:
    """Test setting results"""

    def test_set_results_stores_data(self, sample_results, sample_main_effects):
        """Test that set_results correctly stores data"""
        exporter = ResultsExporter()
        exporter.set_results(sample_results, sample_main_effects)

        assert exporter.results == sample_results
        assert exporter.main_effects == sample_main_effects

    def test_set_results_overwrites_previous(self, sample_results, sample_main_effects):
        """Test that set_results overwrites previous data"""
        exporter = ResultsExporter()

        # Set first time
        exporter.set_results(sample_results, sample_main_effects)
        first_results = exporter.results

        # Set second time with different data
        new_results = {'model_stats': {'AIC': 50.0}}
        new_main_effects = {}
        exporter.set_results(new_results, new_main_effects)

        assert exporter.results != first_results
        assert exporter.results == new_results


class TestResultsExporterExportExcel:
    """Test Excel export functionality"""

    def test_export_creates_file(self, sample_results, sample_main_effects, tmp_path):
        """Test that export creates an Excel file"""
        exporter = ResultsExporter()
        exporter.set_results(sample_results, sample_main_effects)

        filepath = tmp_path / "test_export.xlsx"
        exporter.export_statistics_excel(str(filepath))

        assert filepath.exists()

    def test_export_contains_all_sheets(self, sample_results, sample_main_effects, tmp_path):
        """Test that exported file contains all required sheets"""
        exporter = ResultsExporter()
        exporter.set_results(sample_results, sample_main_effects)

        filepath = tmp_path / "test_export.xlsx"
        exporter.export_statistics_excel(str(filepath))

        # Read back and verify sheets
        excel_file = pd.ExcelFile(filepath)
        sheets = excel_file.sheet_names

        assert 'Model Statistics' in sheets
        assert 'Coefficients' in sheets
        assert 'Main Effects' in sheets
        assert 'Significant Factors' in sheets

    def test_export_model_statistics_content(self, sample_results, sample_main_effects, tmp_path):
        """Test that Model Statistics sheet contains correct data"""
        exporter = ResultsExporter()
        exporter.set_results(sample_results, sample_main_effects)

        filepath = tmp_path / "test_export.xlsx"
        exporter.export_statistics_excel(str(filepath))

        # Read back Model Statistics sheet
        model_stats_df = pd.read_excel(filepath, sheet_name='Model Statistics', index_col=0)

        assert 'AIC' in model_stats_df.index
        assert 'R-squared' in model_stats_df.index
        assert model_stats_df.loc['AIC', 'Value'] == 45.2
        assert model_stats_df.loc['R-squared', 'Value'] == 0.85

    def test_export_coefficients_content(self, sample_results, sample_main_effects, tmp_path):
        """Test that Coefficients sheet contains correct data"""
        exporter = ResultsExporter()
        exporter.set_results(sample_results, sample_main_effects)

        filepath = tmp_path / "test_export.xlsx"
        exporter.export_statistics_excel(str(filepath))

        # Read back Coefficients sheet
        coef_df = pd.read_excel(filepath, sheet_name='Coefficients', index_col=0)

        assert 'Intercept' in coef_df.index
        assert 'Factor1' in coef_df.index
        assert 'coef' in coef_df.columns
        assert coef_df.loc['Factor1', 'coef'] == 0.3

    def test_export_main_effects_content(self, sample_results, sample_main_effects, tmp_path):
        """Test that Main Effects sheet contains correct data"""
        exporter = ResultsExporter()
        exporter.set_results(sample_results, sample_main_effects)

        filepath = tmp_path / "test_export.xlsx"
        exporter.export_statistics_excel(str(filepath))

        # Read back Main Effects sheet
        main_effects_df = pd.read_excel(filepath, sheet_name='Main Effects')

        # Check that data is present (column names may vary due to pandas export/import)
        assert 'Level' in main_effects_df.columns
        assert 'Mean Response' in main_effects_df.columns
        # Factor values should be in the dataframe somewhere
        assert 'Factor1' in main_effects_df.values
        assert 'Factor2' in main_effects_df.values

    def test_export_significant_factors_filters_correctly(self, sample_results, sample_main_effects, tmp_path):
        """Test that Significant Factors sheet filters p < 0.05"""
        exporter = ResultsExporter()
        exporter.set_results(sample_results, sample_main_effects)

        filepath = tmp_path / "test_export.xlsx"
        exporter.export_statistics_excel(str(filepath))

        # Read back Significant Factors sheet
        sig_factors_df = pd.read_excel(filepath, sheet_name='Significant Factors', index_col=0)

        # All factors should have p < 0.05
        assert len(sig_factors_df) == 3  # Factor1, Factor2, Factor1:Factor2 (not Intercept)
        assert 'Intercept' not in sig_factors_df.index

    def test_export_pvalues_present(self, sample_results, sample_main_effects, tmp_path):
        """Test that p-values are exported correctly"""
        exporter = ResultsExporter()
        exporter.set_results(sample_results, sample_main_effects)

        filepath = tmp_path / "test_export.xlsx"
        exporter.export_statistics_excel(str(filepath))

        # Read back Coefficients sheet
        coef_df = pd.read_excel(filepath, sheet_name='Coefficients', index_col=0)

        # p-values should be present and numeric (Excel may convert string back to float)
        p_value = coef_df.loc['Factor1', 'p-value']
        assert p_value is not None
        # Should be small (< 0.05 for Factor1)
        assert float(p_value) < 0.001


class TestResultsExporterIntegration:
    """Integration tests for complete export workflow"""

    def test_full_export_workflow(self, sample_results, sample_main_effects, tmp_path):
        """Test complete workflow from initialization to export"""
        exporter = ResultsExporter()

        # Set results
        exporter.set_results(sample_results, sample_main_effects)

        # Export
        filepath = tmp_path / "full_export.xlsx"
        exporter.export_statistics_excel(str(filepath))

        # Verify
        assert filepath.exists()
        excel_file = pd.ExcelFile(filepath)
        assert len(excel_file.sheet_names) == 4

    def test_multiple_exports(self, sample_results, sample_main_effects, tmp_path):
        """Test that multiple exports work correctly"""
        exporter = ResultsExporter()
        exporter.set_results(sample_results, sample_main_effects)

        # First export
        filepath1 = tmp_path / "export1.xlsx"
        exporter.export_statistics_excel(str(filepath1))

        # Second export with new data
        new_results = sample_results.copy()
        new_results['model_stats']['AIC'] = 60.0
        exporter.set_results(new_results, sample_main_effects)

        filepath2 = tmp_path / "export2.xlsx"
        exporter.export_statistics_excel(str(filepath2))

        # Both should exist and be different
        assert filepath1.exists()
        assert filepath2.exists()

        model_stats1 = pd.read_excel(filepath1, sheet_name='Model Statistics', index_col=0)
        model_stats2 = pd.read_excel(filepath2, sheet_name='Model Statistics', index_col=0)

        assert model_stats1.loc['AIC', 'Value'] == 45.2
        assert model_stats2.loc['AIC', 'Value'] == 60.0
