"""
Results Export Functions
Extracted from analysis_tab.py
"""
import pandas as pd
from typing import Dict, Any
from core.constants import SIGNIFICANCE_LEVEL, P_VALUE_PRECISION


class ResultsExporter:
    """Exports statistical analysis results to various formats"""

    def __init__(self) -> None:
        """Initialize exporter with empty results"""
        self.results: Dict[str, Any] = None
        self.main_effects: Dict[str, pd.DataFrame] = None

    def set_results(self, results: Dict[str, Any], main_effects: Dict[str, pd.DataFrame]) -> None:
        """
        Set results to export

        Args:
            results: Dictionary containing model statistics, coefficients, and fitted values
            main_effects: Dictionary mapping factor names to their main effects DataFrames
        """
        self.results = results
        self.main_effects = main_effects

    @staticmethod
    def _format_pvalue(pvalue: float) -> str:
        """
        Format p-value in scientific notation with consistent precision

        Args:
            pvalue: P-value to format

        Returns:
            Formatted p-value string (e.g., "1.234567e-05")
        """
        return f"{pvalue:.{P_VALUE_PRECISION}e}"

    def export_statistics_excel(self, filepath: str) -> None:
        """
        Export statistical analysis results to multi-sheet Excel file

        Creates sheets for:
        - Model Statistics (R², AIC, BIC, etc.)
        - Coefficients (all model terms with p-values)
        - Main Effects (mean/std per factor level)
        - Significant Factors (only terms with p < SIGNIFICANCE_LEVEL)

        Args:
            filepath: Path where Excel file should be saved
        """
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Model Statistics
            model_stats_df = pd.DataFrame([self.results['model_stats']]).T
            model_stats_df.columns = ['Value']
            model_stats_df.index.name = 'Statistic'
            model_stats_df.to_excel(writer, sheet_name='Model Statistics')

            # Coefficients
            coef_df = self.results['coefficients'].copy()
            coef_df['p-value'] = coef_df['p-value'].apply(self._format_pvalue)
            coef_df.to_excel(writer, sheet_name='Coefficients')

            # Main Effects
            main_effects_combined = []
            for factor, effects_df in self.main_effects.items():
                effects_df_copy = effects_df.copy()
                effects_df_copy.insert(0, 'Factor', factor)
                effects_df_copy.reset_index(inplace=True)
                effects_df_copy.rename(columns={effects_df_copy.columns[1]: 'Level'}, inplace=True)
                main_effects_combined.append(effects_df_copy)

            combined_df = pd.concat(main_effects_combined, ignore_index=True)
            combined_df.to_excel(writer, sheet_name='Main Effects', index=False)

            # Significant Factors
            sig_factors = self.results['coefficients'][
                self.results['coefficients']['p-value'].astype(float) < SIGNIFICANCE_LEVEL
            ].copy()
            sig_factors = sig_factors[sig_factors.index != 'Intercept']
            sig_factors['p-value'] = sig_factors['p-value'].apply(self._format_pvalue)
            sig_factors.to_excel(writer, sheet_name='Significant Factors')
