"""
Results Export Functions
Extracted from analysis_tab.py
"""
import pandas as pd
from typing import Dict


class ResultsExporter:
    """Exports results to various formats"""

    def __init__(self):
        self.results = None
        self.main_effects = None

    def set_results(self, results: Dict, main_effects: Dict):
        """Set results to export"""
        self.results = results
        self.main_effects = main_effects

    def export_statistics_excel(self, filepath: str):
        """Export statistics to Excel"""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Model Statistics
            model_stats_df = pd.DataFrame([self.results['model_stats']]).T
            model_stats_df.columns = ['Value']
            model_stats_df.index.name = 'Statistic'
            model_stats_df.to_excel(writer, sheet_name='Model Statistics')

            # Coefficients
            coef_df = self.results['coefficients'].copy()
            coef_df['p-value'] = coef_df['p-value'].apply(lambda x: f"{x:.6e}")
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
                self.results['coefficients']['p-value'].astype(float) < 0.05
            ].copy()
            sig_factors = sig_factors[sig_factors.index != 'Intercept']
            sig_factors['p-value'] = sig_factors['p-value'].apply(lambda x: f"{x:.6e}")
            sig_factors.to_excel(writer, sheet_name='Significant Factors')
