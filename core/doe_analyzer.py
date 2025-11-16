"""
DoE Statistical Analysis Logic
Extracted from doe_analysis_gui.pyw
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from typing import Dict, List, Optional


class DoEAnalyzer:
    """Statistical analysis via regression"""

    MODEL_TYPES = {
        'mean': 'Mean (intercept only)',
        'linear': 'Linear (main effects only)',
        'interactions': 'Linear with 2-way interactions',
        'quadratic': 'Quadratic (full second-order)',
        'purequadratic': 'Pure quadratic (squared terms only)',
        'reduced': 'Reduced Quadratic (backward elimination)'
    }

    def __init__(self):
        self.data = None
        self.model = None
        self.model_type = 'linear'
        self.factor_columns = []
        self.categorical_factors = []
        self.numeric_factors = []
        self.response_column = None
        self.results = None

    def set_data(self, data: pd.DataFrame, factor_columns: List[str],
                 categorical_factors: List[str], numeric_factors: List[str],
                 response_column: str):
        """Set data and factor information"""
        self.data = data.copy()
        self.factor_columns = factor_columns
        self.categorical_factors = categorical_factors
        self.numeric_factors = numeric_factors
        self.response_column = response_column

    def _build_interaction_terms(self, factor_terms: List[str]) -> List[str]:
        """Build interaction terms for all factor combinations"""
        interactions = []
        for i in range(len(factor_terms)):
            for j in range(i + 1, len(factor_terms)):
                interactions.append(f"{factor_terms[i]}:{factor_terms[j]}")
        return interactions

    def build_formula(self, model_type: str = 'linear') -> str:
        """
        Build regression formula based on model type

        Args:
            model_type: One of 'mean', 'linear', 'interactions', 'quadratic', 'purequadratic', 'reduced'

        Returns:
            Regression formula string for statsmodels
        """
        self.model_type = model_type

        # Mean model - intercept only
        if model_type == 'mean':
            formula = f"Q('{self.response_column}') ~ 1"
            return formula

        # Prepare factor terms
        factor_terms = []
        for factor in self.factor_columns:
            if factor in self.categorical_factors:
                # C() treats as categorical, Q() quotes column names with spaces
                factor_terms.append(f"C(Q('{factor}'))")
            else:
                factor_terms.append(f"Q('{factor}')")

        # Build formula based on model type
        if model_type == 'linear':
            formula = f"Q('{self.response_column}') ~ " + " + ".join(factor_terms)

        elif model_type == 'interactions':
            main_effects = " + ".join(factor_terms)
            interactions = self._build_interaction_terms(factor_terms)
            formula = f"Q('{self.response_column}') ~ {main_effects}"
            if interactions:
                formula += " + " + " + ".join(interactions)

        elif model_type == 'quadratic':
            main_effects = " + ".join(factor_terms)
            interactions = self._build_interaction_terms(factor_terms)
            squared_terms = []
            for factor in self.numeric_factors:
                squared_terms.append(f"I(Q('{factor}')**2)")
            formula = f"Q('{self.response_column}') ~ {main_effects}"
            if interactions:
                formula += " + " + " + ".join(interactions)
            if squared_terms:
                formula += " + " + " + ".join(squared_terms)

        elif model_type == 'purequadratic':
            main_effects = " + ".join(factor_terms)
            squared_terms = []
            for factor in self.numeric_factors:
                squared_terms.append(f"I(Q('{factor}')**2)")
            formula = f"Q('{self.response_column}') ~ {main_effects}"
            if squared_terms:
                formula += " + " + " + ".join(squared_terms)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return formula

    def fit_model(self, model_type: str = 'linear') -> Dict:
        """
        Fit regression model

        Args:
            model_type: One of 'linear', 'interactions', 'quadratic', 'purequadratic'

        Returns:
            Dict with model results
        """
        if self.data is None:
            raise ValueError("No data set")

        formula = self.build_formula(model_type)
        self.model = smf.ols(formula=formula, data=self.data).fit()
        self.results = self._extract_results()
        return self.results

    def _extract_results(self) -> Dict:
        """Extract and organize model results"""
        summary_df = pd.DataFrame({
            'Coefficient': self.model.params,
            'Std Error': self.model.bse,
            't-statistic': self.model.tvalues,
            'p-value': self.model.pvalues,
            'Significant': self.model.pvalues < 0.05
        })

        model_stats = {
            'R-squared': self.model.rsquared,
            'Adjusted R-squared': self.model.rsquared_adj,
            'RMSE': np.sqrt(self.model.mse_resid),
            'F-statistic': self.model.fvalue,
            'F p-value': self.model.f_pvalue,
            'AIC': self.model.aic,
            'BIC': self.model.bic,
            'Observations': int(self.model.nobs),
            'DF Residuals': int(self.model.df_resid),
            'DF Model': int(self.model.df_model)
        }

        return {
            'coefficients': summary_df,
            'model_stats': model_stats,
            'model_type': self.model_type,
            'formula': self.model.model.formula,
            'predictions': self.model.fittedvalues,
            'residuals': self.model.resid
        }

    def get_significant_factors(self, alpha: float = 0.05) -> List[str]:
        """Get list of significant factors"""
        if self.results is None:
            raise ValueError("No results available")

        coef_df = self.results['coefficients']
        significant = coef_df[coef_df['p-value'] < alpha]
        return [idx for idx in significant.index if idx != 'Intercept']

    def calculate_main_effects(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate main effects for each factor

        Returns:
            Dict mapping factor_name → DataFrame with mean/std/count per level
        """
        if self.data is None:
            raise ValueError("No data available")

        main_effects = {}
        for factor in self.factor_columns:
            # Group by factor levels and calculate statistics
            effects = self.data.groupby(factor)[self.response_column].agg(['mean', 'std', 'count'])
            effects.columns = ['Mean Response', 'Std Dev', 'Count']
            main_effects[factor] = effects

        return main_effects

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict response for new data

        Args:
            X: DataFrame with factor columns

        Returns:
            Array of predicted response values
        """
        if self.model is None:
            raise ValueError("No model fitted")

        return self.model.predict(X)

    def fit_reduced_quadratic(self, p_remove: float = 0.10):
        """
        Fit reduced quadratic model using backward elimination
        Starts with full quadratic and removes non-significant terms

        Args:
            p_remove: P-value threshold for removing terms (default 0.10)

        Returns:
            Fitted statsmodels regression model
        """
        if self.data is None:
            raise ValueError("No data available")

        formula = self.build_formula('quadratic')
        current_model = smf.ols(formula=formula, data=self.data).fit()

        while True:
            pvalues = current_model.pvalues.drop('Intercept', errors='ignore')

            if pvalues.empty or pvalues.max() < p_remove:
                break

            worst_term = pvalues.idxmax()

            try:
                current_formula = current_model.model.formula
                terms = current_formula.split('~')[1].strip().split('+')
                terms = [t.strip() for t in terms if t.strip()]

                term_to_remove = None
                for term in terms:
                    if worst_term in term or term in worst_term:
                        term_to_remove = term
                        break

                if term_to_remove:
                    terms.remove(term_to_remove)
                    if not terms:
                        break
                    new_formula = f"Q('{self.response_column}') ~ " + " + ".join(terms)
                    new_model = smf.ols(formula=new_formula, data=self.data).fit()
                    current_model = new_model
                else:
                    break

            except Exception:
                break

        return current_model

    def compare_all_models(self) -> Dict:
        """
        Fit all model types and return comparison statistics

        Returns:
            dict: Comparison data with keys:
                - 'models': dict mapping model_type -> stats dict
                - 'comparison_table': DataFrame with all models side-by-side
                - 'fitted_models': dict mapping model_type -> fitted model object
                - 'errors': dict mapping model_type -> error message (if failed)
        """
        if self.data is None:
            raise ValueError("No data available")

        model_types = ['mean', 'linear', 'interactions', 'quadratic', 'purequadratic', 'reduced']
        comparison_data = {
            'models': {},
            'fitted_models': {},
            'errors': {}
        }

        for model_type in model_types:
            try:
                if model_type == 'reduced':
                    fitted_model = self.fit_reduced_quadratic(p_remove=0.10)
                else:
                    formula = self.build_formula(model_type)
                    fitted_model = smf.ols(formula=formula, data=self.data).fit()

                stats = {
                    'Model Type': self.MODEL_TYPES[model_type],
                    'R²': fitted_model.rsquared,
                    'Adj R²': fitted_model.rsquared_adj,
                    'RMSE': np.sqrt(fitted_model.mse_resid),
                    'AIC': fitted_model.aic,
                    'BIC': fitted_model.bic,
                    'DF Model': int(fitted_model.df_model),
                    'DF Resid': int(fitted_model.df_resid),
                    'F-statistic': fitted_model.fvalue,
                    'F p-value': fitted_model.f_pvalue
                }

                comparison_data['models'][model_type] = stats
                comparison_data['fitted_models'][model_type] = fitted_model

            except Exception as e:
                comparison_data['errors'][model_type] = str(e)

        if comparison_data['models']:
            comparison_df = pd.DataFrame(comparison_data['models']).T
            comparison_data['comparison_table'] = comparison_df
        else:
            comparison_data['comparison_table'] = None

        return comparison_data

    def select_best_model(self, comparison_data: Dict) -> Dict:
        """
        Analyze model comparison and recommend the best model

        Uses these criteria in priority order:
        1. Adjusted R² (higher is better) - penalizes overfitting
        2. BIC (lower is better) - penalizes complexity more than AIC
        3. Model parsimony (prefer simpler if similar performance)

        Args:
            comparison_data: Output from compare_all_models()

        Returns:
            dict: {
                'recommended_model': str (model type),
                'reason': str (explanation),
                'scores': dict (ranking scores for each model)
            }
        """
        models = comparison_data['models']

        if not models:
            return {
                'recommended_model': None,
                'reason': "No models successfully fitted",
                'scores': {}
            }

        scores = {}

        for model_type in models.keys():
            stats = models[model_type]

            adj_r2_score = stats['Adj R²'] * 100

            bic_values = [m['BIC'] for m in models.values()]
            min_bic = min(bic_values)
            max_bic = max(bic_values)
            if max_bic > min_bic:
                bic_score = 100 * (1 - (stats['BIC'] - min_bic) / (max_bic - min_bic))
            else:
                bic_score = 100

            complexity_penalty = stats['DF Model'] * 2

            combined_score = (0.6 * adj_r2_score + 0.3 * bic_score - complexity_penalty)

            scores[model_type] = {
                'adj_r2_score': adj_r2_score,
                'bic_score': bic_score,
                'complexity_penalty': complexity_penalty,
                'combined_score': combined_score,
                'adj_r2': stats['Adj R²'],
                'bic': stats['BIC'],
                'rmse': stats['RMSE']
            }

        best_model = max(scores.keys(), key=lambda k: scores[k]['combined_score'])
        best_stats = models[best_model]
        best_score = scores[best_model]

        reason_parts = []
        reason_parts.append(f"Best Adj R² = {best_score['adj_r2']:.4f}")
        reason_parts.append(f"BIC = {best_score['bic']:.1f}")
        reason_parts.append(f"RMSE = {best_score['rmse']:.4f}")

        if best_score['adj_r2'] < 0.5:
            reason_parts.append("(Warning: Low R² - model may not fit data well)")
        elif best_score['adj_r2'] > 0.9:
            reason_parts.append("(Excellent fit)")

        model_order = ['mean', 'linear', 'interactions', 'quadratic', 'purequadratic', 'reduced']
        best_idx = model_order.index(best_model) if best_model in model_order else -1

        for i in range(best_idx):
            simpler_model = model_order[i]
            if simpler_model in scores:
                simpler_score = scores[simpler_model]
                adj_r2_diff = best_score['adj_r2'] - simpler_score['adj_r2']

                if adj_r2_diff < 0.05:
                    reason_parts.append(f"(Consider {simpler_model} for parsimony)")
                    break

        return {
            'recommended_model': best_model,
            'reason': "; ".join(reason_parts),
            'scores': scores
        }
