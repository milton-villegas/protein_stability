"""
DoE Statistical Analysis Logic
Extracted from doe_analysis_gui.pyw
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from typing import Dict, List, Optional, Any
from core.constants import (
    SIGNIFICANCE_LEVEL,
    BACKWARD_ELIMINATION_THRESHOLD,
    ADJ_R2_WEIGHT,
    BIC_WEIGHT,
    COMPLEXITY_PENALTY,
    R2_LOW_THRESHOLD,
    R2_EXCELLENT_THRESHOLD,
    ADJ_R2_SIMILARITY_THRESHOLD
)


class DoEAnalyzer:
    """Statistical analysis of Design of Experiments via regression modeling"""

    MODEL_TYPES = {
        'mean': 'Mean (intercept only)',
        'linear': 'Linear (main effects only)',
        'interactions': 'Linear with 2-way interactions',
        'quadratic': 'Quadratic (full second-order)',
        'purequadratic': 'Pure quadratic (squared terms only)',
        'reduced': 'Reduced Quadratic (backward elimination)'
    }

    def __init__(self) -> None:
        """Initialize analyzer with empty data and model"""
        self.data: Optional[pd.DataFrame] = None
        self.model = None
        self.model_type: str = 'linear'
        self.factor_columns: List[str] = []
        self.categorical_factors: List[str] = []
        self.numeric_factors: List[str] = []
        self.response_column: Optional[str] = None  # Backward compatibility
        self.response_columns: List[str] = []  # New: support multiple responses
        self.results: Optional[Dict[str, Any]] = None
        self.all_results: Dict[str, Dict[str, Any]] = {}  # New: results per response

    def set_data(self, data: pd.DataFrame, factor_columns: List[str],
                 categorical_factors: List[str], numeric_factors: List[str],
                 response_column: str = None, response_columns: List[str] = None) -> None:
        """
        Set experimental data and factor configuration

        Args:
            data: DataFrame containing experimental results
            factor_columns: List of column names that are experimental factors
            categorical_factors: Subset of factor_columns that are categorical
            numeric_factors: Subset of factor_columns that are numeric/continuous
            response_column: Name of the column containing response variable (backward compatibility)
            response_columns: List of response column names (new multi-response support)
        """
        self.data = data.copy()
        self.factor_columns = factor_columns
        self.categorical_factors = categorical_factors
        self.numeric_factors = numeric_factors

        # Support both old (single) and new (multiple) response specification
        if response_columns is not None:
            self.response_columns = response_columns if isinstance(response_columns, list) else [response_columns]
            self.response_column = self.response_columns[0] if self.response_columns else None
        elif response_column is not None:
            self.response_column = response_column
            self.response_columns = [response_column]
        else:
            raise ValueError("Must specify either response_column or response_columns")

    @staticmethod
    def _build_squared_terms(numeric_factors: List[str]) -> List[str]:
        """
        Build squared terms for numeric factors (for quadratic models)

        Args:
            numeric_factors: List of numeric factor names

        Returns:
            List of squared term strings (e.g., ["I(Q('pH')**2)", ...])
        """
        return [f"I(Q('{factor}')**2)" for factor in numeric_factors]

    def _build_interaction_terms(self, factor_terms: List[str]) -> List[str]:
        """
        Build 2-way interaction terms for all factor combinations

        Args:
            factor_terms: List of factor term strings (already formatted for statsmodels)

        Returns:
            List of interaction term strings (e.g., ["factor1:factor2", ...])
        """
        interactions = []
        for i in range(len(factor_terms)):
            for j in range(i + 1, len(factor_terms)):
                interactions.append(f"{factor_terms[i]}:{factor_terms[j]}")
        return interactions

    def build_formula(self, model_type: str = 'linear', response_name: str = None) -> str:
        """
        Build regression formula based on model type

        Args:
            model_type: One of 'mean', 'linear', 'interactions', 'quadratic', 'purequadratic', 'reduced'
            response_name: Name of response column (defaults to self.response_column)

        Returns:
            Regression formula string for statsmodels
        """
        self.model_type = model_type
        response = response_name if response_name is not None else self.response_column

        # Mean model - intercept only
        if model_type == 'mean':
            formula = f"Q('{response}') ~ 1"
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
            formula = f"Q('{response}') ~ " + " + ".join(factor_terms)

        elif model_type == 'interactions':
            main_effects = " + ".join(factor_terms)
            interactions = self._build_interaction_terms(factor_terms)
            formula = f"Q('{response}') ~ {main_effects}"
            if interactions:
                formula += " + " + " + ".join(interactions)

        elif model_type == 'quadratic':
            main_effects = " + ".join(factor_terms)
            interactions = self._build_interaction_terms(factor_terms)
            squared_terms = self._build_squared_terms(self.numeric_factors)
            formula = f"Q('{response}') ~ {main_effects}"
            if interactions:
                formula += " + " + " + ".join(interactions)
            if squared_terms:
                formula += " + " + " + ".join(squared_terms)

        elif model_type == 'purequadratic':
            main_effects = " + ".join(factor_terms)
            squared_terms = self._build_squared_terms(self.numeric_factors)
            formula = f"Q('{response}') ~ {main_effects}"
            if squared_terms:
                formula += " + " + " + ".join(squared_terms)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return formula

    def fit_model(self, model_type: str = 'linear', response_name: str = None) -> Dict:
        """
        Fit regression model

        Args:
            model_type: One of 'linear', 'interactions', 'quadratic', 'purequadratic'
            response_name: Name of response column (defaults to self.response_column)

        Returns:
            Dict with model results
        """
        if self.data is None:
            raise ValueError("No data set")

        formula = self.build_formula(model_type, response_name=response_name)
        self.model = smf.ols(formula=formula, data=self.data).fit()
        self.results = self._extract_results()
        return self.results

    def _extract_results(self) -> Dict[str, Any]:
        """
        Extract and organize regression model results

        Returns:
            Dictionary containing coefficients, model statistics, predictions, and residuals
        """
        summary_df = pd.DataFrame({
            'Coefficient': self.model.params,
            'Std Error': self.model.bse,
            't-statistic': self.model.tvalues,
            'p-value': self.model.pvalues,
            'Significant': self.model.pvalues < SIGNIFICANCE_LEVEL
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

    def get_significant_factors(self, alpha: float = SIGNIFICANCE_LEVEL) -> List[str]:
        """
        Get list of significant factors based on p-value threshold

        Args:
            alpha: Significance level threshold (default from constants)

        Returns:
            List of factor names with p-value < alpha (excluding intercept)
        """
        if self.results is None:
            raise ValueError("No results available")

        coef_df = self.results['coefficients']
        significant = coef_df[coef_df['p-value'] < alpha]
        return [idx for idx in significant.index if idx != 'Intercept']

    def calculate_main_effects(self, response_name: str = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate main effects for each factor

        Args:
            response_name: Name of response column (defaults to self.response_column)

        Returns:
            Dict mapping factor_name → DataFrame with mean/std/count per level
        """
        if self.data is None:
            raise ValueError("No data available")

        response = response_name if response_name is not None else self.response_column
        main_effects = {}
        for factor in self.factor_columns:
            # Group by factor levels and calculate statistics
            effects = self.data.groupby(factor)[response].agg(['mean', 'std', 'count'])
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

    def fit_reduced_quadratic(self, p_remove: float = BACKWARD_ELIMINATION_THRESHOLD, response_name: str = None):
        """
        Fit reduced quadratic model using backward elimination

        Starts with full quadratic model and iteratively removes terms with
        highest p-value until all remaining terms are significant.

        Args:
            p_remove: P-value threshold for removing terms (default from constants)
            response_name: Name of response column (defaults to self.response_column)

        Returns:
            Fitted statsmodels OLS regression model
        """
        if self.data is None:
            raise ValueError("No data available")

        response = response_name if response_name is not None else self.response_column
        formula = self.build_formula('quadratic', response_name=response)
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
                    new_formula = f"Q('{response}') ~ " + " + ".join(terms)
                    new_model = smf.ols(formula=new_formula, data=self.data).fit()
                    current_model = new_model
                else:
                    break

            except Exception:
                break

        return current_model

    def compare_all_models(self, response_name: str = None) -> Dict:
        """
        Fit all model types and return comparison statistics

        Args:
            response_name: Name of response column (defaults to self.response_column)

        Returns:
            dict: Comparison data with keys:
                - 'models': dict mapping model_type -> stats dict
                - 'comparison_table': DataFrame with all models side-by-side
                - 'fitted_models': dict mapping model_type -> fitted model object
                - 'errors': dict mapping model_type -> error message (if failed)
        """
        if self.data is None:
            raise ValueError("No data available")

        response = response_name if response_name is not None else self.response_column
        model_types = ['mean', 'linear', 'interactions', 'quadratic', 'purequadratic', 'reduced']
        comparison_data = {
            'models': {},
            'fitted_models': {},
            'errors': {}
        }

        for model_type in model_types:
            try:
                if model_type == 'reduced':
                    fitted_model = self.fit_reduced_quadratic(p_remove=BACKWARD_ELIMINATION_THRESHOLD, response_name=response)
                else:
                    formula = self.build_formula(model_type, response_name=response)
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

            complexity_penalty = stats['DF Model'] * COMPLEXITY_PENALTY

            combined_score = (ADJ_R2_WEIGHT * adj_r2_score + BIC_WEIGHT * bic_score - complexity_penalty)

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

        if best_score['adj_r2'] < R2_LOW_THRESHOLD:
            reason_parts.append("(Warning: Low R² - model may not fit data well)")
        elif best_score['adj_r2'] > R2_EXCELLENT_THRESHOLD:
            reason_parts.append("(Excellent fit)")

        model_order = ['mean', 'linear', 'interactions', 'quadratic', 'purequadratic', 'reduced']
        best_idx = model_order.index(best_model) if best_model in model_order else -1

        for i in range(best_idx):
            simpler_model = model_order[i]
            if simpler_model in scores:
                simpler_score = scores[simpler_model]
                adj_r2_diff = best_score['adj_r2'] - simpler_score['adj_r2']

                if adj_r2_diff < ADJ_R2_SIMILARITY_THRESHOLD:
                    reason_parts.append(f"(Consider {simpler_model} for parsimony)")
                    break

        return {
            'recommended_model': best_model,
            'reason': "; ".join(reason_parts),
            'scores': scores
        }

    # ===== Multi-Response Methods =====

    def fit_model_all_responses(self, model_type: str = 'linear') -> Dict[str, Dict]:
        """
        Fit regression model for all response columns

        Args:
            model_type: One of 'linear', 'interactions', 'quadratic', 'purequadratic'

        Returns:
            Dict mapping response_name → model results dict
        """
        if self.data is None:
            raise ValueError("No data set")

        if not self.response_columns:
            raise ValueError("No response columns configured")

        self.all_results = {}
        for response_name in self.response_columns:
            formula = self.build_formula(model_type, response_name=response_name)
            model = smf.ols(formula=formula, data=self.data).fit()

            # Store results for this response
            summary_df = pd.DataFrame({
                'Coefficient': model.params,
                'Std Error': model.bse,
                't-statistic': model.tvalues,
                'p-value': model.pvalues,
                'Significant': model.pvalues < SIGNIFICANCE_LEVEL
            })

            model_stats = {
                'R-squared': model.rsquared,
                'Adjusted R-squared': model.rsquared_adj,
                'RMSE': np.sqrt(model.mse_resid),
                'F-statistic': model.fvalue,
                'F p-value': model.f_pvalue,
                'AIC': model.aic,
                'BIC': model.bic,
                'Observations': int(model.nobs),
                'DF Residuals': int(model.df_resid),
                'DF Model': int(model.df_model)
            }

            self.all_results[response_name] = {
                'coefficients': summary_df,
                'model_stats': model_stats,
                'model_type': model_type,
                'formula': model.model.formula,
                'predictions': model.fittedvalues,
                'residuals': model.resid,
                'model_object': model
            }

        return self.all_results

    def compare_all_models_all_responses(self) -> Dict[str, Dict]:
        """
        Fit all model types for all response columns and return comparison statistics

        Returns:
            Dict mapping response_name → comparison_data dict (same format as compare_all_models)
        """
        if self.data is None:
            raise ValueError("No data available")

        if not self.response_columns:
            raise ValueError("No response columns configured")

        all_comparisons = {}
        for response_name in self.response_columns:
            comparison_data = self.compare_all_models(response_name=response_name)
            all_comparisons[response_name] = comparison_data

        return all_comparisons

    def calculate_main_effects_all_responses(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate main effects for each factor across all response columns

        Returns:
            Dict mapping response_name → {factor_name → DataFrame with mean/std/count}
        """
        if self.data is None:
            raise ValueError("No data available")

        if not self.response_columns:
            raise ValueError("No response columns configured")

        all_main_effects = {}
        for response_name in self.response_columns:
            main_effects = self.calculate_main_effects(response_name=response_name)
            all_main_effects[response_name] = main_effects

        return all_main_effects
