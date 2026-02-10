"""
Model Panel Mixin for DoE Analysis Tab.

This module provides the ModelPanelMixin class which contains methods for
model comparison, selection, fitting, and statistical results display.
These methods are designed to be mixed into the main AnalysisTab class.

The mixin handles:
- Model comparison across all model types (mean, linear, interactions, quadratic, reduced)
- Automatic model selection based on Adjusted R-squared, BIC, and parsimony
- Manual model selection by user preference
- Multi-response analysis support
- Statistical results display with warnings and recommendations
"""

import tkinter as tk
from tkinter import messagebox

# Check if Ax (Adaptive Experimentation) is available for Bayesian Optimization
try:
    from ax.plot.contour import plot_contour
    from ax.plot.feature_importances import plot_feature_importance_by_feature
    AX_AVAILABLE = True
except ImportError:
    AX_AVAILABLE = False


class ModelPanelMixin:
    """
    Mixin class providing model analysis and statistics display functionality.

    This mixin should be combined with a class that provides the following attributes:
        - self.filepath: Path to the data file
        - self.debug_log: List for storing debug messages
        - self.has_analysis_error: Boolean flag for analysis errors
        - self.selected_responses: List of selected response variable names
        - self.response_directions: Dict mapping responses to optimization directions
        - self.response_constraints: Dict mapping responses to constraint values
        - self.main_window: Reference to main application window
        - self.handler: Data handler instance
        - self.analyzer: DoE analyzer instance
        - self.optimizer: Bayesian optimizer instance (optional)
        - self.model_selection_var: Tkinter StringVar for model selection dropdown
        - self.stats_text: Tkinter Text widget for statistics display
        - self.export_stats_btn: Export statistics button
        - self.export_plots_btn: Export plots button

    Methods provided:
        - analyze_data(): Main analysis method that fits models
        - _display_single_response_statistics(): Helper for multi-response statistics display
        - display_statistics(): Display statistical results
    """

    def analyze_data(self):
        """
        Perform DoE analysis on the loaded data.

        This method:
        1. Validates that data and responses are selected
        2. Detects and preprocesses data columns
        3. Compares all model types (mean, linear, interactions, quadratic, reduced)
        4. Selects the best model (auto or user-specified)
        5. Fits the chosen model and calculates main effects
        6. Displays statistics, plots, and recommendations

        For multi-response analysis, each response gets its own model selection
        when in Auto mode, or all responses use the same user-selected model.

        Raises:
            Shows messagebox warnings/errors for:
            - No data file selected
            - No response variables selected
            - Model fitting failures
        """
        if not self.filepath:
            messagebox.showwarning("Warning", "Please select a data file first.")
            return

        # Clear debug log and error flag at start of new analysis
        self.debug_log = []
        self.has_analysis_error = False

        # Update selected responses to capture latest dropdown/constraint values
        self._update_selected_responses()

        if not self.selected_responses:
            messagebox.showwarning("Warning", "Please select at least one response variable.")
            return

        self._debug_log(f"\n[DEBUG ANALYZE] Starting analysis with:")
        self._debug_log(f"  - Selected responses: {self.selected_responses}")
        self._debug_log(f"  - Response directions: {self.response_directions}")
        self._debug_log(f"  - Response constraints: {self.response_constraints}")

        self.main_window.update_status("Analyzing...")
        self.update()

        try:
            # Detect which columns are factors vs response
            self.handler.detect_columns(response_columns=self.selected_responses)
            clean_data = self.handler.preprocess_data()

            # Pass cleaned data to analyzer with all selected responses
            self.analyzer.set_data(
                data=clean_data,
                factor_columns=self.handler.factor_columns,
                categorical_factors=self.handler.categorical_factors,
                numeric_factors=self.handler.numeric_factors,
                response_columns=self.selected_responses
            )

            # Get user's model selection
            selected_option = self.model_selection_var.get()

            # Map dropdown options to model types
            model_mapping = {
                'Auto (Recommended)': None,
                'Mean (intercept only)': 'mean',
                'Linear (main effects)': 'linear',
                'Interactions (2-way)': 'interactions',
                'Quadratic (full)': 'quadratic',
                'Reduced (backward elimination)': 'reduced'
            }

            selected_model = model_mapping[selected_option]

            if len(self.selected_responses) > 1:
                self.main_window.update_status(f"Analyzing {len(self.selected_responses)} responses...")
                self.update()

                # Compare all models for all responses (this fits all models)
                self.model_comparison_all = self.analyzer.compare_all_models_all_responses()

                # Select best model and fit for each response
                self.model_selection_all = {}
                self.chosen_models = {}
                self.all_results = {}
                self.all_main_effects = {}

                for response_name in self.selected_responses:
                    comparison_data = self.model_comparison_all[response_name]

                    # Determine which model to use for this response
                    if selected_model is None:
                        # AUTO MODE - select best model for each response
                        model_sel = self.analyzer.select_best_model(comparison_data)
                        chosen = model_sel['recommended_model']

                        if chosen is None:
                            raise ValueError(f"No models could be fitted for {response_name}. Check your data.")

                        self.model_selection_all[response_name] = model_sel
                        self.chosen_models[response_name] = chosen
                    else:
                        # MANUAL MODE - use same model for all responses
                        chosen = selected_model
                        self.model_selection_all[response_name] = {
                            'recommended_model': chosen,
                            'reason': 'User selected'
                        }
                        self.chosen_models[response_name] = chosen

                    self.main_window.update_status(f"Fitting {response_name} with {chosen} model...")
                    self.update()

                    self.all_results[response_name] = self.analyzer.fit_model(chosen, response_name=response_name)
                    self.all_main_effects[response_name] = self.analyzer.calculate_main_effects(response_name=response_name)

                # For backward compatibility, use first response as primary
                self.model_comparison = self.model_comparison_all[self.selected_responses[0]]
                self.model_selection = self.model_selection_all[self.selected_responses[0]]
                chosen_model = self.chosen_models[self.selected_responses[0]]
                self.results = self.all_results[self.selected_responses[0]]
                self.main_effects = self.all_main_effects[self.selected_responses[0]]

                mode_description = "Auto-selected" if selected_model is None else "User selected"

            else:
                self.main_window.update_status("Comparing all models...")
                self.update()

                self.model_comparison = self.analyzer.compare_all_models()

                # Determine which model to use
                if selected_model is None:
                    # AUTO MODE - select best model
                    self.model_selection = self.analyzer.select_best_model(self.model_comparison)
                    chosen_model = self.model_selection['recommended_model']

                    if chosen_model is None:
                        raise ValueError("No models could be fitted successfully. Check your data.")

                    mode_description = "Auto-selected"
                else:
                    # MANUAL MODE - use user's choice
                    chosen_model = selected_model
                    self.model_selection = {'recommended_model': chosen_model, 'reason': 'User selected'}
                    mode_description = "User selected"

                self.main_window.update_status(f"Using {mode_description.lower()} model: {chosen_model}...")
                self.update()

                # Fit the chosen model for detailed analysis
                self.results = self.analyzer.fit_model(chosen_model)
                self.main_effects = self.analyzer.calculate_main_effects()

            self.display_statistics()
            self.display_plots()
            self.display_recommendations()

            # Display optimization plot if available, initialized, and BO enabled
            if AX_AVAILABLE and self.bo_enabled_var.get() and self.optimizer and self.optimizer.is_initialized:
                self.display_optimization_plot()

            self.export_stats_btn.config(state='normal')
            self.export_plots_btn.config(state='normal')

            # Show completion status (only if no analysis errors occurred)
            if self.has_analysis_error:
                # Error was already shown, don't show success
                return

            chosen_model_name = self.analyzer.MODEL_TYPES[chosen_model]

            if len(self.selected_responses) > 1:
                avg_r2 = sum(r['model_stats']['R-squared'] for r in self.all_results.values()) / len(self.all_results)
                self.main_window.update_status(f"Analysis complete! {len(self.selected_responses)} responses analyzed | Avg R-squared = {avg_r2:.4f}")

                messagebox.showinfo("Success",
                                  f"Multi-response analysis completed successfully!\n\n"
                                  f"Responses analyzed: {len(self.selected_responses)}\n"
                                  f"Average R-squared: {avg_r2:.4f}\n"
                                  f"Observations: {self.results['model_stats']['Observations']}\n\n"
                                  f"Check the Results tab for detailed analysis of each response.")
            else:
                self.main_window.update_status(f"Analysis complete! Model: {chosen_model_name} | R-squared = {self.results['model_stats']['R-squared']:.4f}")

                messagebox.showinfo("Success",
                                  f"Analysis completed successfully!\n\n"
                                  f"{mode_description} Model: {chosen_model_name}\n"
                                  f"Observations: {self.results['model_stats']['Observations']}\n"
                                  f"R-squared: {self.results['model_stats']['R-squared']:.4f}\n"
                                  f"Adjusted R-squared: {self.results['model_stats']['Adjusted R-squared']:.4f}\n\n"
                                  f"Check the Results tab for detailed analysis.")

        except Exception as e:
            import traceback
            print(f"\n{'='*60}")
            print(f"ANALYSIS ERROR")
            print(f"{'='*60}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Selected responses: {self.selected_responses}")
            print(f"{'='*60}")
            traceback.print_exc()
            print(f"{'='*60}\n")

            messagebox.showerror("Analysis Failed",
                f"Statistical analysis could not be completed.\n\n"
                f"Error: {str(e)}\n\n"
                f"Check that your data includes all required columns and valid numeric values.")
            self.main_window.update_status("Analysis failed")

    def _display_single_response_statistics(self, results, model_comparison, model_selection, response_name):
        """
        Display statistics for a single response (helper for multi-response display).

        This method formats and displays detailed statistical results for one response
        variable, including model comparison, warnings, coefficients, and significant
        factors.

        Args:
            results: Dictionary containing model results with keys:
                - 'model_stats': Dict of model statistics (R-squared, Observations, etc.)
                - 'coefficients': DataFrame with coefficient estimates and p-values
            model_comparison: Dictionary containing model comparison data with keys:
                - 'comparison_table': DataFrame comparing all models
                - 'models': Dict of individual model statistics
            model_selection: Dictionary containing model selection info with keys:
                - 'recommended_model': String identifier of selected model
                - 'reason': String explaining selection rationale
            response_name: Name of the response variable being displayed
        """
        # MODEL COMPARISON SECTION
        if model_comparison:
            self.stats_text.insert(tk.END, "MODEL COMPARISON\n")
            self.stats_text.insert(tk.END, "-"*80 + "\n\n")

            comparison_table = model_comparison['comparison_table']

            if comparison_table is not None and not comparison_table.empty:
                self.stats_text.insert(tk.END, "Model Comparison (all 5 models fitted):\n\n")

                header = f"{'Model':<30} {'Adj R-sq':>10} {'BIC':>10} {'RMSE':>10} {'DF':>6}  {'Selected':>10}\n"
                self.stats_text.insert(tk.END, header)
                self.stats_text.insert(tk.END, "-"*80 + "\n")

                recommended_model = model_selection['recommended_model']
                is_auto = self.model_selection_var.get() == 'Auto (Recommended)'

                model_order = ['mean', 'linear', 'interactions', 'quadratic', 'reduced']
                for model_type in model_order:
                    if model_type in model_comparison['models']:
                        stats = model_comparison['models'][model_type]
                        model_name = stats['Model Type']

                        marker = ("AUTO" if is_auto else "USER") if model_type == recommended_model else ""

                        line = (f"{model_name:<30} "
                               f"{stats['Adj R²']:>10.4f} "
                               f"{stats['BIC']:>10.1f} "
                               f"{stats['RMSE']:>10.4f} "
                               f"{stats['DF Model']:>6} "
                               f"{marker:>10}\n")
                        self.stats_text.insert(tk.END, line)

                self.stats_text.insert(tk.END, "\n" + "-"*80 + "\n")
                self.stats_text.insert(tk.END, f"RECOMMENDED: {self.analyzer.MODEL_TYPES[recommended_model]}\n")
                self.stats_text.insert(tk.END, f"   Reason: {model_selection['reason']}\n")

                self.stats_text.insert(tk.END, "\nSelection criteria: Adjusted R-squared (60%), BIC (30%), Parsimony (10%)\n")
                self.stats_text.insert(tk.END, "   Higher Adj R-squared is better | Lower BIC is better | Simpler models preferred\n")

            self.stats_text.insert(tk.END, "\n" + "="*80 + "\n\n")

        # RED FLAGS / WARNINGS
        r_squared = results['model_stats']['R-squared']
        n_obs = results['model_stats']['Observations']

        # Get significant factors for this response
        sig_factors = []
        for idx in results['coefficients'].index:
            if idx != 'Intercept' and results['coefficients'].loc[idx, 'p-value'] < 0.05:
                sig_factors.append(idx)

        warnings = []
        if r_squared < 0.5:
            warnings.append(f"LOW R-squared ({r_squared:.3f}): Model explains only {r_squared*100:.1f}% of variance")
        elif r_squared < 0.7:
            warnings.append(f"MODERATE R-squared ({r_squared:.3f}): Model is acceptable but could be improved")

        if len(sig_factors) == 0:
            warnings.append("NO SIGNIFICANT FACTORS: No factors with p < 0.05 found")

        if n_obs < 20:
            warnings.append(f"SMALL SAMPLE SIZE: Only {n_obs} observations")

        if warnings:
            self.stats_text.insert(tk.END, "RED FLAGS / WARNINGS\n")
            self.stats_text.insert(tk.END, "-"*80 + "\n")
            for warning in warnings:
                self.stats_text.insert(tk.END, f"{warning}\n")
            self.stats_text.insert(tk.END, "\n" + "="*80 + "\n\n")

        # MODEL STATISTICS
        self.stats_text.insert(tk.END, "MODEL STATISTICS\n")
        self.stats_text.insert(tk.END, "-"*80 + "\n")
        for key, value in results['model_stats'].items():
            if isinstance(value, float):
                self.stats_text.insert(tk.END, f"  {key:<25}: {value:>15.6f}\n")
            else:
                self.stats_text.insert(tk.END, f"  {key:<25}: {value:>15}\n")

        self.stats_text.insert(tk.END, "\n" + "="*80 + "\n")
        self.stats_text.insert(tk.END, "COEFFICIENTS\n")
        self.stats_text.insert(tk.END, "-"*80 + "\n\n")

        coef_str = results['coefficients'].to_string()
        self.stats_text.insert(tk.END, coef_str + "\n")

        # SIGNIFICANT FACTORS
        self.stats_text.insert(tk.END, "\n" + "="*80 + "\n")
        self.stats_text.insert(tk.END, f"SIGNIFICANT FACTORS (p < 0.05): {len(sig_factors)} found\n")
        self.stats_text.insert(tk.END, "-"*80 + "\n")

        if sig_factors:
            factor_importance = []
            for factor in sig_factors:
                coef = results['coefficients'].loc[factor, 'Coefficient']
                pval = results['coefficients'].loc[factor, 'p-value']
                factor_importance.append((factor, abs(coef), coef, pval))

            factor_importance.sort(key=lambda x: x[1], reverse=True)

            self.stats_text.insert(tk.END, "Ranked by effect size (most important first):\n\n")
            for rank, (factor, abs_coef, coef, pval) in enumerate(factor_importance, 1):
                self.stats_text.insert(tk.END, f"  {rank}. {factor:<38} coef={coef:>10.4f}  p={pval:.2e}\n")
        else:
            self.stats_text.insert(tk.END, "  None found - no factors are statistically significant.\n")

        # INTERACTIONS
        interaction_factors = [f for f in sig_factors if ':' in f]
        if interaction_factors:
            self.stats_text.insert(tk.END, "\n" + "="*80 + "\n")
            self.stats_text.insert(tk.END, f"SIGNIFICANT INTERACTIONS DETECTED: {len(interaction_factors)}\n")
            self.stats_text.insert(tk.END, "-"*80 + "\n")
            for interaction in interaction_factors:
                coef = results['coefficients'].loc[interaction, 'Coefficient']
                pval = results['coefficients'].loc[interaction, 'p-value']
                self.stats_text.insert(tk.END, f"  {interaction:<40} coef={coef:>10.4f}  p={pval:.2e}\n")
            self.stats_text.insert(tk.END, "\nWARNING: Interactions mean optimal settings depend on factor combinations!\n")

        self.stats_text.insert(tk.END, "\n" + "="*80 + "\n")

    def display_statistics(self):
        """
        Display statistical results with recommendations and warnings.

        This method clears the statistics text widget and displays comprehensive
        analysis results including:
        - Model comparison table (if available)
        - Red flags and warnings for model quality issues
        - Model statistics (R-squared, observations, etc.)
        - Coefficient estimates with p-values
        - Significant factors ranked by effect size
        - Detected interactions

        For multi-response analysis, results for each response are displayed
        sequentially with clear separators.
        """
        self.stats_text.delete('1.0', tk.END)

        self.stats_text.insert(tk.END, "="*80 + "\n")
        self.stats_text.insert(tk.END, "DOE ANALYSIS RESULTS\n")
        self.stats_text.insert(tk.END, "="*80 + "\n\n")

        # Multi-response analysis: display results for each response
        if hasattr(self, 'all_results') and len(self.all_results) > 1:
            self.stats_text.insert(tk.END, f"MULTI-RESPONSE ANALYSIS ({len(self.all_results)} responses selected)\n")
            self.stats_text.insert(tk.END, "="*80 + "\n\n")

            for response_idx, response_name in enumerate(self.selected_responses, 1):
                self.stats_text.insert(tk.END, f"\n{'#'*80}\n")
                self.stats_text.insert(tk.END, f"RESPONSE {response_idx}/{len(self.selected_responses)}: {response_name}\n")
                self.stats_text.insert(tk.END, f"{'#'*80}\n\n")

                # Display statistics for this response
                self._display_single_response_statistics(
                    results=self.all_results[response_name],
                    model_comparison=self.model_comparison_all[response_name],
                    model_selection=self.model_selection_all[response_name],
                    response_name=response_name
                )

            return

        # Single response (backward compatible)

        # MODEL COMPARISON SECTION - Show all models and selection
        if hasattr(self, 'model_comparison') and self.model_comparison:
            self.stats_text.insert(tk.END, "MODEL COMPARISON\n")
            self.stats_text.insert(tk.END, "-"*80 + "\n\n")

            # Display comparison table
            comparison_table = self.model_comparison['comparison_table']

            if comparison_table is not None and not comparison_table.empty:
                # Format the comparison table for display
                self.stats_text.insert(tk.END, "Model Comparison (all 5 models fitted):\n\n")

                # Create formatted header
                header = f"{'Model':<30} {'Adj R-sq':>10} {'BIC':>10} {'RMSE':>10} {'DF':>6}  {'Selected':>10}\n"
                self.stats_text.insert(tk.END, header)
                self.stats_text.insert(tk.END, "-"*80 + "\n")

                # Get selected/recommended model
                recommendation = self.model_selection
                recommended_model = recommendation['recommended_model']
                is_auto = self.model_selection_var.get() == 'Auto (Recommended)'

                # Display each model
                model_order = ['mean', 'linear', 'interactions', 'quadratic', 'reduced']
                for model_type in model_order:
                    if model_type in self.model_comparison['models']:
                        stats = self.model_comparison['models'][model_type]
                        model_name = stats['Model Type']

                        # Mark selected model
                        if model_type == recommended_model:
                            marker = "AUTO" if is_auto else "USER"
                        else:
                            marker = ""

                        line = (f"{model_name:<30} "
                               f"{stats['Adj R²']:>10.4f} "
                               f"{stats['BIC']:>10.1f} "
                               f"{stats['RMSE']:>10.4f} "
                               f"{stats['DF Model']:>6} "
                               f"{marker:>10}\n")
                        self.stats_text.insert(tk.END, line)
                    elif model_type in self.model_comparison['errors']:
                        # Show error for models that failed to fit
                        error_msg = self.model_comparison['errors'][model_type]
                        model_name = self.analyzer.MODEL_TYPES.get(model_type, model_type)
                        self.stats_text.insert(tk.END,
                                             f"{model_name:<30} (Failed: {error_msg[:40]}...)\n")

                # Show recommendation explanation
                self.stats_text.insert(tk.END, "\n" + "-"*80 + "\n")
                self.stats_text.insert(tk.END, f"RECOMMENDED: {self.analyzer.MODEL_TYPES[recommended_model]}\n")
                self.stats_text.insert(tk.END, f"   Reason: {recommendation['reason']}\n")

                # Show note about model selection criteria
                self.stats_text.insert(tk.END, "\nSelection criteria: Adjusted R-squared (60%), BIC (30%), Parsimony (10%)\n")
                self.stats_text.insert(tk.END, "   Higher Adj R-squared is better | Lower BIC is better | Simpler models preferred\n")

            self.stats_text.insert(tk.END, "\n" + "="*80 + "\n\n")

        # RED FLAGS SECTION - Show warnings first if model quality is poor
        r_squared = self.results['model_stats']['R-squared']
        n_obs = self.results['model_stats']['Observations']
        sig_factors = self.analyzer.get_significant_factors()

        warnings = []
        if r_squared < 0.5:
            warnings.append(f"LOW R-squared ({r_squared:.3f}): Model explains only {r_squared*100:.1f}% of variance; inspect residuals for unexplained structure.")
        elif r_squared < 0.7:
            warnings.append(f"MODERATE R-squared ({r_squared:.3f}): Model is acceptable but could be improved.")

        if len(sig_factors) == 0:
            warnings.append("NO SIGNIFICANT FACTORS: No factors with p < 0.05 found. Consider reviewing experimental design.")

        if n_obs < 20:
            warnings.append(f"SMALL SAMPLE SIZE: Only {n_obs} observations. Consider adding replicates for more robust results.")

        if warnings:
            self.stats_text.insert(tk.END, "RED FLAGS / WARNINGS\n")
            self.stats_text.insert(tk.END, "-"*80 + "\n")
            for warning in warnings:
                self.stats_text.insert(tk.END, f"{warning}\n")
            self.stats_text.insert(tk.END, "\n" + "="*80 + "\n\n")

        # MODEL STATISTICS
        self.stats_text.insert(tk.END, "MODEL STATISTICS\n")
        self.stats_text.insert(tk.END, "-"*80 + "\n")
        for key, value in self.results['model_stats'].items():
            if isinstance(value, float):
                self.stats_text.insert(tk.END, f"  {key:<25}: {value:>15.6f}\n")
            else:
                self.stats_text.insert(tk.END, f"  {key:<25}: {value:>15}\n")

        self.stats_text.insert(tk.END, "\n" + "="*80 + "\n")
        self.stats_text.insert(tk.END, "COEFFICIENTS\n")
        self.stats_text.insert(tk.END, "-"*80 + "\n\n")

        coef_str = self.results['coefficients'].to_string()
        self.stats_text.insert(tk.END, coef_str + "\n")

        # SIGNIFICANT FACTORS
        self.stats_text.insert(tk.END, "\n" + "="*80 + "\n")
        self.stats_text.insert(tk.END, f"SIGNIFICANT FACTORS (p < 0.05): {len(sig_factors)} found\n")
        self.stats_text.insert(tk.END, "-"*80 + "\n")

        if sig_factors:
            # Sort by absolute coefficient value to rank importance
            factor_importance = []
            for factor in sig_factors:
                coef = self.results['coefficients'].loc[factor, 'Coefficient']
                pval = self.results['coefficients'].loc[factor, 'p-value']
                factor_importance.append((factor, abs(coef), coef, pval))

            # Rank by effect magnitude
            factor_importance.sort(key=lambda x: x[1], reverse=True)

            self.stats_text.insert(tk.END, "Ranked by effect size (most important first):\n\n")
            for rank, (factor, abs_coef, coef, pval) in enumerate(factor_importance, 1):
                self.stats_text.insert(tk.END, f"  {rank}. {factor:<38} coef={coef:>10.4f}  p={pval:.2e}\n")
        else:
            self.stats_text.insert(tk.END, "  None found - no factors are statistically significant.\n")

        # INTERACTION DETECTION
        interaction_factors = [f for f in sig_factors if ':' in f]
        if interaction_factors:
            self.stats_text.insert(tk.END, "\n" + "="*80 + "\n")
            self.stats_text.insert(tk.END, f"SIGNIFICANT INTERACTIONS DETECTED: {len(interaction_factors)}\n")
            self.stats_text.insert(tk.END, "-"*80 + "\n")
            for interaction in interaction_factors:
                coef = self.results['coefficients'].loc[interaction, 'Coefficient']
                pval = self.results['coefficients'].loc[interaction, 'p-value']
                self.stats_text.insert(tk.END, f"  {interaction:<40} coef={coef:>10.4f}  p={pval:.2e}\n")
            self.stats_text.insert(tk.END, "\nWARNING: Interactions mean optimal settings depend on factor combinations!\n")

        self.stats_text.insert(tk.END, "\n" + "="*80 + "\n")
