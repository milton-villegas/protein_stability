#!/usr/bin/env python3
"""
Optimization Panel Mixin for DoE Analysis Tab

This module provides optimization-related methods for displaying recommendations,
Bayesian optimization suggestions, and Pareto frontier analysis.

Contains methods extracted from analysis_tab.py for better code organization.
"""

# GUI imports
import tkinter as tk
from tkinter import ttk, messagebox

# Plotting imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import validation function from sibling module
from .validation import validate_constraint

# Bayesian Optimization availability check
try:
    from ax.service.ax_client import AxClient, ObjectiveProperties
    AX_AVAILABLE = True
except ImportError:
    AX_AVAILABLE = False


class OptimizationPanelMixin:
    """
    Mixin class providing optimization panel methods for the Analysis Tab.

    This mixin expects the following attributes to be present on the class:
        - self.optimizer: BayesianOptimizer instance
        - self.handler: DataHandler instance
        - self.results: Analysis results dictionary
        - self.analyzer: DoEAnalyzer instance
        - self.response_directions: Dict mapping response names to 'maximize'/'minimize'
        - self.response_constraints: Dict mapping response names to constraint dicts
        - self.selected_responses: List of selected response column names
        - self.recommendations_text: Text widget for recommendations
        - self.optimization_frame: Frame for optimization plots
        - self.main_window: Reference to main window for status updates
        - self.exploration_mode_var: Tkinter variable for exploration mode
        - self.has_analysis_error: Boolean flag for analysis errors
        - self.debug_log: List for debug messages
        - self._debug_log(): Method for logging debug messages
    """

    def display_recommendations(self):
        """
        Display recommendations and optimal conditions.

        This method analyzes the experimental data and model results to provide:
        - Confidence level assessment based on R-squared and significant factors
        - Best observed experiment identification
        - Model-predicted optimal directions
        - Next steps recommendations
        - Bayesian optimization suggestions (if available)
        - Pareto frontier analysis for multi-objective optimization
        """
        self.recommendations_text.delete('1.0', tk.END)

        self.recommendations_text.insert(tk.END, "="*80 + "\n")
        self.recommendations_text.insert(tk.END, "RECOMMENDATIONS & OPTIMAL CONDITIONS\n")
        self.recommendations_text.insert(tk.END, "="*80 + "\n\n")

        # Determine confidence level
        r_squared = self.results['model_stats']['R-squared']
        n_obs = self.results['model_stats']['Observations']
        sig_factors = self.analyzer.get_significant_factors()

        # Determine confidence level based on model quality metrics
        if r_squared >= 0.8 and len(sig_factors) > 0 and n_obs >= 20:
            confidence = "HIGH"
        elif r_squared >= 0.6 and len(sig_factors) > 0:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        self.recommendations_text.insert(tk.END, f"CONFIDENCE LEVEL: {confidence}\n")
        self.recommendations_text.insert(tk.END, "-"*80 + "\n")
        self.recommendations_text.insert(tk.END, f"Based on R² = {r_squared:.3f}, {n_obs} observations, {len(sig_factors)} significant factors\n\n")

        # Find optimal condition from data based on optimization direction
        clean_data = self.handler.clean_data

        # Get direction for this response (default to maximize for backward compatibility)
        response_direction = self.response_directions.get(self.handler.response_column, 'maximize')

        self._debug_log(f"[DEBUG BEST] Finding best experiment for '{self.handler.response_column}'")
        self._debug_log(f"  - Direction: {response_direction}")
        self._debug_log(f"  - Value range: {clean_data[self.handler.response_column].min()} to {clean_data[self.handler.response_column].max()}")
        self._debug_log(f"  - Constraints: {self.response_constraints.get(self.handler.response_column, 'None')}")

        # Filter by constraints if any exist for this response
        filtered_data = clean_data.copy()
        constraint_applied = False
        constraint_msg = ""

        if self.handler.response_column in self.response_constraints:
            constraint = self.response_constraints[self.handler.response_column]
            original_count = len(filtered_data)

            if 'min' in constraint:
                min_val = constraint['min']
                filtered_data = filtered_data[filtered_data[self.handler.response_column] >= min_val]
                constraint_msg += f" >= {min_val}"
                constraint_applied = True

            if 'max' in constraint:
                max_val = constraint['max']
                filtered_data = filtered_data[filtered_data[self.handler.response_column] <= max_val]
                constraint_msg += f" <= {max_val}" if not constraint_msg else f" and <= {max_val}"
                constraint_applied = True

            self._debug_log(f"  - After constraint filtering: {len(filtered_data)}/{original_count} experiments remain")

        self.recommendations_text.insert(tk.END, "="*80 + "\n")
        self.recommendations_text.insert(tk.END, "BEST OBSERVED EXPERIMENT\n")
        self.recommendations_text.insert(tk.END, "-"*80 + "\n")

        # Check if any experiments meet constraints
        if constraint_applied and len(filtered_data) == 0:
            # Detect logical contradictions between direction and constraints
            data_min = clean_data[self.handler.response_column].min()
            data_max = clean_data[self.handler.response_column].max()
            constraint = self.response_constraints[self.handler.response_column]

            is_contradiction = False
            contradiction_msg = ""

            # Check for contradictions
            if response_direction == 'minimize' and 'min' in constraint:
                if constraint['min'] > data_max:
                    is_contradiction = True
                    contradiction_msg = (
                        f"LOGICAL CONTRADICTION DETECTED!\n\n"
                        f"You selected:\n"
                        f"  - Direction: MINIMIZE {self.handler.response_column} (get it as LOW as possible)\n"
                        f"  - Constraint: {self.handler.response_column} >= {constraint['min']} (must be at least {constraint['min']})\n\n"
                        f"But your data range is {data_min:.2f} to {data_max:.2f} (all below {constraint['min']}).\n\n"
                        f"This is contradictory because:\n"
                        f"  - MINIMIZE means you want the LOWEST value\n"
                        f"  - But your constraint requires a HIGH value ({constraint['min']})\n\n"
                        f"Did you mean to MAXIMIZE instead of minimize?\n\n"
                        f"Please fix this by either:\n"
                        f"  1. Change direction to MAXIMIZE (if you want {self.handler.response_column} to be high)\n"
                        f"  2. Remove the minimum constraint (if you truly want the lowest value)\n"
                        f"  3. Adjust the constraint to a realistic value within your data range\n\n"
                    )

            elif response_direction == 'maximize' and 'max' in constraint:
                if constraint['max'] < data_min:
                    is_contradiction = True
                    contradiction_msg = (
                        f"LOGICAL CONTRADICTION DETECTED!\n\n"
                        f"You selected:\n"
                        f"  - Direction: MAXIMIZE {self.handler.response_column} (get it as HIGH as possible)\n"
                        f"  - Constraint: {self.handler.response_column} <= {constraint['max']} (must be at most {constraint['max']})\n\n"
                        f"But your data range is {data_min:.2f} to {data_max:.2f} (all above {constraint['max']}).\n\n"
                        f"This is contradictory because:\n"
                        f"  - MAXIMIZE means you want the HIGHEST value\n"
                        f"  - But your constraint requires a LOW value ({constraint['max']})\n\n"
                        f"Did you mean to MINIMIZE instead of maximize?\n\n"
                        f"Please fix this by either:\n"
                        f"  1. Change direction to MINIMIZE (if you want {self.handler.response_column} to be low)\n"
                        f"  2. Remove the maximum constraint (if you truly want the highest value)\n"
                        f"  3. Adjust the constraint to a realistic value within your data range\n\n"
                    )

            if is_contradiction:
                # Show popup error message
                if response_direction == 'minimize' and 'min' in constraint:
                    messagebox.showerror("Logical Contradiction",
                        f"You selected:\n"
                        f"  - MINIMIZE {self.handler.response_column}\n"
                        f"  - Constraint: {self.handler.response_column} >= {constraint['min']}\n\n"
                        f"Your data range: {data_min:.2f} to {data_max:.2f}\n\n"
                        f"This is contradictory because MINIMIZE means finding the LOWEST value, "
                        f"but your constraint requires a HIGH value ({constraint['min']}).\n\n"
                        f"Did you mean to MAXIMIZE instead?\n\n"
                        f"Please fix by:\n"
                        f"  1. Change direction to MAXIMIZE\n"
                        f"  2. Remove the minimum constraint\n"
                        f"  3. Adjust constraint to match your data range")
                elif response_direction == 'maximize' and 'max' in constraint:
                    messagebox.showerror("Logical Contradiction",
                        f"You selected:\n"
                        f"  - MAXIMIZE {self.handler.response_column}\n"
                        f"  - Constraint: {self.handler.response_column} <= {constraint['max']}\n\n"
                        f"Your data range: {data_min:.2f} to {data_max:.2f}\n\n"
                        f"This is contradictory because MAXIMIZE means finding the HIGHEST value, "
                        f"but your constraint requires a LOW value ({constraint['max']}).\n\n"
                        f"Did you mean to MINIMIZE instead?\n\n"
                        f"Please fix by:\n"
                        f"  1. Change direction to MINIMIZE\n"
                        f"  2. Remove the maximum constraint\n"
                        f"  3. Adjust constraint to match your data range")

                self.main_window.update_status("Error: Logical contradiction in settings")

                # Set error flag to prevent success popup
                self.has_analysis_error = True

                # Also show in recommendations tab
                self.recommendations_text.insert(tk.END, contradiction_msg)
                self.recommendations_text.insert(tk.END, "="*80 + "\n")
                self.recommendations_text.insert(tk.END, "Analysis cannot continue with contradictory settings.\n")
                self.recommendations_text.insert(tk.END, "Please adjust your settings and re-run the analysis.\n")
                return  # Stop the recommendations display
            else:
                # No contradiction, just no data meets constraints
                self.recommendations_text.insert(tk.END, f"WARNING: No experiments meet the constraint {self.handler.response_column}{constraint_msg}\n\n")
                self.recommendations_text.insert(tk.END, f"Your data range: {data_min:.2f} to {data_max:.2f}\n")
                self.recommendations_text.insert(tk.END, f"Constraint requires: {self.handler.response_column}{constraint_msg}\n\n")
                self.recommendations_text.insert(tk.END, "Showing best experiment WITHOUT constraint applied:\n\n")
                filtered_data = clean_data.copy()
                constraint_applied = False

        # Find best based on direction
        if response_direction == 'minimize':
            best_idx = filtered_data[self.handler.response_column].idxmin()
            self._debug_log(f"  - Using idxmin() for minimize")
        else:
            best_idx = filtered_data[self.handler.response_column].idxmax()
            self._debug_log(f"  - Using idxmax() for maximize")

        optimal_response = filtered_data.loc[best_idx, self.handler.response_column]
        self._debug_log(f"  - Best value: {optimal_response} at index {best_idx}")

        # Show ID if available
        direction_text = "lowest" if response_direction == 'minimize' else "highest"
        constraint_text = f" (meeting constraint{constraint_msg})" if constraint_applied else ""
        header = f"This is the experiment with the {direction_text} {self.handler.response_column}{constraint_text}."
        if 'ID' in clean_data.columns:
            exp_id = clean_data.loc[best_idx, 'ID']
            header = f"Best Experiment: ID {exp_id} ({direction_text} {self.handler.response_column}{constraint_text})"
        self.recommendations_text.insert(tk.END, header + "\n\n")

        self.recommendations_text.insert(tk.END, "Experimental Conditions:\n")
        for factor in self.handler.factor_columns:
            value = clean_data.loc[best_idx, factor]
            if isinstance(value, float):
                self.recommendations_text.insert(tk.END, f"  - {factor:<30}: {value:.2f}\n")
            else:
                self.recommendations_text.insert(tk.END, f"  - {factor:<30}: {value}\n")

        self.recommendations_text.insert(tk.END, f"\nResult:\n  - {self.handler.response_column:<30}: {optimal_response:.2f} ({response_direction})\n")

        # Predicted optimal based on model
        self.recommendations_text.insert(tk.END, "\n" + "="*80 + "\n")
        self.recommendations_text.insert(tk.END, "MODEL-PREDICTED OPTIMAL DIRECTION\n")
        self.recommendations_text.insert(tk.END, "-"*80 + "\n")

        if len(sig_factors) > 0:
            # Get only main effect factors (not interactions or quadratic terms)
            main_sig_factors = [f for f in sig_factors
                               if ':' not in f and 'I(' not in f and '**' not in f]

            if main_sig_factors:
                self.recommendations_text.insert(tk.END, "To INCREASE response, adjust these significant factors:\n\n")
                for factor in main_sig_factors:
                    coef = self.results['coefficients'].loc[factor, 'Coefficient']
                    # Positive coef = increase factor to increase response
                    if coef > 0:
                        direction = "INCREASE"
                    else:
                        direction = "DECREASE"

                    # Clean factor name (remove C() and Q() wrappers)
                    clean_factor = factor.replace("C(Q('", "").replace("'))", "").replace("Q('", "").replace("')", "")
                    self.recommendations_text.insert(tk.END, f"  - {clean_factor:<30}: {direction}  (effect: {coef:+.4f})\n")

            interaction_factors = [f for f in sig_factors if ':' in f]
            quadratic_factors = [f for f in sig_factors if 'I(' in f or '**' in f]

            if interaction_factors:
                self.recommendations_text.insert(tk.END, f"\nWARNING: {len(interaction_factors)} interaction(s) detected!\n")
                self.recommendations_text.insert(tk.END, "Optimal levels depend on factor combinations - see Interactions plot.\n")

            if quadratic_factors:
                self.recommendations_text.insert(tk.END, f"\nNOTE: {len(quadratic_factors)} quadratic term(s) detected!\n")
                self.recommendations_text.insert(tk.END, "Response has curvature - optimal values may be within the tested range.\n")
        else:
            self.recommendations_text.insert(tk.END, "No significant factors found. Consider:\n")
            self.recommendations_text.insert(tk.END, "  - Testing wider factor ranges\n")
            self.recommendations_text.insert(tk.END, "  - Adding more factors to the experiment\n")
            self.recommendations_text.insert(tk.END, "  - Checking measurement accuracy\n")

        # Next steps recommendations
        self.recommendations_text.insert(tk.END, "\n" + "="*80 + "\n")
        self.recommendations_text.insert(tk.END, "NEXT STEPS\n")
        self.recommendations_text.insert(tk.END, "-"*80 + "\n")

        if confidence == "HIGH":
            self.recommendations_text.insert(tk.END,
                "1. Run 3-5 confirmation experiments at the predicted optimal condition\n"
                "2. Compare results to model prediction to validate\n"
                "3. If confirmed, implement optimized condition in production\n"
            )
        elif confidence == "MEDIUM":
            self.recommendations_text.insert(tk.END,
                "1. Run confirmation experiments at predicted optimal condition\n"
                "2. Consider additional replicates to improve model confidence\n"
                "3. May need to refine factor ranges or add more data\n"
            )
        else:
            self.recommendations_text.insert(tk.END,
                "1. WARNING: Results may not be reliable enough for immediate use\n"
                "2. Consider running more experiments\n"
                "3. Check for experimental errors or measurement issues\n"
                "4. May need to reconsider factors or expand factor ranges\n"
            )

        self.recommendations_text.insert(tk.END, "\n" + "="*80 + "\n")

        # Add Bayesian Optimization suggestions if available
        if AX_AVAILABLE and self.optimizer:
            self.recommendations_text.insert(tk.END, "\n" + "="*80 + "\n")
            self.recommendations_text.insert(tk.END, "BAYESIAN OPTIMIZATION SUGGESTIONS\n")
            self.recommendations_text.insert(tk.END, "="*80 + "\n")

            try:
                # Initialize optimizer with current data (multi-response support)
                exploration_mode = self.exploration_mode_var.get() if hasattr(self, 'exploration_mode_var') else False

                self._debug_log(f"[DEBUG ANALYSIS] Setting optimizer data:")
                self._debug_log(f"  - Response columns: {self.selected_responses}")
                self._debug_log(f"  - Response directions: {self.response_directions}")
                self._debug_log(f"  - Response constraints: {self.response_constraints}")
                self._debug_log(f"  - Exploration mode: {exploration_mode}")

                self.optimizer.set_data(
                    data=self.handler.clean_data,
                    factor_columns=self.handler.factor_columns,
                    categorical_factors=self.handler.categorical_factors,
                    numeric_factors=self.handler.numeric_factors,
                    response_columns=self.selected_responses,
                    response_directions=self.response_directions,
                    response_constraints=self.response_constraints,
                    exploration_mode=exploration_mode
                )

                self._debug_log(f"[DEBUG ANALYSIS] After set_data:")
                self._debug_log(f"  - Optimizer.response_directions: {self.optimizer.response_directions}")
                self._debug_log(f"  - Optimizer.response_constraints: {self.optimizer.response_constraints}")
                self._debug_log(f"  - Optimizer.is_multi_objective: {self.optimizer.is_multi_objective}")

                # COMPREHENSIVE CONSTRAINT VALIDATION
                # Validate all constraints with severity levels: error, warning, info
                all_validation_results = []
                errors = []
                warnings = []
                infos = []

                if self.response_constraints:
                    clean_data = self.handler.clean_data
                    total_experiments = len(clean_data)

                    for response_name, constraint in self.response_constraints.items():
                        direction = self.response_directions.get(response_name, 'maximize')
                        data_min = clean_data[response_name].min()
                        data_max = clean_data[response_name].max()

                        # Validate this constraint
                        validation_results = validate_constraint(
                            response_name, direction, constraint,
                            data_min, data_max, total_experiments
                        )

                        all_validation_results.extend(validation_results)

                        # Sort by severity
                        for result in validation_results:
                            if result['severity'] == 'error':
                                errors.append(result)
                            elif result['severity'] == 'warning':
                                warnings.append(result)
                            elif result['severity'] == 'info':
                                infos.append(result)

                # Handle ERRORS (stop analysis)
                if errors:
                    if len(errors) == 1:
                        # Single error
                        err = errors[0]
                        messagebox.showerror("Constraint Error",
                            f"{err['message']}\n\n{err.get('detail', '')}\n\n"
                            f"Please fix this issue before running analysis.")
                    else:
                        # Multiple errors
                        error_msg = f"Found {len(errors)} constraint errors:\n\n"
                        for i, err in enumerate(errors, 1):
                            error_msg += f"{i}. {err['message']}\n"
                        error_msg += "\nPlease fix these issues before running analysis."
                        messagebox.showerror("Multiple Constraint Errors", error_msg)

                    self.main_window.update_status("Error: Invalid constraints")
                    self.has_analysis_error = True

                    # Show in recommendations tab
                    self.recommendations_text.insert(tk.END, "CONSTRAINT ERROR(S) DETECTED!\n\n")
                    for err in errors:
                        self.recommendations_text.insert(tk.END, f"- {err['message']}\n")
                    self.recommendations_text.insert(tk.END, "\nAnalysis cannot continue with invalid constraints.\n")
                    self.recommendations_text.insert(tk.END, "Please fix your constraint settings and re-run the analysis.\n")
                    return

                # Handle WARNINGS (continue but notify)
                if warnings:
                    warning_messages = []
                    for warn in warnings:
                        warning_messages.append(warn['message'])

                    if len(warnings) == 1:
                        messagebox.showwarning("Constraint Warning",
                            f"{warnings[0]['message']}\n\n{warnings[0].get('detail', '')}\n\n"
                            f"Analysis will continue, but this constraint may not have the desired effect.")
                    else:
                        warning_msg = f"Found {len(warnings)} constraint warnings:\n\n"
                        for i, warn in enumerate(warnings, 1):
                            warning_msg += f"{i}. {warn['message']}\n"
                        warning_msg += "\nAnalysis will continue, but these constraints may not have the desired effect."
                        messagebox.showwarning("Constraint Warnings", warning_msg)

                self.optimizer.initialize_optimizer()

                # Show constraint information with validation results
                if self.response_constraints:
                    self.recommendations_text.insert(tk.END, "\nActive Constraints:\n")
                    for response, constraint in self.response_constraints.items():
                        parts = []
                        if 'min' in constraint:
                            parts.append(f"{constraint['min']} <= {response}")
                        if 'max' in constraint:
                            parts.append(f"{response} <= {constraint['max']}")
                        if parts:
                            self.recommendations_text.insert(tk.END, f"  - {' and '.join(parts)}\n")

                    # Show validation results (warnings and info)
                    if warnings or infos:
                        self.recommendations_text.insert(tk.END, "\nConstraint Validation:\n")

                        # Show warnings
                        for warn in warnings:
                            self.recommendations_text.insert(tk.END, f"  WARNING: {warn['message']}\n")

                        # Show infos
                        for info in infos:
                            self.recommendations_text.insert(tk.END, f"  INFO: {info['message']}\n")

                        self.recommendations_text.insert(tk.END, "\n")

                    if exploration_mode:
                        self.recommendations_text.insert(tk.END, "Exploration Mode: ON\n")
                        self.recommendations_text.insert(tk.END, "Suggestions may explore outside constraint boundaries.\n\n")
                    else:
                        self.recommendations_text.insert(tk.END, "Note: Constraints are used as optimization targets.\n")
                        self.recommendations_text.insert(tk.END, "Bayesian optimization will tend toward constraint boundaries.\n\n")
                else:
                    self.recommendations_text.insert(tk.END, "\nNo constraints applied.\n\n")

                self.recommendations_text.insert(tk.END, "\nSuggested next experiments:\n\n")

                # Get suggestions
                suggestions = self.optimizer.get_next_suggestions(n=5)

                for i, suggestion in enumerate(suggestions, 1):
                    self.recommendations_text.insert(tk.END, f"Suggested Experiment #{i}:\n")
                    for factor, value in suggestion.items():
                        if isinstance(value, float):
                            self.recommendations_text.insert(tk.END, f"  - {factor:<30}: {value:.4f}\n")
                        else:
                            self.recommendations_text.insert(tk.END, f"  - {factor:<30}: {value}\n")
                    self.recommendations_text.insert(tk.END, "\n")

                # Add Pareto frontier analysis for multi-objective
                if self.optimizer.is_multi_objective:
                    self._debug_log(f"\n[DEBUG DISPLAY] Displaying Pareto frontier...")
                    self._debug_log(f"  response_directions at display time: {self.response_directions}")

                    pareto_points = self.optimizer.get_pareto_frontier()
                    if pareto_points and len(pareto_points) > 0:
                        self.recommendations_text.insert(tk.END, "\n" + "="*80 + "\n")
                        self.recommendations_text.insert(tk.END, "PARETO FRONTIER ANALYSIS (Multi-Objective Trade-offs)\n")
                        self.recommendations_text.insert(tk.END, "="*80 + "\n\n")

                        self.recommendations_text.insert(tk.END,
                            f"Found {len(pareto_points)} Pareto-optimal experiments (best trade-offs):\n")
                        self.recommendations_text.insert(tk.END,
                            "These are experiments you already ran that represent optimal trade-offs.\n\n")

                        # Show first 5 Pareto points
                        for i, point in enumerate(pareto_points[:5], 1):
                            violations = self.optimizer.check_constraint_violations(point['objectives'])

                            status = "OK" if not violations else "WARNING"

                            # Display header with ID
                            header = f"Pareto Point #{i}: {status}"
                            if point.get('id') is not None:
                                header += f" (ID: {point['id']})"
                            self.recommendations_text.insert(tk.END, header + "\n")

                            if i == 1:  # Debug first point
                                self._debug_log(f"\n[DEBUG DISPLAY] First Pareto point:")
                                self._debug_log(f"  ID: {point.get('id')}")
                                self._debug_log(f"  Objectives: {point['objectives']}")
                                self._debug_log(f"  Checking directions for each objective:")

                            # Display experimental conditions
                            if point.get('parameters'):
                                self.recommendations_text.insert(tk.END, "  Experimental Conditions:\n")
                                for param_name, param_value in point['parameters'].items():
                                    if isinstance(param_value, float):
                                        self.recommendations_text.insert(tk.END, f"    - {param_name}: {param_value:.2f}\n")
                                    else:
                                        self.recommendations_text.insert(tk.END, f"    - {param_name}: {param_value}\n")

                            # Display objectives
                            self.recommendations_text.insert(tk.END, "  Results:\n")
                            for resp, value in point['objectives'].items():
                                direction = self.response_directions.get(resp, 'maximize')
                                if i == 1:  # Debug first point
                                    self._debug_log(f"    {resp}: direction={direction}")
                                arrow = '^' if direction == 'maximize' else 'v'
                                direction_text = '(maximize)' if direction == 'maximize' else '(minimize)'
                                self.recommendations_text.insert(tk.END, f"    {arrow} {resp:<25}: {value:.4f} {direction_text}\n")

                            if violations:
                                self.recommendations_text.insert(tk.END, "  Constraint Violations:\n")
                                for violation in violations:
                                    self.recommendations_text.insert(tk.END, f"    - {violation}\n")

                            self.recommendations_text.insert(tk.END, "\n")

                        if len(pareto_points) > 5:
                            self.recommendations_text.insert(tk.END,
                                f"  ... and {len(pareto_points) - 5} more Pareto-optimal points\n\n")

                        self.recommendations_text.insert(tk.END,
                            "TRADE-OFF INSIGHT:\n"
                            "   No single solution is best for ALL objectives.\n"
                            "   Choose a Pareto point based on your priorities.\n"
                            "   View 'Optimization Details' tab for Pareto frontier visualization.\n\n")

                self.recommendations_text.insert(tk.END,
                    "TIP: View 'Optimization Details' tab for visualization of predicted response surface.\n"
                )

                # Enable export button after successful BO initialization
                if hasattr(self, 'export_bo_button'):
                    self.export_bo_button.config(state='normal')

                self._create_optimization_export_button()

            except Exception as e:
                self.recommendations_text.insert(tk.END,
                    f"Could not generate BO suggestions: {str(e)}\n"
                    "This may require more data points or only numeric factors.\n"
                )
                # Disable export button if BO failed
                if hasattr(self, 'export_bo_button'):
                    self.export_bo_button.config(state='disabled')

            self.recommendations_text.insert(tk.END, "\n" + "="*80 + "\n")

        # Display collected debug log at the end (only if debug mode enabled)
        # Set SHOW_DEBUG_LOG = True if you need to see internal debugging information
        SHOW_DEBUG_LOG = False
        if SHOW_DEBUG_LOG and self.debug_log:
            self.recommendations_text.insert(tk.END, "\nDEBUG LOG\n")
            self.recommendations_text.insert(tk.END, "="*80 + "\n")
            for log_entry in self.debug_log:
                self.recommendations_text.insert(tk.END, log_entry + "\n")
            self.recommendations_text.insert(tk.END, "="*80 + "\n")

    def _create_optimization_export_button(self):
        """
        Create appropriate export button based on optimization type.

        Creates either a Pareto plots button for multi-objective optimization
        or a BO plots button for single-objective optimization.
        """
        if not AX_AVAILABLE or not hasattr(self, 'export_frame_opt'):
            return

        for widget in self.export_frame_opt.winfo_children():
            widget.destroy()

        if self.optimizer.is_multi_objective:
            self.export_pareto_button = ttk.Button(
                self.export_frame_opt,
                text="Pareto Plots",
                command=self.export_pareto_plots_gui,
                state='normal'
            )
            self.export_pareto_button.pack(side='left', padx=5)
        else:
            self.export_bo_plots_button = ttk.Button(
                self.export_frame_opt,
                text="BO Plots",
                command=self.export_bo_plots_gui,
                state='normal'
            )
            self.export_bo_plots_button.pack(side='left', padx=5)

    def display_optimization_plot(self):
        """
        Display Bayesian Optimization predicted response surface or Pareto frontier.

        For multi-objective optimization, displays the Pareto frontier visualization.
        For single-objective optimization, displays the acquisition/response surface plot.

        Requirements:
        - Ax platform must be available (AX_AVAILABLE = True)
        - Optimizer must be initialized
        - At least 2 numeric factors for single-objective plots
        """
        if not AX_AVAILABLE or not self.optimizer:
            return

        for widget in self.optimization_frame.winfo_children():
            widget.destroy()

        try:
            # Multi-objective: Show Pareto frontier
            if self.optimizer.is_multi_objective:
                fig = self.optimizer.plot_pareto_frontier()

                if fig is None:
                    message_label = ttk.Label(
                        self.optimization_frame,
                        text="Pareto frontier plot requires 2-3 objectives.\n"
                             f"Your selection has {len(self.selected_responses)} responses.",
                        font=('TkDefaultFont', 12),
                        justify='center'
                    )
                    message_label.pack(expand=True)
                    return

                canvas = FigureCanvasTkAgg(fig, master=self.optimization_frame)
                canvas.draw()
                canvas_widget = canvas.get_tk_widget()
                canvas_widget.pack(fill='both', expand=True)

                # Bind mousewheel
                if hasattr(self.optimization_frame, '_bind_mousewheel'):
                    self.optimization_frame._bind_mousewheel(canvas_widget)

                # Reset scroll position to top
                if hasattr(self.optimization_frame, '_scroll_canvas'):
                    self.optimization_frame._scroll_canvas.update_idletasks()
                    self.optimization_frame._scroll_canvas.yview_moveto(0)

                plt.close(fig)
                return

            # Single-objective: Show acquisition plot (original behavior)
            # Check if we have enough numeric factors
            num_numeric = len(self.handler.numeric_factors)

            if num_numeric < 2:
                # Show message if plot can't be generated
                message_label = ttk.Label(
                    self.optimization_frame,
                    text=f"Optimization plot requires at least 2 numeric factors.\n"
                         f"Your data has {num_numeric} numeric factor(s).\n\n"
                         f"Check the Recommendations tab for suggested experiments.",
                    font=('TkDefaultFont', 12),
                    justify='center'
                )
                message_label.pack(expand=True)
                return

            fig = self.optimizer.get_acquisition_plot()

            if fig is None:
                # Show message if plot generation failed
                message_label = ttk.Label(
                    self.optimization_frame,
                    text="Could not generate optimization plot.\n"
                         "Check the Recommendations tab for suggested experiments.",
                    font=('TkDefaultFont', 12),
                    justify='center'
                )
                message_label.pack(expand=True)
                return

            canvas = FigureCanvasTkAgg(fig, master=self.optimization_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill='both', expand=True)

            # Bind mousewheel to the matplotlib canvas widget
            if hasattr(self.optimization_frame, '_bind_mousewheel'):
                self.optimization_frame._bind_mousewheel(canvas_widget)

            # Reset scroll position to top
            if hasattr(self.optimization_frame, '_scroll_canvas'):
                self.optimization_frame._scroll_canvas.update_idletasks()
                self.optimization_frame._scroll_canvas.yview_moveto(0)

            plt.close(fig)

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"DISPLAY OPTIMIZATION PLOT ERROR")
            print(f"{'='*60}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"{'='*60}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")

            error_label = ttk.Label(
                self.optimization_frame,
                text=f"Could not generate optimization plot:\n{str(e)}\n\n"
                     "This feature works best with 2+ numeric factors and sufficient data.\n"
                     "Check the Recommendations tab for suggested experiments.",
                font=('TkDefaultFont', 10),
                justify='center',
                wraplength=600
            )
            error_label.pack(expand=True, pady=20)
