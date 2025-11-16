"""
Analysis Tab - DoE Analysis and Bayesian Optimization
Simplified version adapted from doe_analysis_gui.pyw
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

from core.project import DoEProject, METADATA_COLUMNS
from core.doe_analyzer import DoEAnalyzer
from core.optimizer import BayesianOptimizer, AX_AVAILABLE


class AnalysisTab(ttk.Frame):
    """DoE analysis and optimization interface"""

    def __init__(self, parent, project: DoEProject, main_window):
        super().__init__(parent)
        self.project = project
        self.main_window = main_window
        self.analyzer = DoEAnalyzer()
        self.optimizer = BayesianOptimizer()

        self._create_ui()

    def _create_ui(self):
        """Setup UI elements"""
        # Main container
        main_container = ttk.Frame(self, padding=10)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left panel - Controls
        left_panel = ttk.Frame(main_container, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)

        # Data loading
        data_frame = ttk.LabelFrame(left_panel, text="Data", padding=10)
        data_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(data_frame, text="Load Results (Excel)", command=self.load_results).pack(fill=tk.X, pady=(0, 5))

        self.data_status = ttk.Label(data_frame, text="No data loaded", foreground="gray")
        self.data_status.pack()

        # Response selection
        response_frame = ttk.LabelFrame(left_panel, text="Response", padding=10)
        response_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(response_frame, text="Select Response:").pack(anchor=tk.W)
        self.response_var = tk.StringVar()
        self.response_dropdown = ttk.Combobox(response_frame, textvariable=self.response_var, state="readonly")
        self.response_dropdown.pack(fill=tk.X, pady=(5, 0))
        self.response_dropdown.bind('<<ComboboxSelected>>', lambda e: self._detect_columns())

        # Analysis
        analysis_frame = ttk.LabelFrame(left_panel, text="Analysis", padding=10)
        analysis_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(analysis_frame, text="Model Type:").pack(anchor=tk.W)
        self.model_var = tk.StringVar(value="linear")
        models = [("Linear", "linear"), ("With Interactions", "interactions"),
                 ("Quadratic", "quadratic")]

        for label, value in models:
            ttk.Radiobutton(analysis_frame, text=label, variable=self.model_var, value=value).pack(anchor=tk.W)

        ttk.Button(analysis_frame, text="Run Analysis", command=self._run_analysis).pack(fill=tk.X, pady=(10, 0))

        # Bayesian Optimization
        bo_frame = ttk.LabelFrame(left_panel, text="Bayesian Optimization", padding=10)
        bo_frame.pack(fill=tk.X, pady=(0, 10))

        if not AX_AVAILABLE:
            ttk.Label(bo_frame, text="Install ax-platform\nfor BO support", foreground="gray").pack()
        else:
            self.bo_minimize_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(bo_frame, text="Minimize Response", variable=self.bo_minimize_var).pack(anchor=tk.W)

            ttk.Button(bo_frame, text="Initialize BO", command=self._init_bo).pack(fill=tk.X, pady=(5, 0))
            ttk.Button(bo_frame, text="Get Suggestions", command=self._get_suggestions).pack(fill=tk.X, pady=(5, 0))

        # Right panel - Results
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Notebook for different views
        self.results_notebook = ttk.Notebook(right_panel)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)

        # Summary tab
        summary_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(summary_frame, text="Summary")

        self.summary_text = scrolledtext.ScrolledText(summary_frame, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        # Plot tab
        plot_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(plot_frame, text="Plots")

        self.plot_canvas_frame = ttk.Frame(plot_frame)
        self.plot_canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Suggestions tab
        suggestions_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(suggestions_frame, text="BO Suggestions")

        self.suggestions_text = scrolledtext.ScrolledText(suggestions_frame, wrap=tk.WORD)
        self.suggestions_text.pack(fill=tk.BOTH, expand=True)

    def load_results(self):
        """Load experimental results from Excel"""
        filepath = filedialog.askopenfilename(
            title="Load Results",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )

        if filepath:
            try:
                self.project.load_results(filepath)
                self.project.load_stock_concentrations_from_sheet(filepath)

                # Update UI
                columns = [col for col in self.project.results_data.columns if col not in METADATA_COLUMNS]
                self.response_dropdown['values'] = columns

                if 'Response' in columns:
                    self.response_var.set('Response')
                    self._detect_columns()
                elif columns:
                    self.response_var.set(columns[0])
                    self._detect_columns()

                n_rows = len(self.project.results_data)
                self.data_status.config(text=f"Loaded: {n_rows} experiments", foreground="green")
                self.main_window.update_status(f"Loaded {n_rows} experiments")

            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load data:\n{e}")

    def _detect_columns(self):
        """Detect factor and response columns"""
        response = self.response_var.get()
        if not response or self.project.results_data is None:
            return

        try:
            self.project.detect_columns(response)
            self.project.preprocess_data()

            summary = f"Response: {response}\n"
            summary += f"Numeric factors: {', '.join(self.project.numeric_factors)}\n"
            summary += f"Categorical factors: {', '.join(self.project.categorical_factors)}\n"
            summary += f"Clean data: {len(self.project.clean_data)} rows"

            self.summary_text.delete('1.0', tk.END)
            self.summary_text.insert('1.0', summary)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _run_analysis(self):
        """Run statistical analysis"""
        if self.project.clean_data is None:
            messagebox.showwarning("No Data", "Load and process data first")
            return

        try:
            # Set analyzer data
            self.analyzer.set_data(
                self.project.clean_data,
                self.project.factor_columns,
                self.project.categorical_factors,
                self.project.numeric_factors,
                self.project.response_column
            )

            # Fit model
            model_type = self.model_var.get()
            results = self.analyzer.fit_model(model_type)

            # Display results
            summary = f"=== {DoEAnalyzer.MODEL_TYPES[model_type]} ===\n\n"
            summary += f"R-squared: {results['model_stats']['R-squared']:.4f}\n"
            summary += f"Adjusted R-squared: {results['model_stats']['Adjusted R-squared']:.4f}\n"
            summary += f"RMSE: {results['model_stats']['RMSE']:.4f}\n"
            summary += f"F p-value: {results['model_stats']['F p-value']:.4e}\n\n"

            summary += "=== Coefficients ===\n"
            summary += results['coefficients'].to_string()

            self.summary_text.delete('1.0', tk.END)
            self.summary_text.insert('1.0', summary)

            # Plot main effects
            self._plot_main_effects()

            self.main_window.update_status("Analysis complete")

        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))

    def _plot_main_effects(self):
        """Plot main effects"""
        try:
            main_effects = self.analyzer.calculate_main_effects()

            # Clear previous plots
            for widget in self.plot_canvas_frame.winfo_children():
                widget.destroy()

            # Create figure
            n_factors = len(main_effects)
            if n_factors == 0:
                return

            fig, axes = plt.subplots(1, min(n_factors, 3), figsize=(12, 4))
            if n_factors == 1:
                axes = [axes]

            for idx, (factor, effects_df) in enumerate(list(main_effects.items())[:3]):
                ax = axes[idx]
                effects_df['Mean Response'].plot(kind='bar', ax=ax, color='steelblue')
                ax.set_title(f"Main Effect: {factor}")
                ax.set_ylabel(self.project.response_column)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, self.plot_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            print(f"Plot error: {e}")

    def _init_bo(self):
        """Initialize Bayesian Optimization"""
        if not AX_AVAILABLE:
            messagebox.showerror("Not Available", "Install ax-platform:\npip install ax-platform")
            return

        if self.project.clean_data is None:
            messagebox.showwarning("No Data", "Load data first")
            return

        try:
            self.optimizer.set_data(
                self.project.clean_data,
                self.project.factor_columns,
                self.project.categorical_factors,
                self.project.numeric_factors,
                self.project.response_column
            )

            minimize = self.bo_minimize_var.get()
            self.optimizer.initialize_optimizer(minimize=minimize)

            messagebox.showinfo("Success", "Bayesian Optimization initialized")
            self.main_window.update_status("BO initialized")

        except Exception as e:
            messagebox.showerror("BO Error", str(e))

    def _get_suggestions(self):
        """Get BO suggestions"""
        if not self.optimizer.is_initialized:
            messagebox.showwarning("Not Initialized", "Initialize BO first")
            return

        try:
            suggestions = self.optimizer.get_next_suggestions(n=5)

            # Display suggestions
            output = "=== Next 5 Suggested Experiments ===\n\n"
            for idx, params in enumerate(suggestions, 1):
                output += f"Suggestion {idx}:\n"
                for factor, value in params.items():
                    output += f"  {factor}: {value}\n"
                output += "\n"

            self.suggestions_text.delete('1.0', tk.END)
            self.suggestions_text.insert('1.0', output)

            self.results_notebook.select(2)  # Switch to suggestions tab
            self.main_window.update_status("Generated 5 BO suggestions")

        except Exception as e:
            messagebox.showerror("BO Error", str(e))

    def refresh(self):
        """Refresh UI from project data"""
        self.summary_text.delete('1.0', tk.END)
        self.suggestions_text.delete('1.0', tk.END)

        if self.project.results_data is not None:
            n_rows = len(self.project.results_data)
            self.data_status.config(text=f"Loaded: {n_rows} experiments", foreground="green")
        else:
            self.data_status.config(text="No data loaded", foreground="gray")
