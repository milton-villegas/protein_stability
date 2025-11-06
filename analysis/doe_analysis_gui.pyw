#!/usr/bin/env python3
"""
DoE Data Analysis Tool
Statistical analysis GUI for Design of Experiments data.
Replicates MATLAB fitlm (EDA CAPKIN MATLAB code), main effects, and interaction plots.

Milton F. Villegas - v0.3.1
"""

# GUI and system imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Data processing imports
import pandas as pd
import numpy as np

# Statistical analysis imports
import statsmodels.formula.api as smf

# Plotting imports
import matplotlib
matplotlib.use('TkAgg')  # Backend for embedding plots in tkinter GUI
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats as scipy_stats
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 9


# Data Handler

class DataHandler:
    """Data loading and preprocessing"""
    
    # Metadata columns to exclude from analysis
    METADATA_COLUMNS = ['ID', 'Plate_96', 'Well_96', 'Well_384']
    
    def __init__(self):
        self.data = None
        self.clean_data = None
        self.factor_columns = []
        self.categorical_factors = []
        self.numeric_factors = []
        self.response_column = None
        
    def load_excel(self, filepath: str):
        """Load data from Excel file"""
        self.data = pd.read_excel(filepath)
    
    def detect_columns(self, response_column: str):
        """Detect factor types, exclude metadata"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        self.response_column = response_column
        
        # All columns except response and metadata are factors
        self.factor_columns = [
            col for col in self.data.columns 
            if col != response_column and col not in self.METADATA_COLUMNS
        ]
        
        # Detect categorical vs numeric factors
        self.categorical_factors = []
        self.numeric_factors = []
        
        for col in self.factor_columns:
            if self.data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(self.data[col]):
                self.categorical_factors.append(col)
            else:
                self.numeric_factors.append(col)
    
    def preprocess_data(self):
        """Clean and prepare data for analysis"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Create copy
        self.clean_data = self.data.copy()
        
        # Drop metadata columns
        columns_to_drop = [col for col in self.METADATA_COLUMNS if col in self.clean_data.columns]
        if columns_to_drop:
            self.clean_data = self.clean_data.drop(columns=columns_to_drop)
        
        # Drop rows with missing response
        self.clean_data = self.clean_data.dropna(subset=[self.response_column])
        
        # Handle categorical variables - fill NaN with 'None'
        for col in self.categorical_factors:
            self.clean_data[col] = self.clean_data[col].fillna('None')
            self.clean_data[col] = self.clean_data[col].astype(str)
        
        # Handle numeric variables - drop rows with NaN in factors
        for col in self.numeric_factors:
            self.clean_data = self.clean_data.dropna(subset=[col])
        
        return self.clean_data


# DoE Analyzer

class DoEAnalyzer:
    """Statistical analysis via regression"""
    
    MODEL_TYPES = {
        'linear': 'Linear (main effects only)',
        'interactions': 'Linear with 2-way interactions',
        'quadratic': 'Quadratic (interactions + squared terms)',
        'purequadratic': 'Pure quadratic (squared terms only)'
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
        
    def set_data(self, data, factor_columns, categorical_factors, numeric_factors, response_column):
        """Set data and factor information"""
        self.data = data.copy()
        self.factor_columns = factor_columns
        self.categorical_factors = categorical_factors
        self.numeric_factors = numeric_factors
        self.response_column = response_column
    
    def _build_interaction_terms(self, factor_terms):
        """Build interaction terms for all factor combinations"""
        interactions = []
        for i in range(len(factor_terms)):
            for j in range(i+1, len(factor_terms)):
                interactions.append(f"{factor_terms[i]}:{factor_terms[j]}")
        return interactions
        
    def build_formula(self, model_type='linear'):
        """Build regression formula based on model type"""
        self.model_type = model_type
        
        # Prepare factor terms
        factor_terms = []
        for factor in self.factor_columns:
            if factor in self.categorical_factors:
                # C() treats as categorical, Q() quotes column names with spaces
                factor_terms.append(f"C(Q('{factor}'))")
            else:
                factor_terms.append(f"Q('{factor}')")
        
        # Build formula
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
                # I() treats expression as-is (prevents ** being interpreted as interaction)
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
                # I() treats expression as-is
                squared_terms.append(f"I(Q('{factor}')**2)")
            formula = f"Q('{self.response_column}') ~ {main_effects}"
            if squared_terms:
                formula += " + " + " + ".join(squared_terms)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return formula
    
    def fit_model(self, model_type='linear'):
        """Fit regression model"""
        if self.data is None:
            raise ValueError("No data set")
        
        formula = self.build_formula(model_type)
        self.model = smf.ols(formula=formula, data=self.data).fit()
        self.results = self._extract_results()
        return self.results
    
    def _extract_results(self):
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
    
    def get_significant_factors(self, alpha=0.05):
        """Get list of significant factors"""
        if self.results is None:
            raise ValueError("No results available")
        
        coef_df = self.results['coefficients']
        significant = coef_df[coef_df['p-value'] < alpha]
        return [idx for idx in significant.index if idx != 'Intercept']
    
    def calculate_main_effects(self):
        """Calculate main effects for each factor"""
        if self.data is None:
            raise ValueError("No data available")
        
        main_effects = {}
        for factor in self.factor_columns:
            # Group by factor levels and calculate mean/std/count for each
            effects = self.data.groupby(factor)[self.response_column].agg(['mean', 'std', 'count'])
            effects.columns = ['Mean Response', 'Std Dev', 'Count']
            main_effects[factor] = effects
        
        return main_effects


# Plotter class for visualizations

class DoEPlotter:
    """Plotting functions for DoE"""
    
    def __init__(self):
        self.data = None
        self.factor_columns = []
        self.response_column = None
        
    def set_data(self, data, factor_columns, response_column):
        """Set data for plotting"""
        self.data = data
        self.factor_columns = factor_columns
        self.response_column = response_column
    
    def plot_main_effects(self, save_path=None):
        """Create main effects plot"""
        num_factors = len(self.factor_columns)
        ncols = min(3, num_factors)
        nrows = (num_factors + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 2*nrows))
        if num_factors == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, factor in enumerate(self.factor_columns):
            ax = axes[idx]
            grouped = self.data.groupby(factor)[self.response_column].agg(['mean', 'std'])
            levels = sorted(self.data[factor].unique())
            means = [grouped.loc[level, 'mean'] for level in levels]
            stds = [grouped.loc[level, 'std'] for level in levels]
            
            ax.plot(range(len(levels)), means, 'o-', linewidth=2, markersize=8, color='steelblue')
            # Shaded region shows ± 1 std dev
            ax.fill_between(range(len(levels)), 
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.2, color='steelblue')
            
            ax.set_xlabel(factor, fontsize=11, fontweight='bold')
            ax.set_ylabel('Mean Response', fontsize=11, fontweight='bold')
            ax.set_title(f'Main Effect: {factor}', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(levels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        
        for idx in range(num_factors, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout(pad=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_interaction_effects(self, save_path=None, max_factors=6):
        """Create interaction plot matrix"""
        factors_to_plot = self.factor_columns[:max_factors]
        num_factors = len(factors_to_plot)
        
        if num_factors < 2:
            return None
        
        fig, axes = plt.subplots(num_factors, num_factors, 
                                figsize=(1.8*num_factors, 1.8*num_factors))
        
        for i, factor1 in enumerate(factors_to_plot):
            for j, factor2 in enumerate(factors_to_plot):
                ax = axes[i, j]
                
                # Diagonal: main effects
                if i == j:
                    grouped = self.data.groupby(factor1)[self.response_column].mean()
                    levels = sorted(self.data[factor1].unique())
                    means = [grouped.loc[level] for level in levels]
                    
                    ax.plot(range(len(levels)), means, 'o-', linewidth=2, color='steelblue')
                    ax.set_xticks(range(len(levels)))
                    ax.set_xticklabels(levels, rotation=45, ha='right', fontsize=8)
                    
                    if j == 0:
                        ax.set_ylabel('Mean\nResponse', fontsize=9)
                    
                # Lower triangle: interaction plots
                elif i > j:
                    levels1 = sorted(self.data[factor1].unique())
                    levels2 = sorted(self.data[factor2].unique())
                    
                    for level2 in levels2:
                        subset = self.data[self.data[factor2] == level2]
                        grouped = subset.groupby(factor1)[self.response_column].mean()
                        means = [grouped.loc[level1] if level1 in grouped.index else np.nan 
                                for level1 in levels1]
                        ax.plot(range(len(levels1)), means, 'o-', linewidth=1.5, 
                               label=f'{factor2}={level2}', alpha=0.7)
                    
                    ax.set_xticks(range(len(levels1)))
                    ax.set_xticklabels(levels1, rotation=45, ha='right', fontsize=8)
                    
                    if j == 0:
                        ax.set_ylabel('Mean\nResponse', fontsize=9)
                    
                    if i == 1 and j == 0 and len(levels2) <= 5:
                        ax.legend(fontsize=7, loc='best')
                    
                # Upper triangle: just label the comparison
                else:
                    ax.text(0.5, 0.5, f'{factor1}\nvs\n{factor2}', 
                           ha='center', va='center', fontsize=9,
                           transform=ax.transAxes, style='italic')
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                ax.grid(True, alpha=0.3)
                
                if i == 0:
                    ax.set_title(factor2, fontsize=10, fontweight='bold')
                
                if j == 0 and i > 0:
                    ax.set_ylabel(factor1, fontsize=10, fontweight='bold', rotation=0, 
                                 ha='right', va='center', labelpad=20)
        
        plt.tight_layout(pad=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_residuals(self, predictions, residuals, save_path=None):
        """Create residual diagnostic plots"""
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        
        # Residuals vs Fitted
        axes[0, 0].scatter(predictions, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Fitted Values', fontsize=11)
        axes[0, 0].set_ylabel('Residuals', fontsize=11)
        axes[0, 0].set_title('Residuals vs Fitted', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q plot (This was just made to have inverted axes to match PRISM style)
        qq_data = scipy_stats.probplot(residuals, dist="norm")
        theoretical_quantiles = qq_data[0][0]  # Y-axis
        ordered_residuals = qq_data[0][1]      # X-axis
        slope = qq_data[1][0]
        intercept = qq_data[1][1]
        
        # Plot data points with swapped axes
        axes[0, 1].plot(ordered_residuals, theoretical_quantiles, 'o', markersize=5)
        
        # Reference line: Since normal relationship is y = slope*x + intercept
        # When we swap axes, the line becomes: theoretical = (actual - intercept) / slope
        # Or simplified: theoretical = (1/slope) * actual - (intercept/slope)
        x_line = np.array([ordered_residuals.min(), ordered_residuals.max()])
        y_line = (x_line - intercept) / slope
        axes[0, 1].plot(x_line, y_line, 'r-', linewidth=2)
        
        axes[0, 1].set_xlabel('Actual residual', fontsize=11)
        axes[0, 1].set_ylabel('Predicted residual', fontsize=11)
        axes[0, 1].set_title('Normal Q-Q Plot', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scale-Location
        standardized_resid = residuals / residuals.std()
        axes[1, 0].scatter(predictions, np.sqrt(np.abs(standardized_resid)), alpha=0.5)
        axes[1, 0].set_xlabel('Fitted Values', fontsize=11)
        axes[1, 0].set_ylabel('√|Standardized Residuals|', fontsize=11)
        axes[1, 0].set_title('Scale-Location', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Histogram
        axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Residuals', fontsize=11)
        axes[1, 1].set_ylabel('Frequency', fontsize=11)
        axes[1, 1].set_title('Histogram of Residuals', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout(pad=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# Results Exporter

class ResultsExporter:
    """Exports results to various formats"""
    
    def __init__(self):
        self.results = None
        self.main_effects = None
        
    def set_results(self, results, main_effects):
        """Set results to export"""
        self.results = results
        self.main_effects = main_effects
    
    def export_statistics_excel(self, filepath):
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

# GUI Application

class DoEAnalysisGUI:
    """Main GUI Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("DoE Data Analysis Tool v0.3.1")
        self.root.geometry("1100x900")  # Adjusted window size for smaller plots
        self.root.minsize(1000, 800)  # Minimum size adjusted for smaller plots
        
        # Initialize components
        self.handler = DataHandler()
        self.analyzer = DoEAnalyzer()
        self.plotter = DoEPlotter()
        self.exporter = ResultsExporter()
        
        # Data storage
        self.filepath = None
        self.results = None
        self.main_effects = None
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup GUI layout"""
        
        # Title
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill='x', padx=10, pady=5)

        # File selection
        file_frame = ttk.LabelFrame(self.root, text="1. Select Data File", padding=10)
        file_frame.pack(fill='x', padx=10, pady=5)
        
        self.file_label = ttk.Label(file_frame, text="No file selected", foreground='gray')
        self.file_label.pack(side='left', padx=5)
        
        browse_btn = ttk.Button(file_frame, text="Browse...", command=self.browse_file)
        browse_btn.pack(side='right', padx=5)
        
        # Configuration
        config_frame = ttk.LabelFrame(self.root, text="2. Configure Analysis", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)
        
        # Model Type row with info button
        ttk.Label(config_frame, text="Model Type:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        
        self.model_type_var = tk.StringVar(value='linear')
        model_combo = ttk.Combobox(config_frame, textvariable=self.model_type_var,
                                   values=['linear', 'interactions', 'purequadratic', 'quadratic'],
                                   state='readonly', width=25)
        model_combo.grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        # info button
        info_btn = ttk.Button(config_frame, text="?", width=2, command=self.show_model_guide)
        info_btn.grid(row=0, column=2, sticky='w', padx=(2, 0), pady=5)
        
        self.analyze_btn = ttk.Button(config_frame, text="Analyze Data", 
                                      command=self.analyze_data, state='disabled')
        self.analyze_btn.grid(row=0, column=3, padx=20, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor='w')
        status_bar.pack(fill='x', side='bottom')
        
        # Export buttons
        export_frame = ttk.LabelFrame(self.root, text="4. Export Results", padding=10)
        export_frame.pack(fill='x', side='bottom', padx=10, pady=5)
        
        self.export_stats_btn = ttk.Button(export_frame, text="Export Statistics (.xlxs)",
                                          command=self.export_statistics, state='disabled')
        self.export_stats_btn.pack(side='left', padx=5)
        
        self.export_plots_btn = ttk.Button(export_frame, text="Export Plots (.png)",
                                          command=self.export_plots, state='disabled')
        self.export_plots_btn.pack(side='left', padx=5)
        
        # Results notebook
        results_frame = ttk.LabelFrame(self.root, text="3. Results", padding=5)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # It was needed to create a canvas with scrollbar for results to prevent expanding too much
        results_container = ttk.Frame(results_frame)
        results_container.pack(fill='both', expand=True)
        
        self.notebook = ttk.Notebook(results_container)
        self.notebook.pack(fill='both', expand=True)
        
        # Tab 1: Statistics
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="Statistics")
        
        self.stats_text = scrolledtext.ScrolledText(self.stats_frame, wrap=tk.WORD, font=('Courier', 14))
        self.stats_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tab 2: Main Effects
        main_effects_container = ttk.Frame(self.notebook)
        self.notebook.add(main_effects_container, text="Main Effects")
        
        # Tooltip button for main effects
        me_header = ttk.Frame(main_effects_container)
        me_header.pack(fill='x', padx=5, pady=2)
        ttk.Label(me_header, text="Main Effects Plot", font=('TkDefaultFont', 10, 'bold')).pack(side='left')
        ttk.Button(me_header, text="ℹ️ How to Read", width=15,
                  command=lambda: self.show_tooltip("main_effects")).pack(side='right', padx=5)
        
        # Scrollable frame for main effects plot
        self.main_effects_frame = self.create_scrollable_frame(main_effects_container)
        
        # Tab 3: Interactions
        interactions_container = ttk.Frame(self.notebook)
        self.notebook.add(interactions_container, text="Interactions")
        
        # Tooltip button for interactions
        int_header = ttk.Frame(interactions_container)
        int_header.pack(fill='x', padx=5, pady=2)
        ttk.Label(int_header, text="Interaction Effects Plot", font=('TkDefaultFont', 10, 'bold')).pack(side='left')
        ttk.Button(int_header, text="ℹ️ How to Read", width=15,
                  command=lambda: self.show_tooltip("interactions")).pack(side='right', padx=5)
        
        # Scrollable frame for interactions plot
        self.interactions_frame = self.create_scrollable_frame(interactions_container)
        
        # Tab 4: Residuals
        residuals_container = ttk.Frame(self.notebook)
        self.notebook.add(residuals_container, text="Residuals")
        
        # Tooltip button for residuals
        res_header = ttk.Frame(residuals_container)
        res_header.pack(fill='x', padx=5, pady=2)
        ttk.Label(res_header, text="Residuals Diagnostic Plot", font=('TkDefaultFont', 10, 'bold')).pack(side='left')
        ttk.Button(res_header, text="ℹ️ How to Read", width=15,
                  command=lambda: self.show_tooltip("residuals")).pack(side='right', padx=5)
        
        # Scrollable frame for residuals plot
        self.residuals_frame = self.create_scrollable_frame(residuals_container)
        
        # Tab 5: Recommendations
        self.recommendations_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.recommendations_frame, text="Recommendations")
        
        self.recommendations_text = scrolledtext.ScrolledText(self.recommendations_frame, 
                                                             wrap=tk.WORD, font=('Courier', 14))
        self.recommendations_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def create_scrollable_frame(self, parent):
        """Create a scrollable frame for plot display with comprehensive mousewheel support"""
        import platform
        
        # Canvas with scrollbars and white background
        canvas = tk.Canvas(parent, bg='white', highlightthickness=0)
        v_scrollbar = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(parent, orient='horizontal', command=canvas.xview)
        
        # Create frame inside canvas
        scrollable_frame = ttk.Frame(canvas)
        
        # Store canvas reference in frame
        scrollable_frame._scroll_canvas = canvas
        
        # Canvas config
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        scrollable_frame.bind("<Configure>", on_frame_configure)

        canvas.create_window((0, 0), window=scrollable_frame, anchor="center")
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Configure smooth scrolling
        canvas.configure(yscrollincrement='10')  # Scroll with 10 pixels at a time for smooth but faster scrolling
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')
        canvas.pack(side='left', fill='both', expand=True)
        
        # MOUSEWHEEL SOLUTION (Mousewheel wasn't being detected, this is the patch)
        # Detect platform
        system = platform.system()
        
        # Mousewheel handler
        def on_mousewheel(event):
            # Use smaller scroll increments for smooth scrolling
            if system == 'Windows':
                # Windows: delta is typically ±120, divide further for smoothness
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            elif system == 'Darwin':  # macOS
                # macOS: smaller delta values, use directly but scale down
                canvas.yview_scroll(int(-1*event.delta), "units")
            else:  # Linux/Unix
                if event.num == 4:
                    canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    canvas.yview_scroll(1, "units")
        
        # Recursive binding function to bind all widgets
        def bind_tree(widget):
            """Recursively bind mousewheel to widget and all its children"""
            if system == 'Windows' or system == 'Darwin':
                widget.bind('<MouseWheel>', on_mousewheel, add='+')
            else:  # Linux
                widget.bind('<Button-4>', on_mousewheel, add='+')
                widget.bind('<Button-5>', on_mousewheel, add='+')
            
            # Recursively bind to all children
            for child in widget.winfo_children():
                bind_tree(child)
        
        # Bind to canvas
        if system == 'Windows' or system == 'Darwin':
            canvas.bind('<MouseWheel>', on_mousewheel)
        else:  # Linux
            canvas.bind('<Button-4>', on_mousewheel)
            canvas.bind('<Button-5>', on_mousewheel)
        
        # Bind to scrollable frame
        bind_tree(scrollable_frame)
        
        # Store binding function for later use when new widgets are added
        scrollable_frame._bind_mousewheel = bind_tree

        def on_canvas_configure(event):
            # Center the window
            canvas_width = event.width
            frame_width = scrollable_frame.winfo_reqwidth()
            x_position = max(0, (canvas_width - frame_width) // 2)
            if canvas.find_withtag("all"):
                canvas.coords(canvas.find_withtag("all")[0], x_position, 0)
        
        canvas.bind("<Configure>", on_canvas_configure)
        
        return scrollable_frame
    
    def show_tooltip(self, plot_type):
        """Show interpretation guidance for different plot types"""
        tooltips = {
            "main_effects": (
                "How to Read Main Effects Plot\n\n"
                "• Steeper slopes = Stronger factor influence on response\n"
                "• Upward slope = Increasing factor increases response\n"
                "• Downward slope = Increasing factor decreases response\n"
                "• Flat line = Factor has little/no effect\n"
                "• Compare slopes to identify most important factors\n\n"
                "Example: If Salt has a steep upward slope and Buffer pH is flat,\n"
                "Salt is more important for your response."
            ),
            "interactions": (
                "How to Read Interaction Plot\n\n"
                "• Parallel lines = NO interaction (factors work independently)\n"
                "• Non-parallel lines = Interaction present (factors affect each other)\n"
                "• Crossing lines = Strong interaction (optimal level depends on other factor)\n\n"
                "Example: If glycerol lines cross for different salt levels,\n"
                "the best glycerol amount depends on which salt level you use."
            ),
            "residuals": (
                "How to Read Residuals Plot\n\n"
                "• Random scatter around zero = Good model fit ✓\n"
                "• Pattern/curve = Model may be inappropriate ✗\n"
                "• Funnel shape = Variance increases with response ✗\n"
                "• Outliers far from others = Check those data points\n\n"
                "If residuals look bad, your statistical conclusions may be unreliable.\n"
                "Consider trying a different model type or checking for data errors."
            )
        }
        
        messagebox.showinfo("Plot Interpretation Guide", tooltips.get(plot_type, "No guide available"))
    
    def show_model_guide(self):
        """Show model selection guide in a popup window"""
        guide_text = (
            "Linear: Simplest model, main effects only (A + B + C)\n"
            "  • Use when: You want to understand which factors matter\n"
            "  • Assumes: Each factor works independently\n"
            "  • Example: Salt effect + Buffer effect\n\n"
            
            "Interactions: Use if factors work together (A + B + A×B)\n"
            "  • Use when: Factors may influence each other\n"
            "  • Detects: Combined effects of multiple factors\n"
            "  • Example: Salt effect depends on Buffer level\n\n"
            
            "Pure Quadratic: For curved responses (A + B + A² + B²)\n"
            "  • Use when: Response has a peak or valley\n"
            "  • Assumes: No interactions, but curved relationships\n"
            "  • Example: Optimum temperature between two extremes\n\n"
            
            "Quadratic: Full model (A + B + A×B + A² + B²)\n"
            "  • Use when: You expect both curvature and interactions\n"
            "  • Most complex: Requires more data points\n"
            "  • Example: Complex biological systems with multiple effects"
        )
        
        messagebox.showinfo("Model Selection Guide", guide_text)
    
    def browse_file(self):
        """Open file dialog"""
        filepath = filedialog.askopenfilename(
            title="Select DoE Data File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        
        if filepath:
            self.filepath = filepath
            # Keep label white/colored when file selected
            self.file_label.config(text=os.path.basename(filepath), foreground='white')
            self.status_var.set(f"Loaded: {os.path.basename(filepath)}")
            
            try:
                self.handler.load_excel(filepath)
                
                # Check if "Response" column exists
                if 'Response' not in self.handler.data.columns:
                    messagebox.showerror("Error", 
                                       "Excel file must have a column named 'Response'\n\n"
                                       "Please rename your response column to 'Response' and try again.")
                    self.status_var.set("Error: No 'Response' column found")
                    return
                
                # Enable analyze button
                self.analyze_btn.config(state='normal')
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
                self.status_var.set("Error loading file")
    
    def analyze_data(self):
        """Perform DoE analysis"""
        if not self.filepath:
            messagebox.showwarning("Warning", "Please select a data file first.")
            return
        
        self.status_var.set("Analyzing...")
        self.root.update()
        
        try:
            # Always use "Response" column in the xlxs file
            response_col = "Response"
            model_type = self.model_type_var.get()
            # Detect which columns are factors vs response
            self.handler.detect_columns(response_col)
            clean_data = self.handler.preprocess_data()
            
            # Pass cleaned data to analyzer
            self.analyzer.set_data(
                data=clean_data,
                factor_columns=self.handler.factor_columns,
                categorical_factors=self.handler.categorical_factors,
                numeric_factors=self.handler.numeric_factors,
                response_column=self.handler.response_column
            )
            
            # Fit regression model then calculations are made
            self.results = self.analyzer.fit_model(model_type)
            self.main_effects = self.analyzer.calculate_main_effects()
            
            self.display_statistics()
            self.display_plots()
            self.display_recommendations()
            
            self.export_stats_btn.config(state='normal')
            self.export_plots_btn.config(state='normal')
            
            self.status_var.set(f"Analysis complete! R² = {self.results['model_stats']['R-squared']:.4f}")
            
            messagebox.showinfo("Success", 
                              f"Analysis completed successfully!\n\n"
                              f"Model: {model_type}\n"
                              f"Observations: {self.results['model_stats']['Observations']}\n"
                              f"R-squared: {self.results['model_stats']['R-squared']:.4f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")
            self.status_var.set("Analysis failed")
            import traceback
            traceback.print_exc()
    
    def display_statistics(self):
        """Display statistical results with recommendations and warnings"""
        self.stats_text.delete('1.0', tk.END)
        
        self.stats_text.insert(tk.END, "="*80 + "\n")
        self.stats_text.insert(tk.END, "DOE ANALYSIS RESULTS\n")
        self.stats_text.insert(tk.END, "="*80 + "\n\n")
        
        # RED FLAGS SECTION - Show warnings first if model quality is poor
        r_squared = self.results['model_stats']['R-squared']
        n_obs = self.results['model_stats']['Observations']
        sig_factors = self.analyzer.get_significant_factors()
        
        warnings = []
        if r_squared < 0.5:
            warnings.append(f"⚠️  LOW R² ({r_squared:.3f}): Model explains only {r_squared*100:.1f}% of variance; inspect residuals for unexplained structure.")
        elif r_squared < 0.7:
            warnings.append(f"⚠️  MODERATE R² ({r_squared:.3f}): Model is acceptable but could be improved.")
        
        if len(sig_factors) == 0:
            warnings.append("⚠️  NO SIGNIFICANT FACTORS: No factors with p < 0.05 found. Consider reviewing experimental design.")
        
        if n_obs < 20:
            warnings.append(f"⚠️  SMALL SAMPLE SIZE: Only {n_obs} observations. Consider adding replicates for more robust results.")
        
        if warnings:
            self.stats_text.insert(tk.END, "🚩 RED FLAGS / WARNINGS\n")
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
            self.stats_text.insert(tk.END, f"⚡ SIGNIFICANT INTERACTIONS DETECTED: {len(interaction_factors)}\n")
            self.stats_text.insert(tk.END, "-"*80 + "\n")
            for interaction in interaction_factors:
                coef = self.results['coefficients'].loc[interaction, 'Coefficient']
                pval = self.results['coefficients'].loc[interaction, 'p-value']
                self.stats_text.insert(tk.END, f"  {interaction:<40} coef={coef:>10.4f}  p={pval:.2e}\n")
            self.stats_text.insert(tk.END, "\n⚠️  Interactions mean optimal settings depend on factor combinations!\n")
        
        self.stats_text.insert(tk.END, "\n" + "="*80 + "\n")
    
    def display_plots(self):
        """Display all plots"""
        self.plotter.set_data(
            self.handler.clean_data,
            self.handler.factor_columns,
            self.handler.response_column
        )
        
        self.display_main_effects_plot()
        self.display_interaction_plot()
        self.display_residuals_plot()
    
    def display_main_effects_plot(self):
        """Display main effects plot"""
        for widget in self.main_effects_frame.winfo_children():
            widget.destroy()
        
        fig = self.plotter.plot_main_effects()
        canvas = FigureCanvasTkAgg(fig, master=self.main_effects_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill='both', expand=True)
        
        # Bind mousewheel to the matplotlib canvas widget
        if hasattr(self.main_effects_frame, '_bind_mousewheel'):
            self.main_effects_frame._bind_mousewheel(canvas_widget)
        
        # Reset scroll position to top
        if hasattr(self.main_effects_frame, '_scroll_canvas'):
            self.main_effects_frame._scroll_canvas.update_idletasks()
            self.main_effects_frame._scroll_canvas.yview_moveto(0)
        
        plt.close(fig)
    
    def display_interaction_plot(self):
        """Display interaction plot"""
        for widget in self.interactions_frame.winfo_children():
            widget.destroy()
        
        fig = self.plotter.plot_interaction_effects()
        if fig:
            canvas = FigureCanvasTkAgg(fig, master=self.interactions_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill='both', expand=True)
            
            # Bind mousewheel to the matplotlib canvas widget
            if hasattr(self.interactions_frame, '_bind_mousewheel'):
                self.interactions_frame._bind_mousewheel(canvas_widget)
            
            # Reset scroll position to top
            if hasattr(self.interactions_frame, '_scroll_canvas'):
                self.interactions_frame._scroll_canvas.update_idletasks()
                self.interactions_frame._scroll_canvas.yview_moveto(0)
            
            plt.close(fig)
    
    def display_residuals_plot(self):
        """Display residuals plot"""
        for widget in self.residuals_frame.winfo_children():
            widget.destroy()
        
        fig = self.plotter.plot_residuals(
            self.results['predictions'],
            self.results['residuals']
        )
        canvas = FigureCanvasTkAgg(fig, master=self.residuals_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill='both', expand=True)
        
        # Bind mousewheel to the matplotlib canvas widget
        if hasattr(self.residuals_frame, '_bind_mousewheel'):
            self.residuals_frame._bind_mousewheel(canvas_widget)
        
        # Reset scroll position to top
        if hasattr(self.residuals_frame, '_scroll_canvas'):
            self.residuals_frame._scroll_canvas.update_idletasks()
            self.residuals_frame._scroll_canvas.yview_moveto(0)
        
        plt.close(fig)
    
    def display_recommendations(self):
        """Display recommendations and optimal conditions"""
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
        
        # Find optimal condition from data (row with highest response)
        clean_data = self.handler.clean_data
        max_idx = clean_data[self.handler.response_column].idxmax()
        optimal_response = clean_data.loc[max_idx, self.handler.response_column]
        
        self.recommendations_text.insert(tk.END, "="*80 + "\n")
        self.recommendations_text.insert(tk.END, "BEST OBSERVED CONDITION (from your data)\n")
        self.recommendations_text.insert(tk.END, "-"*80 + "\n")
        self.recommendations_text.insert(tk.END, f"Response Value: {optimal_response:.2f}\n\n")
        
        for factor in self.handler.factor_columns:
            value = clean_data.loc[max_idx, factor]
            self.recommendations_text.insert(tk.END, f"  • {factor:<30}: {value}\n")
        
        # Predicted optimal based on model
        self.recommendations_text.insert(tk.END, "\n" + "="*80 + "\n")
        self.recommendations_text.insert(tk.END, "MODEL-PREDICTED OPTIMAL DIRECTION\n")
        self.recommendations_text.insert(tk.END, "-"*80 + "\n")
        
        if len(sig_factors) > 0:
            # Get only main effect factors (not interactions)
            main_sig_factors = [f for f in sig_factors if ':' not in f]
            
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
                    self.recommendations_text.insert(tk.END, f"  • {clean_factor:<30}: {direction}  (effect: {coef:+.4f})\n")
            
            interaction_factors = [f for f in sig_factors if ':' in f]
            if interaction_factors:
                self.recommendations_text.insert(tk.END, f"\nWARNING: {len(interaction_factors)} interaction(s) detected!\n")
                self.recommendations_text.insert(tk.END, "Optimal levels depend on factor combinations - see Interactions plot.\n")
        else:
            self.recommendations_text.insert(tk.END, "No significant factors found. Consider:\n")
            self.recommendations_text.insert(tk.END, "  • Testing wider factor ranges\n")
            self.recommendations_text.insert(tk.END, "  • Adding more factors to the experiment\n")
            self.recommendations_text.insert(tk.END, "  • Checking measurement accuracy\n")
        
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
    
    def export_statistics(self):
        """Export statistics to Excel"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile=f"DoE_Statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
        
        if filepath:
            try:
                self.exporter.set_results(self.results, self.main_effects)
                self.exporter.export_statistics_excel(filepath)
                messagebox.showinfo("Success", f"Statistics exported to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def export_plots(self):
        """Export plots as PNG"""
        directory = filedialog.askdirectory(title="Select directory to save plots")
        
        if directory:
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                fig1 = self.plotter.plot_main_effects()
                fig1.savefig(os.path.join(directory, f'main_effects_{timestamp}.png'), 
                            dpi=300, bbox_inches='tight')
                plt.close(fig1)
                
                fig2 = self.plotter.plot_interaction_effects()
                if fig2:
                    fig2.savefig(os.path.join(directory, f'interactions_{timestamp}.png'),
                                dpi=300, bbox_inches='tight')
                    plt.close(fig2)
                
                fig3 = self.plotter.plot_residuals(
                    self.results['predictions'],
                    self.results['residuals']
                )
                fig3.savefig(os.path.join(directory, f'residuals_{timestamp}.png'),
                            dpi=300, bbox_inches='tight')
                plt.close(fig3)
                
                messagebox.showinfo("Success", f"Plots exported to:\n{directory}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")


# Main

def main():
    """Main entry point"""
    root = tk.Tk()
    app = DoEAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
