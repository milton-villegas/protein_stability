#!/usr/bin/env python3
"""
DoE Data Analysis Tool - v0.4.1
Statistical analysis GUI for Design of Experiments data
Replicates MATLAB fitlm (EDA CAPKIN MATLAB code), main effects, and interaction plots
Includes Bayesian Optimization for intelligent experiment suggestions

v0.4.1 Changes:
- Fixed pH handling in Bayesian Optimization
- pH now treated as ordered categorical parameter (only suggests tested pH values)
- Prevents extrapolation to untested pH values
- Removed pH rounding logic (no longer needed)

Milton F. Villegas
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

# Bayesian Optimization imports
try:
    from ax.service.ax_client import AxClient, ObjectiveProperties
    from ax.plot.contour import plot_contour
    from ax.plot.feature_importances import plot_feature_importance_by_feature
    AX_AVAILABLE = True
except ImportError:
    AX_AVAILABLE = False
    print("Warning: Ax not available. Install with: pip install ax-platform")

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

# Suppress verbose Ax logging
import logging
logging.getLogger('ax').setLevel(logging.WARNING)

# Import consolidated modules
from core.doe_analyzer import DoEAnalyzer
from core.optimizer import BayesianOptimizer
from utils.constants import METADATA_COLUMNS
from utils.sanitization import smart_factor_match


# Data Handler

class DataHandler:
    """Data loading and preprocessing"""

    def __init__(self):
        self.data = None
        self.clean_data = None
        self.factor_columns = []
        self.categorical_factors = []
        self.numeric_factors = []
        self.response_column = None
        self.stock_concentrations = {}  # Store stock concentrations from metadata
        
    def load_excel(self, filepath: str):
        """Load data from Excel file"""
        self.data = pd.read_excel(filepath)
        
        # Try to load stock concentrations from metadata sheet
        self._load_stock_concentrations(filepath)
    
    def _load_stock_concentrations(self, filepath: str):
        """Load stock concentrations from Stock_Concentrations sheet if it exists"""
        try:
            # Try to read the Stock_Concentrations sheet
            stock_df = pd.read_excel(filepath, sheet_name="Stock_Concentrations")
            
            # Parse the stock concentrations with intelligent matching
            self.stock_concentrations = {}
            for _, row in stock_df.iterrows():
                factor_name = str(row['Factor Name']).strip()
                stock_value_raw = row['Stock Value']

                # Skip rows with None or NaN values
                if pd.isna(stock_value_raw):
                    continue

                stock_value = float(stock_value_raw)

                # Smart matching algorithm - convert display name to internal key
                internal_name = smart_factor_match(factor_name)

                if internal_name:
                    self.stock_concentrations[internal_name] = stock_value

            print(f"✓ Loaded stock concentrations from metadata: {self.stock_concentrations}")

        except Exception as e:
            # Sheet doesn't exist or error reading - that's okay, will use dialog
            print(f"Note: Stock concentrations sheet not found or error reading ({e})")
            self.stock_concentrations = {}
    
    def get_stock_concentrations(self) -> dict:
        """Get stock concentrations (either from metadata or empty dict)"""
        return self.stock_concentrations.copy()
    
    def detect_columns(self, response_column: str):
        """Detect factor types, exclude metadata"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        self.response_column = response_column
        
        # All columns except response and metadata are factors
        self.factor_columns = [
            col for col in self.data.columns
            if col != response_column and col not in METADATA_COLUMNS
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
        columns_to_drop = [col for col in METADATA_COLUMNS if col in self.clean_data.columns]
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


# Plotter class for visualizations

class DoEPlotter:
    """Plotting functions for DoE"""

    # Professional colorblind-safe palette (Okabe-Ito)
    COLORS = {
        'primary': '#0173B2',      # Blue
        'secondary': '#DE8F05',    # Orange
        'accent': '#CC78BC',       # Reddish Purple
        'warning': '#D55E00',      # Vermillion
        'success': '#029E73',      # Bluish Green
        'palette': ['#0173B2', '#DE8F05', '#029E73', '#D55E00',
                   '#56B4E9', '#CC78BC', '#ECE133', '#000000']
    }

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
            
            ax.plot(range(len(levels)), means, 'o-', linewidth=2, markersize=8, color=self.COLORS['primary'])
            # Shaded region shows ± 1 std dev
            ax.fill_between(range(len(levels)),
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.2, color=self.COLORS['primary'])
            
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

                    ax.plot(range(len(levels)), means, 'o-', linewidth=2, color=self.COLORS['primary'])
                    ax.set_xticks(range(len(levels)))
                    ax.set_xticklabels(levels, rotation=45, ha='right', fontsize=8)
                    
                    if j == 0:
                        ax.set_ylabel('Mean\nResponse', fontsize=9)
                    
                # Lower triangle: interaction plots
                elif i > j:
                    levels1 = sorted(self.data[factor1].unique())
                    levels2 = sorted(self.data[factor2].unique())

                    for idx, level2 in enumerate(levels2):
                        subset = self.data[self.data[factor2] == level2]
                        grouped = subset.groupby(factor1)[self.response_column].mean()
                        means = [grouped.loc[level1] if level1 in grouped.index else np.nan
                                for level1 in levels1]
                        color = self.COLORS['palette'][idx % len(self.COLORS['palette'])]
                        ax.plot(range(len(levels1)), means, 'o-', linewidth=1.5,
                               label=f'{factor2}={level2}', alpha=0.85, color=color)
                    
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
        axes[0, 0].scatter(predictions, residuals, alpha=0.6, color=self.COLORS['primary'], edgecolors='white', linewidth=0.5)
        axes[0, 0].axhline(y=0, color=self.COLORS['warning'], linestyle='--', linewidth=2)
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
        axes[0, 1].plot(ordered_residuals, theoretical_quantiles, 'o', markersize=6,
                       color=self.COLORS['primary'], alpha=0.7, markeredgecolor='white', markeredgewidth=0.5)

        # Reference line: Since normal relationship is y = slope*x + intercept
        # When we swap axes, the line becomes: theoretical = (actual - intercept) / slope
        # Or simplified: theoretical = (1/slope) * actual - (intercept/slope)
        x_line = np.array([ordered_residuals.min(), ordered_residuals.max()])
        y_line = (x_line - intercept) / slope
        axes[0, 1].plot(x_line, y_line, '-', linewidth=2, color=self.COLORS['warning'])
        
        axes[0, 1].set_xlabel('Actual residual', fontsize=11)
        axes[0, 1].set_ylabel('Predicted residual', fontsize=11)
        axes[0, 1].set_title('Normal Q-Q Plot', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scale-Location
        standardized_resid = residuals / residuals.std()
        axes[1, 0].scatter(predictions, np.sqrt(np.abs(standardized_resid)), alpha=0.6,
                          color=self.COLORS['primary'], edgecolors='white', linewidth=0.5)
        axes[1, 0].set_xlabel('Fitted Values', fontsize=11)
        axes[1, 0].set_ylabel('√|Standardized Residuals|', fontsize=11)
        axes[1, 0].set_title('Scale-Location', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Histogram
        axes[1, 1].hist(residuals, bins=30, color=self.COLORS['primary'],
                       edgecolor='white', alpha=0.8, linewidth=1.2)
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

class AnalysisTab(ttk.Frame):
    """Main GUI Application - Analysis Tab"""

    def __init__(self, parent, project, main_window):
        super().__init__(parent)
        self.project = project
        self.main_window = main_window

        # Initialize components
        self.handler = DataHandler()
        self.analyzer = DoEAnalyzer()
        self.plotter = DoEPlotter()
        self.exporter = ResultsExporter()
        self.optimizer = BayesianOptimizer() if AX_AVAILABLE else None

        # Data storage
        self.filepath = None
        self.results = None
        self.main_effects = None

        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup GUI layout"""
        
        # Title
        title_frame = ttk.Frame(self)
        title_frame.pack(fill='x', padx=10, pady=5)

        # File selection
        file_frame = ttk.LabelFrame(self, text="1. Select Data File", padding=10)
        file_frame.pack(fill='x', padx=10, pady=5)
        
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side='left', padx=5)
        
        browse_btn = ttk.Button(file_frame, text="Browse...", command=self.browse_file)
        browse_btn.pack(side='right', padx=5)
        
        # Configuration
        config_frame = ttk.LabelFrame(self, text="2. Configure Analysis", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)
        
        # Model Type row with info button - AUTO SELECTION
        ttk.Label(config_frame, text="Model Selection:").grid(row=0, column=0, sticky='w', padx=5, pady=5)

        # Label indicating automatic selection
        auto_label = ttk.Label(config_frame, text="✓ Automatic (compares all 5 models)",
                              foreground='#029E73', font=('TkDefaultFont', 9, 'bold'))
        auto_label.grid(row=0, column=1, sticky='w', padx=5, pady=5)

        # info button
        info_btn = ttk.Button(config_frame, text="?", width=2, command=self.show_model_guide)
        info_btn.grid(row=0, column=2, sticky='w', padx=(2, 0), pady=5)
        
        self.analyze_btn = ttk.Button(config_frame, text="Analyze Data", 
                                      command=self.analyze_data, state='disabled')
        self.analyze_btn.grid(row=0, column=3, padx=20, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor='w')
        status_bar.pack(fill='x', side='bottom')
        
        # Export buttons
        export_frame = ttk.LabelFrame(self, text="4. Export Results", padding=10)
        export_frame.pack(fill='x', side='bottom', padx=10, pady=5)
        
        self.export_stats_btn = ttk.Button(export_frame, text="Export Statistics (.xlxs)",
                                          command=self.export_statistics, state='disabled')
        self.export_stats_btn.pack(side='left', padx=5)
        
        self.export_plots_btn = ttk.Button(export_frame, text="Export Plots (.png)",
                                          command=self.export_plots, state='disabled')
        self.export_plots_btn.pack(side='left', padx=5)
        
        # Results notebook
        results_frame = ttk.LabelFrame(self, text="3. Results", padding=5)
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

        # Add button frame at top for BO batch export
        if AX_AVAILABLE:
            button_frame = ttk.Frame(self.recommendations_frame)
            button_frame.pack(fill='x', padx=5, pady=5)

            self.export_bo_button = ttk.Button(button_frame, text="📤 Export BO Batch to Files",
                                              command=self.export_bo_batch, state='disabled')
            self.export_bo_button.pack(side='left', padx=5)

            ttk.Label(button_frame, text="(Available after analysis with BO suggestions)",
                     font=('TkDefaultFont', 9)).pack(side='left', padx=5)

        self.recommendations_text = scrolledtext.ScrolledText(self.recommendations_frame,
                                                             wrap=tk.WORD, font=('Courier', 14))
        self.recommendations_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tab 6: Optimization Details (Bayesian Optimization plot)
        if AX_AVAILABLE:
            optimization_container = ttk.Frame(self.notebook)
            self.notebook.add(optimization_container, text="Optimization Details")

            # Tooltip button for optimization and export button
            opt_header = ttk.Frame(optimization_container)
            opt_header.pack(fill='x', padx=5, pady=2)
            ttk.Label(opt_header, text="Bayesian Optimization Analysis",
                     font=('TkDefaultFont', 10, 'bold')).pack(side='left')

            # Export BO Plots button
            self.export_bo_plots_button = ttk.Button(opt_header, text="📊 Export BO Plots",
                                                     command=self.export_bo_plots_gui, state='disabled')
            self.export_bo_plots_button.pack(side='right', padx=5)

            ttk.Button(opt_header, text="ℹ️ How to Read", width=15,
                      command=lambda: self.show_tooltip("optimization")).pack(side='right', padx=5)

            # Scrollable frame for optimization plot
            self.optimization_frame = self.create_scrollable_frame(optimization_container)
    
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
            ),
            "optimization": (
                "How to Read Optimization Details\n\n"
                "This plot shows where Bayesian Optimization predicts you should explore:\n\n"
                "• Hot colors (yellow/green) = Predicted high response regions\n"
                "• Cool colors (blue/purple) = Predicted low response regions\n"
                "• Red dots = Your existing experiments\n"
                "• Gaps between dots = Unexplored regions to test\n\n"
                "The algorithm learns from your data to suggest intelligent next experiments\n"
                "that balance exploring new areas with exploiting promising regions."
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
            self.file_label.config(text=os.path.basename(filepath))
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
        self.update()
        
        try:
            # Always use "Response" column in the xlxs file
            response_col = "Response"

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

            # AUTOMATIC MODEL SELECTION - Compare all 5 models
            self.status_var.set("Comparing all models...")
            self.update()

            self.model_comparison = self.analyzer.compare_all_models()
            self.model_selection = self.analyzer.select_best_model(self.model_comparison)

            # Use the recommended model for detailed analysis
            recommended_model = self.model_selection['recommended_model']

            if recommended_model is None:
                raise ValueError("No models could be fitted successfully. Check your data.")

            self.status_var.set(f"Using recommended model: {recommended_model}...")
            self.update()

            # Fit the recommended model for detailed analysis
            self.results = self.analyzer.fit_model(recommended_model)
            self.main_effects = self.analyzer.calculate_main_effects()

            self.display_statistics()
            self.display_plots()
            self.display_recommendations()

            # Display optimization plot if available
            if AX_AVAILABLE and self.optimizer:
                self.display_optimization_plot()

            self.export_stats_btn.config(state='normal')
            self.export_plots_btn.config(state='normal')

            # Show completion status with recommended model
            recommended_model_name = self.analyzer.MODEL_TYPES[recommended_model]
            self.status_var.set(f"Analysis complete! Model: {recommended_model_name} | R² = {self.results['model_stats']['R-squared']:.4f}")

            messagebox.showinfo("Success",
                              f"Analysis completed successfully!\n\n"
                              f"Recommended Model: {recommended_model_name}\n"
                              f"Observations: {self.results['model_stats']['Observations']}\n"
                              f"R-squared: {self.results['model_stats']['R-squared']:.4f}\n"
                              f"Adjusted R-squared: {self.results['model_stats']['Adjusted R-squared']:.4f}\n\n"
                              f"All 5 models were compared. See Statistics tab for details.")

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

        # MODEL COMPARISON SECTION - Show all models and recommendation
        if hasattr(self, 'model_comparison') and self.model_comparison:
            self.stats_text.insert(tk.END, "📊 AUTOMATIC MODEL SELECTION\n")
            self.stats_text.insert(tk.END, "-"*80 + "\n\n")

            # Display comparison table
            comparison_table = self.model_comparison['comparison_table']

            if comparison_table is not None and not comparison_table.empty:
                # Format the comparison table for display
                self.stats_text.insert(tk.END, "Model Comparison (all 5 models fitted):\n\n")

                # Create formatted header
                header = f"{'Model':<30} {'Adj R²':>10} {'BIC':>10} {'RMSE':>10} {'DF':>6}  {'Recommend':>10}\n"
                self.stats_text.insert(tk.END, header)
                self.stats_text.insert(tk.END, "-"*80 + "\n")

                # Get recommendation
                recommendation = self.model_selection
                recommended_model = recommendation['recommended_model']

                # Display each model
                model_order = ['mean', 'linear', 'interactions', 'quadratic', 'reduced']
                for model_type in model_order:
                    if model_type in self.model_comparison['models']:
                        stats = self.model_comparison['models'][model_type]
                        model_name = stats['Model Type']

                        # Mark recommended model with checkmark
                        if model_type == recommended_model:
                            marker = "✓ BEST"
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
                self.stats_text.insert(tk.END, f"🎯 RECOMMENDED: {self.analyzer.MODEL_TYPES[recommended_model]}\n")
                self.stats_text.insert(tk.END, f"   Reason: {recommendation['reason']}\n")

                # Show note about model selection criteria
                self.stats_text.insert(tk.END, "\n💡 Selection criteria: Adjusted R² (60%), BIC (30%), Parsimony (10%)\n")
                self.stats_text.insert(tk.END, "   Higher Adj R² is better | Lower BIC is better | Simpler models preferred\n")

            self.stats_text.insert(tk.END, "\n" + "="*80 + "\n\n")

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
        
        # Add Bayesian Optimization suggestions if available
        if AX_AVAILABLE and self.optimizer:
            self.recommendations_text.insert(tk.END, "\n" + "="*80 + "\n")
            self.recommendations_text.insert(tk.END, "BAYESIAN OPTIMIZATION SUGGESTIONS\n")
            self.recommendations_text.insert(tk.END, "="*80 + "\n")
            self.recommendations_text.insert(tk.END, "Intelligently suggested next experiments (balancing exploration & exploitation):\n\n")
            
            try:
                # Initialize optimizer with current data
                self.optimizer.set_data(
                    data=self.handler.clean_data,
                    factor_columns=self.handler.factor_columns,
                    categorical_factors=self.handler.categorical_factors,
                    numeric_factors=self.handler.numeric_factors,
                    response_column=self.handler.response_column
                )
                self.optimizer.initialize_optimizer(minimize=False)  # Assume maximize response
                
                # Get suggestions
                suggestions = self.optimizer.get_next_suggestions(n=5)
                
                for i, suggestion in enumerate(suggestions, 1):
                    self.recommendations_text.insert(tk.END, f"Suggested Experiment #{i}:\n")
                    for factor, value in suggestion.items():
                        if isinstance(value, float):
                            self.recommendations_text.insert(tk.END, f"  • {factor:<30}: {value:.4f}\n")
                        else:
                            self.recommendations_text.insert(tk.END, f"  • {factor:<30}: {value}\n")
                    self.recommendations_text.insert(tk.END, "\n")
                
                self.recommendations_text.insert(tk.END, 
                    "💡 TIP: These suggestions use machine learning to predict where to test next.\n"
                    "   View 'Optimization Details' tab for visualization of predicted response surface.\n"
                )
                
                # Enable export button after successful BO initialization
                if hasattr(self, 'export_bo_button'):
                    self.export_bo_button.config(state='normal')
                if hasattr(self, 'export_bo_plots_button'):
                    self.export_bo_plots_button.config(state='normal')
                
            except Exception as e:
                self.recommendations_text.insert(tk.END, 
                    f"Could not generate BO suggestions: {str(e)}\n"
                    "This may require more data points or only numeric factors.\n"
                )
                # Disable export button if BO failed
                if hasattr(self, 'export_bo_button'):
                    self.export_bo_button.config(state='disabled')
                if hasattr(self, 'export_bo_plots_button'):
                    self.export_bo_plots_button.config(state='disabled')
            
            self.recommendations_text.insert(tk.END, "\n" + "="*80 + "\n")
    
    def display_optimization_plot(self):
        """Display Bayesian Optimization predicted response surface"""
        if not AX_AVAILABLE or not self.optimizer:
            return
        
        for widget in self.optimization_frame.winfo_children():
            widget.destroy()
        
        try:
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
    
    def export_statistics(self):
        """Export statistics to Excel"""
        date_str = datetime.now().strftime('%Y%m%d')

        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile="Analysis1.xlsx"
        )

        if filepath:
            try:
                # Generate path with naming convention: [UserName]_Statistics_[Date]
                base_path = os.path.splitext(filepath)[0]

                # Add standardized suffix
                final_path = f"{base_path}_Statistics_{date_str}.xlsx"

                self.exporter.set_results(self.results, self.main_effects)
                self.exporter.export_statistics_excel(final_path)
                messagebox.showinfo("Success", f"Statistics exported to:\n{final_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def _prompt_for_stock_concentrations(self):
        """Prompt user for stock concentrations when metadata is not available"""
        # Create dialog
        stock_dialog = tk.Toplevel(self)
        stock_dialog.title("Enter Stock Concentrations")
        stock_dialog.geometry("500x400")
        stock_dialog.transient(self)
        stock_dialog.grab_set()
        
        ttk.Label(stock_dialog, text="Stock Concentrations Not Found in File",
                 font=('TkDefaultFont', 12, 'bold')).pack(pady=10)
        ttk.Label(stock_dialog, text="Please enter stock concentrations for each factor:",
                 font=('TkDefaultFont', 10)).pack(pady=5)
        
        # Frame for entries
        entries_frame = ttk.Frame(stock_dialog, padding=20)
        entries_frame.pack(fill='both', expand=True)
        
        # Create entry for each numeric factor
        stock_vars = {}
        row = 0
        for factor in self.handler.numeric_factors:
            # Display name and unit
            if 'buffer' in factor.lower() and 'conc' in factor.lower():
                display_name = "Buffer Concentration"
                unit = "mM"
                default = "1000"
            elif 'salt' in factor.lower():
                display_name = factor.replace('_', ' ').title()
                unit = "mM"
                default = "5000"
            elif 'glycerol' in factor.lower():
                display_name = "Glycerol"
                unit = "%"
                default = "100"
            elif 'dmso' in factor.lower():
                display_name = "DMSO"
                unit = "%"
                default = "100"
            elif 'detergent' in factor.lower():
                display_name = "Detergent"
                unit = "%"
                default = "10"
            else:
                display_name = factor.replace('_', ' ').title()
                unit = ""
                default = ""
            
            ttk.Label(entries_frame, text=f"{display_name}:").grid(row=row, column=0, sticky='e', padx=5, pady=5)
            var = tk.StringVar(value=default)
            entry = ttk.Entry(entries_frame, textvariable=var, width=15)
            entry.grid(row=row, column=1, padx=5, pady=5)
            ttk.Label(entries_frame, text=unit).grid(row=row, column=2, sticky='w', padx=5, pady=5)
            
            stock_vars[factor] = var
            row += 1
        
        result = {'confirmed': False, 'stocks': {}}
        
        def confirm():
            try:
                stocks = {}
                for factor, var in stock_vars.items():
                    value_str = var.get().strip()
                    if value_str:
                        stocks[factor] = float(value_str)
                
                result['confirmed'] = True
                result['stocks'] = stocks
                stock_dialog.destroy()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numeric values for all fields.")
        
        def cancel():
            stock_dialog.destroy()
        
        # Buttons
        button_frame = ttk.Frame(stock_dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Confirm", command=confirm).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side='left', padx=5)
        
        # Wait for dialog to close
        self.wait_window(stock_dialog)
        
        if result['confirmed']:
            return result['stocks']
        else:
            return None
    
    def _ask_plot_format(self):
        """Ask user for plot export format"""
        dialog = tk.Toplevel(self)
        dialog.title("Select Plot Format")
        dialog.geometry("500x400")
        dialog.transient(self)
        dialog.grab_set()

        result = {'format': None, 'dpi': 300}

        ttk.Label(dialog, text="Select Export Format", font=('TkDefaultFont', 12, 'bold')).pack(pady=10)

        # Format selection
        format_frame = ttk.LabelFrame(dialog, text="File Format", padding=10)
        format_frame.pack(fill='x', padx=20, pady=10)

        format_var = tk.StringVar(value="png")

        formats = [
            ("PNG - Good for presentations, web (300 DPI)", "png"),
            ("TIFF - Publication quality, lossless (300 DPI)", "tiff"),
            ("PDF - Vector format, scalable", "pdf"),
            ("EPS - Vector format, publication standard", "eps")
        ]

        for text, value in formats:
            ttk.Radiobutton(format_frame, text=text, variable=format_var, value=value).pack(anchor='w', pady=2)

        # DPI selection (only for raster formats)
        dpi_frame = ttk.LabelFrame(dialog, text="Resolution (PNG/TIFF only)", padding=10)
        dpi_frame.pack(fill='x', padx=20, pady=10)

        dpi_var = tk.IntVar(value=300)

        dpi_options = [
            ("300 DPI - Standard publication quality", 300),
            ("600 DPI - High quality publication", 600),
            ("150 DPI - Screen/web quality", 150)
        ]

        for text, value in dpi_options:
            ttk.Radiobutton(dpi_frame, text=text, variable=dpi_var, value=value).pack(anchor='w', pady=2)

        def confirm():
            result['format'] = format_var.get()
            result['dpi'] = dpi_var.get()
            dialog.destroy()

        def cancel():
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="OK", command=confirm).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel).pack(side='left', padx=5)

        self.wait_window(dialog)
        return result

    def export_plots(self):
        """Export plots in user-selected format"""
        # Ask for format first
        format_options = self._ask_plot_format()
        if not format_options['format']:
            return

        file_format = format_options['format']
        dpi = format_options['dpi']
        date_str = datetime.now().strftime('%Y%m%d')

        # File type mapping for dialog
        format_map = {
            'png': ("PNG files", "*.png"),
            'tiff': ("TIFF files", "*.tiff"),
            'pdf': ("PDF files", "*.pdf"),
            'eps': ("EPS files", "*.eps")
        }

        # Ask user for base name
        base_name = filedialog.asksaveasfilename(
            defaultextension=f".{file_format}",
            filetypes=[format_map[file_format], ("All files", "*.*")],
            initialfile=f"Plots1.{file_format}",
            title="Choose base name for plots (will create multiple files)"
        )

        if not base_name:
            return

        # Extract directory and base name without extension
        directory = os.path.dirname(base_name)
        base_path = os.path.splitext(os.path.basename(base_name))[0]

        try:
            # Export main effects plot
            fig1 = self.plotter.plot_main_effects()
            fig1.savefig(os.path.join(directory, f'{base_path}_MainEffects_{date_str}.{file_format}'),
                        dpi=dpi, bbox_inches='tight')
            plt.close(fig1)

            # Export interactions plot
            fig2 = self.plotter.plot_interaction_effects()
            if fig2:
                fig2.savefig(os.path.join(directory, f'{base_path}_Interactions_{date_str}.{file_format}'),
                            dpi=dpi, bbox_inches='tight')
                plt.close(fig2)

            # Export residuals plot
            fig3 = self.plotter.plot_residuals(
                self.results['predictions'],
                self.results['residuals']
            )
            fig3.savefig(os.path.join(directory, f'{base_path}_Residuals_{date_str}.{file_format}'),
                        dpi=dpi, bbox_inches='tight')
            plt.close(fig3)

            messagebox.showinfo("Success", f"Plots exported to:\n{directory}\n\n"
                              f"Format: {file_format.upper()}, DPI: {dpi if file_format in ['png', 'tiff'] else 'Vector'}\n\n"
                              f"Files created:\n"
                              f"- {base_path}_MainEffects_{date_str}.{file_format}\n"
                              f"- {base_path}_Interactions_{date_str}.{file_format}\n"
                              f"- {base_path}_Residuals_{date_str}.{file_format}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def export_bo_batch(self):
        """Export BO suggestions to Excel and Opentrons CSV"""
        if not AX_AVAILABLE or not self.optimizer or not self.optimizer.is_initialized:
            messagebox.showerror("Error", "Bayesian Optimization not available or not initialized.")
            return
        
        if not self.filepath:
            messagebox.showerror("Error", "No Excel file loaded. Please load data first.")
            return
        
        # Dialog to get parameters
        dialog = tk.Toplevel(self)
        dialog.title("Export BO Batch")
        dialog.geometry("500x400")
        dialog.transient(self)
        dialog.grab_set()
        
        # Number of suggestions
        ttk.Label(dialog, text="Number of BO Suggestions:", font=('TkDefaultFont', 10, 'bold')).pack(pady=(20,5))
        n_var = tk.IntVar(value=5)
        n_spin = ttk.Spinbox(dialog, from_=1, to=20, textvariable=n_var, width=10)
        n_spin.pack()
        
        # Batch number
        ttk.Label(dialog, text="Batch Number:", font=('TkDefaultFont', 10, 'bold')).pack(pady=(15,5))
        
        # Auto-detect next batch number from loaded data
        try:
            if 'Batch' in self.handler.data.columns:
                max_batch = int(self.handler.data['Batch'].max())
                next_batch = max_batch + 1
            else:
                next_batch = 1
        except:
            next_batch = 1
        
        batch_var = tk.IntVar(value=next_batch)
        batch_frame = ttk.Frame(dialog)
        batch_frame.pack()
        ttk.Label(batch_frame, text=f"Suggested: {next_batch} (next after current data)").pack(side='left', padx=5)
        batch_spin = ttk.Spinbox(batch_frame, from_=1, to=100, textvariable=batch_var, width=10)
        batch_spin.pack(side='left')
        
        # Final volume
        ttk.Label(dialog, text="Final Volume (µL):", font=('TkDefaultFont', 10, 'bold')).pack(pady=(15,5))
        vol_var = tk.DoubleVar(value=100.0)
        vol_entry = ttk.Entry(dialog, textvariable=vol_var, width=15)
        vol_entry.pack()
        
        # Stock concentrations note
        stock_concs_from_metadata = self.handler.get_stock_concentrations()
        if stock_concs_from_metadata:
            ttk.Label(dialog, text="✓ Using stock concentrations from file metadata",
                     font=('TkDefaultFont', 9, 'bold'), foreground='green').pack(pady=(15,5))
        else:
            ttk.Label(dialog, text="⚠ No metadata found - will prompt for stock concentrations",
                     font=('TkDefaultFont', 9), foreground='orange').pack(pady=(15,5))
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        
        def do_export():
            try:
                n_suggestions = n_var.get()
                batch_number = batch_var.get()
                final_volume = vol_var.get()
                
                # Get stock concentrations from metadata first
                stock_concs = self.handler.get_stock_concentrations()
                
                # If no metadata, show dialog to ask user
                if not stock_concs:
                    stock_concs = self._prompt_for_stock_concentrations()
                    if stock_concs is None:
                        # User cancelled
                        return
                
                # Get buffer pH values from data
                buffer_ph_values = []
                if 'buffer pH' in self.handler.data.columns or 'Buffer pH' in self.handler.data.columns:
                    ph_col = 'buffer pH' if 'buffer pH' in self.handler.data.columns else 'Buffer pH'
                    unique_phs = sorted(self.handler.data[ph_col].dropna().unique())
                    buffer_ph_values = [str(ph) for ph in unique_phs]
                
                # Export
                result = self.optimizer.export_bo_batch_to_files(
                    n_suggestions=n_suggestions,
                    batch_number=batch_number,
                    excel_path=self.filepath,
                    stock_concs=stock_concs,
                    final_volume=final_volume,
                    buffer_ph_values=buffer_ph_values
                )
                
                if result:
                    xlsx_path, csv_path = result
                    dialog.destroy()
                    messagebox.showinfo("Export Successful!",
                        f"BO Batch exported successfully!\n\n"
                        f"📊 Excel updated:\n{xlsx_path}\n\n"
                        f"🤖 Opentrons CSV:\n{csv_path}\n\n"
                        f"Batch {batch_number}: {n_suggestions} new experiments added")
                else:
                    messagebox.showerror("Error", "Export failed. Check console for details.")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
        
        ttk.Button(button_frame, text="Export", command=do_export).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side='left', padx=5)

    def export_bo_plots_gui(self):
        """GUI wrapper for exporting BO plots"""
        if not AX_AVAILABLE or not self.optimizer or not self.optimizer.is_initialized:
            messagebox.showerror("Error", "Bayesian Optimization not available or not initialized.")
            return

        # Ask for format first
        format_options = self._ask_plot_format()
        if not format_options['format']:
            return

        file_format = format_options['format']
        dpi = format_options['dpi']
        date_str = datetime.now().strftime('%Y%m%d')

        # File type mapping for dialog
        format_map = {
            'png': ("PNG files", "*.png"),
            'tiff': ("TIFF files", "*.tiff"),
            'pdf': ("PDF files", "*.pdf"),
            'eps': ("EPS files", "*.eps")
        }

        # Ask user for base name
        base_name = filedialog.asksaveasfilename(
            defaultextension=f".{file_format}",
            filetypes=[format_map[file_format], ("All files", "*.*")],
            initialfile=f"Plots1.{file_format}",
            title="Choose base name for BO plots (will create multiple files)"
        )

        if not base_name:
            return

        # Extract directory and base name
        directory = os.path.dirname(base_name)
        base_path = os.path.splitext(os.path.basename(base_name))[0]

        try:
            exported_files = self.optimizer.export_bo_plots(directory, base_path, date_str, file_format, dpi)

            if exported_files:
                filenames = "\\n".join([os.path.basename(f) for f in exported_files])
                messagebox.showinfo("Success",
                                  f"Exported {len(exported_files)} BO plots\\n\\n"
                                  f"Format: {file_format.upper()}, DPI: {dpi if file_format in ['png', 'tiff'] else 'Vector'}\\n\\n"
                                  f"{filenames}\\n\\nLocation: {directory}")
            else:
                messagebox.showwarning("Warning", "No plots were exported. Check console for details.")

        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\\n{str(e)}")


    def load_results(self):
        """Load results from Excel (called from main window menu)"""
        self.browse_file()

    def refresh(self):
        """Refresh UI from project data (called when switching tabs)"""
        # Reload display if data exists
        if self.handler.data is not None:
            self._update_display()
