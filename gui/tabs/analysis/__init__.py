#!/usr/bin/env python3
"""
DoE Data Analysis Tool - v0.4.1
Statistical analysis GUI for Design of Experiments data
Replicates MATLAB fitlm (EDA CAPKIN MATLAB code), main effects, and interaction plots
Includes Bayesian Optimization for experiment suggestions

Milton F. Villegas
"""

# GUI and system imports
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import platform
import warnings
warnings.filterwarnings('ignore')

# Bayesian Optimization imports
try:
    from ax.service.ax_client import AxClient, ObjectiveProperties
    AX_AVAILABLE = True
except ImportError:
    AX_AVAILABLE = False
    print("Warning: Ax not available. Install with: pip install ax-platform")

# Plotting imports
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
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
from core.data_handler import DataHandler
from core.plotter import DoEPlotter
from core.exporter import ResultsExporter

# Import mixins
from .data_panel import DataPanelMixin
from .model_panel import ModelPanelMixin
from .visualization_panel import VisualizationPanelMixin
from .optimization_panel import OptimizationPanelMixin
from .export_panel import ExportPanelMixin


class AnalysisTab(
    DataPanelMixin,
    ModelPanelMixin,
    VisualizationPanelMixin,
    OptimizationPanelMixin,
    ExportPanelMixin,
    ttk.Frame
):
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

        # Response selection tracking
        self.response_checkboxes = {}
        self.selected_responses = []
        self.response_directions = {}

        # Debug log collector
        self.debug_log = []
        self.CONSOLE_DEBUG = False

        # Error tracking
        self.has_analysis_error = False

        # Setup GUI
        self.setup_gui()

    def setup_gui(self):
        """Setup GUI layout"""
        # Title
        title_frame = ttk.Frame(self)
        title_frame.pack(fill='x', padx=10, pady=5)

        # File selection
        file_frame = ttk.LabelFrame(self, text="Select Data File", padding=10)
        file_frame.pack(fill='x', padx=10, pady=5)

        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side='left', padx=5)

        browse_btn = ttk.Button(file_frame, text="Browse...", command=self.browse_file)
        browse_btn.pack(side='right', padx=5)

        # Response selection
        self.response_frame = ttk.LabelFrame(self, text="Select Response Variables", padding=10)
        self.response_frame.pack(fill='x', padx=10, pady=5)

        self.response_label = ttk.Label(self.response_frame,
                                       text="Load a file to see available response columns")
        self.response_label.pack(padx=5, pady=5)

        # Configuration
        config_frame = ttk.LabelFrame(self, text="Configure Analysis", padding=10)
        config_frame.pack(fill='x', padx=10, pady=5)

        # Model Type row with info button
        ttk.Label(config_frame, text="Model Selection:").grid(row=0, column=0, sticky='w', padx=5, pady=5)

        # Dropdown for model selection
        model_options = [
            'Auto (Recommended)',
            'Mean (intercept only)',
            'Linear (main effects)',
            'Interactions (2-way)',
            'Quadratic (full)',
            'Reduced (backward elimination)'
        ]
        self.model_selection_var = tk.StringVar(value='Auto (Recommended)')
        self.model_dropdown = ttk.Combobox(config_frame, textvariable=self.model_selection_var,
                                          values=model_options, state='readonly', width=30)
        self.model_dropdown.grid(row=0, column=1, sticky='w', padx=5, pady=5)

        # info button
        info_btn = ttk.Button(config_frame, text="?", width=2, command=self.show_model_guide)
        info_btn.grid(row=0, column=2, sticky='w', padx=(2, 0), pady=5)

        self.analyze_btn = ttk.Button(config_frame, text="Analyze Data",
                                      command=self.analyze_data, state='disabled')
        self.analyze_btn.grid(row=0, column=3, padx=20, pady=5)

    def create_results_tab(self):
        """Create the Results tab with results notebook and export buttons"""
        parent_notebook = self.main_window.notebook

        # Create Results tab
        results_tab = ttk.Frame(parent_notebook)
        parent_notebook.add(results_tab, text="Results")

        # Export buttons at top
        export_frame = ttk.LabelFrame(results_tab, text="Export Results", padding=10)
        export_frame.pack(fill='x', padx=10, pady=5)

        self.export_stats_btn = ttk.Button(export_frame, text="Export Statistics",
                                          command=self.export_statistics, state='disabled')
        self.export_stats_btn.pack(side='left', padx=5)

        self.export_plots_btn = ttk.Button(export_frame, text="Export Plots",
                                          command=self.export_plots, state='disabled')
        self.export_plots_btn.pack(side='left', padx=5)

        # Results notebook
        results_frame = ttk.LabelFrame(results_tab, text="Analysis Results", padding=5)
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)

        results_container = ttk.Frame(results_frame)
        results_container.pack(fill='both', expand=True)

        self.notebook = ttk.Notebook(results_container)
        self.notebook.pack(fill='both', expand=True)

        # Tab 1: Statistics
        stats_container = ttk.Frame(self.notebook)
        self.notebook.add(stats_container, text="Statistics")

        stats_header = ttk.Frame(stats_container)
        stats_header.pack(fill='x', padx=5, pady=2)
        ttk.Label(stats_header, text="Statistical Analysis Results", font=('TkDefaultFont', 10, 'bold')).pack(side='left')
        ttk.Button(stats_header, text="i", width=3,
                  command=lambda: self.show_tooltip("statistics")).pack(side='right', padx=5)

        self.stats_text = scrolledtext.ScrolledText(stats_container, wrap=tk.WORD, font=('Courier', 14))
        self.stats_text.pack(fill='both', expand=True, padx=5, pady=5)

        # Tab 2: Main Effects
        main_effects_container = ttk.Frame(self.notebook)
        self.notebook.add(main_effects_container, text="Main Effects")

        me_header = ttk.Frame(main_effects_container)
        me_header.pack(fill='x', padx=5, pady=2)
        ttk.Label(me_header, text="Main Effects Plot", font=('TkDefaultFont', 10, 'bold')).pack(side='left')
        ttk.Button(me_header, text="i", width=3,
                  command=lambda: self.show_tooltip("main_effects")).pack(side='right', padx=5)

        self.main_effects_frame = self.create_scrollable_frame(main_effects_container)

        # Tab 3: Interactions
        interactions_container = ttk.Frame(self.notebook)
        self.notebook.add(interactions_container, text="Interactions")

        int_header = ttk.Frame(interactions_container)
        int_header.pack(fill='x', padx=5, pady=2)
        ttk.Label(int_header, text="Interaction Effects Plot", font=('TkDefaultFont', 10, 'bold')).pack(side='left')
        ttk.Button(int_header, text="i", width=3,
                  command=lambda: self.show_tooltip("interactions")).pack(side='right', padx=5)

        self.interactions_frame = self.create_scrollable_frame(interactions_container)

        # Tab 4: Residuals
        residuals_container = ttk.Frame(self.notebook)
        self.notebook.add(residuals_container, text="Residuals")

        res_header = ttk.Frame(residuals_container)
        res_header.pack(fill='x', padx=5, pady=2)
        ttk.Label(res_header, text="Residuals Diagnostic Plot", font=('TkDefaultFont', 10, 'bold')).pack(side='left')
        ttk.Button(res_header, text="i", width=3,
                  command=lambda: self.show_tooltip("residuals")).pack(side='right', padx=5)

        self.residuals_frame = self.create_scrollable_frame(residuals_container)

        # Tab 5: Recommendations
        recommendations_container = ttk.Frame(self.notebook)
        self.notebook.add(recommendations_container, text="Recommendations")

        rec_header = ttk.Frame(recommendations_container)
        rec_header.pack(fill='x', padx=5, pady=2)
        ttk.Label(rec_header, text="Next Experiments", font=('TkDefaultFont', 10, 'bold')).pack(side='left')
        ttk.Button(rec_header, text="i", width=3,
                  command=lambda: self.show_tooltip("recommendations")).pack(side='right', padx=5)

        if AX_AVAILABLE:
            export_frame_rec = ttk.LabelFrame(recommendations_container, text="Export", padding=10)
            export_frame_rec.pack(fill='x', padx=10, pady=5)

            self.export_bo_button = ttk.Button(export_frame_rec, text="Export BO Batch",
                                              command=self.export_bo_batch, state='disabled')
            self.export_bo_button.pack(side='left', padx=5)

            ttk.Label(export_frame_rec, text="(Available after analysis with BO suggestions)",
                     font=('TkDefaultFont', 9)).pack(side='left', padx=5)

        self.recommendations_text = scrolledtext.ScrolledText(recommendations_container,
                                                             wrap=tk.WORD, font=('Courier', 14))
        self.recommendations_text.pack(fill='both', expand=True, padx=5, pady=5)

        # Tab 6: Optimization Details
        if AX_AVAILABLE:
            optimization_container = ttk.Frame(self.notebook)
            self.notebook.add(optimization_container, text="Optimization Details")

            opt_header = ttk.Frame(optimization_container)
            opt_header.pack(fill='x', padx=5, pady=2)
            ttk.Label(opt_header, text="Bayesian Optimization Analysis", font=('TkDefaultFont', 10, 'bold')).pack(side='left')
            ttk.Button(opt_header, text="i", width=3,
                      command=lambda: self.show_tooltip("optimization")).pack(side='right', padx=5)

            self.export_frame_opt = ttk.LabelFrame(optimization_container, text="Export", padding=10)
            self.export_frame_opt.pack(fill='x', padx=10, pady=5)

            self.optimization_frame = self.create_scrollable_frame(optimization_container)

    def create_scrollable_frame(self, parent):
        """Create a scrollable frame for plot display with comprehensive mousewheel support"""
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
        canvas.configure(yscrollincrement='10')

        # Pack scrollbars and canvas
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')
        canvas.pack(side='left', fill='both', expand=True)

        # MOUSEWHEEL SOLUTION
        system = platform.system()

        def on_mousewheel(event):
            if system == 'Windows':
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            elif system == 'Darwin':
                canvas.yview_scroll(int(-1*event.delta), "units")
            else:
                if event.num == 4:
                    canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    canvas.yview_scroll(1, "units")

        def bind_tree(widget):
            """Recursively bind mousewheel to widget and all its children"""
            if system == 'Windows' or system == 'Darwin':
                widget.bind('<MouseWheel>', on_mousewheel, add='+')
            else:
                widget.bind('<Button-4>', on_mousewheel, add='+')
                widget.bind('<Button-5>', on_mousewheel, add='+')

            for child in widget.winfo_children():
                bind_tree(child)

        # Bind to canvas
        if system == 'Windows' or system == 'Darwin':
            canvas.bind('<MouseWheel>', on_mousewheel)
        else:
            canvas.bind('<Button-4>', on_mousewheel)
            canvas.bind('<Button-5>', on_mousewheel)

        bind_tree(scrollable_frame)
        scrollable_frame._bind_mousewheel = bind_tree

        def on_canvas_configure(event):
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
            "statistics": (
                "How to Read Statistical Analysis Results\n\n"
                "R-squared (R2) = How well the model fits (0-1, higher is better)\n"
                "p-value < 0.05 = Factor is statistically significant\n"
                "Coefficient = Effect size and direction (+ or -)\n"
                "Adjusted R2 = R2 adjusted for number of factors\n\n"
                "Look for:\n"
                "High R2 (>0.7) indicates good model fit\n"
                "Significant factors (p < 0.05) with asterisks ***\n"
                "Large absolute coefficients = stronger effects"
            ),
            "main_effects": (
                "How to Read Main Effects Plot\n\n"
                "Steeper slopes = Stronger factor influence on response\n"
                "Upward slope = Increasing factor increases response\n"
                "Downward slope = Increasing factor decreases response\n"
                "Flat line = Factor has little/no effect\n"
                "Compare slopes to identify most important factors"
            ),
            "interactions": (
                "How to Read Interaction Plot\n\n"
                "Parallel lines = NO interaction (factors work independently)\n"
                "Non-parallel lines = Interaction present\n"
                "Crossing lines = Strong interaction"
            ),
            "residuals": (
                "How to Read Residuals Plot\n\n"
                "Random scatter around zero = Good model fit\n"
                "Pattern/curve = Model may be inappropriate\n"
                "Funnel shape = Variance increases with response\n"
                "Outliers far from others = Check those data points"
            ),
            "recommendations": (
                "How to Read Next Experiments\n\n"
                "Bayesian Optimization (BO) suggests next experiments based on your data:\n\n"
                "Suggestions are ranked by expected improvement\n"
                "BO balances exploration vs exploitation\n"
                "Each suggestion shows predicted factor values\n"
                "Run suggested experiments to improve your response"
            ),
            "optimization": (
                "How to Read Optimization Details\n\n"
                "This plot shows where BO predicts you should explore:\n\n"
                "Hot colors (yellow/green) = Predicted high response regions\n"
                "Cool colors (blue/purple) = Predicted low response regions\n"
                "Red dots = Your existing experiments\n"
                "Gaps between dots = Unexplored regions to test"
            )
        }

        messagebox.showinfo("Plot Interpretation Guide", tooltips.get(plot_type, "No guide available"))

    def show_model_guide(self):
        """Show model selection guide in a popup window"""
        guide_text = (
            "Linear: Simplest model, main effects only (A + B + C)\n"
            "  Use when: You want to understand which factors matter\n"
            "  Assumes: Each factor works independently\n\n"

            "Interactions: Use if factors work together (A + B + AxB)\n"
            "  Use when: Factors may influence each other\n"
            "  Detects: Combined effects of multiple factors\n\n"

            "Pure Quadratic: For curved responses (A + B + A2 + B2)\n"
            "  Use when: Response has a peak or valley\n"
            "  Assumes: No interactions, but curved relationships\n\n"

            "Quadratic: Full model (A + B + AxB + A2 + B2)\n"
            "  Use when: You expect both curvature and interactions\n"
            "  Most complex: Requires more data points"
        )

        messagebox.showinfo("Model Selection Guide", guide_text)

    def _update_display(self):
        """Update display based on current data (placeholder for compatibility)"""
        pass
