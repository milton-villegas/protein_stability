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
from core.data_handler import DataHandler
from core.plotter import DoEPlotter
from core.exporter import ResultsExporter
from utils.constants import METADATA_COLUMNS
from utils.sanitization import smart_factor_match


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

        # Response selection tracking
        self.response_checkboxes = {}  # {column_name: (var, direction_var)}
        self.selected_responses = []  # List of selected response column names
        self.response_directions = {}  # {column_name: 'maximize' or 'minimize'}

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
        # Get the parent notebook from main_window
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
        ttk.Button(stats_header, text="ℹ️", width=3,
                  command=lambda: self.show_tooltip("statistics")).pack(side='right', padx=5)

        self.stats_text = scrolledtext.ScrolledText(stats_container, wrap=tk.WORD, font=('Courier', 14))
        self.stats_text.pack(fill='both', expand=True, padx=5, pady=5)

        # Tab 2: Main Effects
        main_effects_container = ttk.Frame(self.notebook)
        self.notebook.add(main_effects_container, text="Main Effects")

        me_header = ttk.Frame(main_effects_container)
        me_header.pack(fill='x', padx=5, pady=2)
        ttk.Label(me_header, text="Main Effects Plot", font=('TkDefaultFont', 10, 'bold')).pack(side='left')
        ttk.Button(me_header, text="ℹ️", width=3,
                  command=lambda: self.show_tooltip("main_effects")).pack(side='right', padx=5)

        self.main_effects_frame = self.create_scrollable_frame(main_effects_container)

        # Tab 3: Interactions
        interactions_container = ttk.Frame(self.notebook)
        self.notebook.add(interactions_container, text="Interactions")

        int_header = ttk.Frame(interactions_container)
        int_header.pack(fill='x', padx=5, pady=2)
        ttk.Label(int_header, text="Interaction Effects Plot", font=('TkDefaultFont', 10, 'bold')).pack(side='left')
        ttk.Button(int_header, text="ℹ️", width=3,
                  command=lambda: self.show_tooltip("interactions")).pack(side='right', padx=5)

        self.interactions_frame = self.create_scrollable_frame(interactions_container)

        # Tab 4: Residuals
        residuals_container = ttk.Frame(self.notebook)
        self.notebook.add(residuals_container, text="Residuals")

        res_header = ttk.Frame(residuals_container)
        res_header.pack(fill='x', padx=5, pady=2)
        ttk.Label(res_header, text="Residuals Diagnostic Plot", font=('TkDefaultFont', 10, 'bold')).pack(side='left')
        ttk.Button(res_header, text="ℹ️", width=3,
                  command=lambda: self.show_tooltip("residuals")).pack(side='right', padx=5)

        self.residuals_frame = self.create_scrollable_frame(residuals_container)

        # Tab 5: Recommendations
        recommendations_container = ttk.Frame(self.notebook)
        self.notebook.add(recommendations_container, text="Recommendations")

        rec_header = ttk.Frame(recommendations_container)
        rec_header.pack(fill='x', padx=5, pady=2)
        ttk.Label(rec_header, text="Next Experiments", font=('TkDefaultFont', 10, 'bold')).pack(side='left')
        ttk.Button(rec_header, text="ℹ️", width=3,
                  command=lambda: self.show_tooltip("recommendations")).pack(side='right', padx=5)

        if AX_AVAILABLE:
            export_frame_rec = ttk.LabelFrame(recommendations_container, text="Export", padding=10)
            export_frame_rec.pack(fill='x', padx=10, pady=5)

            self.export_bo_button = ttk.Button(export_frame_rec, text="📤 Export BO Batch",
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
            ttk.Button(opt_header, text="ℹ️", width=3,
                      command=lambda: self.show_tooltip("optimization")).pack(side='right', padx=5)

            self.export_frame_opt = ttk.LabelFrame(optimization_container, text="Export", padding=10)
            self.export_frame_opt.pack(fill='x', padx=10, pady=5)

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
            "statistics": (
                "How to Read Statistical Analysis Results\n\n"
                "• R-squared (R²) = How well the model fits (0-1, higher is better)\n"
                "• p-value < 0.05 = Factor is statistically significant\n"
                "• Coefficient = Effect size and direction (+ or -)\n"
                "• Adjusted R² = R² adjusted for number of factors\n\n"
                "Look for:\n"
                "• High R² (>0.7) indicates good model fit\n"
                "• Significant factors (p < 0.05) with asterisks ***\n"
                "• Large absolute coefficients = stronger effects"
            ),
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
            "recommendations": (
                "How to Read Next Experiments\n\n"
                "Bayesian Optimization (BO) suggests intelligent next experiments:\n\n"
                "• Suggestions are ranked by expected improvement\n"
                "• BO balances exploration (new regions) vs exploitation (promising regions)\n"
                "• Each suggestion shows predicted factor values\n"
                "• Run suggested experiments to improve your response\n\n"
                "Tips:\n"
                "• Start with top 3-5 suggestions\n"
                "• Add results back to dataset and re-analyze\n"
                "• BO gets smarter with each iteration"
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
            self.file_label.config(text=os.path.basename(filepath))
            self.main_window.update_status(f"Loaded: {os.path.basename(filepath)}")
            
            try:
                self.handler.load_excel(filepath)

                # Get potential response columns (numeric columns excluding metadata)
                potential_responses = self.handler.get_potential_response_columns()

                if not potential_responses:
                    messagebox.showerror("No Response Columns Found",
                                       "Excel file must have at least one numeric column\n"
                                       "that can be used as a response variable.\n\n"
                                       "Numeric columns are needed for optimization.")
                    self.main_window.update_status("Error: No numeric columns found")
                    return

                self._populate_response_selection(potential_responses)

                self.main_window.update_status(f"Loaded: {len(potential_responses)} potential response(s) found")

            except Exception as e:
                messagebox.showerror("File Load Failed",
                    f"Could not open the selected Excel file.\n\n"
                    f"Details: {str(e)}\n\n"
                    f"Make sure the file is not open in another program.")
                self.main_window.update_status("Error loading file")

    def _populate_response_selection(self, potential_responses):
        """Create checkboxes for response selection with maximize/minimize options"""
        # Clear existing widgets in response frame
        for widget in self.response_frame.winfo_children():
            widget.destroy()

        # Reset tracking
        self.response_checkboxes = {}
        self.selected_responses = []
        self.response_directions = {}
        self.response_constraints = {}

        # Header
        header_label = ttk.Label(self.response_frame,
                                text="Select response variable(s) to optimize:",
                                font=('TkDefaultFont', 9, 'bold'))
        header_label.grid(row=0, column=0, columnspan=5, sticky='w', padx=5, pady=5)

        # Column headers
        ttk.Label(self.response_frame, text="Constraint Min", font=('TkDefaultFont', 8)).grid(row=0, column=2, padx=5)
        ttk.Label(self.response_frame, text="Constraint Max", font=('TkDefaultFont', 8)).grid(row=0, column=3, padx=5)

        # Create checkbox for each potential response
        for idx, col_name in enumerate(potential_responses):
            # Checkbox
            var = tk.BooleanVar(value=False)

            # Auto-select if column name contains "response" (case-insensitive)
            if 'response' in col_name.lower():
                var.set(True)

            checkbox = ttk.Checkbutton(self.response_frame, text=col_name, variable=var,
                                      command=self._update_selected_responses)
            checkbox.grid(row=idx+1, column=0, sticky='w', padx=20, pady=2)

            # Direction dropdown (Maximize/Minimize)
            direction_var = tk.StringVar(value='Maximize')
            direction_combo = ttk.Combobox(self.response_frame, textvariable=direction_var,
                                          values=['Maximize', 'Minimize'],
                                          state='readonly', width=10)
            direction_combo.grid(row=idx+1, column=1, padx=10, pady=2)
            direction_combo.bind('<<ComboboxSelected>>', lambda e: self._update_selected_responses())

            # Min constraint entry
            min_var = tk.StringVar(value='')
            min_entry = ttk.Entry(self.response_frame, textvariable=min_var, width=10)
            min_entry.grid(row=idx+1, column=2, padx=5, pady=2)
            min_entry.bind('<FocusOut>', lambda e: self._update_selected_responses())
            min_entry.bind('<Return>', lambda e: self._update_selected_responses())

            # Max constraint entry
            max_var = tk.StringVar(value='')
            max_entry = ttk.Entry(self.response_frame, textvariable=max_var, width=10)
            max_entry.grid(row=idx+1, column=3, padx=5, pady=2)
            max_entry.bind('<FocusOut>', lambda e: self._update_selected_responses())
            max_entry.bind('<Return>', lambda e: self._update_selected_responses())

            # Store references
            self.response_checkboxes[col_name] = (var, direction_var, min_var, max_var)

        # Note label
        note_label = ttk.Label(self.response_frame,
                              text="Note: Constraints are optional. Leave blank for no constraint.",
                              font=('TkDefaultFont', 8, 'italic'))
        note_label.grid(row=len(potential_responses)+1, column=0, columnspan=5,
                       sticky='w', padx=5, pady=5)

        # Exploration mode checkbox
        self.exploration_mode_var = tk.BooleanVar(value=False)
        exploration_cb = ttk.Checkbutton(
            self.response_frame,
            text="Allow exploration outside bounds (20% of suggestions may violate constraints)",
            variable=self.exploration_mode_var
        )
        exploration_cb.grid(row=len(potential_responses)+2, column=0, columnspan=5,
                           sticky='w', padx=20, pady=5)

        # Initial update
        self._update_selected_responses()

    def _update_selected_responses(self):
        """Update selected responses list and enable/disable analyze button"""
        self.selected_responses = []
        self.response_directions = {}
        self.response_constraints = {}

        for col_name, (var, direction_var, min_var, max_var) in self.response_checkboxes.items():
            if var.get():
                self.selected_responses.append(col_name)
                direction = 'minimize' if direction_var.get() == 'Minimize' else 'maximize'
                self.response_directions[col_name] = direction

                print(f"[DEBUG UI] Response '{col_name}': direction={direction} (UI dropdown value: '{direction_var.get()}')")

                min_val = min_var.get().strip()
                max_val = max_var.get().strip()

                constraint = {}
                if min_val:
                    try:
                        constraint['min'] = float(min_val)
                        print(f"[DEBUG UI] Response '{col_name}': min constraint = {constraint['min']}")
                    except ValueError:
                        print(f"[DEBUG UI] Response '{col_name}': invalid min value '{min_val}'")
                        pass
                if max_val:
                    try:
                        constraint['max'] = float(max_val)
                        print(f"[DEBUG UI] Response '{col_name}': max constraint = {constraint['max']}")
                    except ValueError:
                        print(f"[DEBUG UI] Response '{col_name}': invalid max value '{max_val}'")
                        pass

                if constraint:
                    self.response_constraints[col_name] = constraint

        if self.selected_responses:
            self.analyze_btn.config(state='normal')
            self.main_window.update_status(f"Ready to analyze {len(self.selected_responses)} response(s)")
        else:
            self.analyze_btn.config(state='disabled')
            self.main_window.update_status("Please select at least one response variable")

    def analyze_data(self):
        """Perform DoE analysis"""
        if not self.filepath:
            messagebox.showwarning("Warning", "Please select a data file first.")
            return

        # Update selected responses to capture latest dropdown/constraint values
        self._update_selected_responses()

        if not self.selected_responses:
            messagebox.showwarning("Warning", "Please select at least one response variable.")
            return

        print(f"\n[DEBUG ANALYZE] Starting analysis with:")
        print(f"  - Selected responses: {self.selected_responses}")
        print(f"  - Response directions: {self.response_directions}")
        print(f"  - Response constraints: {self.response_constraints}")

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

            # Display optimization plot if available
            if AX_AVAILABLE and self.optimizer:
                self.display_optimization_plot()

            self.export_stats_btn.config(state='normal')
            self.export_plots_btn.config(state='normal')

            # Show completion status
            chosen_model_name = self.analyzer.MODEL_TYPES[chosen_model]

            if len(self.selected_responses) > 1:
                avg_r2 = sum(r['model_stats']['R-squared'] for r in self.all_results.values()) / len(self.all_results)
                self.main_window.update_status(f"Analysis complete! {len(self.selected_responses)} responses analyzed | Avg R² = {avg_r2:.4f}")

                messagebox.showinfo("Success",
                                  f"Multi-response analysis completed successfully!\n\n"
                                  f"Responses analyzed: {len(self.selected_responses)}\n"
                                  f"Average R-squared: {avg_r2:.4f}\n"
                                  f"Observations: {self.results['model_stats']['Observations']}\n\n"
                                  f"Check the Results tab for detailed analysis of each response.")
            else:
                self.main_window.update_status(f"Analysis complete! Model: {chosen_model_name} | R² = {self.results['model_stats']['R-squared']:.4f}")

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
            print(f"❌ ANALYSIS ERROR")
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
        """Display statistics for a single response (helper for multi-response display)"""
        # MODEL COMPARISON SECTION
        if model_comparison:
            self.stats_text.insert(tk.END, "📊 MODEL COMPARISON\n")
            self.stats_text.insert(tk.END, "-"*80 + "\n\n")

            comparison_table = model_comparison['comparison_table']

            if comparison_table is not None and not comparison_table.empty:
                self.stats_text.insert(tk.END, "Model Comparison (all 5 models fitted):\n\n")

                header = f"{'Model':<30} {'Adj R²':>10} {'BIC':>10} {'RMSE':>10} {'DF':>6}  {'Selected':>10}\n"
                self.stats_text.insert(tk.END, header)
                self.stats_text.insert(tk.END, "-"*80 + "\n")

                recommended_model = model_selection['recommended_model']
                is_auto = self.model_selection_var.get() == 'Auto (Recommended)'

                model_order = ['mean', 'linear', 'interactions', 'quadratic', 'reduced']
                for model_type in model_order:
                    if model_type in model_comparison['models']:
                        stats = model_comparison['models'][model_type]
                        model_name = stats['Model Type']

                        marker = ("✓ AUTO" if is_auto else "✓ USER") if model_type == recommended_model else ""

                        line = (f"{model_name:<30} "
                               f"{stats['Adj R²']:>10.4f} "
                               f"{stats['BIC']:>10.1f} "
                               f"{stats['RMSE']:>10.4f} "
                               f"{stats['DF Model']:>6} "
                               f"{marker:>10}\n")
                        self.stats_text.insert(tk.END, line)

                self.stats_text.insert(tk.END, "\n" + "-"*80 + "\n")
                self.stats_text.insert(tk.END, f"🎯 RECOMMENDED: {self.analyzer.MODEL_TYPES[recommended_model]}\n")
                self.stats_text.insert(tk.END, f"   Reason: {model_selection['reason']}\n")

                self.stats_text.insert(tk.END, "\n💡 Selection criteria: Adjusted R² (60%), BIC (30%), Parsimony (10%)\n")
                self.stats_text.insert(tk.END, "   Higher Adj R² is better | Lower BIC is better | Simpler models preferred\n")

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
            warnings.append(f"⚠️  LOW R² ({r_squared:.3f}): Model explains only {r_squared*100:.1f}% of variance")
        elif r_squared < 0.7:
            warnings.append(f"⚠️  MODERATE R² ({r_squared:.3f}): Model is acceptable but could be improved")

        if len(sig_factors) == 0:
            warnings.append("⚠️  NO SIGNIFICANT FACTORS: No factors with p < 0.05 found")

        if n_obs < 20:
            warnings.append(f"⚠️  SMALL SAMPLE SIZE: Only {n_obs} observations")

        if warnings:
            self.stats_text.insert(tk.END, "🚩 RED FLAGS / WARNINGS\n")
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
            self.stats_text.insert(tk.END, f"⚡ SIGNIFICANT INTERACTIONS DETECTED: {len(interaction_factors)}\n")
            self.stats_text.insert(tk.END, "-"*80 + "\n")
            for interaction in interaction_factors:
                coef = results['coefficients'].loc[interaction, 'Coefficient']
                pval = results['coefficients'].loc[interaction, 'p-value']
                self.stats_text.insert(tk.END, f"  {interaction:<40} coef={coef:>10.4f}  p={pval:.2e}\n")
            self.stats_text.insert(tk.END, "\n⚠️  Interactions mean optimal settings depend on factor combinations!\n")

        self.stats_text.insert(tk.END, "\n" + "="*80 + "\n")

    def display_statistics(self):
        """Display statistical results with recommendations and warnings"""
        self.stats_text.delete('1.0', tk.END)

        self.stats_text.insert(tk.END, "="*80 + "\n")
        self.stats_text.insert(tk.END, "DOE ANALYSIS RESULTS\n")
        self.stats_text.insert(tk.END, "="*80 + "\n\n")

        # Multi-response analysis: display results for each response
        if hasattr(self, 'all_results') and len(self.all_results) > 1:
            self.stats_text.insert(tk.END, f"🔬 MULTI-RESPONSE ANALYSIS ({len(self.all_results)} responses selected)\n")
            self.stats_text.insert(tk.END, "="*80 + "\n\n")

            for response_idx, response_name in enumerate(self.selected_responses, 1):
                self.stats_text.insert(tk.END, f"\n{'█'*80}\n")
                self.stats_text.insert(tk.END, f"RESPONSE {response_idx}/{len(self.selected_responses)}: {response_name}\n")
                self.stats_text.insert(tk.END, f"{'█'*80}\n\n")

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
            self.stats_text.insert(tk.END, "📊 MODEL COMPARISON\n")
            self.stats_text.insert(tk.END, "-"*80 + "\n\n")

            # Display comparison table
            comparison_table = self.model_comparison['comparison_table']

            if comparison_table is not None and not comparison_table.empty:
                # Format the comparison table for display
                self.stats_text.insert(tk.END, "Model Comparison (all 5 models fitted):\n\n")

                # Create formatted header
                header = f"{'Model':<30} {'Adj R²':>10} {'BIC':>10} {'RMSE':>10} {'DF':>6}  {'Selected':>10}\n"
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
                            marker = "✓ AUTO" if is_auto else "✓ USER"
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
        
        # Find optimal condition from data based on optimization direction
        clean_data = self.handler.clean_data

        # Get direction for this response (default to maximize for backward compatibility)
        response_direction = self.response_directions.get(self.handler.response_column, 'maximize')

        print(f"[DEBUG BEST] Finding best experiment for '{self.handler.response_column}'")
        print(f"  - Direction: {response_direction}")
        print(f"  - Value range: {clean_data[self.handler.response_column].min()} to {clean_data[self.handler.response_column].max()}")
        print(f"  - Constraints: {self.response_constraints.get(self.handler.response_column, 'None')}")

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

            print(f"  - After constraint filtering: {len(filtered_data)}/{original_count} experiments remain")

        self.recommendations_text.insert(tk.END, "="*80 + "\n")
        self.recommendations_text.insert(tk.END, "BEST OBSERVED EXPERIMENT\n")
        self.recommendations_text.insert(tk.END, "-"*80 + "\n")

        # Check if any experiments meet constraints
        if constraint_applied and len(filtered_data) == 0:
            self.recommendations_text.insert(tk.END, f"⚠️  WARNING: No experiments meet the constraint {self.handler.response_column}{constraint_msg}\n\n")
            self.recommendations_text.insert(tk.END, f"Your data range: {clean_data[self.handler.response_column].min():.2f} to {clean_data[self.handler.response_column].max():.2f}\n")
            self.recommendations_text.insert(tk.END, f"Constraint requires: {self.handler.response_column}{constraint_msg}\n\n")
            self.recommendations_text.insert(tk.END, "Showing best experiment WITHOUT constraint applied:\n\n")
            filtered_data = clean_data.copy()
            constraint_applied = False

        # Find best based on direction
        if response_direction == 'minimize':
            best_idx = filtered_data[self.handler.response_column].idxmin()
            print(f"  - Using idxmin() for minimize")
        else:
            best_idx = filtered_data[self.handler.response_column].idxmax()
            print(f"  - Using idxmax() for maximize")

        optimal_response = filtered_data.loc[best_idx, self.handler.response_column]
        print(f"  - Best value: {optimal_response} at index {best_idx}")

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
                self.recommendations_text.insert(tk.END, f"  • {factor:<30}: {value:.2f}\n")
            else:
                self.recommendations_text.insert(tk.END, f"  • {factor:<30}: {value}\n")

        self.recommendations_text.insert(tk.END, f"\nResult:\n  • {self.handler.response_column:<30}: {optimal_response:.2f} ({response_direction})\n")
        
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
                    self.recommendations_text.insert(tk.END, f"  • {clean_factor:<30}: {direction}  (effect: {coef:+.4f})\n")
            
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

            try:
                # Initialize optimizer with current data (multi-response support)
                exploration_mode = self.exploration_mode_var.get() if hasattr(self, 'exploration_mode_var') else False

                print(f"[DEBUG ANALYSIS] Setting optimizer data:")
                print(f"  - Response columns: {self.selected_responses}")
                print(f"  - Response directions: {self.response_directions}")
                print(f"  - Response constraints: {self.response_constraints}")
                print(f"  - Exploration mode: {exploration_mode}")

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

                print(f"[DEBUG ANALYSIS] After set_data:")
                print(f"  - Optimizer.response_directions: {self.optimizer.response_directions}")
                print(f"  - Optimizer.response_constraints: {self.optimizer.response_constraints}")
                print(f"  - Optimizer.is_multi_objective: {self.optimizer.is_multi_objective}")

                self.optimizer.initialize_optimizer()

                # Show constraint information
                if self.response_constraints:
                    self.recommendations_text.insert(tk.END, "\nActive Constraints:\n")
                    for response, constraint in self.response_constraints.items():
                        parts = []
                        if 'min' in constraint:
                            parts.append(f"{constraint['min']} ≤ {response}")
                        if 'max' in constraint:
                            parts.append(f"{response} ≤ {constraint['max']}")
                        if parts:
                            self.recommendations_text.insert(tk.END, f"  • {' and '.join(parts)}\n")

                    if exploration_mode:
                        self.recommendations_text.insert(tk.END, "\n⚠️  Exploration Mode: ON\n")
                        self.recommendations_text.insert(tk.END, "   Some suggestions may violate constraints to discover unexpected solutions.\n\n")
                    else:
                        self.recommendations_text.insert(tk.END, "\n✓ All suggestions will respect these constraints.\n\n")
                else:
                    self.recommendations_text.insert(tk.END, "\nNo constraints applied.\n\n")

                self.recommendations_text.insert(tk.END, "\nIntelligently suggested next experiments (balancing exploration & exploitation):\n\n")

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

                # Add Pareto frontier analysis for multi-objective
                if self.optimizer.is_multi_objective:
                    print(f"\n[DEBUG DISPLAY] Displaying Pareto frontier...")
                    print(f"  response_directions at display time: {self.response_directions}")

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

                            status = "✓" if not violations else "⚠️"

                            # Display header with ID
                            header = f"Pareto Point #{i}: {status}"
                            if point.get('id') is not None:
                                header += f" (ID: {point['id']})"
                            self.recommendations_text.insert(tk.END, header + "\n")

                            if i == 1:  # Debug first point
                                print(f"\n[DEBUG DISPLAY] First Pareto point:")
                                print(f"  ID: {point.get('id')}")
                                print(f"  Objectives: {point['objectives']}")
                                print(f"  Checking directions for each objective:")

                            # Display experimental conditions
                            if point.get('parameters'):
                                self.recommendations_text.insert(tk.END, "  Experimental Conditions:\n")
                                for param_name, param_value in point['parameters'].items():
                                    if isinstance(param_value, float):
                                        self.recommendations_text.insert(tk.END, f"    • {param_name}: {param_value:.2f}\n")
                                    else:
                                        self.recommendations_text.insert(tk.END, f"    • {param_name}: {param_value}\n")

                            # Display objectives
                            self.recommendations_text.insert(tk.END, "  Results:\n")
                            for resp, value in point['objectives'].items():
                                direction = self.response_directions.get(resp, 'maximize')
                                if i == 1:  # Debug first point
                                    print(f"    {resp}: direction={direction}")
                                arrow = '↑' if direction == 'maximize' else '↓'
                                direction_text = '(maximize)' if direction == 'maximize' else '(minimize)'
                                self.recommendations_text.insert(tk.END, f"    {arrow} {resp:<25}: {value:.4f} {direction_text}\n")

                            if violations:
                                self.recommendations_text.insert(tk.END, "  ⚠️  Constraint Violations:\n")
                                for violation in violations:
                                    self.recommendations_text.insert(tk.END, f"    • {violation}\n")

                            self.recommendations_text.insert(tk.END, "\n")

                        if len(pareto_points) > 5:
                            self.recommendations_text.insert(tk.END,
                                f"  ... and {len(pareto_points) - 5} more Pareto-optimal points\n\n")

                        self.recommendations_text.insert(tk.END,
                            "💡 TRADE-OFF INSIGHT:\n"
                            "   No single solution is best for ALL objectives.\n"
                            "   Choose a Pareto point based on your priorities.\n"
                            "   View 'Optimization Details' tab for Pareto frontier visualization.\n\n")

                self.recommendations_text.insert(tk.END,
                    "💡 TIP: These suggestions use machine learning to predict where to test next.\n"
                    "   View 'Optimization Details' tab for visualization of predicted response surface.\n"
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

    def _create_optimization_export_button(self):
        """Create appropriate export button based on optimization type"""
        if not AX_AVAILABLE or not hasattr(self, 'export_frame_opt'):
            return

        for widget in self.export_frame_opt.winfo_children():
            widget.destroy()

        if self.optimizer.is_multi_objective:
            self.export_pareto_button = ttk.Button(
                self.export_frame_opt,
                text="📊 Pareto Plots",
                command=self.export_pareto_plots_gui,
                state='normal'
            )
            self.export_pareto_button.pack(side='left', padx=5)
        else:
            self.export_bo_plots_button = ttk.Button(
                self.export_frame_opt,
                text="📊 BO Plots",
                command=self.export_bo_plots_gui,
                state='normal'
            )
            self.export_bo_plots_button.pack(side='left', padx=5)

    def display_optimization_plot(self):
        """Display Bayesian Optimization predicted response surface or Pareto frontier"""
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
            print(f"❌ DISPLAY OPTIMIZATION PLOT ERROR")
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

                # Extract filename and directory for clean message
                filename = os.path.basename(final_path)
                directory = os.path.dirname(final_path)

                messagebox.showinfo("Export Complete",
                    f"File saved:\n\n"
                    f"    {filename}\n\n"
                    f"Location:\n"
                    f"    {directory}")
            except Exception as e:
                messagebox.showerror("Export Failed",
                    f"Could not export statistics to Excel.\n\n"
                    f"Error: {str(e)}\n\n"
                    f"Check that you have write permissions for the selected location.")
    
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

            messagebox.showinfo("Export Complete",
                              f"Files saved:\n\n"
                              f"    {base_path}_MainEffects_{date_str}.{file_format}\n"
                              f"    {base_path}_Interactions_{date_str}.{file_format}\n"
                              f"    {base_path}_Residuals_{date_str}.{file_format}\n\n"
                              f"Location:\n"
                              f"    {directory}")
        except Exception as e:
            messagebox.showerror("Export Failed",
                f"Could not export plots to image files.\n\n"
                f"Error: {str(e)}\n\n"
                f"Check that you have write permissions for the selected location.")
    
    def export_bo_batch(self):
        """Export BO suggestions to Excel and Opentrons CSV"""
        if not AX_AVAILABLE or not self.optimizer or not self.optimizer.is_initialized:
            messagebox.showerror("Bayesian Optimization Unavailable",
                "Bayesian Optimization is not available or not initialized.\n\n"
                "Make sure you have installed ax-platform:\\n"
                "pip install ax-platform\n\n"
                "Then run the analysis to initialize the optimizer.")
            return
        
        if not self.filepath:
            messagebox.showerror("No Data Loaded",
                "No Excel file has been loaded.\n\n"
                "Please load your experimental data before exporting BO suggestions.")
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

                    # Extract filenames and directory for clean message
                    xlsx_filename = os.path.basename(xlsx_path)
                    csv_filename = os.path.basename(csv_path)
                    directory = os.path.dirname(xlsx_path)

                    messagebox.showinfo("Export Complete",
                        f"Files saved:\n\n"
                        f"    {xlsx_filename}\n"
                        f"    {csv_filename}\n\n"
                        f"Location:\n"
                        f"    {directory}")
                else:
                    messagebox.showerror("Export Failed",
                        "Could not export BO batch to files.\n\n"
                        "Check the console output for detailed error information.")

            except Exception as e:
                messagebox.showerror("Export Failed",
                    f"Could not export BO batch to files.\n\n"
                    f"Error: {str(e)}\n\n"
                    f"Check that all parameters are valid and you have write permissions.")

        ttk.Button(button_frame, text="Export", command=do_export).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side='left', padx=5)

    def export_bo_plots_gui(self):
        """GUI wrapper for exporting BO plots"""
        if not AX_AVAILABLE or not self.optimizer or not self.optimizer.is_initialized:
            messagebox.showerror("Bayesian Optimization Unavailable",
                "Bayesian Optimization is not available or not initialized.\n\n"
                "Make sure you have installed ax-platform:\\n"
                "pip install ax-platform\n\n"
                "Then run the analysis to initialize the optimizer.")
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
                # Build file list with proper formatting
                filenames = "\n".join([f"    {os.path.basename(f)}" for f in exported_files])
                messagebox.showinfo("Export Complete",
                                  f"Files saved:\n\n"
                                  f"{filenames}\n\n"
                                  f"Location:\n"
                                  f"    {directory}")
            else:
                messagebox.showwarning("Warning", "No plots were exported. Check console for details.")

        except Exception as e:
            messagebox.showerror("Export Failed",
                f"Could not export BO plots to image files.\n\n"
                f"Error: {str(e)}\n\n"
                f"Check that you have write permissions for the selected location.")

    def export_pareto_plots_gui(self):
        """GUI wrapper for exporting Pareto frontier plots"""
        if not AX_AVAILABLE or not self.optimizer or not self.optimizer.is_initialized:
            messagebox.showerror("Bayesian Optimization Unavailable",
                "Bayesian Optimization is not available or not initialized.\n\n"
                "Make sure you have installed ax-platform:\n"
                "pip install ax-platform\n\n"
                "Then run the analysis to initialize the optimizer.")
            return

        if not self.optimizer.is_multi_objective:
            messagebox.showwarning("Not Multi-Objective",
                "Pareto plots are only available for multi-objective optimization.\n\n"
                "This analysis has only one response variable.")
            return

        format_options = self._ask_plot_format()
        if not format_options['format']:
            return

        file_format = format_options['format']
        dpi = format_options['dpi']

        format_map = {
            'png': ("PNG files", "*.png"),
            'tiff': ("TIFF files", "*.tiff"),
            'pdf': ("PDF files", "*.pdf"),
            'eps': ("EPS files", "*.eps")
        }

        filepath = filedialog.asksaveasfilename(
            defaultextension=f".{file_format}",
            filetypes=[format_map[file_format], ("All files", "*.*")],
            initialfile=f"ParetoFrontier.{file_format}",
            title="Save Pareto Frontier Plot"
        )

        if not filepath:
            return

        try:
            fig = self.optimizer.plot_pareto_frontier()

            if fig is None:
                messagebox.showwarning("Export Failed",
                    "Could not generate Pareto frontier plot.\n\n"
                    "Pareto visualization requires 2-3 objectives.")
                return

            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)

            messagebox.showinfo("Export Complete",
                f"Pareto frontier plot saved:\n\n"
                f"{os.path.basename(filepath)}\n\n"
                f"Location:\n"
                f"{os.path.dirname(filepath)}")

        except Exception as e:
            messagebox.showerror("Export Failed",
                f"Could not export Pareto plot to file.\n\n"
                f"Error: {str(e)}\n\n"
                f"Check that you have write permissions for the selected location.")

    def load_results(self):
        """Load results from Excel (called from main window menu)"""
        self.browse_file()

    def refresh(self):
        """Refresh UI from project data (called when switching tabs)"""
        # Reload display if data exists
        if self.handler.data is not None:
            self._update_display()
