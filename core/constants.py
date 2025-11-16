"""
Core module constants for DoE analysis
Extracted magic numbers for better maintainability
"""

# Statistical Analysis Constants
SIGNIFICANCE_LEVEL = 0.05  # Alpha level for hypothesis testing
P_VALUE_PRECISION = 6  # Decimal precision for p-value formatting
BACKWARD_ELIMINATION_THRESHOLD = 0.10  # P-value threshold for backward elimination

# Model Selection Constants
ADJ_R2_WEIGHT = 0.6  # Weight for adjusted R² in model selection
BIC_WEIGHT = 0.3  # Weight for BIC in model selection
COMPLEXITY_PENALTY = 2  # Penalty per degree of freedom

# Model Quality Thresholds
R2_LOW_THRESHOLD = 0.5  # Below this is considered a poor fit
R2_EXCELLENT_THRESHOLD = 0.9  # Above this is considered excellent fit
ADJ_R2_SIMILARITY_THRESHOLD = 0.05  # Max difference for considering models equivalent

# Plotting Constants
PLOT_DPI = 300  # Resolution for saved plots
PLOT_PAD = 0.5  # Padding for tight_layout
SUBPLOT_COLS_MAX = 3  # Maximum columns in subplot grid
PLOT_LINEWIDTH = 2  # Default line width for main plots
PLOT_LINEWIDTH_INTERACTION = 1.5  # Line width for interaction plots
PLOT_MARKERSIZE = 8  # Default marker size
PLOT_MARKERSIZE_INTERACTION = 6  # Marker size for interaction plots
PLOT_ALPHA_FILL = 0.2  # Alpha for shaded regions
PLOT_ALPHA_LINE = 0.85  # Alpha for interaction lines
PLOT_ALPHA_SCATTER = 0.6  # Alpha for scatter plots
PLOT_ALPHA_GRID = 0.3  # Alpha for grid lines
PLOT_ALPHA_HISTOGRAM = 0.8  # Alpha for histogram bars
HISTOGRAM_BINS = 30  # Number of bins for histogram
EDGE_LINE_WIDTH = 0.5  # Edge line width for scatter plots
HISTOGRAM_LINE_WIDTH = 1.2  # Edge line width for histogram

# Interaction Plot Constants
MAX_FACTORS_INTERACTION = 6  # Maximum factors to show in interaction matrix
SUBPLOT_SIZE_INTERACTION = 1.8  # Figure size multiplier per factor

# Font Sizes
FONT_SIZE_TITLE = 12
FONT_SIZE_LABEL = 11
FONT_SIZE_AXIS = 9
FONT_SIZE_LEGEND = 7
FONT_SIZE_INTERACTION_AXIS = 8
FONT_SIZE_INTERACTION_TITLE = 10
