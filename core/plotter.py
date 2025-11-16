"""
DoE Plotting Functions
Extracted from analysis_tab.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import stats as scipy_stats
from typing import Optional
from core.constants import (
    PLOT_DPI, PLOT_PAD, SUBPLOT_COLS_MAX,
    PLOT_LINEWIDTH, PLOT_LINEWIDTH_INTERACTION,
    PLOT_MARKERSIZE, PLOT_MARKERSIZE_INTERACTION,
    PLOT_ALPHA_FILL, PLOT_ALPHA_LINE, PLOT_ALPHA_SCATTER,
    PLOT_ALPHA_GRID, PLOT_ALPHA_HISTOGRAM,
    HISTOGRAM_BINS, EDGE_LINE_WIDTH, HISTOGRAM_LINE_WIDTH,
    MAX_FACTORS_INTERACTION, SUBPLOT_SIZE_INTERACTION,
    FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_AXIS,
    FONT_SIZE_LEGEND, FONT_SIZE_INTERACTION_AXIS, FONT_SIZE_INTERACTION_TITLE
)


class DoEPlotter:
    """Publication-quality plotting functions for Design of Experiments analysis"""

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

    def __init__(self) -> None:
        """Initialize plotter with empty data"""
        self.data: Optional[pd.DataFrame] = None
        self.factor_columns: list = []
        self.response_column: Optional[str] = None

    def set_data(self, data: pd.DataFrame, factor_columns: list, response_column: str) -> None:
        """
        Set data for plotting

        Args:
            data: DataFrame containing experimental data
            factor_columns: List of column names that are factors
            response_column: Name of the response variable column
        """
        self.data = data
        self.factor_columns = factor_columns
        self.response_column = response_column

    @staticmethod
    def _save_plot(fig: Figure, save_path: Optional[str]) -> None:
        """
        Save plot to file with consistent settings

        Args:
            fig: Matplotlib figure to save
            save_path: Path where plot should be saved (if provided)
        """
        if save_path:
            plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')

    def plot_main_effects(self, save_path: Optional[str] = None) -> Figure:
        """
        Create main effects plot showing mean response per factor level

        Plots mean ± 1 standard deviation for each factor level.
        Useful for identifying which factor levels produce highest/lowest responses.

        Args:
            save_path: Optional path to save the figure

        Returns:
            Matplotlib Figure object
        """
        num_factors = len(self.factor_columns)
        ncols = min(SUBPLOT_COLS_MAX, num_factors)
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

            ax.plot(range(len(levels)), means, 'o-',
                   linewidth=PLOT_LINEWIDTH, markersize=PLOT_MARKERSIZE,
                   color=self.COLORS['primary'])
            # Shaded region shows ± 1 std dev
            ax.fill_between(range(len(levels)),
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=PLOT_ALPHA_FILL, color=self.COLORS['primary'])

            ax.set_xlabel(factor, fontsize=FONT_SIZE_LABEL, fontweight='bold')
            ax.set_ylabel('Mean Response', fontsize=FONT_SIZE_LABEL, fontweight='bold')
            ax.set_title(f'Main Effect: {factor}', fontsize=FONT_SIZE_TITLE, fontweight='bold')
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(levels, rotation=45, ha='right')
            ax.grid(True, alpha=PLOT_ALPHA_GRID)

        for idx in range(num_factors, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout(pad=PLOT_PAD)
        self._save_plot(fig, save_path)

        return fig

    def plot_interaction_effects(self, save_path: Optional[str] = None,
                                max_factors: int = MAX_FACTORS_INTERACTION) -> Optional[Figure]:
        """
        Create interaction plot matrix showing factor interactions

        Displays diagonal main effects and lower-triangle interaction plots.
        Helps identify if factors interact (non-parallel lines indicate interaction).

        Args:
            save_path: Optional path to save the figure
            max_factors: Maximum number of factors to include (default from constants)

        Returns:
            Matplotlib Figure object, or None if less than 2 factors
        """
        factors_to_plot = self.factor_columns[:max_factors]
        num_factors = len(factors_to_plot)

        if num_factors < 2:
            return None

        fig, axes = plt.subplots(num_factors, num_factors,
                                figsize=(SUBPLOT_SIZE_INTERACTION*num_factors,
                                        SUBPLOT_SIZE_INTERACTION*num_factors))

        for i, factor1 in enumerate(factors_to_plot):
            for j, factor2 in enumerate(factors_to_plot):
                ax = axes[i, j]

                # Diagonal: main effects
                if i == j:
                    grouped = self.data.groupby(factor1)[self.response_column].mean()
                    levels = sorted(self.data[factor1].unique())
                    means = [grouped.loc[level] for level in levels]

                    ax.plot(range(len(levels)), means, 'o-',
                           linewidth=PLOT_LINEWIDTH, color=self.COLORS['primary'])
                    ax.set_xticks(range(len(levels)))
                    ax.set_xticklabels(levels, rotation=45, ha='right',
                                      fontsize=FONT_SIZE_INTERACTION_AXIS)

                    if j == 0:
                        ax.set_ylabel('Mean\nResponse', fontsize=FONT_SIZE_AXIS)

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
                        ax.plot(range(len(levels1)), means, 'o-',
                               linewidth=PLOT_LINEWIDTH_INTERACTION,
                               label=f'{factor2}={level2}',
                               alpha=PLOT_ALPHA_LINE, color=color)

                    ax.set_xticks(range(len(levels1)))
                    ax.set_xticklabels(levels1, rotation=45, ha='right',
                                      fontsize=FONT_SIZE_INTERACTION_AXIS)

                    if j == 0:
                        ax.set_ylabel('Mean\nResponse', fontsize=FONT_SIZE_AXIS)

                    if i == 1 and j == 0 and len(levels2) <= 5:
                        ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best')

                # Upper triangle: just label the comparison
                else:
                    ax.text(0.5, 0.5, f'{factor1}\nvs\n{factor2}',
                           ha='center', va='center', fontsize=FONT_SIZE_AXIS,
                           transform=ax.transAxes, style='italic')
                    ax.set_xticks([])
                    ax.set_yticks([])

                ax.grid(True, alpha=PLOT_ALPHA_GRID)

                if i == 0:
                    ax.set_title(factor2, fontsize=FONT_SIZE_INTERACTION_TITLE,
                               fontweight='bold')

                if j == 0 and i > 0:
                    ax.set_ylabel(factor1, fontsize=FONT_SIZE_INTERACTION_TITLE,
                                fontweight='bold', rotation=0,
                                ha='right', va='center', labelpad=20)

        plt.tight_layout(pad=PLOT_PAD)
        self._save_plot(fig, save_path)

        return fig

    def plot_residuals(self, predictions, residuals, save_path: Optional[str] = None) -> Figure:
        """
        Create 4-panel residual diagnostic plots for regression models

        Generates:
        - Residuals vs Fitted (check for non-linearity and heteroscedasticity)
        - Normal Q-Q Plot (check if residuals are normally distributed)
        - Scale-Location Plot (check for homoscedasticity)
        - Histogram of Residuals (check distribution shape)

        Args:
            predictions: Model predicted values
            residuals: Model residuals (observed - predicted)
            save_path: Optional path to save the figure

        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))

        # Residuals vs Fitted
        axes[0, 0].scatter(predictions, residuals,
                          alpha=PLOT_ALPHA_SCATTER,
                          color=self.COLORS['primary'],
                          edgecolors='white',
                          linewidth=EDGE_LINE_WIDTH)
        axes[0, 0].axhline(y=0, color=self.COLORS['warning'],
                          linestyle='--', linewidth=PLOT_LINEWIDTH)
        axes[0, 0].set_xlabel('Fitted Values', fontsize=FONT_SIZE_LABEL)
        axes[0, 0].set_ylabel('Residuals', fontsize=FONT_SIZE_LABEL)
        axes[0, 0].set_title('Residuals vs Fitted', fontsize=FONT_SIZE_TITLE, fontweight='bold')
        axes[0, 0].grid(True, alpha=PLOT_ALPHA_GRID)

        # Q-Q plot (This was just made to have inverted axes to match PRISM style)
        qq_data = scipy_stats.probplot(residuals, dist="norm")
        theoretical_quantiles = qq_data[0][0]  # Y-axis
        ordered_residuals = qq_data[0][1]      # X-axis
        slope = qq_data[1][0]
        intercept = qq_data[1][1]

        # Plot data points with swapped axes
        axes[0, 1].plot(ordered_residuals, theoretical_quantiles, 'o',
                       markersize=PLOT_MARKERSIZE_INTERACTION,
                       color=self.COLORS['primary'],
                       alpha=0.7,
                       markeredgecolor='white',
                       markeredgewidth=EDGE_LINE_WIDTH)

        # Reference line: Since normal relationship is y = slope*x + intercept
        # When we swap axes, the line becomes: theoretical = (actual - intercept) / slope
        # Or simplified: theoretical = (1/slope) * actual - (intercept/slope)
        x_line = np.array([ordered_residuals.min(), ordered_residuals.max()])
        y_line = (x_line - intercept) / slope
        axes[0, 1].plot(x_line, y_line, '-',
                       linewidth=PLOT_LINEWIDTH,
                       color=self.COLORS['warning'])

        axes[0, 1].set_xlabel('Actual residual', fontsize=FONT_SIZE_LABEL)
        axes[0, 1].set_ylabel('Predicted residual', fontsize=FONT_SIZE_LABEL)
        axes[0, 1].set_title('Normal Q-Q Plot', fontsize=FONT_SIZE_TITLE, fontweight='bold')
        axes[0, 1].grid(True, alpha=PLOT_ALPHA_GRID)

        # Scale-Location
        standardized_resid = residuals / residuals.std()
        axes[1, 0].scatter(predictions, np.sqrt(np.abs(standardized_resid)),
                          alpha=PLOT_ALPHA_SCATTER,
                          color=self.COLORS['primary'],
                          edgecolors='white',
                          linewidth=EDGE_LINE_WIDTH)
        axes[1, 0].set_xlabel('Fitted Values', fontsize=FONT_SIZE_LABEL)
        axes[1, 0].set_ylabel('√|Standardized Residuals|', fontsize=FONT_SIZE_LABEL)
        axes[1, 0].set_title('Scale-Location', fontsize=FONT_SIZE_TITLE, fontweight='bold')
        axes[1, 0].grid(True, alpha=PLOT_ALPHA_GRID)

        # Histogram
        axes[1, 1].hist(residuals, bins=HISTOGRAM_BINS,
                       color=self.COLORS['primary'],
                       edgecolor='white',
                       alpha=PLOT_ALPHA_HISTOGRAM,
                       linewidth=HISTOGRAM_LINE_WIDTH)
        axes[1, 1].set_xlabel('Residuals', fontsize=FONT_SIZE_LABEL)
        axes[1, 1].set_ylabel('Frequency', fontsize=FONT_SIZE_LABEL)
        axes[1, 1].set_title('Histogram of Residuals', fontsize=FONT_SIZE_TITLE, fontweight='bold')
        axes[1, 1].grid(True, alpha=PLOT_ALPHA_GRID)

        plt.tight_layout(pad=PLOT_PAD)
        self._save_plot(fig, save_path)

        return fig
