"""
DoE Plotting Functions
Extracted from analysis_tab.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from typing import Optional


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

    def plot_main_effects(self, save_path: Optional[str] = None):
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

    def plot_interaction_effects(self, save_path: Optional[str] = None, max_factors: int = 6):
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

    def plot_residuals(self, predictions, residuals, save_path: Optional[str] = None):
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
