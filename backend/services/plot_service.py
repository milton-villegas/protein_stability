"""Plot service - converts matplotlib Figures to base64 images"""

import io
import base64
import logging
import traceback

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from core.plotter import DoEPlotter

logger = logging.getLogger(__name__)


def figure_to_base64(fig: Figure, fmt: str = "png", dpi: int = 150) -> str:
    """Convert a matplotlib Figure to a base64-encoded data URI"""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/{fmt};base64,{encoded}"


def generate_main_effects_plot(plotter: DoEPlotter) -> str:
    """Generate main effects plot and return as base64"""
    logger.info(f"[PLOT.SVC] main_effects: data={plotter.data is not None}, "
                f"factors={getattr(plotter, 'factor_columns', None)}, "
                f"response={getattr(plotter, 'response_column', None)}")
    try:
        fig = plotter.plot_main_effects()
        return figure_to_base64(fig)
    except Exception as e:
        logger.error(f"[PLOT.SVC] main_effects ERROR: {e}\n{traceback.format_exc()}")
        raise


def generate_interaction_plot(plotter: DoEPlotter) -> str:
    """Generate interaction effects plot and return as base64"""
    logger.info("[PLOT.SVC] interaction_effects")
    try:
        fig = plotter.plot_interaction_effects()
        return figure_to_base64(fig)
    except Exception as e:
        logger.error(f"[PLOT.SVC] interaction ERROR: {e}\n{traceback.format_exc()}")
        raise


def generate_residuals_plot(plotter: DoEPlotter, predictions=None, residuals=None) -> str:
    """Generate residuals plot and return as base64"""
    logger.info("[PLOT.SVC] residuals")
    try:
        if predictions is None or residuals is None:
            raise ValueError("Predictions and residuals required. Run analysis first.")
        fig = plotter.plot_residuals(predictions, residuals)
        return figure_to_base64(fig)
    except Exception as e:
        logger.error(f"[PLOT.SVC] residuals ERROR: {e}\n{traceback.format_exc()}")
        raise


def generate_predictions_plot(plotter: DoEPlotter, predictions=None, residuals=None) -> str:
    """Generate predictions vs actual scatter plot and return as base64"""
    logger.info("[PLOT.SVC] predictions_vs_actual")
    try:
        if predictions is None:
            raise ValueError("Predictions required. Run analysis first.")
        # DoEPlotter doesn't have plot_predictions_vs_actual, build it inline
        actual = predictions + residuals if residuals is not None else predictions
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(actual, predictions, alpha=0.7, edgecolors='white', linewidth=0.5)
        lims = [min(actual.min(), predictions.min()), max(actual.max(), predictions.max())]
        ax.plot(lims, lims, '--', color='red', linewidth=1)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Predictions vs Actual')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return figure_to_base64(fig)
    except Exception as e:
        logger.error(f"[PLOT.SVC] predictions ERROR: {e}\n{traceback.format_exc()}")
        raise


def generate_response_distribution_plot(plotter: DoEPlotter) -> str:
    """Generate response distribution histogram and return as base64"""
    logger.info("[PLOT.SVC] response_distribution")
    try:
        if plotter.data is None or plotter.response_column is None:
            raise ValueError("No data set. Run analysis first.")
        response_data = plotter.data[plotter.response_column].dropna()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(response_data, bins=15,
                color=plotter.COLORS['primary'],
                edgecolor='white', alpha=0.8, linewidth=0.5)
        ax.set_xlabel(plotter.response_column, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'Distribution of {plotter.response_column}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return figure_to_base64(fig)
    except Exception as e:
        logger.error(f"[PLOT.SVC] distribution ERROR: {e}\n{traceback.format_exc()}")
        raise


def generate_qq_plot(plotter: DoEPlotter, residuals=None) -> str:
    """Generate standalone Q-Q plot (PRISM-style inverted axes) and return as base64"""
    logger.info("[PLOT.SVC] qq_plot")
    try:
        from scipy import stats as scipy_stats
        import numpy as np

        if residuals is None:
            raise ValueError("Residuals required. Run analysis first.")

        qq_data = scipy_stats.probplot(residuals, dist="norm")
        theoretical_quantiles = qq_data[0][0]
        ordered_residuals = qq_data[0][1]
        slope = qq_data[1][0]
        intercept = qq_data[1][1]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(ordered_residuals, theoretical_quantiles, 'o',
                markersize=5, color=plotter.COLORS['primary'],
                alpha=0.7, markeredgecolor='white', markeredgewidth=0.5)

        x_line = np.array([ordered_residuals.min(), ordered_residuals.max()])
        y_line = (x_line - intercept) / slope
        ax.plot(x_line, y_line, '-', linewidth=1.5, color=plotter.COLORS['warning'])

        ax.set_xlabel('Actual residual', fontsize=10)
        ax.set_ylabel('Predicted residual', fontsize=10)
        ax.set_title('Normal Q-Q Plot', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return figure_to_base64(fig)
    except Exception as e:
        logger.error(f"[PLOT.SVC] qq ERROR: {e}\n{traceback.format_exc()}")
        raise
