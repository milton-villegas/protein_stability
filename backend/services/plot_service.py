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
        import numpy as np
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
    """Generate response distribution plot and return as base64"""
    logger.info("[PLOT.SVC] response_distribution")
    try:
        fig = plotter.plot_response_distribution()
        return figure_to_base64(fig)
    except Exception as e:
        logger.error(f"[PLOT.SVC] distribution ERROR: {e}\n{traceback.format_exc()}")
        raise


def generate_qq_plot(plotter: DoEPlotter) -> str:
    """Generate Q-Q plot and return as base64"""
    logger.info("[PLOT.SVC] qq_plot")
    try:
        fig = plotter.plot_qq_plot()
        return figure_to_base64(fig)
    except Exception as e:
        logger.error(f"[PLOT.SVC] qq ERROR: {e}\n{traceback.format_exc()}")
        raise
