"""Plot service - converts matplotlib Figures to base64 images"""

import io
import base64

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from core.plotter import DoEPlotter


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
    fig = plotter.plot_main_effects()
    return figure_to_base64(fig)


def generate_interaction_plot(plotter: DoEPlotter) -> str:
    """Generate interaction effects plot and return as base64"""
    fig = plotter.plot_interaction_effects()
    return figure_to_base64(fig)


def generate_residuals_plot(plotter: DoEPlotter) -> str:
    """Generate residuals plot and return as base64"""
    fig = plotter.plot_residuals()
    return figure_to_base64(fig)


def generate_predictions_plot(plotter: DoEPlotter) -> str:
    """Generate predictions vs actual plot and return as base64"""
    fig = plotter.plot_predictions_vs_actual()
    return figure_to_base64(fig)


def generate_response_distribution_plot(plotter: DoEPlotter) -> str:
    """Generate response distribution plot and return as base64"""
    fig = plotter.plot_response_distribution()
    return figure_to_base64(fig)


def generate_qq_plot(plotter: DoEPlotter) -> str:
    """Generate Q-Q plot and return as base64"""
    fig = plotter.plot_qq_plot()
    return figure_to_base64(fig)
