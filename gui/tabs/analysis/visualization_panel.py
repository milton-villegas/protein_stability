"""
Visualization Panel Mixin for Analysis Tab.

This module provides plotting functionality for the analysis tab,
including main effects, interaction effects, and residuals plots.
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class VisualizationPanelMixin:
    """
    Mixin class providing visualization and plotting methods for analysis.

    This mixin contains methods for displaying various statistical plots
    including main effects, interaction effects, and residuals plots.

    Expected attributes on the mixed-in class:
        - plotter: Plotter instance for generating figures
        - handler: Data handler with clean_data, factor_columns, response_column
        - results: Dictionary containing 'predictions' and 'residuals'
        - main_effects_frame: Frame widget for main effects plot
        - interactions_frame: Frame widget for interaction plot
        - residuals_frame: Frame widget for residuals plot
    """

    def display_plots(self):
        """
        Display all analysis plots.

        Sets the data on the plotter and calls individual plot display methods
        for main effects, interactions, and residuals.
        """
        self.plotter.set_data(
            self.handler.clean_data,
            self.handler.factor_columns,
            self.handler.response_column
        )

        self.display_main_effects_plot()
        self.display_interaction_plot()
        self.display_residuals_plot()

    def display_main_effects_plot(self):
        """
        Display the main effects plot.

        Clears existing widgets in the main effects frame, generates
        a new main effects plot, and embeds it in the frame with
        proper mousewheel binding and scroll position reset.
        """
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
        """
        Display the interaction effects plot.

        Clears existing widgets in the interactions frame, generates
        a new interaction effects plot (if available), and embeds it
        in the frame with proper mousewheel binding and scroll position reset.
        """
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
        """
        Display the residuals plot.

        Clears existing widgets in the residuals frame, generates
        a new residuals plot using predictions and residuals from
        the analysis results, and embeds it in the frame with
        proper mousewheel binding and scroll position reset.
        """
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
