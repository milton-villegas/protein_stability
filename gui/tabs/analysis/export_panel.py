"""
Export Panel Mixin for Analysis Tab

This module contains the ExportPanelMixin class which provides export functionality
for statistics, plots, and Bayesian Optimization results.

Methods included:
- export_statistics: Export statistics to Excel
- _prompt_for_stock_concentrations: Dialog for stock concentrations input
- _ask_plot_format: Dialog for plot format selection
- export_plots: Export analysis plots
- export_bo_batch: Export BO suggestions to Excel and Opentrons CSV
- export_bo_plots_gui: GUI wrapper for BO plot export
- export_pareto_plots_gui: GUI wrapper for Pareto plot export
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from datetime import datetime

import matplotlib.pyplot as plt

# Bayesian Optimization availability check
try:
    from ax.service.ax_client import AxClient, ObjectiveProperties
    AX_AVAILABLE = True
except ImportError:
    AX_AVAILABLE = False


class ExportPanelMixin:
    """
    Mixin class providing export functionality for the Analysis Tab.

    This mixin provides methods for exporting statistics, plots, and
    Bayesian Optimization results to various file formats.

    Expected instance attributes:
        - self.exporter: Statistics exporter object
        - self.plotter: Plot generator object
        - self.optimizer: Bayesian optimizer object
        - self.handler: Data handler object
        - self.filepath: Path to loaded Excel file
        - self.results: Analysis results dictionary
        - self.main_effects: Main effects data
    """

    def export_statistics(self):
        """
        Export statistics to Excel file.

        Saves statistical analysis results to an Excel file with naming convention:
        [UserName]_Statistics_[Date].xlsx

        Uses self.exporter to generate the Excel file with formatted statistics.
        """
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
        """
        Prompt user for stock concentrations when metadata is not available.

        Creates a dialog window with entry fields for each numeric factor,
        pre-populated with sensible defaults based on factor names.

        Returns:
            dict: Dictionary mapping factor names to stock concentration values,
                  or None if user cancels the dialog.
        """
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
        """
        Ask user for plot export format.

        Displays a dialog with options for:
        - File format: PNG, TIFF, PDF, EPS
        - DPI resolution: 150, 300, 600 (for raster formats)

        Returns:
            dict: Dictionary with 'format' (str or None) and 'dpi' (int) keys.
        """
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
        """
        Export plots in user-selected format.

        Exports three main analysis plots:
        - Main Effects plot
        - Interaction Effects plot
        - Residuals plot

        Files are named with pattern: [BaseName]_[PlotType]_[Date].[format]
        """
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
        """
        Export BO suggestions to Excel and Opentrons CSV.

        Creates a dialog for configuring export parameters:
        - Number of suggestions (1-20)
        - Batch number (auto-detected from data)
        - Final volume in microliters
        - Stock concentrations (from metadata or user input)

        Generates two files:
        - Excel file with BO suggestions
        - CSV file for Opentrons robot
        """
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
        ttk.Label(dialog, text="Final Volume (uL):", font=('TkDefaultFont', 10, 'bold')).pack(pady=(15,5))
        vol_var = tk.DoubleVar(value=100.0)
        vol_entry = ttk.Entry(dialog, textvariable=vol_var, width=15)
        vol_entry.pack()

        # Stock concentrations note
        stock_concs_from_metadata = self.handler.get_stock_concentrations()
        if stock_concs_from_metadata:
            ttk.Label(dialog, text="Using stock concentrations from file metadata",
                     font=('TkDefaultFont', 9, 'bold'), foreground='green').pack(pady=(15,5))
        else:
            ttk.Label(dialog, text="No metadata found - will prompt for stock concentrations",
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
        """
        GUI wrapper for exporting BO plots.

        Exports Bayesian Optimization visualization plots in user-selected format.
        Prompts for format/DPI and base filename, then exports all available BO plots.
        """
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
        """
        GUI wrapper for exporting Pareto frontier plots.

        Exports Pareto frontier visualization for multi-objective optimization.
        Only available when optimizer has multiple objectives configured.
        """
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
