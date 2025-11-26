#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimental Modeling Suite v0.2.0
Comprehensive Experimental Design & Modeling Platform
Creates advanced experiment designs, fits models, checks assumptions,
and suggests next experiments. Exports to XLSX and Opentrons formats.

Milton F. Villegas

Design Types Available:
  - Full Factorial - All possible combinations
  - Latin Hypercube Sampling - Space-filling designs
  - 2-Level Fractional Factorial - Efficient screening (Resolution III, IV, V)
  - Plackett-Burman - Ultra-efficient screening for many factors
  - Central Composite Design - Response surface optimization
  - Box-Behnken - Response surface without extreme corners
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Tuple

from utils.constants import AVAILABLE_FACTORS

# Optional pyDOE3 for advanced designs
try:
    import pyDOE3
    HAS_PYDOE3 = True
except Exception:
    HAS_PYDOE3 = False

# Optional SMT for optimized LHS
try:
    from smt.sampling_methods import LHS
    HAS_SMT = True
except Exception:
    HAS_SMT = False

# Import models and dialogs
from .models import FactorModel, validate_single_numeric_input
from .dialogs import FactorEditDialog

# Import mixins
from .design_panel import DesignPanelMixin
from .export_panel import ExportPanelMixin


class DesignerTab(DesignPanelMixin, ExportPanelMixin, ttk.Frame):
    """DoE Designer GUI tab for creating experimental designs"""

    # Define categorical factor pairings as class constant
    CATEGORICAL_PAIRS = {
        "buffer pH": "buffer_concentration",
        "detergent": "detergent_concentration",
        "reducing_agent": "reducing_agent_concentration"
    }

    def __init__(self, parent, project, main_window):
        """
        Initialize the Designer Tab.

        Args:
            parent: Parent tkinter widget
            project: Project instance for data persistence
            main_window: Main application window reference
        """
        super().__init__(parent)
        self.project = project
        self.main_window = main_window

        # Use shared project instead of separate FactorModel
        self._build_ui()
        self._update_display()

    def _build_ui(self):
        """Build the complete Designer Tab user interface"""
        # Configure grid
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Main container with reduced padding
        main_container = ttk.Frame(self, padding=8)
        main_container.grid(row=0, column=0, sticky="nsew")
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=2)
        main_container.rowconfigure(0, weight=1)
        main_container.rowconfigure(1, weight=0)

        # Left panel: Factor Selection
        left_panel = ttk.Frame(main_container)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(0, weight=0)
        left_panel.rowconfigure(1, weight=1)

        # Quick actions: Custom factor
        quick_frame = ttk.LabelFrame(left_panel, text="Quick Actions", padding=8)
        quick_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        ttk.Button(quick_frame, text="Add Custom Factor",
                  command=self._add_custom_factor).pack(fill=tk.X, pady=2)
        ttk.Button(quick_frame, text="Clear All",
                  command=self._clear_all).pack(fill=tk.X, pady=2)

        # Available factors
        factors_frame = ttk.LabelFrame(left_panel, text="Available Factors", padding=8)
        factors_frame.grid(row=1, column=0, sticky="nsew")
        factors_frame.columnconfigure(0, weight=1)
        factors_frame.rowconfigure(1, weight=1)

        ttk.Label(factors_frame, text="Double-click to add:",
                 font=("TkDefaultFont", 9, "italic")).grid(row=0, column=0, sticky="w", pady=(0, 4))

        self.available_listbox = tk.Listbox(factors_frame, exportselection=False)
        self.available_listbox.grid(row=1, column=0, sticky="nsew")
        self.available_listbox.bind('<Double-Button-1>', lambda e: self._quick_add_factor())

        scrollbar = ttk.Scrollbar(factors_frame, orient="vertical",
                                 command=self.available_listbox.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.available_listbox.configure(yscrollcommand=scrollbar.set)

        # Populate available factors - organized by category
        factor_categories = [
            ("--- BUFFER SYSTEM ---", ["buffer pH", "buffer_concentration"]),
            ("--- DETERGENTS ---", ["detergent", "detergent_concentration"]),
            ("--- REDUCING AGENTS ---", ["reducing_agent", "reducing_agent_concentration"]),
            ("--- SALTS ---", ["nacl", "kcl"]),
            ("--- METALS ---", ["zinc", "magnesium", "calcium"]),
            ("--- ADDITIVES ---", ["glycerol", "dmso"])
        ]

        for category_name, factor_keys in factor_categories:
            self.available_listbox.insert(tk.END, category_name)
            for key in factor_keys:
                display_name = AVAILABLE_FACTORS.get(key, key)
                self.available_listbox.insert(tk.END, f"  {display_name}")
            self.available_listbox.insert(tk.END, "")

        # Right panel: Current Design
        right_panel = ttk.Frame(main_container)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=0)

        # Design table
        design_frame = ttk.LabelFrame(right_panel, text="Current Design", padding=8)
        design_frame.grid(row=0, column=0, sticky="nsew")
        design_frame.columnconfigure(0, weight=1)
        design_frame.rowconfigure(0, weight=1)

        # Treeview for factors
        self.tree = ttk.Treeview(design_frame,
                                columns=("factor", "levels", "count", "stock"),
                                show="headings", height=10, selectmode="browse")
        self.tree.heading("factor", text="Factor")
        self.tree.heading("levels", text="Levels")
        self.tree.heading("count", text="# Levels")
        self.tree.heading("stock", text="Stock Conc")

        self.tree.column("factor", width=130, anchor="w")
        self.tree.column("levels", width=250, anchor="w")
        self.tree.column("count", width=70, anchor="center")
        self.tree.column("stock", width=90, anchor="center")

        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind('<Double-Button-1>', lambda e: self._edit_factor())

        tree_scroll = ttk.Scrollbar(design_frame, orient="vertical", command=self.tree.yview)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=tree_scroll.set)

        # Instructions
        ttk.Label(design_frame, text="Double-click a factor to edit",
                 font=("TkDefaultFont", 8, "italic")).grid(row=1, column=0, sticky="w", pady=(4, 0))

        # Factor controls
        ctrl_frame = ttk.Frame(design_frame)
        ctrl_frame.grid(row=2, column=0, sticky="ew", pady=(6, 0))

        ttk.Button(ctrl_frame, text="Edit",
                  command=self._edit_factor).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="Delete",
                  command=self._delete_factor).pack(side=tk.LEFT, padx=2)

        # Design Type Selection
        design_type_frame = ttk.LabelFrame(right_panel, text="Design Type", padding=8)
        design_type_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        design_type_frame.columnconfigure(0, weight=1)

        # Dropdown for design type
        dropdown_frame = ttk.Frame(design_type_frame)
        dropdown_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(dropdown_frame, text="Select Design:").pack(side=tk.LEFT, padx=(0, 5))

        self.design_type_var = tk.StringVar(value="full_factorial")

        design_options = [
            ("full_factorial", "Full Factorial (all combinations)"),
            ("lhs", "Latin Hypercube (space-filling)"),
            ("d_optimal", "D-Optimal (model-optimized)"),
            ("fractional", "2-Level Fractional Factorial (screening)"),
            ("plackett_burman", "Plackett-Burman (efficient screening)"),
            ("central_composite", "Central Composite (optimization)"),
            ("box_behnken", "Box-Behnken (optimization)")
        ]

        self.design_dropdown = ttk.Combobox(dropdown_frame,
                                           textvariable=self.design_type_var,
                                           values=[desc for _, desc in design_options],
                                           state="readonly", width=40)
        self.design_dropdown.pack(side=tk.LEFT)
        self.design_dropdown.current(0)
        self.design_dropdown.bind('<<ComboboxSelected>>', lambda e: self._on_design_type_changed())

        # Map display names back to internal values
        self.design_map = {desc: val for val, desc in design_options}
        self.design_map_reverse = {val: desc for val, desc in design_options}

        if not HAS_PYDOE3:
            warning_label = ttk.Label(dropdown_frame, text="pyDOE3 required for advanced designs",
                                     foreground="red", font=("TkDefaultFont", 8))
            warning_label.pack(side=tk.LEFT, padx=(10, 0))

        # Container for design-specific controls
        self.design_controls_frame = ttk.Frame(design_type_frame)
        self.design_controls_frame.pack(fill=tk.X, pady=(5, 0))

        # LHS-specific controls
        self.lhs_controls = ttk.Frame(self.design_controls_frame)

        sample_frame = ttk.Frame(self.lhs_controls)
        sample_frame.pack(fill=tk.X)

        ttk.Label(sample_frame, text="Sample Size:").pack(side=tk.LEFT, padx=(0, 5))
        self.sample_size_var = tk.IntVar(value=96)
        vcmd = (self.register(validate_single_numeric_input), '%d', '%S', '%P')
        ttk.Entry(sample_frame, textvariable=self.sample_size_var,
                 width=8, validate='key', validatecommand=vcmd).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(sample_frame, text="(8-384)",
                 font=("TkDefaultFont", 8, "italic")).pack(side=tk.LEFT)

        optimize_frame = ttk.Frame(self.lhs_controls)
        optimize_frame.pack(fill=tk.X, pady=(5, 0))

        self.optimize_lhs_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(optimize_frame,
                       text="Optimize LHS (SMT - better space filling)",
                       variable=self.optimize_lhs_var).pack(side=tk.LEFT)

        if not HAS_SMT:
            ttk.Label(optimize_frame, text="(requires SMT)",
                     foreground="orange", font=("TkDefaultFont", 8)).pack(side=tk.LEFT, padx=5)

        # D-Optimal controls
        self.d_optimal_controls = ttk.Frame(self.design_controls_frame)

        d_opt_sample_frame = ttk.Frame(self.d_optimal_controls)
        d_opt_sample_frame.pack(fill=tk.X)

        ttk.Label(d_opt_sample_frame, text="Sample Size:").pack(side=tk.LEFT, padx=(0, 5))
        self.d_optimal_sample_var = tk.IntVar(value=24)
        vcmd_d_opt = (self.register(validate_single_numeric_input), '%d', '%S', '%P')
        ttk.Entry(d_opt_sample_frame, textvariable=self.d_optimal_sample_var,
                 width=8, validate='key', validatecommand=vcmd_d_opt).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(d_opt_sample_frame, text="(exact number of runs)",
                 font=("TkDefaultFont", 8, "italic")).pack(side=tk.LEFT)

        d_opt_model_frame = ttk.Frame(self.d_optimal_controls)
        d_opt_model_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(d_opt_model_frame, text="Model Type:").pack(side=tk.LEFT, padx=(0, 5))
        self.d_optimal_model_var = tk.StringVar(value="quadratic")
        d_opt_model_combo = ttk.Combobox(d_opt_model_frame, textvariable=self.d_optimal_model_var,
                                        values=["linear", "interactions", "quadratic"],
                                        state="readonly", width=12)
        d_opt_model_combo.pack(side=tk.LEFT, padx=(0, 10))
        d_opt_model_combo.current(2)

        ttk.Label(d_opt_model_frame, text="(linear=main effects, interactions=+pairs, quadratic=+curves)",
                 font=("TkDefaultFont", 8, "italic"), foreground="gray").pack(side=tk.LEFT)

        # Fractional Factorial controls
        self.fractional_controls = ttk.Frame(self.design_controls_frame)

        resolution_frame = ttk.Frame(self.fractional_controls)
        resolution_frame.pack(fill=tk.X)

        ttk.Label(resolution_frame, text="Resolution:").pack(side=tk.LEFT, padx=(0, 5))
        self.resolution_var = tk.StringVar(value="IV")
        resolution_combo = ttk.Combobox(resolution_frame, textvariable=self.resolution_var,
                                       values=["III", "IV", "V"], state="readonly", width=8)
        resolution_combo.pack(side=tk.LEFT, padx=(0, 10))
        resolution_combo.current(1)

        ttk.Label(resolution_frame, text="(III=screening, IV=interactions, V=full)",
                 font=("TkDefaultFont", 8, "italic"), foreground="gray").pack(side=tk.LEFT)

        # Plackett-Burman controls
        self.pb_controls = ttk.Frame(self.design_controls_frame)

        pb_info = ttk.Label(self.pb_controls,
                           text="Automatically determines optimal run count based on number of factors",
                           font=("TkDefaultFont", 8, "italic"), foreground="gray")
        pb_info.pack(anchor="w")

        # Central Composite controls
        self.ccd_controls = ttk.Frame(self.design_controls_frame)

        alpha_frame = ttk.Frame(self.ccd_controls)
        alpha_frame.pack(fill=tk.X)

        ttk.Label(alpha_frame, text="Design Type:").pack(side=tk.LEFT, padx=(0, 5))
        self.ccd_type_var = tk.StringVar(value="faced")
        ccd_combo = ttk.Combobox(alpha_frame, textvariable=self.ccd_type_var,
                                values=["faced", "inscribed", "circumscribed"],
                                state="readonly", width=15)
        ccd_combo.pack(side=tk.LEFT, padx=(0, 10))
        ccd_combo.current(0)

        ttk.Label(alpha_frame, text="(faced=practical, inscribed=scaled, circumscribed=extended)",
                 font=("TkDefaultFont", 8, "italic"), foreground="gray").pack(side=tk.LEFT)

        # Box-Behnken controls
        self.bb_controls = ttk.Frame(self.design_controls_frame)

        bb_info = ttk.Label(self.bb_controls,
                           text="Requires 3+ factors. Does not test extreme corner combinations.",
                           font=("TkDefaultFont", 8, "italic"), foreground="gray")
        bb_info.pack(anchor="w")

        # Hide all controls initially
        self._hide_all_design_controls()

        # Bottom panel: Export Settings
        bottom_panel = ttk.LabelFrame(main_container, text="Export", padding=8)
        bottom_panel.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        bottom_panel.columnconfigure(1, weight=1)

        # Final volume
        ttk.Label(bottom_panel, text="Final Volume (uL):").grid(row=0, column=0, sticky="w")
        self.final_vol_var = tk.StringVar(value="100")
        vcmd = (self.register(validate_single_numeric_input), '%d', '%S', '%P')
        ttk.Entry(bottom_panel, textvariable=self.final_vol_var, width=12,
                 validate='key', validatecommand=vcmd).grid(row=0, column=1, sticky="w", padx=5)

        # Protein concentration inputs
        protein_frame = ttk.Frame(bottom_panel)
        protein_frame.grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))

        ttk.Label(protein_frame, text="Protein Stock (mg/mL):").pack(side=tk.LEFT)
        self.protein_stock_var = tk.StringVar(value="")
        ttk.Entry(protein_frame, textvariable=self.protein_stock_var, width=8,
                 validate='key', validatecommand=vcmd).pack(side=tk.LEFT, padx=(5, 15))

        ttk.Label(protein_frame, text="Final (mg/mL):").pack(side=tk.LEFT)
        self.protein_final_var = tk.StringVar(value="")
        ttk.Entry(protein_frame, textvariable=self.protein_final_var, width=8,
                 validate='key', validatecommand=vcmd).pack(side=tk.LEFT, padx=5)

        # Dynamic protein volume display
        self.protein_vol_var = tk.StringVar(value="")
        self.protein_vol_label = ttk.Label(protein_frame, textvariable=self.protein_vol_var,
                                           font=("TkDefaultFont", 11, "bold"), foreground="#2E7D32")
        self.protein_vol_label.pack(side=tk.LEFT, padx=(10, 0))

        # Add traces to update protein volume display
        self.protein_stock_var.trace_add("write", self._update_protein_volume_display)
        self.protein_final_var.trace_add("write", self._update_protein_volume_display)
        self.final_vol_var.trace_add("write", self._update_protein_volume_display)

        # Status display
        status_frame = ttk.Frame(bottom_panel)
        status_frame.grid(row=0, column=2, sticky="e")

        ttk.Label(status_frame, text="Combinations:").pack(side=tk.LEFT, padx=(0, 3))
        self.combo_var = tk.StringVar(value="0")
        combo_label = ttk.Label(status_frame, textvariable=self.combo_var,
                               font=("TkDefaultFont", 10, "bold"), foreground="#2196F3")
        combo_label.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(status_frame, text="Plates:").pack(side=tk.LEFT, padx=(0, 3))
        self.plates_var = tk.StringVar(value="0")
        plates_label = ttk.Label(status_frame, textvariable=self.plates_var,
                                font=("TkDefaultFont", 10, "bold"), foreground="#4CAF50")
        plates_label.pack(side=tk.LEFT)

        # Export button
        ttk.Button(bottom_panel, text="Export Design",
                  command=self._export_both).grid(row=2, column=0, columnspan=3, pady=(8, 0))

    def _hide_all_design_controls(self):
        """Hide all design-specific control frames"""
        self.lhs_controls.pack_forget()
        self.d_optimal_controls.pack_forget()
        self.fractional_controls.pack_forget()
        self.pb_controls.pack_forget()
        self.ccd_controls.pack_forget()
        self.bb_controls.pack_forget()

    def _update_protein_volume_display(self, *args):
        """Update the protein volume display when inputs change."""
        try:
            stock_str = self.protein_stock_var.get().strip()
            final_str = self.protein_final_var.get().strip()
            vol_str = self.final_vol_var.get().strip()

            if stock_str and final_str and vol_str:
                stock = float(stock_str)
                final = float(final_str)
                final_vol = float(vol_str)

                if stock > 0 and final > 0 and final_vol > 0:
                    # C1*V1 = C2*V2 -> V1 = (C2*V2)/C1
                    protein_vol = round((final * final_vol) / stock, 2)
                    self.protein_vol_var.set(f"= {protein_vol} uL/well")
                    return

            self.protein_vol_var.set("")
        except (ValueError, ZeroDivisionError):
            self.protein_vol_var.set("")

    def _on_design_type_changed(self):
        """Handle design type dropdown change"""
        display_text = self.design_type_var.get()
        if display_text in self.design_map:
            design_type = self.design_map[display_text]
        else:
            design_type = display_text

        self._hide_all_design_controls()

        if design_type == "lhs":
            self.lhs_controls.pack(fill=tk.X, pady=(5, 0))
        elif design_type == "d_optimal":
            self.d_optimal_controls.pack(fill=tk.X, pady=(5, 0))
        elif design_type == "fractional":
            self.fractional_controls.pack(fill=tk.X, pady=(5, 0))
        elif design_type == "plackett_burman":
            self.pb_controls.pack(fill=tk.X, pady=(5, 0))
        elif design_type == "central_composite":
            self.ccd_controls.pack(fill=tk.X, pady=(5, 0))
        elif design_type == "box_behnken":
            self.bb_controls.pack(fill=tk.X, pady=(5, 0))

        # Force GUI to recalculate layout
        self.update_idletasks()

        # For Full Factorial, explicitly shrink the controls frame
        if design_type == "full_factorial":
            # Unpack and repack the frame to force resize
            self.design_controls_frame.pack_forget()
            self.design_controls_frame.pack(fill=tk.X, pady=(5, 0))
            self.update_idletasks()

        self._update_display()

    def _add_custom_factor(self):
        """Add a completely custom factor (e.g., KCl, MgCl2, etc.) with units"""
        dialog = tk.Toplevel(self)
        dialog.title("Add Custom Factor")
        dialog.geometry("450x250")
        dialog.transient(self)
        dialog.grab_set()

        result = {"name": None}

        frame = ttk.Frame(dialog, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Enter factor name:",
                 font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(0, 5))

        ttk.Label(frame, text="Examples: KCl, MgCl2, Enzyme X, Temperature, etc.",
                 font=("TkDefaultFont", 9, "italic"),
                 foreground="gray").pack(anchor="w", pady=(0, 10))

        name_var = tk.StringVar()
        name_entry = ttk.Entry(frame, textvariable=name_var, font=("TkDefaultFont", 11))
        name_entry.pack(fill=tk.X, pady=(0, 15))
        name_entry.focus()

        def save():
            name = name_var.get().strip()
            if not name:
                messagebox.showerror("Invalid Name", "Please enter a factor name.")
                return

            factors = self.project.get_factors()
            if name in factors:
                messagebox.showerror("Duplicate", f"Factor '{name}' already exists.")
                return

            result["name"] = name
            dialog.destroy()

        name_entry.bind('<Return>', lambda e: save())

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(20, 0))

        ttk.Button(btn_frame, text="Cancel",
                  command=dialog.destroy).pack(side=tk.RIGHT, padx=2)
        ttk.Button(btn_frame, text="Continue",
                  command=save).pack(side=tk.RIGHT, padx=2)

        self.wait_window(dialog)

        if result["name"]:
            factor_name = result["name"]
            editor = FactorEditDialog(self, factor_name, parent_model=self.project)
            self.wait_window(editor)

            if editor.result_levels:
                try:
                    # In per-level mode, don't add the concentration factor to the model
                    # The concentrations will be looked up from per_level_concs during export
                    if editor.result_per_level_concs:
                        cat_factor = self._get_categorical_factor_for_conc(factor_name)
                        print(f"[DEBUG ADD] Setting per-level for {cat_factor}: {editor.result_per_level_concs}")
                        self.project.set_per_level_concs(cat_factor, editor.result_per_level_concs)
                        print(f"[DEBUG ADD] Verification - stored data: {self.project.get_per_level_concs(cat_factor)}")
                        # Don't add the concentration factor itself
                    else:
                        # Normal mode: add the factor with levels and stock
                        self.project.add_factor(factor_name, editor.result_levels, editor.result_stock)
                    self._update_display()
                except ValueError as e:
                    messagebox.showerror("Invalid Factor",
                        f"Could not add factor '{factor_name}'.\n\n"
                        f"Error: {str(e)}")

    def _quick_add_factor(self):
        """Quick add from available factors"""
        selection = self.available_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        display_name = self.available_listbox.get(idx)

        if display_name.startswith("---") or display_name.strip() == "":
            return

        display_name = display_name.strip()

        factor_key = None
        for key, disp in AVAILABLE_FACTORS.items():
            if disp == display_name:
                factor_key = key
                break

        if not factor_key:
            return

        factors = self.project.get_factors()
        if factor_key in factors:
            messagebox.showinfo("Already Added",
                f"Factor '{display_name}' is already in your design.")
            return

        # Get existing per-level concs for concentration factors
        per_level_concs = None
        if factor_key in ["detergent_concentration", "reducing_agent_concentration"]:
            cat_factor = self._get_categorical_factor_for_conc(factor_key)
            per_level_concs = self.project.get_per_level_concs(cat_factor)

        editor = FactorEditDialog(self, factor_key, per_level_concs=per_level_concs, parent_model=self.project)
        self.wait_window(editor)

        if editor.result_levels:
            try:
                # In per-level mode, don't add the concentration factor to the model
                # The concentrations will be looked up from per_level_concs during export
                if editor.result_per_level_concs:
                    cat_factor = self._get_categorical_factor_for_conc(factor_key)
                    self.project.set_per_level_concs(cat_factor, editor.result_per_level_concs)
                    # Don't add the concentration factor itself
                else:
                    # Normal mode: add the factor with levels and stock
                    self.project.add_factor(factor_key, editor.result_levels, editor.result_stock)
                self._update_display()
            except ValueError as e:
                messagebox.showerror("Invalid Factor",
                    f"Could not add factor.\n\nError: {str(e)}")

    def _edit_factor(self):
        """Edit selected factor"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select a factor to edit.")
            return

        item = selection[0]
        factor_name = self.tree.item(item, "values")[0]

        factor_key = None
        for key, disp in AVAILABLE_FACTORS.items():
            if disp == factor_name:
                factor_key = key
                break

        if not factor_key:
            factor_key = factor_name

        factors = self.project.get_factors()

        # Check if it's a per-level concentration factor (not in factors but has per_level_concs)
        per_level_concs = None
        if factor_key in ["detergent_concentration", "reducing_agent_concentration"]:
            cat_factor = self._get_categorical_factor_for_conc(factor_key)
            per_level_concs = self.project.get_per_level_concs(cat_factor)

        # If not in factors and no per-level concs, can't edit
        if factor_key not in factors and not per_level_concs:
            return

        # Get existing levels and stock (if in normal mode)
        existing_levels = factors.get(factor_key, [])
        stock_conc = self.project.get_stock_conc(factor_key)

        editor = FactorEditDialog(self, factor_key, existing_levels, stock_conc, per_level_concs, parent_model=self.project)
        self.wait_window(editor)

        if editor.result_levels:
            try:
                cat_factor = self._get_categorical_factor_for_conc(factor_key)

                # Handle per-level mode
                if editor.result_per_level_concs:
                    # Switching to or staying in per-level mode
                    print(f"[DEBUG EDIT] Setting per-level for {cat_factor}: {editor.result_per_level_concs}")
                    self.project.set_per_level_concs(cat_factor, editor.result_per_level_concs)
                    print(f"[DEBUG EDIT] Verification - stored data: {self.project.get_per_level_concs(cat_factor)}")
                    # Remove the concentration factor from model if it exists
                    if factor_key in self.project.get_factors():
                        self.project.remove_factor(factor_key)
                else:
                    # Normal mode: update or add the factor
                    if factor_key in self.project.get_factors():
                        self.project.update_factor(factor_key, editor.result_levels, editor.result_stock)
                    else:
                        self.project.add_factor(factor_key, editor.result_levels, editor.result_stock)

                    # Clear per-level concs if switching from per-level to normal mode
                    if factor_key in ["detergent_concentration", "reducing_agent_concentration"]:
                        if self.project.has_per_level_concs(cat_factor):
                            self.project.clear_per_level_concs(cat_factor)

                self._update_display()
            except ValueError as e:
                messagebox.showerror("Invalid Factor Update",
                    f"Could not update factor '{factor_key}'.\n\nError: {str(e)}")

    def _delete_factor(self):
        """Delete selected factor"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select a factor to delete.")
            return

        item = selection[0]
        factor_name = self.tree.item(item, "values")[0]

        factor_key = None
        for key, disp in AVAILABLE_FACTORS.items():
            if disp == factor_name:
                factor_key = key
                break

        if not factor_key:
            factor_key = factor_name

        result = messagebox.askyesno("Confirm Delete",
            f"Are you sure you want to delete '{factor_name}'?")

        if result:
            self.project.remove_factor(factor_key)

            # Also clear per-level concentrations if deleting a concentration factor
            if factor_key in ["detergent_concentration", "reducing_agent_concentration"]:
                cat_factor = self._get_categorical_factor_for_conc(factor_key)
                if self.project.has_per_level_concs(cat_factor):
                    self.project.clear_per_level_concs(cat_factor)

            self._update_display()

    def _clear_all(self):
        """Clear all factors"""
        if not self.project.get_factors():
            messagebox.showinfo("No Factors", "There are no factors to clear.")
            return

        result = messagebox.askyesno("Confirm Clear All",
            "Are you sure you want to clear all factors?")

        if result:
            self.project.clear()
            self._update_display()

    def _get_categorical_factor_for_conc(self, factor_key: str) -> str:
        """Get the categorical factor name for a concentration factor.

        Args:
            factor_key: The concentration factor key (e.g., "detergent_concentration")

        Returns:
            The categorical factor name (e.g., "detergent")
        """
        if factor_key == "detergent_concentration":
            return "detergent"
        elif factor_key == "reducing_agent_concentration":
            return "reducing_agent"
        return factor_key

    def _update_display(self):
        """Update the treeview and combination counter"""
        for item in self.tree.get_children():
            self.tree.delete(item)

        factors = self.project.get_factors()
        per_level_concs = self.project.get_all_per_level_concs()

        # Display regular factors
        for factor_key, levels in factors.items():
            display_name = AVAILABLE_FACTORS.get(factor_key, factor_key)
            levels_str = ", ".join(str(l) for l in levels[:5])
            if len(levels) > 5:
                levels_str += "..."

            count = len(levels)
            stock = self.project.get_stock_conc(factor_key)

            # Check if this factor has per-level concentrations
            if factor_key == "detergent" and "detergent" in per_level_concs:
                # Show that detergent uses per-level concentrations
                stock_str = "Per-level"
            elif factor_key == "reducing_agent" and "reducing_agent" in per_level_concs:
                # Show that reducing agent uses per-level concentrations
                stock_str = "Per-level"
            else:
                stock_str = f"{stock}" if stock else "N/A"

            self.tree.insert("", tk.END, values=(display_name, levels_str, count, stock_str))

        display_text = self.design_type_var.get()
        if display_text in self.design_map:
            design_type = self.design_map[display_text]
        else:
            design_type = display_text

        if not factors:
            self.combo_var.set("0")
            self.plates_var.set("0")
            return

        n_factors = len(factors)

        try:
            if design_type == "full_factorial":
                total = self.project.total_combinations()
                self.combo_var.set(str(total))

            elif design_type == "lhs":
                sample_size = self.sample_size_var.get()
                self.combo_var.set(f"{sample_size} (LHS)")
                total = sample_size

            elif design_type == "d_optimal":
                sample_size = self.d_optimal_sample_var.get()
                model_type = self.d_optimal_model_var.get()
                self.combo_var.set(f"{sample_size} (D-Opt)")
                total = sample_size

            elif design_type == "fractional":
                resolution = self.resolution_var.get()
                if resolution == "III":
                    runs = 2 ** (n_factors - max(1, n_factors - 4))
                elif resolution == "IV":
                    runs = 2 ** (n_factors - max(0, n_factors - 5))
                else:
                    runs = 2 ** (n_factors - max(0, n_factors - 6))
                self.combo_var.set(f"~{runs} (Frac)")
                total = runs

            elif design_type == "plackett_burman":
                runs = ((n_factors // 4) + 1) * 4
                if runs < 12:
                    runs = 12
                self.combo_var.set(f"~{runs} (PB)")
                total = runs

            elif design_type == "central_composite":
                runs = (2 ** n_factors) + (2 * n_factors) + 4
                self.combo_var.set(f"~{runs} (CCD)")
                total = runs

            elif design_type == "box_behnken":
                if n_factors >= 3:
                    runs = 2 * n_factors * (n_factors - 1) + 3
                    self.combo_var.set(f"~{runs} (BB)")
                    total = runs
                else:
                    self.combo_var.set("N/A (need 3+)")
                    total = 0

            else:
                total = 0
                self.combo_var.set("?")

            if total > 0:
                plates = (total + 95) // 96
                self.plates_var.set(str(plates))
            else:
                self.plates_var.set("0")

        except Exception:
            self.combo_var.set("?")
            self.plates_var.set("?")
