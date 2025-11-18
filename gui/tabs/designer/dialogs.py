"""
Dialog classes for the Designer tab.

This module contains dialog windows used for editing factors and other
interactive elements in the experiment designer interface.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Tuple, Optional

from utils.constants import AVAILABLE_FACTORS
from .models import (
    validate_numeric_input,
    validate_single_numeric_input,
    validate_alphanumeric_input
)


class FactorEditDialog(tk.Toplevel):
    """
    Inline factor editor dialog with mandatory stock concentration.

    This dialog allows users to edit factor levels and stock concentrations
    for experimental design. It supports both numeric and categorical factors
    with appropriate validation for each type.

    Attributes:
        factor_name: The internal name of the factor being edited.
        result_levels: The list of levels after saving, or None if cancelled.
        result_stock: The stock concentration after saving, or None if not applicable.
    """

    def __init__(self, parent, factor_name: str, existing_levels: List[str] = None,
                 stock_conc: Optional[float] = None):
        """
        Initialize the factor edit dialog.

        Args:
            parent: The parent widget.
            factor_name: The internal name of the factor to edit.
            existing_levels: Optional list of existing level values to populate.
            stock_conc: Optional existing stock concentration value.
        """
        super().__init__(parent)

        display_name = AVAILABLE_FACTORS.get(factor_name, factor_name)

        self.title(f"Edit Factor: {display_name}")
        self.geometry("550x600")
        self.transient(parent)
        self.grab_set()

        self.factor_name = factor_name
        self.result_levels = None
        self.result_stock = None

        # Store entry widgets for key binding
        self.stock_entry_widget = None
        self.level_entry_widget = None

        # Validation commands with action code for proper backspace handling
        vcmd = (self.register(validate_numeric_input), '%d', '%S', '%P')  # For levels (allows commas)
        vcmd_stock = (self.register(validate_single_numeric_input), '%d', '%S', '%P')  # For stock (no commas)

        # For detergent and reducing_agent (categorical text), allow alphanumeric input
        # Buffer pH still uses numeric validation (values 1-14)
        if factor_name in ["detergent", "reducing_agent"]:
            vcmd_levels = (self.register(validate_alphanumeric_input), '%d', '%S', '%P')
        else:
            vcmd_levels = vcmd

        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Factor info
        info_frame = ttk.LabelFrame(main_frame, text="Factor Information", padding=8)
        info_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(info_frame, text=f"Factor: {display_name}",
                 font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        # Stock concentration
        if factor_name not in ["buffer pH", "detergent", "reducing_agent"]:  # Categorical factors don't need stock
            stock_frame = ttk.LabelFrame(info_frame, text="Stock Concentration (required)", padding=8)
            stock_frame.pack(fill=tk.X, pady=(8, 0))

            # Concentration entry and unit dropdown
            entry_frame = ttk.Frame(stock_frame)
            entry_frame.pack(fill=tk.X)

            ttk.Label(entry_frame, text="Value:").pack(side=tk.LEFT, padx=(0, 5))
            self.stock_var = tk.StringVar(value=str(stock_conc) if stock_conc else "")
            stock_entry = ttk.Entry(entry_frame, textvariable=self.stock_var, width=12,
                                   validate='key', validatecommand=vcmd_stock)
            stock_entry.pack(side=tk.LEFT, padx=(0, 10))
            stock_entry.bind('<Return>', lambda e: self._try_save())  # Enter key to proceed
            self.stock_entry_widget = stock_entry  # Store for key binding

            ttk.Label(entry_frame, text="Unit:").pack(side=tk.LEFT, padx=(0, 5))

            # Determine unit options based on factor name
            unit_options = self._get_unit_options(factor_name)
            self.unit_var = tk.StringVar(value=unit_options[0])
            unit_dropdown = ttk.Combobox(entry_frame, textvariable=self.unit_var,
                                        values=unit_options, state="readonly", width=8)
            unit_dropdown.pack(side=tk.LEFT)

        # Levels editor
        levels_frame = ttk.LabelFrame(main_frame, text="Levels", padding=8)
        levels_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # Hints - different for categorical vs numeric factors
        if factor_name == "detergent":
            hint_text = "Tip: Enter detergent names (e.g., Tween-20, Triton X-100, None)"
        elif factor_name == "reducing_agent":
            hint_text = "Tip: Enter reducing agent names (e.g., DTT, TCEP, BME, None)"
        elif factor_name == "buffer pH":
            hint_text = "Tip: Enter pH values from 1-14 (e.g., 7.0, 7.5, 8.0)"
        else:
            hint_text = "Tip: Use commas to add multiple values at once (e.g., 2, 4, 6, 8)"

        hint_label = ttk.Label(levels_frame,
                              text=hint_text,
                              foreground="gray", font=("TkDefaultFont", 9, "italic"))
        hint_label.pack(anchor="w", pady=(0, 5))

        # Frame
        quick_frame = ttk.Frame(levels_frame)
        quick_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(quick_frame, text="Add level:").pack(side=tk.LEFT)
        self.level_var = tk.StringVar()
        level_entry = ttk.Entry(quick_frame, textvariable=self.level_var, width=20,
                               validate='key', validatecommand=vcmd_levels)
        level_entry.pack(side=tk.LEFT, padx=5)
        level_entry.bind('<Return>', lambda e: self._on_level_entry_enter())
        self.level_entry_widget = level_entry  # Store for key binding

        ttk.Button(quick_frame, text="Add",
                  command=self._add_level).pack(side=tk.LEFT, padx=2)

        # Listbox
        list_frame = ttk.Frame(levels_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.levels_listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE, height=10)
        self.levels_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bind Backspace and Delete keys to remove selected item
        self.levels_listbox.bind('<BackSpace>', lambda e: self._delete_level())
        self.levels_listbox.bind('<Delete>', lambda e: self._delete_level())

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical",
                                 command=self.levels_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.levels_listbox.configure(yscrollcommand=scrollbar.set)

        # Load existing levels
        if existing_levels:
            for level in existing_levels:
                self.levels_listbox.insert(tk.END, level)

        # Buttons
        btn_frame = ttk.Frame(levels_frame)
        btn_frame.pack(fill=tk.X, pady=(8, 0))

        ttk.Button(btn_frame, text="Delete Selected",
                  command=self._delete_level).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Clear All",
                  command=self._clear_levels).pack(side=tk.LEFT, padx=2)

        # Generate sequence
        seq_frame = ttk.LabelFrame(main_frame, text="Generate Sequence", padding=8)
        seq_frame.pack(fill=tk.X, pady=(0, 8))

        seq_input_frame = ttk.Frame(seq_frame)
        seq_input_frame.pack(fill=tk.X)

        ttk.Label(seq_input_frame, text="From:").pack(side=tk.LEFT)
        self.seq_start = tk.StringVar()
        ttk.Entry(seq_input_frame, textvariable=self.seq_start, width=8,
                 validate='key', validatecommand=vcmd).pack(side=tk.LEFT, padx=5)

        ttk.Label(seq_input_frame, text="To:").pack(side=tk.LEFT, padx=(10, 0))
        self.seq_end = tk.StringVar()
        ttk.Entry(seq_input_frame, textvariable=self.seq_end, width=8,
                 validate='key', validatecommand=vcmd).pack(side=tk.LEFT, padx=5)

        ttk.Label(seq_input_frame, text="Step:").pack(side=tk.LEFT, padx=(10, 0))
        self.seq_step = tk.StringVar()
        ttk.Entry(seq_input_frame, textvariable=self.seq_step, width=8,
                 validate='key', validatecommand=vcmd).pack(side=tk.LEFT, padx=5)

        ttk.Button(seq_input_frame, text="Generate",
                  command=self._generate_sequence).pack(side=tk.LEFT, padx=5)

        # Final buttons
        final_btn_frame = ttk.Frame(main_frame)
        final_btn_frame.pack(fill=tk.X)

        ttk.Button(final_btn_frame, text="Save",
                  command=self._save).pack(side=tk.RIGHT, padx=2)
        ttk.Button(final_btn_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT, padx=2)

        # Bind Enter key globally for the dialog
        def on_enter_key(event):
            # If focus is on level entry, add the level
            if event.widget == self.level_entry_widget:
                return  # Let _on_level_entry_enter handle it
            # If focus is on stock entry, don't save yet
            if hasattr(self, 'stock_entry_widget') and event.widget == self.stock_entry_widget:
                return
            # Otherwise, if we have levels, save the dialog
            if self.levels_listbox.size() > 0:
                self._save()

        self.bind('<Return>', on_enter_key)

        # Set initial focus
        if self.stock_entry_widget:
            # Factor has stock concentration - focus there first
            self.stock_entry_widget.focus()
        else:
            # Buffer pH (no stock) - focus on level entry
            level_entry.focus()

    def _get_unit_options(self, factor_name: str) -> List[str]:
        """
        Get appropriate unit options for a factor.

        Args:
            factor_name: The internal name of the factor.

        Returns:
            A list of unit strings appropriate for the factor type.
        """
        if factor_name in ["buffer_concentration", "nacl", "kcl", "zinc", "magnesium", "calcium", "reducing_agent_concentration"]:
            return ["mM", "M", "µM"]
        elif factor_name in ["glycerol", "dmso", "detergent_concentration"]:
            return ["%", "% v/v", "% w/v"]
        else:
            # Custom factor - allow common units
            return ["mM", "%", "M", "µM", "mg/mL", "µg/mL", "U/mL"]

    def _on_level_entry_enter(self):
        """
        Handle Enter key in level entry field.

        If the entry has text, adds the level. If empty and levels exist,
        saves the dialog.
        """
        level_text = self.level_var.get().strip()

        if level_text:
            # Has text - add the level
            self._add_level()
        else:
            # Empty box - if we have levels, save the dialog
            if self.levels_listbox.size() > 0:
                self._save()

    def _add_level(self):
        """
        Add one or more levels to the listbox.

        Supports comma-separated values for adding multiple levels at once.
        Validates each value and prevents duplicates.
        """
        level_text = self.level_var.get().strip()
        if not level_text:
            return

        # Check for comma-separated values
        if ',' in level_text:
            # Split and add multiple
            parts = [p.strip() for p in level_text.split(',') if p.strip()]
            for part in parts:
                # Parse and validate each part
                parsed_value = self._parse_numeric_value(part)
                if parsed_value is not None:
                    # Validate range (pH, percentage, etc.)
                    is_valid, error_msg = self._validate_range(parsed_value, self.factor_name)
                    if not is_valid:
                        messagebox.showerror("Invalid Value", error_msg)
                        return
                    level_clean = str(parsed_value)
                else:
                    level_clean = part

                if level_clean not in self.levels_listbox.get(0, tk.END):
                    self.levels_listbox.insert(tk.END, level_clean)
        else:
            # Single value - parse and validate
            parsed_value = self._parse_numeric_value(level_text)
            if parsed_value is not None:
                # Validate range (pH, percentage, etc.)
                is_valid, error_msg = self._validate_range(parsed_value, self.factor_name)
                if not is_valid:
                    messagebox.showerror("Invalid Value", error_msg)
                    return
                level_clean = str(parsed_value)
            else:
                level_clean = level_text

            if level_clean not in self.levels_listbox.get(0, tk.END):
                self.levels_listbox.insert(tk.END, level_clean)

        # Clear entry and keep focus
        self.level_var.set("")
        self.level_entry_widget.focus()

    def _delete_level(self):
        """
        Delete the selected level from the listbox.

        Automatically selects the next available item after deletion
        to allow continuous deletion with keyboard.
        """
        selection = self.levels_listbox.curselection()
        if selection:
            index = selection[0]
            self.levels_listbox.delete(index)

            # Auto-select the next available item for continuous deletion
            size = self.levels_listbox.size()
            if size > 0:
                # If we deleted the last item, select the new last item
                if index >= size:
                    self.levels_listbox.selection_set(size - 1)
                else:
                    # Select the item that took the deleted item's place
                    self.levels_listbox.selection_set(index)

    def _clear_levels(self):
        """Clear all levels from the listbox."""
        self.levels_listbox.delete(0, tk.END)

    def _generate_sequence(self):
        """
        Generate a sequence of numeric levels.

        Uses the start, end, and step values from the sequence input fields
        to generate evenly spaced levels. Validates inputs and prevents
        duplicate entries.
        """
        try:
            start = float(self.seq_start.get())
            end = float(self.seq_end.get())
            step = float(self.seq_step.get())

            if step <= 0:
                messagebox.showerror("Invalid Step", "Step must be positive.")
                return

            if start >= end:
                messagebox.showerror("Invalid Range", "Start must be less than End.")
                return

            # Generate sequence
            current = start
            while current <= end:
                level_str = str(current) if current % 1 != 0 else str(int(current))
                if level_str not in self.levels_listbox.get(0, tk.END):
                    self.levels_listbox.insert(tk.END, level_str)
                current += step

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values.")

    def _try_save(self):
        """
        Try to save the dialog.

        Called by Enter key binding on stock entry field.
        """
        self._save()

    def _parse_numeric_value(self, value_str: str) -> Optional[float]:
        """
        Parse a numeric value from string.

        Args:
            value_str: The string to parse.

        Returns:
            The parsed float value, or None if parsing fails.
        """
        try:
            return float(value_str)
        except ValueError:
            return None

    def _validate_range(self, value: float, factor_name: str) -> Tuple[bool, str]:
        """
        Validate that a value is in a reasonable range for the factor.

        Args:
            value: The numeric value to validate.
            factor_name: The internal name of the factor.

        Returns:
            A tuple of (is_valid, error_message). If valid, error_message is empty.
        """
        # pH validation: 1-14
        if factor_name == "buffer pH":
            if value < 1 or value > 14:
                return False, (f"Invalid pH value: {value}\n\n"
                             f"pH must be between 1.0 and 14.0\n"
                             f"(Typical range: 2-12 for most buffers)")

        # Percentage validation: 0-100
        elif factor_name in ["glycerol", "dmso", "detergent_concentration"]:
            if value < 0 or value > 100:
                return False, (f"Invalid percentage: {value}%\n\n"
                             f"Percentage must be between 0 and 100%")

        # Concentration validation: 0-10000 mM
        elif factor_name in ["buffer_concentration", "nacl", "kcl", "zinc", "magnesium", "calcium", "reducing_agent_concentration"]:
            if value < 0:
                return False, f"Concentration cannot be negative: {value}"
            if value > 10000:
                return False, (f"Very high concentration: {value} mM\n\n"
                             f"Please verify this is correct.\n"
                             f"(Typical range: 10-1000 mM)")

        return True, ""

    def _save(self):
        """
        Save the dialog results and close.

        Validates all inputs including:
        - At least one level is defined
        - Stock concentration is provided (for non-categorical factors)
        - All levels are within valid ranges
        - No level exceeds the stock concentration

        Sets result_levels and result_stock before destroying the dialog.
        """
        # Get levels
        levels = list(self.levels_listbox.get(0, tk.END))
        if not levels:
            messagebox.showerror("No Levels", "Please add at least one level.")
            return

        # Get stock concentration only for non-categorical factors
        stock = None
        if self.factor_name not in ["buffer pH", "detergent", "reducing_agent"]:
            stock_text = self.stock_var.get().strip()
            if not stock_text:
                messagebox.showerror("Missing Stock Concentration",
                    "Stock concentration is required for this factor.")
                return

            try:
                stock_base = float(stock_text)
                if stock_base <= 0:
                    messagebox.showerror("Invalid Stock",
                        "Stock concentration must be positive.")
                    return
            except ValueError:
                messagebox.showerror("Invalid Stock",
                    "Stock concentration must be a number.")
                return

            # Get unit and convert to base unit if needed
            unit = self.unit_var.get()

            # Validate based on expected range
            is_valid, error_msg = self._validate_range(stock_base, self.factor_name)
            if not is_valid:
                messagebox.showerror("Invalid Range", error_msg)
                return

            stock = stock_base

            # Check that no level exceeds stock concentration
            invalid_levels = []
            for level in levels:
                level_value = self._parse_numeric_value(level)
                if level_value is None:
                    try:
                        level_value = float(level)
                    except ValueError:
                        continue

                is_valid, error_msg = self._validate_range(level_value, self.factor_name)
                if not is_valid:
                    messagebox.showerror("Invalid Level Range", error_msg)
                    return

                # Check if exceeds stock
                if level_value > stock_base:
                    invalid_levels.append(f"{level} (parsed as {level_value})")

            if invalid_levels:
                messagebox.showerror("Invalid Concentrations",
                    f"The following concentration(s) EXCEED the stock concentration:\n\n"
                    f"Stock: {stock_base} {unit}\n"
                    f"Invalid levels: {', '.join(invalid_levels)}\n\n"
                    f"You cannot make a solution more concentrated than your stock!\n\n"
                    f"Either:\n"
                    f"- Increase stock concentration, OR\n"
                    f"- Reduce the level values")
                return

        # Sort levels numerically if possible
        try:
            levels = sorted(levels, key=lambda x: float(x))
        except ValueError:
            # fallback to string sort if not all numeric
            levels = sorted(levels)

        self.result_levels = levels
        self.result_stock = stock
        self.destroy()
