#!/usr/bin/env python3
"""
Data panel mixin for DoE Analysis Tab.
Handles file browsing and response selection.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os


class DataPanelMixin:
    """Mixin providing data loading and response selection functionality."""

    def browse_file(self):
        """Open file dialog"""
        filepath = filedialog.askopenfilename(
            title="Select DoE Data File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )

        if filepath:
            self.filepath = filepath
            self.file_label.config(text=os.path.basename(filepath))
            self.main_window.update_status(f"Loaded: {os.path.basename(filepath)}")

            try:
                self.handler.load_excel(filepath)

                # Get potential response columns (numeric columns excluding metadata)
                potential_responses = self.handler.get_potential_response_columns()

                if not potential_responses:
                    messagebox.showerror("No Response Columns Found",
                                       "Excel file must have at least one numeric column\n"
                                       "that can be used as a response variable.\n\n"
                                       "Numeric columns are needed for optimization.")
                    self.main_window.update_status("Error: No numeric columns found")
                    return

                self._populate_response_selection(potential_responses)

                self.main_window.update_status(f"Loaded: {len(potential_responses)} potential response(s) found")

            except Exception as e:
                messagebox.showerror("File Load Failed",
                    f"Could not open the selected Excel file.\n\n"
                    f"Details: {str(e)}\n\n"
                    f"Make sure the file is not open in another program.")
                self.main_window.update_status("Error loading file")

    def _populate_response_selection(self, potential_responses):
        """Create checkboxes for response selection with maximize/minimize options"""
        # Clear existing widgets in response frame
        for widget in self.response_frame.winfo_children():
            widget.destroy()

        # Reset tracking
        self.response_checkboxes = {}
        self.selected_responses = []
        self.response_directions = {}
        self.response_constraints = {}

        # Header
        header_label = ttk.Label(self.response_frame,
                                text="Select response variable(s) to optimize:",
                                font=('TkDefaultFont', 9, 'bold'))
        header_label.grid(row=0, column=0, columnspan=5, sticky='w', padx=5, pady=5)

        # Column headers
        ttk.Label(self.response_frame, text="Constraint Min", font=('TkDefaultFont', 8)).grid(row=0, column=2, padx=5)
        ttk.Label(self.response_frame, text="Constraint Max", font=('TkDefaultFont', 8)).grid(row=0, column=3, padx=5)

        # Create checkbox for each potential response
        for idx, col_name in enumerate(potential_responses):
            # Checkbox
            var = tk.BooleanVar(value=False)

            # Auto-select if column name contains "response" (case-insensitive)
            if 'response' in col_name.lower():
                var.set(True)

            checkbox = ttk.Checkbutton(self.response_frame, text=col_name, variable=var,
                                      command=self._update_selected_responses)
            checkbox.grid(row=idx+1, column=0, sticky='w', padx=20, pady=2)

            # Direction dropdown (Maximize/Minimize)
            direction_var = tk.StringVar(value='Maximize')
            direction_combo = ttk.Combobox(self.response_frame, textvariable=direction_var,
                                          values=['Maximize', 'Minimize'],
                                          state='readonly', width=10)
            direction_combo.grid(row=idx+1, column=1, padx=10, pady=2)
            direction_combo.bind('<<ComboboxSelected>>', lambda e: self._update_selected_responses())

            # Min constraint entry
            min_var = tk.StringVar(value='')
            min_entry = ttk.Entry(self.response_frame, textvariable=min_var, width=10)
            min_entry.grid(row=idx+1, column=2, padx=5, pady=2)
            min_entry.bind('<FocusOut>', lambda e: self._update_selected_responses())
            min_entry.bind('<Return>', lambda e: self._update_selected_responses())

            # Max constraint entry
            max_var = tk.StringVar(value='')
            max_entry = ttk.Entry(self.response_frame, textvariable=max_var, width=10)
            max_entry.grid(row=idx+1, column=3, padx=5, pady=2)
            max_entry.bind('<FocusOut>', lambda e: self._update_selected_responses())
            max_entry.bind('<Return>', lambda e: self._update_selected_responses())

            # Store references
            self.response_checkboxes[col_name] = (var, direction_var, min_var, max_var)

        # Note label
        note_label = ttk.Label(self.response_frame,
                              text="Note: Constraints are optional. Leave blank for no constraint.",
                              font=('TkDefaultFont', 8, 'italic'))
        note_label.grid(row=len(potential_responses)+1, column=0, columnspan=5,
                       sticky='w', padx=5, pady=5)

        # Exploration mode checkbox
        self.exploration_mode_var = tk.BooleanVar(value=False)
        exploration_cb = ttk.Checkbutton(
            self.response_frame,
            text="Allow exploration outside bounds (20% of suggestions may violate constraints)",
            variable=self.exploration_mode_var
        )
        exploration_cb.grid(row=len(potential_responses)+2, column=0, columnspan=5,
                           sticky='w', padx=20, pady=5)

        # Initial update
        self._update_selected_responses()

    def _debug_log(self, message):
        """Collect debug messages to display at the end"""
        self.debug_log.append(message)
        # Set CONSOLE_DEBUG = True in __init__ if you need console output
        if hasattr(self, 'CONSOLE_DEBUG') and self.CONSOLE_DEBUG:
            print(message)

    def _update_selected_responses(self):
        """Update selected responses list and enable/disable analyze button"""
        self.selected_responses = []
        self.response_directions = {}
        self.response_constraints = {}

        for col_name, (var, direction_var, min_var, max_var) in self.response_checkboxes.items():
            if var.get():
                self.selected_responses.append(col_name)
                direction = 'minimize' if direction_var.get() == 'Minimize' else 'maximize'
                self.response_directions[col_name] = direction

                self._debug_log(f"[DEBUG UI] Response '{col_name}': direction={direction} (UI dropdown value: '{direction_var.get()}')")

                min_val = min_var.get().strip()
                max_val = max_var.get().strip()

                constraint = {}
                if min_val:
                    try:
                        constraint['min'] = float(min_val)
                        self._debug_log(f"[DEBUG UI] Response '{col_name}': min constraint = {constraint['min']}")
                    except ValueError:
                        self._debug_log(f"[DEBUG UI] Response '{col_name}': invalid min value '{min_val}'")
                        pass
                if max_val:
                    try:
                        constraint['max'] = float(max_val)
                        self._debug_log(f"[DEBUG UI] Response '{col_name}': max constraint = {constraint['max']}")
                    except ValueError:
                        self._debug_log(f"[DEBUG UI] Response '{col_name}': invalid max value '{max_val}'")
                        pass

                if constraint:
                    self.response_constraints[col_name] = constraint

        if self.selected_responses:
            self.analyze_btn.config(state='normal')
            self.main_window.update_status(f"Ready to analyze {len(self.selected_responses)} response(s)")
        else:
            self.analyze_btn.config(state='disabled')
            self.main_window.update_status("Please select at least one response variable")

    def load_results(self):
        """Load results from Excel (called from main window menu)"""
        self.browse_file()

    def refresh(self):
        """Refresh UI from project data (called when switching tabs)"""
        # Reload display if data exists
        if self.handler.data is not None:
            self._update_display()
