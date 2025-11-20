"""
Export Panel Mixin for Designer Tab.

This module provides export functionality for factorial designs,
including Excel and CSV export capabilities for Opentrons.
"""

import csv
import itertools
import os
from datetime import datetime
from tkinter import filedialog, messagebox
from typing import Dict, List, Tuple

# Try to import openpyxl for Excel export
try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

# Import AVAILABLE_FACTORS from utils
from utils.constants import AVAILABLE_FACTORS


class ExportPanelMixin:
    """
    Mixin class providing export functionality for factorial designs.

    This mixin provides methods for exporting experimental designs to
    Excel (.xlsx) and CSV formats suitable for Opentrons liquid handling.

    Attributes expected from parent class:
        self.model: DesignerModel instance
        self.main_window: Main application window
        self.final_vol_var: tkinter variable for final volume
        self.design_type_var: tkinter variable for design type
        self.design_map: Dict mapping display names to design types
        self.sample_size_var: tkinter variable for LHS sample size
        self.resolution_var: tkinter variable for fractional factorial resolution
        self.ccd_type_var: tkinter variable for CCD type
        self.optimize_lhs_var: tkinter variable for LHS optimization
        self.CATEGORICAL_PAIRS: Dict mapping categorical factors to concentration factors

    Methods expected from parent class:
        self._filter_categorical_combinations()
        self._generate_lhs_design()
        self._generate_fractional_factorial()
        self._generate_plackett_burman()
        self._generate_central_composite()
        self._generate_box_behnken()
        self._generate_well_position()
        self._convert_96_to_384_well()
        self._update_display()
    """

    def _extract_unique_categorical_values(self, factors: Dict[str, List[str]],
                                           factor_name: str,
                                           skip_none: bool = False) -> List[str]:
        """
        Extract unique values for categorical factors.

        Args:
            factors: Dictionary of factor names to levels
            factor_name: Name of the categorical factor
            skip_none: If True, skip None/0/empty values

        Returns:
            Sorted list of unique values
        """
        if factor_name not in factors:
            return []

        unique_values = set()
        for val in factors[factor_name]:
            val_str = str(val).strip()

            if skip_none:
                # Skip empty, None, or 0 values
                if val_str and val_str.lower() not in ['none', '0', 'nan', '']:
                    unique_values.add(val_str)
            else:
                unique_values.add(val_str)

        return sorted(unique_values)

    def _build_excel_headers(self, factor_names: List[str]) -> List[str]:
        """
        Build Excel headers with proper ordering of categorical factors.

        Args:
            factor_names: List of factor names in the design

        Returns:
            List of column headers for Excel export
        """
        excel_headers = ["ID", "Plate_96", "Well_96", "Well_384", "Source", "Batch"]

        # Track which factors we've already added
        added_factors = set()

        # Add factors in order, but group categorical with their concentrations
        for fn in factor_names:
            if fn in added_factors:
                continue

            # Add the factor
            excel_headers.append(AVAILABLE_FACTORS.get(fn, fn))
            added_factors.add(fn)

            # If it's a categorical factor with a paired concentration, add that next
            if fn in self.CATEGORICAL_PAIRS:
                paired_factor = self.CATEGORICAL_PAIRS[fn]
                if paired_factor in factor_names and paired_factor not in added_factors:
                    excel_headers.append(AVAILABLE_FACTORS.get(paired_factor, paired_factor))
                    added_factors.add(paired_factor)

        # Add Response column for TM data entry
        excel_headers.append("Response")

        return excel_headers

    def _build_volume_headers(self, factor_names: List[str],
                             buffer_ph_values: List[str],
                             detergent_values: List[str],
                             reducing_agent_values: List[str]) -> List[str]:
        """
        Build volume headers for Opentrons CSV.

        Args:
            factor_names: List of factor names
            buffer_ph_values: List of unique buffer pH values
            detergent_values: List of unique detergent types
            reducing_agent_values: List of unique reducing agent types

        Returns:
            List of column headers for volume CSV
        """
        volume_headers = ["ID"]

        # Add buffer pH columns (one column per pH value)
        for ph in buffer_ph_values:
            volume_headers.append(f"buffer_{ph}")

        # Add detergent columns (one column per detergent type)
        for det in detergent_values:
            # Clean up name for column header (lowercase, replace spaces/hyphens with underscores)
            det_clean = det.replace(' ', '_').replace('-', '_').lower()
            volume_headers.append(det_clean)

        # Add reducing agent columns (one column per reducing agent type)
        for agent in reducing_agent_values:
            # Clean up name for column header (lowercase, replace spaces/hyphens with underscores)
            agent_clean = agent.replace(' ', '_').replace('-', '_').lower()
            volume_headers.append(agent_clean)

        # Add other volume headers (skip categorical factors and their concentrations)
        for factor in factor_names:
            if factor in ["buffer pH", "buffer_concentration", "detergent", "detergent_concentration",
                         "reducing_agent", "reducing_agent_concentration"]:
                continue
            volume_headers.append(factor)

        # Add water column at the end
        volume_headers.append("water")

        return volume_headers

    def _build_factorial_with_volumes(self) -> Tuple[List[str], List[List], List[str], List[List]]:
        """
        Build factorial design (full or LHS) and calculate volumes.

        Returns:
            Tuple containing:
                - excel_headers: List of column headers for Excel export
                - excel_rows: List of data rows for Excel export
                - volume_headers: List of column headers for volume CSV
                - volume_rows: List of volume data rows for Opentrons CSV

        Raises:
            ValueError: If no factors defined, invalid volume, sample size issues,
                       or impossible design (negative water volumes)
        """
        factors = self.model.get_factors()
        if not factors:
            raise ValueError("No factors defined.")

        # Get final volume
        try:
            final_vol = float(self.final_vol_var.get())
        except ValueError:
            raise ValueError("Invalid final volume value.")

        stock_concs = self.model.get_all_stock_concs()

        # Get design type and generate combinations
        display_text = self.design_type_var.get()
        if display_text in self.design_map:
            design_type = self.design_map[display_text]
        else:
            design_type = display_text

        factor_names = list(factors.keys())

        if design_type == "full_factorial":
            # Original full factorial logic
            level_lists = [factors[f] for f in factor_names]
            combinations = list(itertools.product(*level_lists))
            # Filter out illogical categorical-concentration pairings
            combinations = self._filter_categorical_combinations(combinations, factor_names)

        elif design_type == "lhs":
            # Latin Hypercube Sampling
            n_samples = self.sample_size_var.get()

            # Validate sample size
            if n_samples > 384:
                raise ValueError("Sample size cannot exceed 384 (4 plates of 96 wells).")
            if n_samples < 1:
                raise ValueError("Sample size must be at least 1.")

            combinations = self._generate_lhs_design(factors, n_samples)
            combinations = self._filter_categorical_combinations(combinations, factor_names)

        elif design_type == "fractional":
            # 2-Level Fractional Factorial
            resolution = self.resolution_var.get()
            combinations = self._generate_fractional_factorial(factors, resolution)
            combinations = self._filter_categorical_combinations(combinations, factor_names)

        elif design_type == "plackett_burman":
            # Plackett-Burman screening design
            combinations = self._generate_plackett_burman(factors)
            combinations = self._filter_categorical_combinations(combinations, factor_names)

        elif design_type == "central_composite":
            # Central Composite Design
            ccd_type = self.ccd_type_var.get()
            combinations = self._generate_central_composite(factors, ccd_type)
            combinations = self._filter_categorical_combinations(combinations, factor_names)

        elif design_type == "box_behnken":
            # Box-Behnken design
            combinations = self._generate_box_behnken(factors)
            combinations = self._filter_categorical_combinations(combinations, factor_names)

        else:
            raise ValueError(f"Unknown design type: {design_type}")

        # Extract unique values for categorical factors
        buffer_ph_values = self._extract_unique_categorical_values(factors, "buffer pH")
        detergent_values = self._extract_unique_categorical_values(factors, "detergent", skip_none=True)
        reducing_agent_values = self._extract_unique_categorical_values(factors, "reducing_agent", skip_none=True)

        # Build headers
        excel_headers = self._build_excel_headers(factor_names)
        volume_headers = self._build_volume_headers(factor_names, buffer_ph_values,
                                                     detergent_values, reducing_agent_values)

        # Calculate volumes
        excel_rows = []
        volume_rows = []

        for idx, combo in enumerate(combinations):
            row_dict = {factor_names[i]: combo[i] for i in range(len(factor_names))}

            plate_num, well_pos = self._generate_well_position(idx)
            well_384 = self._convert_96_to_384_well(plate_num, well_pos)

            # Excel row - use same order as headers
            excel_row = [idx + 1, plate_num, well_pos, well_384, design_type.upper(), 0]

            # Track which factors we've already added
            added_factors_row = set()

            # Add factor values in same order as headers
            for fn in factor_names:
                if fn in added_factors_row:
                    continue

                # Add the factor value
                excel_row.append(row_dict.get(fn, ""))
                added_factors_row.add(fn)

                # If it's a categorical factor with a paired concentration, add that next
                if fn in self.CATEGORICAL_PAIRS:
                    paired_factor = self.CATEGORICAL_PAIRS[fn]
                    if paired_factor in factor_names and paired_factor not in added_factors_row:
                        excel_row.append(row_dict.get(paired_factor, ""))
                        added_factors_row.add(paired_factor)

            # Add empty Response column for manual data entry
            excel_row.append("")
            excel_rows.append(excel_row)

            # Volume calculations
            volumes = {}
            total_volume_used = 0

            # Handle buffer pH (categorical - one column per pH value)
            if "buffer pH" in row_dict:
                buffer_ph = str(row_dict["buffer pH"])
                for ph in buffer_ph_values:
                    volumes[f"buffer_{ph}"] = 0

                if "buffer_concentration" in row_dict and "buffer_concentration" in stock_concs:
                    try:
                        desired_conc = float(row_dict["buffer_concentration"])
                        buffer_stock = stock_concs["buffer_concentration"]
                        # C1*V1 = C2*V2 -> V1 = (C2*V2)/C1
                        volume = (desired_conc * final_vol) / buffer_stock
                        volumes[f"buffer_{buffer_ph}"] = round(volume, 2)
                        total_volume_used += volumes[f"buffer_{buffer_ph}"]
                    except (ValueError, ZeroDivisionError):
                        volumes[f"buffer_{buffer_ph}"] = 0

            # Handle detergent (categorical - one column per detergent type)
            if "detergent" in row_dict:
                detergent_type = str(row_dict["detergent"]).strip()

                # Initialize all detergent columns to 0
                for det in detergent_values:
                    det_clean = det.replace(' ', '_').replace('-', '_').lower()
                    volumes[det_clean] = 0

                # Only add volume if detergent is not None/empty
                if detergent_type and detergent_type.lower() not in ['none', '0', 'nan', '']:
                    if "detergent_concentration" in row_dict and "detergent_concentration" in stock_concs:
                        try:
                            desired_conc = float(row_dict["detergent_concentration"])
                            detergent_stock = stock_concs["detergent_concentration"]
                            # C1*V1 = C2*V2 -> V1 = (C2*V2)/C1
                            volume = (desired_conc * final_vol) / detergent_stock
                            det_clean = detergent_type.replace(' ', '_').replace('-', '_').lower()
                            volumes[det_clean] = round(volume, 2)
                            total_volume_used += volumes[det_clean]
                        except (ValueError, ZeroDivisionError):
                            pass

            # Handle reducing_agent (categorical - one column per reducing agent type)
            if "reducing_agent" in row_dict:
                agent_type = str(row_dict["reducing_agent"]).strip()

                # Initialize all reducing agent columns to 0
                for agent in reducing_agent_values:
                    agent_clean = agent.replace(' ', '_').replace('-', '_').lower()
                    volumes[agent_clean] = 0

                # Only add volume if reducing agent is not None/empty
                if agent_type and agent_type.lower() not in ['none', '0', 'nan', '']:
                    if "reducing_agent_concentration" in row_dict and "reducing_agent_concentration" in stock_concs:
                        try:
                            desired_conc = float(row_dict["reducing_agent_concentration"])
                            agent_stock = stock_concs["reducing_agent_concentration"]
                            # C1*V1 = C2*V2 -> V1 = (C2*V2)/C1
                            volume = (desired_conc * final_vol) / agent_stock
                            agent_clean = agent_type.replace(' ', '_').replace('-', '_').lower()
                            volumes[agent_clean] = round(volume, 2)
                            total_volume_used += volumes[agent_clean]
                        except (ValueError, ZeroDivisionError):
                            pass

            # Calculate volumes for other factors (NaCl, Zinc, Glycerol, etc.)
            for factor in factor_names:
                if factor in ["buffer pH", "buffer_concentration", "detergent", "detergent_concentration",
                             "reducing_agent", "reducing_agent_concentration"]:
                    continue
                if factor in row_dict and factor in stock_concs:
                    try:
                        desired_conc = float(row_dict[factor])
                        stock_conc = stock_concs[factor]
                        volume = (desired_conc * final_vol) / stock_conc
                        volumes[factor] = round(volume, 2)
                        total_volume_used += volumes[factor]
                    except (ValueError, ZeroDivisionError):
                        volumes[factor] = 0

            # Calculate water to reach final volume
            water_volume = round(final_vol - total_volume_used, 2)
            volumes["water"] = water_volume

            # Build volume row in correct order matching headers
            volume_row = [idx + 1]  # ID first
            for h in volume_headers[1:]:  # Skip ID column in headers
                volume_row.append(volumes.get(h, 0))
            volume_rows.append(volume_row)

        # Check for negative water volumes
        negative_water_wells = []
        for idx, volume_row in enumerate(volume_rows):
            water_idx = volume_headers.index("water")
            water_vol = volume_row[water_idx]
            if water_vol < 0:
                well_id = excel_rows[idx][0]  # ID
                well_pos = excel_rows[idx][2]  # Well position
                negative_water_wells.append((well_id, well_pos, water_vol))

        if negative_water_wells:
            # Build error message for impossible designs
            error_msg = "IMPOSSIBLE DESIGN DETECTED\n\n"
            error_msg += f"The following wells require NEGATIVE water volumes:\n\n"

            # Show problematic wells
            for well_id, well_pos, water_vol in negative_water_wells[:5]:
                error_msg += f"  - Well {well_pos} (ID {well_id}): {water_vol} uL water\n"

            if len(negative_water_wells) > 5:
                error_msg += f"  ... and {len(negative_water_wells) - 5} more wells\n"

            error_msg += f"\nTotal problematic wells: {len(negative_water_wells)}\n\n"
            error_msg += "This means the sum of component volumes EXCEEDS the final volume!\n\n"
            error_msg += "Solutions:\n"
            error_msg += "  1. INCREASE stock concentrations (recommended)\n"
            error_msg += "  2. INCREASE final volume\n"
            error_msg += "  3. REDUCE desired concentration levels\n\n"
            error_msg += "Example: If stock is 50 mM and you want 100 mM,\n"
            error_msg += "you'd need to add 200 uL of stock to make 100 uL final volume.\n"
            error_msg += "This is physically impossible!"

            raise ValueError(error_msg)

        return excel_headers, excel_rows, volume_headers, volume_rows

    def _export_both(self):
        """
        Export XLSX and CSV files with single-step file dialog.

        Creates two files:
            - Excel file with sample tracking and stock concentrations
            - CSV file with volumes for Opentrons liquid handling

        The files are named with a standardized convention:
            [UserName]_Design_[Date].xlsx
            [UserName]_Design_[Date]_Opentron.csv
        """
        if not HAS_OPENPYXL:
            messagebox.showerror("Missing Library",
                               "openpyxl is required. Install with: pip install openpyxl")
            return

        try:
            # Validate stock concentrations
            factors = self.model.get_factors()
            stock_concs = self.model.get_all_stock_concs()

            missing_stocks = []
            for factor in factors.keys():
                # Skip categorical factors that don't need stock concentrations
                if factor in ["buffer pH", "detergent", "reducing_agent"]:
                    continue
                if factor not in stock_concs:
                    missing_stocks.append(AVAILABLE_FACTORS.get(factor, factor))

            if missing_stocks:
                messagebox.showerror("Missing Stock Concentrations",
                    f"The following factors need stock concentrations:\n\n" +
                    "\n".join(f"- {f}" for f in missing_stocks) +
                    "\n\nEdit each factor to add stock concentrations.")
                return

            # Build design
            excel_headers, excel_rows, volume_headers, volume_rows = self._build_factorial_with_volumes()

            total = len(excel_rows)
            if total > 384:
                messagebox.showerror("Too Many Combinations",
                    f"Design has {total} combinations.\n\n"
                    f"Maximum: 384 (4 plates of 96 wells)\n\n"
                    f"Please reduce factors/levels or sample size.")
                return

            # Single-step file save dialog with suggested name
            date_str = datetime.now().strftime('%Y%m%d')

            path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                title="Save Factorial Design",
                initialfile="Design1.xlsx"
            )

            if not path:
                return  # User cancelled

            # Generate paths with naming convention: [UserName]_Design_[Date]
            base_path = os.path.splitext(path)[0]

            # Add standardized suffix
            xlsx_path = f"{base_path}_Design_{date_str}.xlsx"
            csv_path = f"{base_path}_Design_{date_str}_Opentron.csv"

            # Export XLSX
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Sample Tracking"

            # Headers
            for col_idx, header in enumerate(excel_headers, start=1):
                cell = ws.cell(row=1, column=col_idx, value=header)
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center")
                cell.fill = PatternFill(start_color="4CAF50", end_color="4CAF50", fill_type="solid")

            # Data - convert numeric strings to numbers
            for row_idx, row_data in enumerate(excel_rows, start=2):
                for col_idx, value in enumerate(row_data, start=1):
                    # Try to convert to number if possible
                    try:
                        numeric_value = float(value)
                        ws.cell(row=row_idx, column=col_idx, value=numeric_value)
                    except (ValueError, TypeError):
                        # Keep as string if not numeric
                        ws.cell(row=row_idx, column=col_idx, value=value)

            # Auto-adjust columns
            for col in ws.columns:
                max_length = 0
                col_letter = col[0].column_letter
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                ws.column_dimensions[col_letter].width = adjusted_width

            # CREATE STOCK CONCENTRATIONS METADATA SHEET
            stock_sheet = wb.create_sheet(title="Stock_Concentrations")

            # Headers for stock concentrations sheet
            stock_headers = ["Factor Name", "Stock Value", "Unit"]
            for col_idx, header in enumerate(stock_headers, start=1):
                cell = stock_sheet.cell(row=1, column=col_idx, value=header)
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center")
                cell.fill = PatternFill(start_color="2196F3", end_color="2196F3", fill_type="solid")

            # Write stock concentration data
            row_idx = 2
            for factor_name, stock_value in stock_concs.items():
                # Get display name
                display_name = AVAILABLE_FACTORS.get(factor_name, factor_name)

                # Determine unit based on factor name
                if "pH" in factor_name:
                    unit = "pH"
                elif "conc" in factor_name.lower() or "salt" in factor_name.lower():
                    unit = "mM"
                elif "glycerol" in factor_name.lower() or "dmso" in factor_name.lower() or "detergent" in factor_name.lower():
                    unit = "%"
                else:
                    unit = ""  # Unknown unit

                # Write row
                stock_sheet.cell(row=row_idx, column=1, value=display_name)
                stock_sheet.cell(row=row_idx, column=2, value=stock_value)
                stock_sheet.cell(row=row_idx, column=3, value=unit)
                row_idx += 1

            # Auto-adjust stock sheet columns
            for col in stock_sheet.columns:
                max_length = 0
                col_letter = col[0].column_letter
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 30)
                stock_sheet.column_dimensions[col_letter].width = adjusted_width

            wb.save(xlsx_path)

            # Export CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(volume_headers)
                for vol_row in volume_rows:
                    writer.writerow(vol_row)

            plates = (total + 95) // 96

            # Get design name for display
            display_text = self.design_type_var.get()
            if display_text in self.design_map:
                design_type = self.design_map[display_text]
            else:
                design_type = display_text

            # Build design name with details
            if design_type == "full_factorial":
                design_name = "Full Factorial"
            elif design_type == "lhs":
                if self.optimize_lhs_var.get() and HAS_SMT:
                    design_name = "Latin Hypercube (Optimized - SMT)"
                else:
                    design_name = "Latin Hypercube (Standard - pyDOE3)"
            elif design_type == "fractional":
                resolution = self.resolution_var.get()
                design_name = f"2-Level Fractional Factorial (Resolution {resolution})"
            elif design_type == "plackett_burman":
                design_name = "Plackett-Burman Screening"
            elif design_type == "central_composite":
                ccd_type = self.ccd_type_var.get()
                design_name = f"Central Composite Design ({ccd_type})"
            elif design_type == "box_behnken":
                design_name = "Box-Behnken Design"
            else:
                design_name = display_text

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

        except Exception as e:
            messagebox.showerror("Export Failed", f"Error during export:\n\n{str(e)}")

    def export_excel(self):
        """Export design to Excel (called from main window menu)."""
        self._export_both()

    def export_csv(self):
        """Export design to CSV (called from main window menu)."""
        self._export_both()

    def refresh(self):
        """Refresh UI from project data (called when switching tabs)."""
        self._update_display()


# Check for SMT optimization library (used in _export_both for display)
try:
    from smt.sampling_methods import LHS as SMT_LHS
    HAS_SMT = True
except ImportError:
    HAS_SMT = False
