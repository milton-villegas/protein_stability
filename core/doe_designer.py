"""
DoE Design Generation Logic
Extracted from factorial_designer_gui.pyw
"""
import math
import itertools
from typing import Dict, List, Tuple
import pandas as pd
from core.project import AVAILABLE_FACTORS


class DoEDesigner:
    """Handles factorial design generation and volume calculations"""

    @staticmethod
    def generate_well_position(index: int) -> Tuple[int, str]:
        """
        Generate plate number and well position (A1-H12)
        Fills column-wise: A1, B1, C1...H1, A2, B2...

        Args:
            index: Sample index (0-based)

        Returns:
            (plate_number, well_position)
        """
        plate_number = (index // 96) + 1
        position_in_plate = index % 96

        # Fill down columns first (A1, B1, C1... H1, then A2, B2...)
        row = position_in_plate % 8  # 0-7 for A-H
        col = position_in_plate // 8  # 0-11 for 1-12

        row_letter = chr(ord('A') + row)
        col_number = col + 1
        well = f"{row_letter}{col_number}"

        return plate_number, well

    @staticmethod
    def convert_96_to_384_well(plate_num: int, well_96: str) -> str:
        """
        Convert 96-well to 384-well position

        Mapping rules:
        - Plate 1→ cols 1-6, Plate 2→ cols 7-12, etc.
        - Odd cols (1,3,5) use first row of pair (A→A, B→C, C→E)
        - Even cols (2,4,6) use second row (A→B, B→D, C→F)

        Args:
            plate_num: 96-well plate number (1-based)
            well_96: 96-well position (e.g., "B3")

        Returns:
            384-well position (e.g., "D7")
        """
        # Parse 96-well position
        row_96 = well_96[0]
        col_96 = int(well_96[1:])

        # Convert row letter to index (A=0, B=1, ..., H=7)
        row_96_index = ord(row_96) - ord('A')

        # Map to 384-well column
        col_384 = (plate_num - 1) * 6 + math.ceil(col_96 / 2)

        # Map row based on column parity
        if col_96 % 2 == 1:  # Odd column
            row_384_index = row_96_index * 2
        else:  # Even column
            row_384_index = row_96_index * 2 + 1

        # Convert back to letter
        row_384 = chr(ord('A') + row_384_index)

        return f"{row_384}{col_384}"

    @staticmethod
    def calculate_volumes(factor_values: Dict[str, float],
                         stock_concs: Dict[str, float],
                         final_volume: float,
                         buffer_ph_values: List[str]) -> Dict[str, float]:
        """
        Calculate reagent volumes using C1V1 = C2V2 formula

        Args:
            factor_values: Dict of factor_name → desired concentration
            stock_concs: Dict of factor_name → stock concentration
            final_volume: Total final volume (µL)
            buffer_ph_values: List of unique pH values

        Returns:
            Dict of reagent → volume (µL)
        """
        volumes = {}
        total_volume_used = 0

        # Handle buffer pH
        if "buffer pH" in factor_values:
            buffer_ph = str(factor_values["buffer pH"])

            # Initialize all pH buffer volumes to 0
            for ph in buffer_ph_values:
                volumes[f"buffer {ph}"] = 0

            # Calculate volume for the selected pH buffer
            if "buffer_concentration" in factor_values and "buffer_concentration" in stock_concs:
                try:
                    desired_conc = float(factor_values["buffer_concentration"])
                    buffer_stock = stock_concs["buffer_concentration"]
                    volume = (desired_conc * final_volume) / buffer_stock
                    volumes[f"buffer {buffer_ph}"] = round(volume, 2)
                    total_volume_used += volumes[f"buffer {buffer_ph}"]
                except (ValueError, ZeroDivisionError):
                    volumes[f"buffer {buffer_ph}"] = 0

        # Calculate volumes for other factors
        for factor_name, desired_conc in factor_values.items():
            if factor_name in ["buffer pH", "buffer_concentration"]:
                continue

            if factor_name in stock_concs:
                try:
                    stock_conc = stock_concs[factor_name]
                    volume = (float(desired_conc) * final_volume) / stock_conc
                    volumes[factor_name] = round(volume, 2)
                    total_volume_used += volumes[factor_name]
                except (ValueError, ZeroDivisionError):
                    volumes[factor_name] = 0

        # Calculate water to reach final volume
        water_volume = round(final_volume - total_volume_used, 2)
        volumes["water"] = water_volume

        return volumes

    def build_factorial_design(self,
                               factors: Dict[str, List[str]],
                               stock_concs: Dict[str, float],
                               final_volume: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build full factorial design with Excel and volume data

        Args:
            factors: Dict of factor_name → list of levels
            stock_concs: Dict of factor_name → stock concentration
            final_volume: Total final volume (µL)

        Returns:
            (excel_df, volume_df): DataFrames for Excel export and Opentrons CSV

        Raises:
            ValueError: If design is impossible (negative water volumes)
        """
        if not factors:
            raise ValueError("No factors defined")

        # Determine unique buffer pH values
        buffer_ph_values = []
        if "buffer pH" in factors:
            unique_phs = set(str(ph).strip() for ph in factors["buffer pH"])
            buffer_ph_values = sorted(unique_phs)

        # Build factorial combinations
        factor_names = list(factors.keys())
        level_lists = [factors[f] for f in factor_names]
        combinations = list(itertools.product(*level_lists))

        # Build headers
        excel_headers = ["ID", "Plate_96", "Well_96", "Well_384"]
        for fn in factor_names:
            excel_headers.append(AVAILABLE_FACTORS.get(fn, fn))
        excel_headers.append("Response")

        volume_headers = []
        if "buffer pH" in factors:
            for ph in buffer_ph_values:
                volume_headers.append(f"buffer {ph}")

        for factor in factor_names:
            if factor not in ["buffer pH", "buffer_concentration"]:
                volume_headers.append(factor)

        volume_headers.append("water")

        # Calculate design
        excel_rows = []
        volume_rows = []

        for idx, combo in enumerate(combinations):
            row_dict = {factor_names[i]: combo[i] for i in range(len(factor_names))}

            # Generate well positions
            plate_num, well_pos = self.generate_well_position(idx)
            well_384 = self.convert_96_to_384_well(plate_num, well_pos)

            # Excel row
            excel_row = [idx + 1, plate_num, well_pos, well_384]
            for fn in factor_names:
                excel_row.append(row_dict.get(fn, ""))
            excel_row.append("")  # Empty Response column
            excel_rows.append(excel_row)

            # Volume calculations
            volumes = self.calculate_volumes(row_dict, stock_concs, final_volume, buffer_ph_values)

            volume_row = [volumes.get(h, 0) for h in volume_headers]
            volume_rows.append(volume_row)

        # Check for negative water volumes
        negative_water_wells = []
        for idx, volume_row in enumerate(volume_rows):
            water_idx = volume_headers.index("water")
            water_vol = volume_row[water_idx]
            if water_vol < 0:
                well_id = excel_rows[idx][0]
                well_pos = excel_rows[idx][2]
                negative_water_wells.append((well_id, well_pos, water_vol))

        if negative_water_wells:
            error_msg = "⚠️ IMPOSSIBLE DESIGN DETECTED ⚠️\n\n"
            error_msg += "The following wells require NEGATIVE water volumes:\n\n"

            for well_id, well_pos, water_vol in negative_water_wells[:5]:
                error_msg += f"  • Well {well_pos} (ID {well_id}): {water_vol} µL water\n"

            if len(negative_water_wells) > 5:
                error_msg += f"  ... and {len(negative_water_wells) - 5} more wells\n"

            error_msg += f"\nTotal problematic wells: {len(negative_water_wells)}\n\n"
            error_msg += "Solutions:\n"
            error_msg += "  1. INCREASE stock concentrations\n"
            error_msg += "  2. INCREASE final volume\n"
            error_msg += "  3. REDUCE desired concentration levels\n"

            raise ValueError(error_msg)

        # Create DataFrames
        excel_df = pd.DataFrame(excel_rows, columns=excel_headers)
        volume_df = pd.DataFrame(volume_rows, columns=volume_headers)

        return excel_df, volume_df
