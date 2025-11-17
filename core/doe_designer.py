"""
DoE Design Generation Logic
Refactored to use new service layer
"""
import itertools
from typing import Dict, List, Tuple
import pandas as pd
from core.project import AVAILABLE_FACTORS
from core.well_mapper import WellMapper
from core.volume_calculator import VolumeCalculator, VolumeValidator


class DoEDesigner:
    """Handles factorial design generation and volume calculations"""

    def __init__(self):
        """Initialize DoE Designer with service dependencies"""
        self.well_mapper = WellMapper()
        self.volume_calculator = VolumeCalculator()
        self.volume_validator = VolumeValidator()

    @staticmethod
    def generate_well_position(index: int) -> Tuple[int, str]:
        """
        Generate plate number and well position (A1-H12)

        DEPRECATED: Use WellMapper.generate_well_position() instead
        Kept for backward compatibility
        """
        return WellMapper.generate_well_position(index)

    @staticmethod
    def convert_96_to_384_well(plate_num: int, well_96: str) -> str:
        """
        Convert 96-well to 384-well position

        DEPRECATED: Use WellMapper.convert_96_to_384_well() instead
        Kept for backward compatibility
        """
        return WellMapper.convert_96_to_384_well(plate_num, well_96)

    @staticmethod
    def calculate_volumes(factor_values: Dict[str, float],
                         stock_concs: Dict[str, float],
                         final_volume: float,
                         buffer_ph_values: List[str]) -> Dict[str, float]:
        """
        Calculate reagent volumes using C1V1 = C2V2 formula

        DEPRECATED: Use VolumeCalculator.calculate_volumes() instead
        Kept for backward compatibility
        """
        return VolumeCalculator.calculate_volumes(
            factor_values,
            stock_concs,
            final_volume,
            buffer_ph_values
        )

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
        volumes_list = []
        well_identifiers = []

        for idx, combo in enumerate(combinations):
            row_dict = {factor_names[i]: combo[i] for i in range(len(factor_names))}

            # Generate well positions using WellMapper
            plate_num, well_pos = self.well_mapper.generate_well_position(idx)
            well_384 = self.well_mapper.convert_96_to_384_well(plate_num, well_pos)

            # Excel row
            excel_row = [idx + 1, plate_num, well_pos, well_384]
            for fn in factor_names:
                excel_row.append(row_dict.get(fn, ""))
            excel_row.append("")  # Empty Response column
            excel_rows.append(excel_row)

            # Volume calculations using VolumeCalculator
            volumes = self.volume_calculator.calculate_volumes(
                row_dict, stock_concs, final_volume, buffer_ph_values
            )

            volume_row = [volumes.get(h, 0) for h in volume_headers]
            volume_rows.append(volume_row)
            volumes_list.append(volumes)
            well_identifiers.append((idx + 1, well_pos))

        # Validate design feasibility using VolumeValidator
        is_valid, error_msg = self.volume_validator.validate_design_feasibility(
            volumes_list, well_identifiers
        )

        if not is_valid:
            raise ValueError(error_msg)

        # Create DataFrames
        excel_df = pd.DataFrame(excel_rows, columns=excel_headers)
        volume_df = pd.DataFrame(volume_rows, columns=volume_headers)

        return excel_df, volume_df
