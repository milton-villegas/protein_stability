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
from core.design_validator import DesignValidator
from config.design_config import CATEGORICAL_FACTORS, MAX_TOTAL_WELLS


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

    def _validate_inputs(
        self,
        factors: Dict[str, List[str]],
        stock_concs: Dict[str, float],
        final_volume: float
    ) -> None:
        """
        Validate all inputs before design generation.

        Args:
            factors: Dictionary of factor_name → list of levels
            stock_concs: Dictionary of factor_name → stock concentration
            final_volume: Total final volume (µL)

        Raises:
            ValueError: If any input is invalid with descriptive message

        Examples:
            >>> designer = DoEDesigner()
            >>> factors = {'buffer pH': ['15.0']}  # Invalid pH
            >>> designer._validate_inputs(factors, {}, 200.0)
            Traceback: ValueError: Invalid factor 'buffer pH'...
        """
        # 1. Check factors exist
        if not factors:
            raise ValueError("No factors defined")

        # 2. Validate each factor and its levels
        for factor_name, levels in factors.items():
            if not levels:
                raise ValueError(
                    f"Factor '{factor_name}' has no levels. "
                    f"Please provide at least one level."
                )

            # Validate level values using DesignValidator
            is_valid, msg = DesignValidator.validate_factor_levels(factor_name, levels)
            if not is_valid:
                raise ValueError(f"Invalid factor '{factor_name}': {msg}")

        # 3. Validate stock concentrations for non-categorical factors
        for factor_name in factors.keys():
            if factor_name in CATEGORICAL_FACTORS:
                continue

            stock_conc = stock_concs.get(factor_name)
            is_valid, msg = DesignValidator.validate_stock_concentration(
                factor_name, stock_conc
            )
            if not is_valid:
                raise ValueError(
                    f"Invalid stock concentration for '{factor_name}': {msg}\n"
                    f"Please provide a positive stock concentration value."
                )

        # 4. Estimate design size and check against plate capacity
        estimated_size = 1
        for levels in factors.values():
            estimated_size *= len(levels)

        if estimated_size > MAX_TOTAL_WELLS:
            raise ValueError(
                f"Design too large: {estimated_size} combinations exceeds "
                f"{MAX_TOTAL_WELLS} wells (4 plates of 96 wells each).\n\n"
                f"Solutions:\n"
                f"  1. Reduce the number of factor levels\n"
                f"  2. Use fewer factors\n"
                f"  3. Use Latin Hypercube Sampling (LHS) design instead"
            )

        # 5. Validate final volume
        if final_volume <= 0:
            raise ValueError(
                f"Final volume must be positive. Got: {final_volume} µL"
            )

        if final_volume > 323:
            raise ValueError(
                f"Final volume {final_volume} µL exceeds 96-well plate capacity (323 µL).\n"
                f"Please reduce the final volume to 323 µL or less."
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
            ValueError: If inputs are invalid or design is impossible
        """
        # Validate all inputs before proceeding with design generation
        self._validate_inputs(factors, stock_concs, final_volume)

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
