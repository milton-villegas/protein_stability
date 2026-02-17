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

# Maps categorical factors to their paired concentration factors
CATEGORICAL_PAIRS = {
    "buffer pH": "buffer_concentration",
    "detergent": "detergent_concentration",
    "reducing_agent": "reducing_agent_concentration",
}


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
                               final_volume: float,
                               per_level_concs: Dict = None,
                               protein_stock: float = None,
                               protein_final: float = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build full factorial design with Excel and volume data

        Args:
            factors: Dict of factor_name → list of levels
            stock_concs: Dict of factor_name → stock concentration
            final_volume: Total final volume (µL)
            per_level_concs: Dict of factor → level → {"stock": float, "final": float}
            protein_stock: Protein stock concentration (mg/mL), optional
            protein_final: Desired final protein concentration (mg/mL), optional

        Returns:
            (excel_df, volume_df): DataFrames for Excel export and Opentrons CSV

        Raises:
            ValueError: If inputs are invalid or design is impossible
        """
        if per_level_concs is None:
            per_level_concs = {}

        # Filter out concentration factors when per-level mode is active
        if per_level_concs.get("detergent") and "detergent_concentration" in factors:
            factors = {k: v for k, v in factors.items() if k != "detergent_concentration"}
        if per_level_concs.get("reducing_agent") and "reducing_agent_concentration" in factors:
            factors = {k: v for k, v in factors.items() if k != "reducing_agent_concentration"}

        # Validate all inputs before proceeding with design generation
        self._validate_inputs(factors, stock_concs, final_volume)

        # Determine unique buffer pH values
        buffer_ph_values = []
        if "buffer pH" in factors:
            unique_phs = set(str(ph).strip() for ph in factors["buffer pH"])
            buffer_ph_values = sorted(unique_phs)

        # Extract unique categorical values (skip None values)
        detergent_values = self._extract_unique_categorical_values(factors, "detergent", skip_none=True)
        reducing_agent_values = self._extract_unique_categorical_values(factors, "reducing_agent", skip_none=True)

        # Build factorial combinations
        factor_names = list(factors.keys())
        level_lists = [factors[f] for f in factor_names]
        combinations = list(itertools.product(*level_lists))

        # Build Excel headers matching Tkinter format:
        # [ID, Plate_96, Well_96, Well_384, Source, Batch, factors..., Response]
        excel_headers = ["ID", "Plate_96", "Well_96", "Well_384", "Source", "Batch"]
        added_factors = set()
        for fn in factor_names:
            if fn in added_factors:
                continue
            excel_headers.append(AVAILABLE_FACTORS.get(fn, fn))
            added_factors.add(fn)
            # Group categorical with their concentration pair
            if fn in CATEGORICAL_PAIRS:
                paired = CATEGORICAL_PAIRS[fn]
                if paired in factor_names and paired not in added_factors:
                    excel_headers.append(AVAILABLE_FACTORS.get(paired, paired))
                    added_factors.add(paired)
        excel_headers.append("Response")

        # Build volume headers matching Tkinter format:
        # [ID, buffer_pH_cols..., detergent_type_cols..., reducing_agent_type_cols..., other_factors..., water]
        volume_headers = ["ID"]
        for ph in buffer_ph_values:
            volume_headers.append(f"buffer_{ph}")
        for det in detergent_values:
            det_clean = det.replace(' ', '_').replace('-', '_').lower()
            volume_headers.append(det_clean)
        for agent in reducing_agent_values:
            agent_clean = agent.replace(' ', '_').replace('-', '_').lower()
            volume_headers.append(agent_clean)
        for factor in factor_names:
            if factor in ("buffer pH", "buffer_concentration", "detergent", "detergent_concentration",
                          "reducing_agent", "reducing_agent_concentration"):
                continue
            volume_headers.append(factor)
        volume_headers.append("water")

        # Calculate design
        excel_rows = []
        volume_rows = []
        volumes_list = []
        well_identifiers = []

        for idx, combo in enumerate(combinations):
            row_dict = {factor_names[i]: combo[i] for i in range(len(factor_names))}

            # Generate well positions using WellMapper (384-well reading order)
            plate_num, well_pos, well_384 = self.well_mapper.generate_well_position_384_order(idx)

            # Excel row with Source and Batch
            excel_row = [idx + 1, plate_num, well_pos, well_384, "FULL_FACTORIAL", 0]
            added_factors_row = set()
            for fn in factor_names:
                if fn in added_factors_row:
                    continue
                excel_row.append(row_dict.get(fn, ""))
                added_factors_row.add(fn)
                if fn in CATEGORICAL_PAIRS:
                    paired = CATEGORICAL_PAIRS[fn]
                    if paired in factor_names and paired not in added_factors_row:
                        excel_row.append(row_dict.get(paired, ""))
                        added_factors_row.add(paired)
            excel_row.append("")  # Empty Response column
            excel_rows.append(excel_row)

            # Volume calculations using VolumeCalculator
            volumes = self.volume_calculator.calculate_volumes(
                row_dict, stock_concs, final_volume, buffer_ph_values,
                protein_stock=protein_stock,
                protein_final=protein_final,
                per_level_concs=per_level_concs
            )

            # Build volume row matching volume_headers order
            volume_row = [idx + 1]  # ID first
            for h in volume_headers[1:]:  # Skip ID
                volume_row.append(volumes.get(h, 0))
            volume_rows.append(volume_row)

            # For categorical factors, ensure all columns are populated
            # (VolumeCalculator only sets the active type, we need 0 for others)
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

    @staticmethod
    def _extract_unique_categorical_values(factors: Dict[str, List[str]],
                                           factor_name: str,
                                           skip_none: bool = False) -> List[str]:
        """Extract unique values for categorical factors."""
        if factor_name not in factors:
            return []
        unique_values = set()
        for val in factors[factor_name]:
            val_str = str(val).strip()
            if skip_none:
                if val_str and val_str.lower() not in ('none', '0', 'nan', ''):
                    unique_values.add(val_str)
            else:
                unique_values.add(val_str)
        return sorted(unique_values)
