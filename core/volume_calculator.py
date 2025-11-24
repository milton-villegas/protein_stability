"""
Volume Calculation Service
Handles reagent volume calculations using C1V1 = C2V2 formula
"""

from typing import Dict, List, Optional, Tuple
from config.design_config import CATEGORICAL_FACTORS, NONE_VALUES


class VolumeCalculator:
    """Service for calculating reagent volumes for experimental designs"""

    @staticmethod
    def calculate_volumes(
        factor_values: Dict[str, float],
        stock_concentrations: Dict[str, float],
        final_volume: float,
        buffer_ph_values: Optional[List[str]] = None,
        protein_stock: Optional[float] = None,
        protein_final: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate reagent volumes using C1V1 = C2V2 formula.

        Formula: V1 = (C2 * V2) / C1
        Where:
            C1 = stock concentration
            V1 = volume to add (what we calculate)
            C2 = desired final concentration
            V2 = final volume

        Args:
            factor_values: Dict of factor_name → desired concentration
            stock_concentrations: Dict of factor_name → stock concentration
            final_volume: Total final volume (µL)
            buffer_ph_values: List of unique pH values (for buffer handling)
            protein_stock: Protein stock concentration (mg/mL), optional
            protein_final: Desired final protein concentration (mg/mL), optional

        Returns:
            Dict of reagent → volume (µL)

        Examples:
            >>> calc = VolumeCalculator()
            >>> factor_values = {"nacl": 150.0, "glycerol": 10.0}
            >>> stock_concs = {"nacl": 5000.0, "glycerol": 100.0}
            >>> volumes = calc.calculate_volumes(factor_values, stock_concs, 200.0)
            >>> volumes["nacl"]
            6.0
            >>> volumes["glycerol"]
            20.0
            >>> volumes["water"]
            174.0
        """
        volumes = {}
        total_volume_used = 0.0

        # Handle buffer pH if present
        if "buffer pH" in factor_values and buffer_ph_values:
            buffer_vols, buffer_vol_used = VolumeCalculator._calculate_buffer_volumes(
                factor_values, stock_concentrations, final_volume, buffer_ph_values
            )
            volumes.update(buffer_vols)
            total_volume_used += buffer_vol_used

        # Handle detergent with concentration pairing
        if "detergent" in factor_values and "detergent_concentration" in factor_values:
            det_vols, det_vol_used = VolumeCalculator._calculate_categorical_volumes(
                factor_values["detergent"],
                factor_values.get("detergent_concentration", 0),
                stock_concentrations.get("detergent_concentration", 0),
                final_volume,
                "detergent"
            )
            volumes.update(det_vols)
            total_volume_used += det_vol_used

        # Handle reducing agent with concentration pairing
        if "reducing_agent" in factor_values and "reducing_agent_concentration" in factor_values:
            agent_vols, agent_vol_used = VolumeCalculator._calculate_categorical_volumes(
                factor_values["reducing_agent"],
                factor_values.get("reducing_agent_concentration", 0),
                stock_concentrations.get("reducing_agent_concentration", 0),
                final_volume,
                "reducing_agent"
            )
            volumes.update(agent_vols)
            total_volume_used += agent_vol_used

        # Calculate volumes for other factors
        for factor_name, desired_conc in factor_values.items():
            if factor_name in CATEGORICAL_FACTORS:
                continue
            if factor_name in ["buffer_concentration", "detergent_concentration", "reducing_agent_concentration"]:
                continue

            if factor_name in stock_concentrations:
                vol = VolumeCalculator._calculate_component_volume(
                    desired_conc, stock_concentrations[factor_name], final_volume
                )
                volumes[factor_name] = vol
                total_volume_used += vol

        # Calculate protein volume if concentrations provided
        if protein_stock and protein_final and protein_stock > 0:
            protein_volume = VolumeCalculator._calculate_component_volume(
                protein_final, protein_stock, final_volume
            )
            volumes["protein"] = protein_volume
            total_volume_used += protein_volume
        else:
            volumes["protein"] = 0.0

        # Calculate water to reach final volume
        water_volume = round(final_volume - total_volume_used, 2)
        volumes["water"] = water_volume

        return volumes

    @staticmethod
    def _calculate_component_volume(
        desired_concentration: float,
        stock_concentration: float,
        final_volume: float
    ) -> float:
        """
        Calculate volume for a single component using C1V1 = C2V2.

        Args:
            desired_concentration: Desired final concentration
            stock_concentration: Stock concentration
            final_volume: Final volume

        Returns:
            Volume in µL (rounded to 2 decimal places)
        """
        try:
            if stock_concentration <= 0:
                return 0.0
            volume = (float(desired_concentration) * final_volume) / stock_concentration
            return round(volume, 2)
        except (ValueError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def _calculate_buffer_volumes(
        factor_values: Dict[str, float],
        stock_concentrations: Dict[str, float],
        final_volume: float,
        buffer_ph_values: List[str]
    ) -> Tuple[Dict[str, float], float]:
        """
        Calculate buffer volumes with pH selection.

        Args:
            factor_values: Factor values including "buffer pH"
            stock_concentrations: Stock concentrations
            final_volume: Final volume
            buffer_ph_values: List of unique pH values

        Returns:
            Tuple of (volumes_dict, total_volume_used)
        """
        volumes = {}
        total_volume = 0.0

        buffer_ph = str(factor_values["buffer pH"])

        # Initialize all pH buffer volumes to 0
        for ph in buffer_ph_values:
            volumes[f"buffer_{ph}"] = 0.0

        # Calculate volume for the selected pH buffer
        if "buffer_concentration" in factor_values and "buffer_concentration" in stock_concentrations:
            desired_conc = factor_values["buffer_concentration"]
            stock_conc = stock_concentrations["buffer_concentration"]
            volume = VolumeCalculator._calculate_component_volume(
                desired_conc, stock_conc, final_volume
            )
            volumes[f"buffer_{buffer_ph}"] = volume
            total_volume = volume

        return volumes, total_volume

    @staticmethod
    def _calculate_categorical_volumes(
        categorical_value: str,
        concentration: float,
        stock_concentration: float,
        final_volume: float,
        factor_prefix: str
    ) -> Tuple[Dict[str, float], float]:
        """
        Calculate volumes for categorical factors (detergent, reducing_agent).

        For categorical factors:
        - If the value is "None" or equivalent, volume = 0
        - Otherwise, calculate volume based on concentration

        Args:
            categorical_value: Value of the categorical factor (e.g., "Tween-20")
            concentration: Desired concentration
            stock_concentration: Stock concentration
            final_volume: Final volume
            factor_prefix: Prefix for volume key (e.g., "detergent")

        Returns:
            Tuple of (volumes_dict, total_volume_used)
        """
        volumes = {}
        total_volume = 0.0

        # Normalize the categorical value
        normalized_value = VolumeCalculator._normalize_factor_name(categorical_value)

        # Check if this is a "None" value
        is_none = VolumeCalculator._is_none_value(categorical_value)

        if is_none or concentration == 0:
            volumes[f"{factor_prefix}_{normalized_value}"] = 0.0
        else:
            volume = VolumeCalculator._calculate_component_volume(
                concentration, stock_concentration, final_volume
            )
            volumes[f"{factor_prefix}_{normalized_value}"] = volume
            total_volume = volume

        return volumes, total_volume

    @staticmethod
    def _normalize_factor_name(name: str) -> str:
        """
        Normalize factor names for consistent column headers.

        Replaces spaces and hyphens with underscores, converts to lowercase.

        Args:
            name: Factor name

        Returns:
            Normalized name

        Examples:
            >>> VolumeCalculator._normalize_factor_name("Tween-20")
            'tween_20'
            >>> VolumeCalculator._normalize_factor_name("None")
            'none'
        """
        return str(name).replace(' ', '_').replace('-', '_').lower()

    @staticmethod
    def _is_none_value(value: str) -> bool:
        """
        Check if a string value represents None/empty.

        Args:
            value: String value to check

        Returns:
            True if value represents None, False otherwise

        Examples:
            >>> VolumeCalculator._is_none_value("None")
            True
            >>> VolumeCalculator._is_none_value("0")
            True
            >>> VolumeCalculator._is_none_value("Tween-20")
            False
        """
        normalized = str(value).strip().lower()
        return normalized in NONE_VALUES


class VolumeValidator:
    """Validator for volume calculations and design feasibility"""

    @staticmethod
    def validate_design_feasibility(
        volumes_list: List[Dict[str, float]],
        well_identifiers: List[Tuple[int, str]]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that a design is feasible (no negative water volumes).

        Args:
            volumes_list: List of volume dictionaries for each sample
            well_identifiers: List of (well_id, well_position) tuples

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if design is feasible, False otherwise
            - error_message: Detailed error message if not feasible, None otherwise

        Examples:
            >>> volumes = [{"water": 100.0}, {"water": -10.0}]
            >>> ids = [(1, "A1"), (2, "A2")]
            >>> valid, msg = VolumeValidator.validate_design_feasibility(volumes, ids)
            >>> valid
            False
        """
        negative_water_wells = []

        for idx, volumes in enumerate(volumes_list):
            water_vol = volumes.get("water", 0)
            if water_vol < 0:
                well_id, well_pos = well_identifiers[idx]
                negative_water_wells.append((well_id, well_pos, water_vol))

        if not negative_water_wells:
            return True, None

        # Build detailed error message
        error_msg = "⚠️ IMPOSSIBLE DESIGN DETECTED ⚠️\n\n"
        error_msg += "The following wells require NEGATIVE water volumes:\n\n"

        # Show first 5 problematic wells
        for well_id, well_pos, water_vol in negative_water_wells[:5]:
            error_msg += f"  • Well {well_pos} (ID {well_id}): {water_vol} µL water\n"

        if len(negative_water_wells) > 5:
            error_msg += f"  ... and {len(negative_water_wells) - 5} more wells\n"

        error_msg += f"\nTotal problematic wells: {len(negative_water_wells)}\n\n"
        error_msg += "Solutions:\n"
        error_msg += "  1. INCREASE stock concentrations\n"
        error_msg += "  2. INCREASE final volume\n"
        error_msg += "  3. REDUCE desired concentration levels\n"

        return False, error_msg

    @staticmethod
    def check_stock_concentrations(
        factor_values: Dict[str, float],
        stock_concentrations: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Check that all required factors have stock concentrations defined.

        Args:
            factor_values: Factor values to check
            stock_concentrations: Available stock concentrations

        Returns:
            Tuple of (all_present, missing_factors)
            - all_present: True if all stocks are defined, False otherwise
            - missing_factors: List of factor names missing stock concentrations

        Examples:
            >>> factors = {"nacl": 150.0, "glycerol": 10.0}
            >>> stocks = {"nacl": 5000.0}
            >>> present, missing = VolumeValidator.check_stock_concentrations(factors, stocks)
            >>> present
            False
            >>> missing
            ['glycerol']
        """
        missing = []

        for factor_name in factor_values.keys():
            # Skip categorical factors and concentration sub-factors
            if factor_name in CATEGORICAL_FACTORS:
                continue
            if "_concentration" in factor_name:
                # Check parent factor instead
                parent = factor_name.replace("_concentration", "")
                if parent not in stock_concentrations and factor_name not in stock_concentrations:
                    missing.append(factor_name)
            elif factor_name not in stock_concentrations:
                missing.append(factor_name)

        return len(missing) == 0, missing
