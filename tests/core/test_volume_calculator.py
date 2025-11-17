"""Tests for VolumeCalculator service"""

import pytest
from core.volume_calculator import VolumeCalculator, VolumeValidator


class TestVolumeCalculator:
    """Test suite for VolumeCalculator class"""

    def test_calculate_component_volume_basic(self):
        """Test basic C1V1 = C2V2 calculation"""
        # Want 150 mM in 200 µL from 5000 mM stock
        # V1 = (150 * 200) / 5000 = 6 µL
        volume = VolumeCalculator._calculate_component_volume(150.0, 5000.0, 200.0)
        assert volume == 6.0

    def test_calculate_component_volume_zero_stock(self):
        """Test handling of zero stock concentration"""
        volume = VolumeCalculator._calculate_component_volume(150.0, 0.0, 200.0)
        assert volume == 0.0

    def test_calculate_component_volume_zero_desired(self):
        """Test handling of zero desired concentration"""
        volume = VolumeCalculator._calculate_component_volume(0.0, 5000.0, 200.0)
        assert volume == 0.0

    def test_calculate_volumes_simple_factors(self):
        """Test volume calculation for simple factors"""
        factor_values = {
            "nacl": 150.0,
            "glycerol": 10.0
        }
        stock_concentrations = {
            "nacl": 5000.0,
            "glycerol": 100.0
        }
        final_volume = 200.0

        volumes = VolumeCalculator.calculate_volumes(
            factor_values, stock_concentrations, final_volume
        )

        # NaCl: (150 * 200) / 5000 = 6.0 µL
        assert volumes["nacl"] == 6.0

        # Glycerol: (10 * 200) / 100 = 20.0 µL
        assert volumes["glycerol"] == 20.0

        # Water: 200 - 6 - 20 = 174.0 µL
        assert volumes["water"] == 174.0

    def test_calculate_volumes_with_buffer(self):
        """Test volume calculation with buffer pH"""
        factor_values = {
            "buffer pH": "7.0",
            "buffer_concentration": 50.0,
            "nacl": 150.0
        }
        stock_concentrations = {
            "buffer_concentration": 1000.0,
            "nacl": 5000.0
        }
        final_volume = 200.0
        buffer_ph_values = ["6.5", "7.0", "7.5"]

        volumes = VolumeCalculator.calculate_volumes(
            factor_values, stock_concentrations, final_volume, buffer_ph_values
        )

        # Buffer 7.0: (50 * 200) / 1000 = 10.0 µL
        assert volumes["buffer_7.0"] == 10.0

        # Other pH buffers should be 0
        assert volumes["buffer_6.5"] == 0.0
        assert volumes["buffer_7.5"] == 0.0

        # NaCl: 6.0 µL
        assert volumes["nacl"] == 6.0

        # Water: 200 - 10 - 6 = 184.0 µL
        assert volumes["water"] == 184.0

    def test_normalize_factor_name(self):
        """Test factor name normalization"""
        assert VolumeCalculator._normalize_factor_name("Tween-20") == "tween_20"
        assert VolumeCalculator._normalize_factor_name("Triton X-100") == "triton_x_100"
        assert VolumeCalculator._normalize_factor_name("None") == "none"
        assert VolumeCalculator._normalize_factor_name("DTT") == "dtt"

    def test_is_none_value(self):
        """Test None value detection"""
        assert VolumeCalculator._is_none_value("None") is True
        assert VolumeCalculator._is_none_value("none") is True
        assert VolumeCalculator._is_none_value("NONE") is True
        assert VolumeCalculator._is_none_value("0") is True
        assert VolumeCalculator._is_none_value("") is True
        assert VolumeCalculator._is_none_value("nan") is True
        assert VolumeCalculator._is_none_value("Tween-20") is False
        assert VolumeCalculator._is_none_value("DTT") is False

    def test_calculate_categorical_volumes_with_value(self):
        """Test categorical factor volume calculation with actual value"""
        volumes, total = VolumeCalculator._calculate_categorical_volumes(
            "Tween-20", 0.1, 10.0, 200.0, "detergent"
        )

        # (0.1 * 200) / 10.0 = 2.0 µL
        assert volumes["detergent_tween_20"] == 2.0
        assert total == 2.0

    def test_calculate_categorical_volumes_with_none(self):
        """Test categorical factor volume calculation with None value"""
        volumes, total = VolumeCalculator._calculate_categorical_volumes(
            "None", 0.0, 10.0, 200.0, "detergent"
        )

        assert volumes["detergent_none"] == 0.0
        assert total == 0.0

    def test_calculate_volumes_high_concentration_negative_water(self):
        """Test that high concentrations can lead to negative water"""
        factor_values = {
            "nacl": 5000.0,  # Very high concentration
            "glycerol": 90.0
        }
        stock_concentrations = {
            "nacl": 5000.0,  # Stock same as desired
            "glycerol": 100.0
        }
        final_volume = 200.0

        volumes = VolumeCalculator.calculate_volumes(
            factor_values, stock_concentrations, final_volume
        )

        # NaCl: (5000 * 200) / 5000 = 200 µL
        # Glycerol: (90 * 200) / 100 = 180 µL
        # Total = 380 µL > 200 µL final volume
        # Water should be negative
        assert volumes["water"] < 0


class TestVolumeValidator:
    """Test suite for VolumeValidator class"""

    def test_validate_design_feasibility_valid(self):
        """Test validation of feasible design"""
        volumes_list = [
            {"nacl": 6.0, "glycerol": 20.0, "water": 174.0},
            {"nacl": 10.0, "glycerol": 30.0, "water": 160.0}
        ]
        well_identifiers = [(1, "A1"), (2, "A2")]

        is_valid, error_msg = VolumeValidator.validate_design_feasibility(
            volumes_list, well_identifiers
        )

        assert is_valid is True
        assert error_msg is None

    def test_validate_design_feasibility_invalid(self):
        """Test validation of infeasible design (negative water)"""
        volumes_list = [
            {"nacl": 6.0, "glycerol": 20.0, "water": 174.0},
            {"nacl": 150.0, "glycerol": 80.0, "water": -30.0},  # Negative!
            {"nacl": 200.0, "glycerol": 50.0, "water": -50.0}   # Negative!
        ]
        well_identifiers = [(1, "A1"), (2, "A2"), (3, "A3")]

        is_valid, error_msg = VolumeValidator.validate_design_feasibility(
            volumes_list, well_identifiers
        )

        assert is_valid is False
        assert error_msg is not None
        assert "IMPOSSIBLE DESIGN" in error_msg
        assert "A2" in error_msg
        assert "A3" in error_msg
        assert "-30.0" in error_msg

    def test_validate_design_feasibility_many_errors(self):
        """Test validation with many errors (shows only first 5)"""
        volumes_list = [{"water": -10.0 * i} for i in range(1, 11)]
        well_identifiers = [(i, f"A{i}") for i in range(1, 11)]

        is_valid, error_msg = VolumeValidator.validate_design_feasibility(
            volumes_list, well_identifiers
        )

        assert is_valid is False
        assert "and 5 more wells" in error_msg
        assert "Total problematic wells: 10" in error_msg

    def test_check_stock_concentrations_all_present(self):
        """Test stock concentration check when all are present"""
        factor_values = {"nacl": 150.0, "glycerol": 10.0}
        stock_concentrations = {"nacl": 5000.0, "glycerol": 100.0}

        all_present, missing = VolumeValidator.check_stock_concentrations(
            factor_values, stock_concentrations
        )

        assert all_present is True
        assert len(missing) == 0

    def test_check_stock_concentrations_missing(self):
        """Test stock concentration check when some are missing"""
        factor_values = {"nacl": 150.0, "glycerol": 10.0, "mgcl2": 5.0}
        stock_concentrations = {"nacl": 5000.0}

        all_present, missing = VolumeValidator.check_stock_concentrations(
            factor_values, stock_concentrations
        )

        assert all_present is False
        assert "glycerol" in missing
        assert "mgcl2" in missing
        assert "nacl" not in missing

    def test_check_stock_concentrations_categorical_factors(self):
        """Test that categorical factors are skipped"""
        factor_values = {
            "buffer pH": "7.0",
            "detergent": "Tween-20",
            "nacl": 150.0
        }
        stock_concentrations = {"nacl": 5000.0}

        all_present, missing = VolumeValidator.check_stock_concentrations(
            factor_values, stock_concentrations
        )

        assert all_present is True
        assert len(missing) == 0
