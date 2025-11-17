"""Tests for DoEDesigner input validation"""

import pytest
from core.doe_designer import DoEDesigner


class TestInputValidationBasic:
    """Test basic input validation"""

    def test_empty_factors_raises_error(self):
        """Test that empty factors raises clear error"""
        designer = DoEDesigner()

        with pytest.raises(ValueError, match="No factors defined"):
            designer._validate_inputs({}, {}, 200.0)

    def test_factor_with_no_levels(self):
        """Test that factor with empty levels list raises error"""
        designer = DoEDesigner()

        factors = {
            'nacl': []  # No levels!
        }
        stock_concs = {'nacl': 5000.0}

        with pytest.raises(ValueError, match="has no levels"):
            designer._validate_inputs(factors, stock_concs, 200.0)


class TestFactorLevelValidation:
    """Test validation of factor levels"""

    def test_invalid_ph_value(self):
        """Test that invalid pH value is caught"""
        designer = DoEDesigner()

        # Note: buffer pH is categorical, so any string is valid
        # Test numeric pH constraint instead
        factors = {
            'nacl': ['100', '200'],
            'buffer_concentration': ['-10', '50']  # Negative is invalid
        }
        stock_concs = {
            'nacl': 5000.0,
            'buffer_concentration': 1000.0
        }

        with pytest.raises(ValueError, match="Invalid factor"):
            designer._validate_inputs(factors, stock_concs, 200.0)

    def test_invalid_percentage_value(self):
        """Test that invalid percentage is caught"""
        designer = DoEDesigner()

        factors = {
            'glycerol': ['50', '150']  # 150% is invalid!
        }
        stock_concs = {'glycerol': 100.0}

        with pytest.raises(ValueError, match="Invalid factor 'glycerol'"):
            designer._validate_inputs(factors, stock_concs, 200.0)

    def test_negative_concentration(self):
        """Test that negative concentration is caught"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['-100', '200']  # Negative is invalid!
        }
        stock_concs = {'nacl': 5000.0}

        with pytest.raises(ValueError, match="Invalid factor 'nacl'"):
            designer._validate_inputs(factors, stock_concs, 200.0)

    def test_non_numeric_value_for_numeric_factor(self):
        """Test that non-numeric value is caught"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', 'abc', '300']  # 'abc' is invalid!
        }
        stock_concs = {'nacl': 5000.0}

        with pytest.raises(ValueError, match="Invalid factor 'nacl'"):
            designer._validate_inputs(factors, stock_concs, 200.0)

    def test_valid_factor_levels_pass(self):
        """Test that valid factor levels pass validation"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200', '300'],
            'glycerol': ['5', '10']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0
        }

        # Should not raise
        designer._validate_inputs(factors, stock_concs, 200.0)


class TestStockConcentrationValidation:
    """Test validation of stock concentrations"""

    def test_missing_stock_concentration(self):
        """Test that missing stock concentration is caught"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200'],
            'glycerol': ['5', '10']
        }
        stock_concs = {
            'nacl': 5000.0
            # Missing glycerol!
        }

        with pytest.raises(ValueError, match="Invalid stock concentration for 'glycerol'"):
            designer._validate_inputs(factors, stock_concs, 200.0)

    def test_negative_stock_concentration(self):
        """Test that negative stock concentration is caught"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200']
        }
        stock_concs = {
            'nacl': -5000.0  # Negative!
        }

        with pytest.raises(ValueError, match="Invalid stock concentration for 'nacl'"):
            designer._validate_inputs(factors, stock_concs, 200.0)

    def test_zero_stock_concentration(self):
        """Test that zero stock concentration is caught"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200']
        }
        stock_concs = {
            'nacl': 0.0  # Zero!
        }

        with pytest.raises(ValueError, match="Invalid stock concentration for 'nacl'"):
            designer._validate_inputs(factors, stock_concs, 200.0)

    def test_categorical_factors_dont_need_stock(self):
        """Test that categorical factors don't require stock concentrations"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200'],
            'buffer pH': ['6.0', '7.0']  # Categorical
        }
        stock_concs = {
            'nacl': 5000.0
            # No stock for buffer pH - should be OK
        }

        # Should not raise
        designer._validate_inputs(factors, stock_concs, 200.0)

    def test_detergent_categorical_no_stock_needed(self):
        """Test that detergent (categorical) doesn't need stock"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100'],
            'detergent': ['Tween-20', 'None'],
            'detergent_concentration': ['0.1']
        }
        stock_concs = {
            'nacl': 5000.0,
            'detergent_concentration': 10.0
            # No stock for 'detergent' - should be OK
        }

        # Should not raise
        designer._validate_inputs(factors, stock_concs, 200.0)


class TestDesignSizeValidation:
    """Test validation of design size against plate capacity"""

    def test_design_exceeds_max_wells(self):
        """Test that design exceeding 384 wells is caught"""
        designer = DoEDesigner()

        # 5 * 5 * 4 * 5 = 500 > 384
        factors = {
            'nacl': ['100', '200', '300', '400', '500'],
            'glycerol': ['5', '10', '15', '20', '25'],
            'buffer pH': ['6', '7', '8', '9'],
            'mgcl2': ['1', '2', '3', '4', '5']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0,
            'mgcl2': 1000.0
        }

        with pytest.raises(ValueError, match="Design too large.*500.*384"):
            designer._validate_inputs(factors, stock_concs, 200.0)

    def test_design_at_max_capacity_passes(self):
        """Test that design at exactly 384 wells passes"""
        designer = DoEDesigner()

        # Create design with exactly 384 combinations
        # 8 * 8 * 6 = 384
        factors = {
            'nacl': ['100', '150', '200', '250', '300', '350', '400', '450'],
            'glycerol': ['5', '10', '15', '20', '25', '30', '35', '40'],
            'mgcl2': ['1', '2', '3', '4', '5', '6']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0,
            'mgcl2': 1000.0
        }

        # Should not raise
        designer._validate_inputs(factors, stock_concs, 200.0)

    def test_design_just_over_limit(self):
        """Test that design with 385 combinations fails"""
        designer = DoEDesigner()

        # 5 * 7 * 11 = 385 > 384
        factors = {
            'nacl': ['100', '200', '300', '400', '500'],
            'glycerol': ['5', '10', '15', '20', '25', '30', '35'],
            'mgcl2': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0,
            'mgcl2': 1000.0
        }

        with pytest.raises(ValueError, match="Design too large.*385.*384"):
            designer._validate_inputs(factors, stock_concs, 200.0)

    def test_small_design_passes(self):
        """Test that small design passes"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200'],
            'glycerol': ['5', '10']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0
        }

        # Should not raise
        designer._validate_inputs(factors, stock_concs, 200.0)


class TestFinalVolumeValidation:
    """Test validation of final volume"""

    def test_negative_final_volume(self):
        """Test that negative final volume is caught"""
        designer = DoEDesigner()

        factors = {'nacl': ['100']}
        stock_concs = {'nacl': 5000.0}

        with pytest.raises(ValueError, match="Final volume must be positive"):
            designer._validate_inputs(factors, stock_concs, -200.0)

    def test_zero_final_volume(self):
        """Test that zero final volume is caught"""
        designer = DoEDesigner()

        factors = {'nacl': ['100']}
        stock_concs = {'nacl': 5000.0}

        with pytest.raises(ValueError, match="Final volume must be positive"):
            designer._validate_inputs(factors, stock_concs, 0.0)

    def test_volume_exceeds_plate_capacity(self):
        """Test that volume exceeding 96-well capacity is caught"""
        designer = DoEDesigner()

        factors = {'nacl': ['100']}
        stock_concs = {'nacl': 5000.0}

        with pytest.raises(ValueError, match="exceeds 96-well plate capacity"):
            designer._validate_inputs(factors, stock_concs, 400.0)

    def test_valid_volumes_pass(self):
        """Test that valid volumes pass"""
        designer = DoEDesigner()

        factors = {'nacl': ['100']}
        stock_concs = {'nacl': 5000.0}

        # Test various valid volumes
        for volume in [50.0, 100.0, 200.0, 300.0, 323.0]:
            designer._validate_inputs(factors, stock_concs, volume)


class TestValidationIntegration:
    """Test validation integrated with build_factorial_design"""

    def test_build_catches_invalid_concentration(self):
        """Test that build_factorial_design catches invalid concentration"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['-100', '200']  # Negative is invalid
        }
        stock_concs = {
            'nacl': 5000.0
        }

        with pytest.raises(ValueError, match="Invalid factor"):
            designer.build_factorial_design(factors, stock_concs, 200.0)

    def test_build_catches_missing_stock(self):
        """Test that build_factorial_design catches missing stock"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200']
        }
        stock_concs = {}  # Missing stock!

        with pytest.raises(ValueError, match="Invalid stock concentration"):
            designer.build_factorial_design(factors, stock_concs, 200.0)

    def test_build_catches_oversized_design(self):
        """Test that build_factorial_design catches oversized design"""
        designer = DoEDesigner()

        # 500 combinations
        factors = {
            'nacl': ['100', '200', '300', '400', '500'],
            'glycerol': ['5', '10', '15', '20', '25'],
            'buffer pH': ['6', '7', '8', '9'],
            'mgcl2': ['1', '2', '3', '4', '5']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0,
            'mgcl2': 1000.0
        }

        with pytest.raises(ValueError, match="Design too large"):
            designer.build_factorial_design(factors, stock_concs, 200.0)

    def test_build_succeeds_with_valid_inputs(self):
        """Test that build_factorial_design succeeds with valid inputs"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200'],
            'glycerol': ['5', '10']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0
        }

        # Should succeed
        excel_df, volume_df = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        assert len(excel_df) == 4
        assert len(volume_df) == 4


class TestErrorMessages:
    """Test that error messages are helpful"""

    def test_oversized_design_error_has_solutions(self):
        """Test that oversized design error suggests solutions"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            'glycerol': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            'mgcl2': ['1', '2', '3', '4', '5']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0,
            'mgcl2': 1000.0
        }

        with pytest.raises(ValueError) as exc_info:
            designer._validate_inputs(factors, stock_concs, 200.0)

        error_msg = str(exc_info.value)
        # Check error message includes solutions
        assert "Solutions:" in error_msg
        assert "Reduce" in error_msg or "LHS" in error_msg

    def test_missing_stock_error_is_clear(self):
        """Test that missing stock error is clear"""
        designer = DoEDesigner()

        factors = {'nacl': ['100']}
        stock_concs = {}

        with pytest.raises(ValueError) as exc_info:
            designer._validate_inputs(factors, stock_concs, 200.0)

        error_msg = str(exc_info.value)
        assert "nacl" in error_msg
        assert "stock concentration" in error_msg.lower()

    def test_invalid_concentration_error_is_clear(self):
        """Test that invalid concentration error is clear"""
        designer = DoEDesigner()

        factors = {'nacl': ['-100']}  # Negative concentration
        stock_concs = {'nacl': 5000.0}

        with pytest.raises(ValueError) as exc_info:
            designer._validate_inputs(factors, stock_concs, 200.0)

        error_msg = str(exc_info.value)
        assert "nacl" in error_msg.lower()
        assert "invalid" in error_msg.lower()
