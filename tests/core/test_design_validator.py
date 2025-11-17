"""Tests for DesignValidator service"""

import pytest
from core.design_validator import DesignValidator, CategoricalValidator


class TestDesignValidator:
    """Test suite for DesignValidator class"""

    def test_validate_factor_value_valid_ph(self):
        """Test validation of valid pH values"""
        is_valid, msg = DesignValidator.validate_factor_value("buffer pH", 7.0)
        assert is_valid is True
        assert msg == ""

        is_valid, msg = DesignValidator.validate_factor_value("buffer pH", 1.0)
        assert is_valid is True

        is_valid, msg = DesignValidator.validate_factor_value("buffer pH", 14.0)
        assert is_valid is True

    def test_validate_factor_value_invalid_ph(self):
        """Test validation of invalid pH values"""
        is_valid, msg = DesignValidator.validate_factor_value("buffer pH", 0.5)
        assert is_valid is False
        assert "pH" in msg

        is_valid, msg = DesignValidator.validate_factor_value("buffer pH", 15.0)
        assert is_valid is False

    def test_validate_factor_value_valid_percentage(self):
        """Test validation of valid percentage values"""
        is_valid, msg = DesignValidator.validate_factor_value("glycerol", 50.0)
        assert is_valid is True

        is_valid, msg = DesignValidator.validate_factor_value("glycerol", 0.0)
        assert is_valid is True

        is_valid, msg = DesignValidator.validate_factor_value("glycerol", 100.0)
        assert is_valid is True

    def test_validate_factor_value_invalid_percentage(self):
        """Test validation of invalid percentage values"""
        is_valid, msg = DesignValidator.validate_factor_value("glycerol", -10.0)
        assert is_valid is False

        is_valid, msg = DesignValidator.validate_factor_value("dmso", 150.0)
        assert is_valid is False

    def test_validate_factor_value_unconstrained(self):
        """Test validation of factor without constraints"""
        # Factor not in FACTOR_CONSTRAINTS should be valid
        is_valid, msg = DesignValidator.validate_factor_value("unknown_factor", 999.0)
        assert is_valid is True
        assert msg == ""

    def test_validate_factor_levels_valid(self):
        """Test validation of valid factor levels"""
        is_valid, msg = DesignValidator.validate_factor_levels("nacl", ["100", "200", "300"])
        assert is_valid is True
        assert msg == ""

    def test_validate_factor_levels_empty(self):
        """Test validation with empty levels"""
        is_valid, msg = DesignValidator.validate_factor_levels("nacl", [])
        assert is_valid is False
        assert "at least one level" in msg.lower()

    def test_validate_factor_levels_categorical(self):
        """Test validation of categorical factor levels"""
        is_valid, msg = DesignValidator.validate_factor_levels(
            "detergent",
            ["Tween-20", "Triton X-100", "None"]
        )
        assert is_valid is True

    def test_validate_factor_levels_invalid_numeric(self):
        """Test validation with invalid numeric values"""
        is_valid, msg = DesignValidator.validate_factor_levels("nacl", ["100", "abc", "300"])
        assert is_valid is False
        assert "abc" in msg

    def test_validate_factor_levels_out_of_range(self):
        """Test validation with out-of-range values"""
        is_valid, msg = DesignValidator.validate_factor_levels("buffer pH", ["7.0", "15.0"])
        assert is_valid is False
        assert "15.0" in msg

    def test_validate_sample_size_valid(self):
        """Test validation of valid sample sizes"""
        is_valid, msg = DesignValidator.validate_sample_size(100)
        assert is_valid is True

        is_valid, msg = DesignValidator.validate_sample_size(384)
        assert is_valid is True

    def test_validate_sample_size_too_large(self):
        """Test validation of sample size exceeding capacity"""
        is_valid, msg = DesignValidator.validate_sample_size(500)
        assert is_valid is False
        assert "384" in msg

    def test_validate_sample_size_zero_or_negative(self):
        """Test validation of zero or negative sample sizes"""
        is_valid, msg = DesignValidator.validate_sample_size(0)
        assert is_valid is False

        is_valid, msg = DesignValidator.validate_sample_size(-10)
        assert is_valid is False

    def test_validate_lhs_parameters_valid(self):
        """Test validation of valid LHS parameters"""
        is_valid, msg = DesignValidator.validate_lhs_parameters(50, 3)
        assert is_valid is True

        is_valid, msg = DesignValidator.validate_lhs_parameters(384, 5)
        assert is_valid is True

    def test_validate_lhs_parameters_too_few_samples(self):
        """Test LHS with too few samples"""
        is_valid, msg = DesignValidator.validate_lhs_parameters(5, 3)
        assert is_valid is False
        assert "8" in msg

    def test_validate_lhs_parameters_too_many_samples(self):
        """Test LHS with too many samples"""
        is_valid, msg = DesignValidator.validate_lhs_parameters(500, 3)
        assert is_valid is False

    def test_validate_lhs_parameters_too_few_factors(self):
        """Test LHS with too few factors"""
        is_valid, msg = DesignValidator.validate_lhs_parameters(50, 1)
        assert is_valid is False
        assert "2 factors" in msg

    def test_validate_fractional_factorial_valid(self):
        """Test validation of valid fractional factorial parameters"""
        is_valid, msg = DesignValidator.validate_fractional_factorial(5, "IV")
        assert is_valid is True

        is_valid, msg = DesignValidator.validate_fractional_factorial(7, "V")
        assert is_valid is True

    def test_validate_fractional_factorial_too_few_factors(self):
        """Test fractional factorial with too few factors"""
        is_valid, msg = DesignValidator.validate_fractional_factorial(2, "IV")
        assert is_valid is False
        assert "3 factors" in msg

    def test_validate_fractional_factorial_invalid_resolution(self):
        """Test fractional factorial with invalid resolution"""
        is_valid, msg = DesignValidator.validate_fractional_factorial(5, "II")
        assert is_valid is False
        assert "III, IV, or V" in msg

    def test_validate_design_type_requirements_full_factorial(self):
        """Test requirements for full factorial design"""
        is_valid, msg = DesignValidator.validate_design_type_requirements(
            "full_factorial", 3, has_pydoe3=False
        )
        assert is_valid is True

    def test_validate_design_type_requirements_lhs_with_pydoe3(self):
        """Test LHS requirements with pyDOE3 available"""
        is_valid, msg = DesignValidator.validate_design_type_requirements(
            "lhs", 3, has_pydoe3=True
        )
        assert is_valid is True

    def test_validate_design_type_requirements_lhs_without_pydoe3(self):
        """Test LHS requirements without pyDOE3"""
        is_valid, msg = DesignValidator.validate_design_type_requirements(
            "lhs", 3, has_pydoe3=False
        )
        assert is_valid is False
        assert "pyDOE3" in msg

    def test_validate_design_type_requirements_min_factors(self):
        """Test minimum factor requirements"""
        # Box-Behnken requires at least 3 factors
        is_valid, msg = DesignValidator.validate_design_type_requirements(
            "box_behnken", 2, has_pydoe3=True
        )
        assert is_valid is False

        is_valid, msg = DesignValidator.validate_design_type_requirements(
            "box_behnken", 3, has_pydoe3=True
        )
        assert is_valid is True

    def test_validate_stock_concentration_valid(self):
        """Test validation of valid stock concentrations"""
        is_valid, msg = DesignValidator.validate_stock_concentration("nacl", 5000.0)
        assert is_valid is True

    def test_validate_stock_concentration_negative(self):
        """Test validation of negative stock concentration"""
        is_valid, msg = DesignValidator.validate_stock_concentration("nacl", -100.0)
        assert is_valid is False
        assert "negative" in msg.lower()

    def test_validate_stock_concentration_zero(self):
        """Test validation of zero stock concentration"""
        is_valid, msg = DesignValidator.validate_stock_concentration("nacl", 0.0)
        assert is_valid is False
        assert "zero" in msg.lower()

    def test_validate_stock_concentration_categorical(self):
        """Test that categorical factors don't need stock concentration"""
        is_valid, msg = DesignValidator.validate_stock_concentration("buffer pH", None)
        assert is_valid is True

        is_valid, msg = DesignValidator.validate_stock_concentration("detergent", None)
        assert is_valid is True

    def test_check_categorical_concentration_pairing_complete(self):
        """Test categorical-concentration pairing check with complete pairs"""
        factors = {
            "detergent": ["Tween-20"],
            "detergent_concentration": ["0.1"],
            "reducing_agent": ["DTT"],
            "reducing_agent_concentration": ["5.0"]
        }

        is_valid, warnings = DesignValidator.check_categorical_concentration_pairing(factors)

        assert is_valid is True
        assert len(warnings) == 0

    def test_check_categorical_concentration_pairing_missing(self):
        """Test categorical-concentration pairing check with missing pairs"""
        factors = {
            "detergent": ["Tween-20"],
            # Missing detergent_concentration
            "reducing_agent": ["DTT"]
            # Missing reducing_agent_concentration
        }

        is_valid, warnings = DesignValidator.check_categorical_concentration_pairing(factors)

        assert is_valid is True  # Always True, just warnings
        assert len(warnings) == 2
        assert any("detergent" in w.lower() for w in warnings)
        assert any("reducing" in w.lower() for w in warnings)


class TestCategoricalValidator:
    """Test suite for CategoricalValidator class"""

    def test_filter_valid_combination(self):
        """Test that valid combinations are kept"""
        combinations = [
            {"detergent": "Tween-20", "detergent_concentration": 0.1},
            {"detergent": "Triton X-100", "detergent_concentration": 0.5}
        ]

        filtered = CategoricalValidator.filter_invalid_categorical_combinations(combinations)

        assert len(filtered) == 2

    def test_filter_none_detergent_with_concentration(self):
        """Test that None detergent with non-zero concentration is filtered"""
        combinations = [
            {"detergent": "Tween-20", "detergent_concentration": 0.1},
            {"detergent": "None", "detergent_concentration": 0.1},  # Invalid!
            {"detergent": "None", "detergent_concentration": 0.0},  # Valid
        ]

        filtered = CategoricalValidator.filter_invalid_categorical_combinations(combinations)

        assert len(filtered) == 2
        # Check that the invalid one was filtered out
        detergent_values = [c["detergent"] for c in filtered]
        concentrations = [c["detergent_concentration"] for c in filtered]
        assert not any(d == "None" and c != 0.0 for d, c in zip(detergent_values, concentrations))

    def test_filter_detergent_with_zero_concentration(self):
        """Test that actual detergent with zero concentration is filtered"""
        combinations = [
            {"detergent": "Tween-20", "detergent_concentration": 0.1},
            {"detergent": "Tween-20", "detergent_concentration": 0.0},  # Invalid!
        ]

        filtered = CategoricalValidator.filter_invalid_categorical_combinations(combinations)

        assert len(filtered) == 1
        assert filtered[0]["detergent_concentration"] == 0.1

    def test_filter_reducing_agent_combinations(self):
        """Test filtering of reducing agent combinations"""
        combinations = [
            {"reducing_agent": "DTT", "reducing_agent_concentration": 5.0},
            {"reducing_agent": "None", "reducing_agent_concentration": 5.0},  # Invalid!
            {"reducing_agent": "None", "reducing_agent_concentration": 0.0},  # Valid
            {"reducing_agent": "DTT", "reducing_agent_concentration": 0.0},  # Invalid!
        ]

        filtered = CategoricalValidator.filter_invalid_categorical_combinations(combinations)

        assert len(filtered) == 2
        # Valid combinations: DTT with 5.0, None with 0.0

    def test_filter_both_detergent_and_reducing_agent(self):
        """Test filtering with both detergent and reducing agent"""
        combinations = [
            {
                "detergent": "Tween-20",
                "detergent_concentration": 0.1,
                "reducing_agent": "DTT",
                "reducing_agent_concentration": 5.0
            },
            {
                "detergent": "None",
                "detergent_concentration": 0.1,  # Invalid!
                "reducing_agent": "DTT",
                "reducing_agent_concentration": 5.0
            },
            {
                "detergent": "Tween-20",
                "detergent_concentration": 0.1,
                "reducing_agent": "None",
                "reducing_agent_concentration": 5.0  # Invalid!
            }
        ]

        filtered = CategoricalValidator.filter_invalid_categorical_combinations(combinations)

        # Only the first combination is valid
        assert len(filtered) == 1

    def test_is_valid_combination_no_categorical_factors(self):
        """Test validation with no categorical factors"""
        combination = {"nacl": 150.0, "glycerol": 10.0}

        is_valid = CategoricalValidator._is_valid_combination(combination)

        assert is_valid is True
