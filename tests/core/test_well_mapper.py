"""Tests for WellMapper service"""

import pytest
from core.well_mapper import WellMapper


class TestWellMapper:
    """Test suite for WellMapper class"""

    def test_generate_well_position_first_well(self):
        """Test first well position (A1)"""
        plate, well = WellMapper.generate_well_position(0)
        assert plate == 1
        assert well == "A1"

    def test_generate_well_position_column_fill(self):
        """Test that wells fill down columns first"""
        # Second position should be B1 (down the column)
        plate, well = WellMapper.generate_well_position(1)
        assert plate == 1
        assert well == "B1"

        # 8th position should be H1 (last in first column)
        plate, well = WellMapper.generate_well_position(7)
        assert plate == 1
        assert well == "H1"

        # 9th position should be A2 (start of second column)
        plate, well = WellMapper.generate_well_position(8)
        assert plate == 1
        assert well == "A2"

    def test_generate_well_position_last_well_plate_1(self):
        """Test last well of first plate (H12)"""
        plate, well = WellMapper.generate_well_position(95)
        assert plate == 1
        assert well == "H12"

    def test_generate_well_position_second_plate(self):
        """Test first well of second plate"""
        plate, well = WellMapper.generate_well_position(96)
        assert plate == 2
        assert well == "A1"

    def test_generate_well_position_multiple_plates(self):
        """Test wells across multiple plates"""
        # Plate 3, well A1
        plate, well = WellMapper.generate_well_position(192)
        assert plate == 3
        assert well == "A1"

        # Plate 4, well B5
        plate, well = WellMapper.generate_well_position(96 * 3 + 32)
        assert plate == 4
        assert well == "A5"

    def test_convert_96_to_384_first_plate_odd_column(self):
        """Test 96 to 384 conversion for plate 1, odd columns"""
        # A1 (plate 1, odd column) → A1
        well_384 = WellMapper.convert_96_to_384_well(1, "A1")
        assert well_384 == "A1"

        # B3 (plate 1, odd column) → C2
        well_384 = WellMapper.convert_96_to_384_well(1, "B3")
        assert well_384 == "C2"

    def test_convert_96_to_384_first_plate_even_column(self):
        """Test 96 to 384 conversion for plate 1, even columns"""
        # A2 (plate 1, even column) → B1
        well_384 = WellMapper.convert_96_to_384_well(1, "A2")
        assert well_384 == "B1"

        # B4 (plate 1, even column) → D2
        well_384 = WellMapper.convert_96_to_384_well(1, "B4")
        assert well_384 == "D2"

    def test_convert_96_to_384_second_plate(self):
        """Test 96 to 384 conversion for plate 2"""
        # A1 on plate 2 → A7 (columns 7-12)
        well_384 = WellMapper.convert_96_to_384_well(2, "A1")
        assert well_384 == "A7"

        # B3 on plate 2 → C8
        well_384 = WellMapper.convert_96_to_384_well(2, "B3")
        assert well_384 == "C8"

    def test_convert_96_to_384_all_rows(self):
        """Test 96 to 384 conversion for all rows"""
        # Test all 8 rows with odd column
        expected_rows_odd = ["A", "C", "E", "G", "I", "K", "M", "O"]
        for i, expected_row in enumerate(expected_rows_odd):
            row_letter = chr(ord('A') + i)
            well_384 = WellMapper.convert_96_to_384_well(1, f"{row_letter}1")
            assert well_384[0] == expected_row

        # Test all 8 rows with even column
        expected_rows_even = ["B", "D", "F", "H", "J", "L", "N", "P"]
        for i, expected_row in enumerate(expected_rows_even):
            row_letter = chr(ord('A') + i)
            well_384 = WellMapper.convert_96_to_384_well(1, f"{row_letter}2")
            assert well_384[0] == expected_row

    def test_validate_96_well_position_valid(self):
        """Test validation of valid 96-well positions"""
        assert WellMapper.validate_96_well_position("A1") is True
        assert WellMapper.validate_96_well_position("H12") is True
        assert WellMapper.validate_96_well_position("D7") is True

    def test_validate_96_well_position_invalid_row(self):
        """Test validation with invalid rows"""
        assert WellMapper.validate_96_well_position("I1") is False  # Row I doesn't exist
        assert WellMapper.validate_96_well_position("Z5") is False

    def test_validate_96_well_position_invalid_column(self):
        """Test validation with invalid columns"""
        assert WellMapper.validate_96_well_position("A13") is False  # Col 13 doesn't exist
        assert WellMapper.validate_96_well_position("B0") is False

    def test_validate_96_well_position_invalid_format(self):
        """Test validation with invalid formats"""
        assert WellMapper.validate_96_well_position("") is False
        assert WellMapper.validate_96_well_position("A") is False
        assert WellMapper.validate_96_well_position("1A") is False
        assert WellMapper.validate_96_well_position("AA1") is False

    def test_validate_384_well_position_valid(self):
        """Test validation of valid 384-well positions"""
        assert WellMapper.validate_384_well_position("A1") is True
        assert WellMapper.validate_384_well_position("P24") is True
        assert WellMapper.validate_384_well_position("H12") is True

    def test_validate_384_well_position_invalid_row(self):
        """Test validation with invalid rows for 384-well"""
        assert WellMapper.validate_384_well_position("Q1") is False  # Row Q doesn't exist
        assert WellMapper.validate_384_well_position("Z5") is False

    def test_validate_384_well_position_invalid_column(self):
        """Test validation with invalid columns for 384-well"""
        assert WellMapper.validate_384_well_position("A25") is False  # Col 25 doesn't exist
        assert WellMapper.validate_384_well_position("B0") is False

    def test_calculate_required_plates(self):
        """Test calculation of required plates"""
        assert WellMapper.calculate_required_plates(50) == 1
        assert WellMapper.calculate_required_plates(96) == 1
        assert WellMapper.calculate_required_plates(97) == 2
        assert WellMapper.calculate_required_plates(192) == 2
        assert WellMapper.calculate_required_plates(193) == 3
        assert WellMapper.calculate_required_plates(288) == 3
        assert WellMapper.calculate_required_plates(289) == 4
        assert WellMapper.calculate_required_plates(384) == 4

    def test_calculate_required_plates_edge_cases(self):
        """Test edge cases for plate calculation"""
        assert WellMapper.calculate_required_plates(0) == 0
        assert WellMapper.calculate_required_plates(1) == 1
