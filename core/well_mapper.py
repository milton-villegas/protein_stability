"""
Well Plate Mapping Service
Handles well position generation and plate format conversions
"""

from typing import Tuple
from config.design_config import (
    WELLS_PER_PLATE,
    ROWS_PER_PLATE,
    COLS_PER_PLATE,
    PLATE_384_ROWS,
    PLATE_384_COLS
)


class WellMapper:
    """Service for mapping sample indices to well positions and plate formats"""

    @staticmethod
    def generate_well_position(sample_index: int) -> Tuple[int, str]:
        """
        Generate 96-well plate number and well position from sample index.

        Fills column-wise: A1, B1, C1...H1, A2, B2...H12

        Args:
            sample_index: 0-based sample index

        Returns:
            Tuple of (plate_number, well_position)
            - plate_number: 1-based plate number (1, 2, 3, 4, ...)
            - well_position: Well position string (e.g., "A1", "B3", "H12")

        Examples:
            >>> WellMapper.generate_well_position(0)
            (1, 'A1')
            >>> WellMapper.generate_well_position(8)
            (1, 'A2')
            >>> WellMapper.generate_well_position(96)
            (2, 'A1')
        """
        plate_number = (sample_index // WELLS_PER_PLATE) + 1
        position_in_plate = sample_index % WELLS_PER_PLATE

        # Fill down columns first (A1, B1, C1... H1, then A2, B2...)
        row_index = position_in_plate % ROWS_PER_PLATE  # 0-7 for A-H
        col_index = position_in_plate // ROWS_PER_PLATE  # 0-11 for 1-12

        row_letter = chr(ord('A') + row_index)
        col_number = col_index + 1
        well_position = f"{row_letter}{col_number}"

        return plate_number, well_position

    @staticmethod
    def convert_96_to_384_well(plate_96: int, well_96: str) -> str:
        """
        Convert 96-well plate position to 384-well plate position.

        Mapping rules:
        - Plate 1 → columns 1-6, Plate 2 → columns 7-12, etc.
        - Odd columns (1,3,5,7,9,11) use first row of pair:
          A→A, B→C, C→E, D→G, E→I, F→K, G→M, H→O
        - Even columns (2,4,6,8,10,12) use second row of pair:
          A→B, B→D, C→F, D→H, E→J, F→L, G→N, H→P

        Args:
            plate_96: 96-well plate number (1-based)
            well_96: 96-well position (e.g., "B3", "H12")

        Returns:
            384-well position string (e.g., "D7", "P24")

        Examples:
            >>> WellMapper.convert_96_to_384_well(1, "A1")
            'A1'
            >>> WellMapper.convert_96_to_384_well(1, "A2")
            'B1'
            >>> WellMapper.convert_96_to_384_well(1, "B3")
            'D2'
            >>> WellMapper.convert_96_to_384_well(2, "A1")
            'A7'
        """
        # Parse 96-well position
        row_96_letter = well_96[0]
        col_96 = int(well_96[1:])

        # Convert row letter to index (A=0, B=1, ..., H=7)
        row_96_index = ord(row_96_letter) - ord('A')

        # Map to 384-well column
        # Each 96-well plate occupies 6 columns in the 384-well plate
        # Plate 1: cols 1-6, Plate 2: cols 7-12, Plate 3: cols 13-18, Plate 4: cols 19-24
        base_col_384 = (plate_96 - 1) * 6

        # Within each plate, 96-well columns 1-12 map to:
        # - Columns 1,2 → 384-col 1
        # - Columns 3,4 → 384-col 2
        # - Columns 5,6 → 384-col 3
        # - Columns 7,8 → 384-col 4
        # - Columns 9,10 → 384-col 5
        # - Columns 11,12 → 384-col 6
        import math
        col_384 = base_col_384 + math.ceil(col_96 / 2)

        # Map row based on column parity
        # Odd columns (1,3,5,7,9,11): A→A(0), B→C(2), C→E(4), D→G(6), E→I(8), F→K(10), G→M(12), H→O(14)
        # Even columns (2,4,6,8,10,12): A→B(1), B→D(3), C→F(5), D→H(7), E→J(9), F→L(11), G→N(13), H→P(15)
        if col_96 % 2 == 1:  # Odd column
            row_384_index = row_96_index * 2
        else:  # Even column
            row_384_index = row_96_index * 2 + 1

        # Convert back to letter
        row_384_letter = chr(ord('A') + row_384_index)

        return f"{row_384_letter}{col_384}"

    @staticmethod
    def validate_96_well_position(well: str) -> bool:
        """
        Validate that a string represents a valid 96-well position.

        Args:
            well: Well position string (e.g., "A1", "H12")

        Returns:
            True if valid, False otherwise

        Examples:
            >>> WellMapper.validate_96_well_position("A1")
            True
            >>> WellMapper.validate_96_well_position("H12")
            True
            >>> WellMapper.validate_96_well_position("I1")
            False
            >>> WellMapper.validate_96_well_position("A13")
            False
        """
        if not well or len(well) < 2:
            return False

        try:
            row = well[0]
            col = int(well[1:])

            # Valid row: A-H (0-7)
            row_index = ord(row.upper()) - ord('A')
            if row_index < 0 or row_index >= ROWS_PER_PLATE:
                return False

            # Valid column: 1-12
            if col < 1 or col > COLS_PER_PLATE:
                return False

            return True
        except (ValueError, IndexError):
            return False

    @staticmethod
    def validate_384_well_position(well: str) -> bool:
        """
        Validate that a string represents a valid 384-well position.

        Args:
            well: Well position string (e.g., "A1", "P24")

        Returns:
            True if valid, False otherwise

        Examples:
            >>> WellMapper.validate_384_well_position("A1")
            True
            >>> WellMapper.validate_384_well_position("P24")
            True
            >>> WellMapper.validate_384_well_position("Q1")
            False
            >>> WellMapper.validate_384_well_position("A25")
            False
        """
        if not well or len(well) < 2:
            return False

        try:
            row = well[0]
            col = int(well[1:])

            # Valid row: A-P (0-15)
            row_index = ord(row.upper()) - ord('A')
            if row_index < 0 or row_index >= PLATE_384_ROWS:
                return False

            # Valid column: 1-24
            if col < 1 or col > PLATE_384_COLS:
                return False

            return True
        except (ValueError, IndexError):
            return False

    @staticmethod
    def calculate_required_plates(num_samples: int) -> int:
        """
        Calculate number of 96-well plates required for given sample count.

        Args:
            num_samples: Number of samples

        Returns:
            Number of plates needed (1-4)

        Examples:
            >>> WellMapper.calculate_required_plates(50)
            1
            >>> WellMapper.calculate_required_plates(96)
            1
            >>> WellMapper.calculate_required_plates(97)
            2
            >>> WellMapper.calculate_required_plates(200)
            3
        """
        if num_samples <= 0:
            return 0
        return (num_samples + WELLS_PER_PLATE - 1) // WELLS_PER_PLATE
