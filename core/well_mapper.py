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
    def reverse_map_384_to_96(well_384_index: int) -> Tuple[int, str]:
        """
        Reverse-map from 384-well sequential index to 96-well position.

        This function takes a sample index (0, 1, 2, 3...) and treats it as
        a 384-well position in reading order (A1, A2, A3...), then determines
        which 96-well plate and position would transfer to that 384 location.

        Args:
            well_384_index: 0-based index in 384-well reading order (row-major)

        Returns:
            Tuple of (plate_96, well_96) where the sample should be placed
            - plate_96: 96-well plate number (1-4)
            - well_96: 96-well position string (e.g., "A1", "B3")

        Examples:
            >>> WellMapper.reverse_map_384_to_96(0)  # 384 A1
            (1, 'A1')
            >>> WellMapper.reverse_map_384_to_96(1)  # 384 A2
            (1, 'A3')
            >>> WellMapper.reverse_map_384_to_96(2)  # 384 A3
            (1, 'A5')
        """
        import math

        # Convert index to 384-well position (row-major order: A1, A2, A3...)
        row_384_index = well_384_index // PLATE_384_COLS  # 0-15 for A-P
        col_384 = (well_384_index % PLATE_384_COLS) + 1    # 1-24

        # Determine which 96-well plate (each plate occupies 6 columns in 384)
        # Plate 1: cols 1-6, Plate 2: cols 7-12, Plate 3: cols 13-18, Plate 4: cols 19-24
        plate_96 = ((col_384 - 1) // 6) + 1

        # Determine position within the 6-column block for this plate
        col_within_plate = ((col_384 - 1) % 6) + 1  # 1-6

        # Reverse the column mapping:
        # 384 cols 1,2,3,4,5,6 → 96 cols 1,2,3,4,5,6,7,8,9,10,11,12
        # 384 col 1 → 96 cols 1,2 | 384 col 2 → 96 cols 3,4 | etc.
        base_col_96 = (col_within_plate - 1) * 2 + 1  # 1,3,5,7,9,11

        # Reverse the row mapping:
        # 384 rows A,B (0,1) → 96 row A | 384 rows C,D (2,3) → 96 row B | etc.
        row_96_index = row_384_index // 2  # 0-7 for A-H
        row_96_letter = chr(ord('A') + row_96_index)

        # Determine if odd or even row in 384 determines column offset
        # Even 384 row (0,2,4,6,8,10,12,14) → odd 96 column (1,3,5,7,9,11)
        # Odd 384 row (1,3,5,7,9,11,13,15) → even 96 column (2,4,6,8,10,12)
        if row_384_index % 2 == 0:  # Even row in 384
            col_96 = base_col_96  # Odd column
        else:  # Odd row in 384
            col_96 = base_col_96 + 1  # Even column

        well_96 = f"{row_96_letter}{col_96}"
        return plate_96, well_96

    @staticmethod
    def generate_well_position_384_order(sample_index: int) -> Tuple[int, str, str]:
        """
        Generate 96-well position for a sample, ordered by 384-well column-major
        order within each 96-well plate batch.

        Samples 0-95 fill Plate 1, 96-191 fill Plate 2, etc.
        Within each batch, positions are assigned in column-major order in the
        384-well space (fill down columns first: A1, B1, C1...P1, A2, B2...).

        Args:
            sample_index: 0-based sample index

        Returns:
            Tuple of (plate_96, well_96, well_384)
            - plate_96: 96-well plate number (1-4)
            - well_96: 96-well position where sample should be placed
            - well_384: Final 384-well position (for reference)

        Examples:
            >>> WellMapper.generate_well_position_384_order(0)
            (1, 'A1', 'A1')
            >>> WellMapper.generate_well_position_384_order(1)
            (1, 'A2', 'B1')
        """
        # Determine which plate (1-4), 96 samples per plate
        plate_96 = (sample_index // WELLS_PER_PLATE) + 1
        pos_in_batch = sample_index % WELLS_PER_PLATE

        # Column-major order in the 384-well space:
        # Each plate occupies 6 columns (16 rows x 6 cols = 96 positions)
        # Fill down columns first: row cycles 0-15, then next column
        row_384 = pos_in_batch % PLATE_384_ROWS          # 0-15 (A-P)
        col_within_plate = pos_in_batch // PLATE_384_ROWS  # 0-5

        # Absolute 384-well column (1-24)
        col_384 = (plate_96 - 1) * 6 + col_within_plate + 1

        # Convert to 384-well index (row-major) for reverse mapping
        well_384_index = row_384 * PLATE_384_COLS + (col_384 - 1)

        # Reverse map to get the 96-well position
        _, well_96 = WellMapper.reverse_map_384_to_96(well_384_index)

        # Build 384-well position string
        well_384 = f"{chr(ord('A') + row_384)}{col_384}"

        return plate_96, well_96, well_384

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
