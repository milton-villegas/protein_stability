"""Tests for DoEDesigner class"""

import pytest
import pandas as pd
from core.doe_designer import DoEDesigner


class TestDoEDesignerBasic:
    """Test basic DoEDesigner functionality"""

    def test_initialization(self):
        """Test DoEDesigner initializes correctly"""
        designer = DoEDesigner()
        assert designer.well_mapper is not None
        assert designer.volume_calculator is not None
        assert designer.volume_validator is not None

    def test_build_simple_factorial_design(self):
        """Test building a simple 2x2 factorial design"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200'],
            'glycerol': ['5', '10']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0
        }
        final_volume = 200.0

        excel_df, volume_df = designer.build_factorial_design(
            factors, stock_concs, final_volume
        )

        # Check number of combinations
        assert len(excel_df) == 4
        assert len(volume_df) == 4

        # Check Excel columns
        assert 'ID' in excel_df.columns
        assert 'Plate_96' in excel_df.columns
        assert 'Well_96' in excel_df.columns
        assert 'Well_384' in excel_df.columns
        assert 'NaCl (mM)' in excel_df.columns
        assert 'Glycerol (%)' in excel_df.columns
        assert 'Response' in excel_df.columns

        # Check volume columns
        assert 'nacl' in volume_df.columns
        assert 'glycerol' in volume_df.columns
        assert 'water' in volume_df.columns

        # Check IDs are sequential
        assert excel_df['ID'].tolist() == [1, 2, 3, 4]

        # Check all wells are on plate 1
        assert all(excel_df['Plate_96'] == 1)

    def test_build_three_factor_design(self):
        """Test building a 3-factor design"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200'],
            'glycerol': ['5', '10'],
            'mgcl2': ['1', '5']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0,
            'mgcl2': 1000.0
        }

        excel_df, volume_df = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        # 2 * 2 * 2 = 8 combinations
        assert len(excel_df) == 8
        assert len(volume_df) == 8

    def test_build_multilevel_design(self):
        """Test building design with different number of levels"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200', '300'],
            'glycerol': ['5', '10']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0
        }

        excel_df, volume_df = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        # 3 * 2 = 6 combinations
        assert len(excel_df) == 6


class TestDoEDesignerWellMapping:
    """Test well mapping functionality"""

    def test_well_positions_column_major(self):
        """Test that wells are ordered column-major in 384-well space within each plate"""
        designer = DoEDesigner()

        # Create design with 8 combinations
        # Column-major 384: fills down columns first (A1,B1,C1...P1,A2...)
        # 8 samples all fit on plate 1
        factors = {
            'nacl': ['100', '200'],
            'glycerol': ['5', '10'],
            'mgcl2': ['1', '5']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0,
            'mgcl2': 1000.0
        }

        excel_df, _ = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        # Column-major in 384 space: first 8 positions fill column 1
        # 384 A1→96 A1, 384 B1→96 A2, 384 C1→96 B1, 384 D1→96 B2, etc.
        expected_wells = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2']
        assert excel_df['Well_96'].tolist() == expected_wells
        # All 8 samples on plate 1
        assert all(excel_df['Plate_96'] == 1)

    def test_well_positions_multiple_columns(self):
        """Test well positions with column-major 384-well ordering"""
        designer = DoEDesigner()

        # Create design with 16 combinations
        # Column-major fills first column of 384 (16 rows), all on plate 1
        factors = {
            'nacl': ['100', '200', '300', '400'],
            'glycerol': ['5', '10'],
            'mgcl2': ['1', '5']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0,
            'mgcl2': 1000.0
        }

        excel_df, _ = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        # All 16 samples on plate 1 (96 per plate)
        assert all(excel_df['Plate_96'] == 1)
        # First sample: 384 A1 → 96 A1
        assert excel_df['Well_96'].iloc[0] == 'A1'
        # Sample 16 fills exactly one 384 column (16 rows A-P)
        assert excel_df['Well_96'].iloc[15] == 'H2'

    def test_384_well_conversion(self):
        """Test 96-well to 384-well conversion"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200']
        }
        stock_concs = {
            'nacl': 5000.0
        }

        excel_df, _ = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        # Check 384-well positions exist and are valid
        assert 'Well_384' in excel_df.columns
        assert all(excel_df['Well_384'].str.match(r'^[A-P]\d+$'))

    def test_multiple_plates(self):
        """Test design spanning multiple plates with column-major 384-well ordering"""
        designer = DoEDesigner()

        # Create design with 20 combinations
        # With column-major ordering, 96 samples per plate
        # 20 samples all fit on plate 1
        factors = {
            'nacl': ['100', '200', '300', '400', '500'],
            'glycerol': ['5', '10', '15', '20']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0
        }

        excel_df, _ = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        # 5 * 4 = 20 combinations, all on plate 1
        assert len(excel_df) == 20
        assert excel_df['Plate_96'].max() == 1

        # Test with >96 combinations to span multiple plates
        factors_large = {
            'nacl': ['100', '200', '300', '400', '500'],
            'glycerol': ['5', '10', '15', '20'],
            'mgcl2': ['1', '5', '10', '15', '20']
        }
        stock_concs_large = {
            'nacl': 5000.0,
            'glycerol': 100.0,
            'mgcl2': 1000.0
        }

        excel_df, _ = designer.build_factorial_design(
            factors_large, stock_concs_large, 200.0
        )

        # 5 * 4 * 5 = 100 combinations → 2 plates (96 + 4)
        assert len(excel_df) == 100
        assert excel_df['Plate_96'].max() == 2


class TestDoEDesignerBufferPH:
    """Test buffer pH handling"""

    def test_buffer_ph_single_value(self):
        """Test design with single buffer pH value"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200'],
            'buffer pH': ['7.0']
        }
        stock_concs = {
            'nacl': 5000.0
        }

        excel_df, volume_df = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        # Check buffer pH column in volume_df
        assert 'buffer_7.0' in volume_df.columns
        assert 'nacl' in volume_df.columns
        assert 'water' in volume_df.columns

    def test_buffer_ph_multiple_values(self):
        """Test design with multiple buffer pH values"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200'],
            'buffer pH': ['6.0', '7.0', '8.0']
        }
        stock_concs = {
            'nacl': 5000.0
        }

        excel_df, volume_df = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        # Check all buffer pH columns exist
        assert 'buffer_6.0' in volume_df.columns
        assert 'buffer_7.0' in volume_df.columns
        assert 'buffer_8.0' in volume_df.columns

        # 2 * 3 = 6 combinations
        assert len(excel_df) == 6

    def test_buffer_ph_sorted_order(self):
        """Test that buffer pH values are sorted in volume headers"""
        designer = DoEDesigner()

        factors = {
            'buffer pH': ['8.0', '6.0', '7.0']  # Unsorted
        }
        stock_concs = {}

        excel_df, volume_df = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        # Buffer columns should be sorted
        buffer_cols = [col for col in volume_df.columns if col.startswith('buffer')]
        assert buffer_cols == ['buffer_6.0', 'buffer_7.0', 'buffer_8.0']


class TestDoEDesignerVolumeCalculations:
    """Test volume calculations"""

    def test_volumes_sum_to_final_volume(self):
        """Test that all volumes sum to final volume"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200'],
            'glycerol': ['5', '10']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0
        }
        final_volume = 200.0

        excel_df, volume_df = designer.build_factorial_design(
            factors, stock_concs, final_volume
        )

        # Check each row sums to final volume (exclude ID column)
        vol_cols = [c for c in volume_df.columns if c != "ID"]
        for idx, row in volume_df.iterrows():
            total = row[vol_cols].sum()
            assert abs(total - final_volume) < 0.1  # Allow small floating point errors

    def test_water_volume_positive(self):
        """Test that water volumes are positive"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200'],
            'glycerol': ['5', '10']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0
        }

        excel_df, volume_df = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        # All water volumes should be positive
        assert all(volume_df['water'] > 0)

    def test_negative_water_raises_error(self):
        """Test that impossible designs raise ValueError"""
        designer = DoEDesigner()

        # Design that requires more volume than available
        # Use values within valid range but with low stock concentrations
        # to cause negative water volumes
        factors = {
            'nacl': ['5000'],    # High but within max (10000)
            'glycerol': ['90']   # High but within max (100)
        }
        stock_concs = {
            'nacl': 5000.0,   # Low stock: needs 100% of volume just for nacl!
            'glycerol': 100.0
        }
        final_volume = 200.0

        with pytest.raises(ValueError, match="NEGATIVE water|IMPOSSIBLE DESIGN"):
            designer.build_factorial_design(factors, stock_concs, final_volume)


class TestDoEDesignerEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_factors_raises_error(self):
        """Test that empty factors dict raises ValueError"""
        designer = DoEDesigner()

        with pytest.raises(ValueError, match="No factors defined"):
            designer.build_factorial_design({}, {}, 200.0)

    def test_single_factor_single_level(self):
        """Test design with single factor, single level"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100']
        }
        stock_concs = {
            'nacl': 5000.0
        }

        excel_df, volume_df = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        assert len(excel_df) == 1
        assert len(volume_df) == 1

    def test_large_design_near_max_capacity(self):
        """Test large design approaching 384-well limit"""
        designer = DoEDesigner()

        # Create design with ~200 combinations
        factors = {
            'nacl': ['100', '200', '300', '400', '500'],
            'glycerol': ['5', '10', '15', '20'],
            'buffer pH': ['6.0', '7.0', '8.0'],
            'mgcl2': ['1', '5']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0,
            'mgcl2': 1000.0
        }

        excel_df, volume_df = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        # 5 * 4 * 3 * 2 = 120 combinations
        assert len(excel_df) == 120
        assert len(volume_df) == 120

        # Check maximum plate number
        max_plate = excel_df['Plate_96'].max()
        assert max_plate <= 4  # Should fit within 4 plates

    def test_different_final_volumes(self):
        """Test with different final volumes"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200']
        }
        stock_concs = {
            'nacl': 5000.0
        }

        # Test with 100 µL
        excel_df, volume_df = designer.build_factorial_design(
            factors, stock_concs, 100.0
        )
        vol_cols = [c for c in volume_df.columns if c != "ID"]
        assert abs(volume_df.iloc[0][vol_cols].sum() - 100.0) < 0.1

        # Test with 300 µL
        excel_df, volume_df = designer.build_factorial_design(
            factors, stock_concs, 300.0
        )
        vol_cols = [c for c in volume_df.columns if c != "ID"]
        assert abs(volume_df.iloc[0][vol_cols].sum() - 300.0) < 0.1


class TestDoEDesignerCategoricalFactors:
    """Test categorical factor handling"""

    def test_categorical_detergent(self):
        """Test design with categorical detergent factor"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200'],
            'detergent': ['Tween-20', 'Triton X-100'],
            'detergent_concentration': ['0.1', '0.5']
        }
        stock_concs = {
            'nacl': 5000.0,
            'detergent_concentration': 10.0
        }

        excel_df, volume_df = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        # 2 * 2 * 2 = 8 combinations
        assert len(excel_df) == 8

        # Check detergent column in Excel (should have "Detergent" header)
        assert 'Detergent' in excel_df.columns or 'detergent' in excel_df.columns

    def test_categorical_buffer_ph(self):
        """Test buffer pH as categorical factor"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100'],
            'buffer pH': ['6.0', '7.0', '8.0']
        }
        stock_concs = {
            'nacl': 5000.0
        }

        excel_df, volume_df = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        # Check pH values in Excel df
        assert 'Buffer pH' in excel_df.columns or 'buffer pH' in excel_df.columns


class TestDoEDesignerIntegration:
    """Integration tests with service dependencies"""

    def test_uses_well_mapper_service(self):
        """Test that DoEDesigner uses WellMapper service"""
        designer = DoEDesigner()

        # WellMapper should be initialized
        assert designer.well_mapper is not None

        # Test that well generation works
        factors = {'nacl': ['100']}
        stock_concs = {'nacl': 5000.0}

        excel_df, _ = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        # Should have well positions
        assert 'Well_96' in excel_df.columns
        assert 'Well_384' in excel_df.columns

    def test_uses_volume_calculator_service(self):
        """Test that DoEDesigner uses VolumeCalculator service"""
        designer = DoEDesigner()

        # VolumeCalculator should be initialized
        assert designer.volume_calculator is not None

        # Test volume calculation
        factors = {
            'nacl': ['100', '200']
        }
        stock_concs = {
            'nacl': 5000.0
        }

        _, volume_df = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        # Should have volume columns
        assert 'nacl' in volume_df.columns
        assert 'water' in volume_df.columns

        # Volumes should be calculated correctly
        assert all(volume_df['nacl'] > 0)
        assert all(volume_df['water'] > 0)

    def test_uses_volume_validator_service(self):
        """Test that DoEDesigner uses VolumeValidator service"""
        designer = DoEDesigner()

        # VolumeValidator should be initialized
        assert designer.volume_validator is not None

        # Test that validation catches impossible designs
        # Use a design that will definitely fail (require more than 100% of final volume)
        factors = {
            'nacl': ['100000'],  # Extremely high concentration
            'glycerol': ['99']   # Nearly 100% glycerol
        }
        stock_concs = {
            'nacl': 100000.0,
            'glycerol': 100.0
        }

        with pytest.raises(ValueError):
            designer.build_factorial_design(factors, stock_concs, 200.0)


class TestDoEDesignerDataFrameStructure:
    """Test DataFrame output structure"""

    def test_excel_df_has_correct_dtypes(self):
        """Test that Excel DataFrame has appropriate data types"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200'],
            'glycerol': ['5', '10']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0
        }

        excel_df, _ = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        # ID should be numeric
        assert excel_df['ID'].dtype in ['int64', 'int32']

        # Plate should be numeric
        assert excel_df['Plate_96'].dtype in ['int64', 'int32']

        # Well positions should be strings
        assert excel_df['Well_96'].dtype == 'object'
        assert excel_df['Well_384'].dtype == 'object'

    def test_volume_df_all_numeric(self):
        """Test that volume DataFrame is all numeric"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200'],
            'glycerol': ['5', '10']
        }
        stock_concs = {
            'nacl': 5000.0,
            'glycerol': 100.0
        }

        _, volume_df = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        # All columns should be numeric
        for col in volume_df.columns:
            assert volume_df[col].dtype in ['float64', 'float32', 'int64', 'int32']

    def test_excel_df_no_missing_values(self):
        """Test that Excel DataFrame has no unexpected missing values"""
        designer = DoEDesigner()

        factors = {
            'nacl': ['100', '200'],
            'buffer pH': ['7.0', '8.0']
        }
        stock_concs = {
            'nacl': 5000.0
        }

        excel_df, _ = designer.build_factorial_design(
            factors, stock_concs, 200.0
        )

        # Only Response column should have empty values
        for col in excel_df.columns:
            if col != 'Response':
                assert excel_df[col].notna().all()
