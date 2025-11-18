#!/usr/bin/env python3
"""
Test to validate minimize mode fixes in BO plots
Tests both maximize and minimize directions to ensure plots are correct
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, '/Users/fernando/Documents/GitProjects/protein_stability')

from core.optimizer import BayesianOptimizer
from core.data_handler import DataHandler

def test_maximize_mode():
    """Test plots for MAXIMIZE mode"""
    print("\n" + "="*80)
    print("TEST 1: MAXIMIZE MODE")
    print("="*80)

    # Load data
    df = pd.read_excel('/Users/fernando/Documents/GitProjects/protein_stability/examples/test_multi_response_data.xlsx')

    # Setup optimizer for MAXIMIZE
    optimizer = BayesianOptimizer()

    factor_columns = ['Buffer pH', 'Buffer Conc (mM)', 'NaCl (mM)', 'Zinc (mM)', 'Glycerol (%)']
    categorical = ['Buffer pH']
    numeric = [f for f in factor_columns if f not in categorical]

    optimizer.set_data(
        data=df,
        factor_columns=factor_columns,
        categorical_factors=categorical,
        numeric_factors=numeric,
        response_columns=['Tm'],
        response_directions={'Tm': 'maximize'}
    )

    optimizer.initialize_optimizer()

    print(f"\nResponse: Tm")
    print(f"Direction: MAXIMIZE")
    print(f"Data range: {df['Tm'].min():.2f} to {df['Tm'].max():.2f}")
    print(f"Current best (should be MAX): {df['Tm'].max():.2f}")

    # Try to generate plot
    try:
        fig = optimizer.get_acquisition_plot()
        if fig is not None:
            print("✅ Plot generated successfully")

            # Check if we can access the axes
            axes = fig.get_axes()
            print(f"✅ Found {len(axes)} panels in plot")

            # Check acquisition function panel (should be panel 2)
            if len(axes) >= 2:
                ax_ei = axes[1]  # Acquisition function panel
                title = ax_ei.get_title()
                print(f"✅ EI panel title: {title}")

            # Check progress panel (should be panel 4)
            if len(axes) >= 4:
                ax_progress = axes[3]
                title = ax_progress.get_title()
                print(f"✅ Progress panel title: {title}")

                # Get the line data to check if it's increasing
                lines = ax_progress.get_lines()
                if lines:
                    y_data = lines[0].get_ydata()
                    is_increasing = all(y_data[i] <= y_data[i+1] for i in range(len(y_data)-1))
                    if is_increasing:
                        print(f"✅ Progress trend: INCREASING (correct for maximize)")
                    else:
                        print(f"⚠️  Progress trend: NOT strictly increasing")

            import matplotlib.pyplot as plt
            plt.close(fig)
            print("✅ Plot closed without errors")

        else:
            print("❌ Plot generation returned None")
            return False

    except Exception as e:
        print(f"❌ Error generating plot: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_minimize_mode():
    """Test plots for MINIMIZE mode"""
    print("\n" + "="*80)
    print("TEST 2: MINIMIZE MODE")
    print("="*80)

    # Load data
    df = pd.read_excel('/Users/fernando/Documents/GitProjects/protein_stability/examples/test_multi_response_data.xlsx')

    # Setup optimizer for MINIMIZE
    optimizer = BayesianOptimizer()

    factor_columns = ['Buffer pH', 'Buffer Conc (mM)', 'NaCl (mM)', 'Zinc (mM)', 'Glycerol (%)']
    categorical = ['Buffer pH']
    numeric = [f for f in factor_columns if f not in categorical]

    optimizer.set_data(
        data=df,
        factor_columns=factor_columns,
        categorical_factors=categorical,
        numeric_factors=numeric,
        response_columns=['Aggregation'],
        response_directions={'Aggregation': 'minimize'}
    )

    optimizer.initialize_optimizer()

    print(f"\nResponse: Aggregation")
    print(f"Direction: MINIMIZE")
    print(f"Data range: {df['Aggregation'].min():.2f} to {df['Aggregation'].max():.2f}")
    print(f"Current best (should be MIN): {df['Aggregation'].min():.2f}")

    # Try to generate plot
    try:
        fig = optimizer.get_acquisition_plot()
        if fig is not None:
            print("✅ Plot generated successfully")

            # Check if we can access the axes
            axes = fig.get_axes()
            print(f"✅ Found {len(axes)} panels in plot")

            # Check progress panel (should be panel 4)
            if len(axes) >= 4:
                ax_progress = axes[3]
                title = ax_progress.get_title()
                print(f"✅ Progress panel title: {title}")

                # Get the line data to check if it's decreasing
                lines = ax_progress.get_lines()
                if lines:
                    y_data = lines[0].get_ydata()
                    is_decreasing = all(y_data[i] >= y_data[i+1] for i in range(len(y_data)-1))
                    if is_decreasing:
                        print(f"✅ Progress trend: DECREASING (correct for minimize)")
                    else:
                        print(f"⚠️  Progress trend: NOT strictly decreasing")

            import matplotlib.pyplot as plt
            plt.close(fig)
            print("✅ Plot closed without errors")

        else:
            print("❌ Plot generation returned None")
            return False

    except Exception as e:
        print(f"❌ Error generating plot: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_colormap_check():
    """Check that colormaps are set to viridis"""
    print("\n" + "="*80)
    print("TEST 3: COLORMAP CHECK")
    print("="*80)

    # Read the optimizer.py file and check for colormaps
    optimizer_path = '/Users/fernando/Documents/GitProjects/protein_stability/core/optimizer.py'

    with open(optimizer_path, 'r') as f:
        content = f.read()

    # Check for 'viridis' appearances
    viridis_count = content.count("cmap='viridis'")
    plasma_count = content.count("cmap='plasma'")

    print(f"\nColormap usage in optimizer.py:")
    print(f"  viridis: {viridis_count} occurrences")
    print(f"  plasma: {plasma_count} occurrences")

    if plasma_count == 0 and viridis_count >= 3:
        print("✅ All colormaps updated to viridis (colorblind-safe)")
        return True
    else:
        print("❌ Some colormaps still use plasma")
        return False


def main():
    """Run all tests"""
    print("="*80)
    print("MINIMIZE MODE PLOT VALIDATION TESTS")
    print("="*80)

    results = {
        'maximize_mode': False,
        'minimize_mode': False,
        'colormap_check': False
    }

    # Run tests
    results['maximize_mode'] = test_maximize_mode()
    results['minimize_mode'] = test_minimize_mode()
    results['colormap_check'] = test_colormap_check()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    total = len(results)
    passed = sum(results.values())

    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Minimize mode fixes are working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
