#!/usr/bin/env python3
"""
Standalone test for ID tracking in Pareto frontier
"""

import pandas as pd
import numpy as np
import sys

try:
    from core.optimizer import BayesianOptimizer
    AX_AVAILABLE = True
except ImportError:
    print("❌ Could not import BayesianOptimizer or Ax not available")
    sys.exit(1)

def test_pareto_frontier_with_id():
    """Test that Pareto frontier includes ID from original data"""
    print("Testing Pareto frontier ID tracking...")

    np.random.seed(42)
    n = 15

    # Create data with ID column
    data = pd.DataFrame({
        'ID': range(1, n + 1),
        'Buffer pH': np.random.choice([6.0, 7.0, 8.0, 9.0], n),
        'NaCl (mM)': np.random.uniform(0, 200, n),
        'Glycerol (%)': np.random.uniform(0, 20, n),
        'Tm': np.random.uniform(45, 55, n),
        'Aggregation': np.random.uniform(5, 15, n)
    })

    print(f"Created test data with {len(data)} rows")
    print(f"ID column: {data['ID'].tolist()}")

    optimizer = BayesianOptimizer()
    optimizer.set_data(
        data=data,
        factor_columns=['Buffer pH', 'NaCl (mM)', 'Glycerol (%)'],
        categorical_factors=['Buffer pH'],
        numeric_factors=['NaCl (mM)', 'Glycerol (%)'],
        response_columns=['Tm', 'Aggregation'],
        response_directions={'Tm': 'maximize', 'Aggregation': 'minimize'}
    )

    print("\nInitializing optimizer...")
    optimizer.initialize_optimizer()

    print("Getting Pareto frontier...")
    pareto_points = optimizer.get_pareto_frontier()

    if pareto_points is None:
        print("❌ FAILED: Pareto frontier is None")
        return False

    print(f"\n✓ Found {len(pareto_points)} Pareto points")

    # Check each Pareto point
    all_passed = True
    for i, point in enumerate(pareto_points, 1):
        print(f"\nPareto Point {i}:")

        # Check required fields
        if 'id' not in point:
            print(f"  ❌ FAILED: Missing 'id' field")
            all_passed = False
            continue

        if 'row_index' not in point:
            print(f"  ❌ FAILED: Missing 'row_index' field")
            all_passed = False
            continue

        if 'parameters' not in point:
            print(f"  ❌ FAILED: Missing 'parameters' field")
            all_passed = False
            continue

        if 'objectives' not in point:
            print(f"  ❌ FAILED: Missing 'objectives' field")
            all_passed = False
            continue

        # Print point info
        print(f"  ID: {point['id']}")
        print(f"  Row Index: {point['row_index']}")
        print(f"  Parameters: {point['parameters']}")
        print(f"  Objectives: {point['objectives']}")

        # Validate ID
        if point['id'] is not None:
            if not isinstance(point['id'], (int, np.integer)):
                print(f"  ❌ FAILED: ID is not an integer (type: {type(point['id'])})")
                all_passed = False
            elif not (1 <= point['id'] <= n):
                print(f"  ❌ FAILED: ID {point['id']} is out of range [1, {n}]")
                all_passed = False
            else:
                print(f"  ✓ ID is valid")
        else:
            print(f"  ⚠️  WARNING: ID is None")

        # Validate row_index
        if point['row_index'] is not None:
            if point['row_index'] not in data.index:
                print(f"  ❌ FAILED: row_index {point['row_index']} not in data")
                all_passed = False
            else:
                print(f"  ✓ Row index is valid")
        else:
            print(f"  ⚠️  WARNING: row_index is None")

    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    try:
        success = test_pareto_frontier_with_id()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
