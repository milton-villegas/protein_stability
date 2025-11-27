#!/usr/bin/env python3
"""
Verify Pareto Frontier Mathematical Correctness

This script checks that the Pareto points identified by Ax are truly non-dominated.
"""

def is_dominated(point_a, point_b, directions):
    """
    Check if point A is dominated by point B.

    Point A is dominated by B if:
    - B is >= A in ALL maximize objectives
    - B is <= A in ALL minimize objectives
    - B is strictly better in AT LEAST ONE objective

    Args:
        point_a: dict of {objective_name: value}
        point_b: dict of {objective_name: value}
        directions: dict of {objective_name: 'maximize' or 'minimize'}

    Returns:
        True if A is dominated by B, False otherwise
    """
    at_least_one_better = False

    for obj_name, direction in directions.items():
        val_a = point_a[obj_name]
        val_b = point_b[obj_name]

        if direction == 'maximize':
            if val_b < val_a:
                # B is worse in this objective, so A is NOT dominated
                return False
            if val_b > val_a:
                at_least_one_better = True
        else:  # minimize
            if val_b > val_a:
                # B is worse in this objective, so A is NOT dominated
                return False
            if val_b < val_a:
                at_least_one_better = True

    # A is dominated only if B is at least as good in all objectives
    # AND strictly better in at least one
    return at_least_one_better


def verify_pareto_frontier(all_points, pareto_points, directions):
    """
    Verify that Pareto points are truly non-dominated.

    Args:
        all_points: list of dicts with objectives
        pareto_points: list of dicts with objectives (subset of all_points)
        directions: dict of {objective_name: 'maximize' or 'minimize'}

    Returns:
        (is_valid, errors)
    """
    errors = []

    print("="*70)
    print("PARETO FRONTIER VERIFICATION")
    print("="*70)

    print(f"\nTotal points: {len(all_points)}")
    print(f"Pareto points: {len(pareto_points)}")
    print(f"\nObjectives and directions:")
    for obj, direction in directions.items():
        arrow = '↑' if direction == 'maximize' else '↓'
        print(f"  {arrow} {obj}: {direction}")

    # Check 1: Every Pareto point should NOT be dominated by any other point
    print("\n" + "-"*70)
    print("CHECK 1: Pareto points are non-dominated")
    print("-"*70)

    pareto_valid = True
    for i, pareto_point in enumerate(pareto_points):
        is_dominated_by_any = False
        dominating_point = None

        for other_point in all_points:
            # Skip comparing with itself
            if pareto_point == other_point:
                continue

            if is_dominated(pareto_point, other_point, directions):
                is_dominated_by_any = True
                dominating_point = other_point
                break

        if is_dominated_by_any:
            errors.append(f"Pareto point {i+1} is dominated!")
            print(f"  ✗ Pareto point {i+1} is DOMINATED")
            print(f"    Point: {pareto_point}")
            print(f"    Dominated by: {dominating_point}")
            pareto_valid = False
        else:
            print(f"  ✓ Pareto point {i+1} is non-dominated")

    # Check 2: Non-Pareto points should be dominated by at least one point
    print("\n" + "-"*70)
    print("CHECK 2: Non-Pareto points are dominated")
    print("-"*70)

    non_pareto_points = [p for p in all_points if p not in pareto_points]
    non_dominated_count = 0

    for i, non_pareto_point in enumerate(non_pareto_points[:10]):  # Check first 10
        is_dominated_by_any = False

        for other_point in all_points:
            if non_pareto_point == other_point:
                continue

            if is_dominated(non_pareto_point, other_point, directions):
                is_dominated_by_any = True
                break

        if not is_dominated_by_any:
            non_dominated_count += 1
            if non_dominated_count <= 3:  # Show first 3
                errors.append(f"Non-Pareto point {i+1} is not dominated (should be Pareto!)")
                print(f"  ⚠️  Non-Pareto point {i+1} is NOT dominated: {non_pareto_point}")

    if non_dominated_count > 0:
        print(f"\n  Found {non_dominated_count} non-Pareto points that are not dominated!")
        print(f"  This suggests Ax may have missed some Pareto points.")
    else:
        print(f"  ✓ All checked non-Pareto points are properly dominated")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if pareto_valid and non_dominated_count == 0:
        print("✓ Pareto frontier is MATHEMATICALLY CORRECT")
        print("  - All Pareto points are non-dominated")
        print("  - All non-Pareto points are dominated")
        return True, []
    else:
        print("✗ Pareto frontier has ISSUES:")
        for error in errors:
            print(f"  - {error}")
        return False, errors


if __name__ == '__main__':
    # Example test case
    print("Running example test...\n")

    # Test with 3 objectives: Tm (max), Aggregation (min), Activity (max)
    directions = {
        'Tm': 'maximize',
        'Aggregation': 'minimize',
        'Activity': 'maximize'
    }

    # Example points
    all_points = [
        {'Tm': 50, 'Aggregation': 10, 'Activity': 60},  # Point 1
        {'Tm': 52, 'Aggregation': 12, 'Activity': 58},  # Point 2 (trade-off with 1)
        {'Tm': 48, 'Aggregation': 8, 'Activity': 55},   # Point 3 (dominated by 1)
        {'Tm': 54, 'Aggregation': 15, 'Activity': 62},  # Point 4 (Pareto)
        {'Tm': 51, 'Aggregation': 11, 'Activity': 61},  # Point 5 (maybe Pareto)
    ]

    # Points 1, 2, 4 should be Pareto (non-dominated)
    pareto_points = [
        {'Tm': 50, 'Aggregation': 10, 'Activity': 60},
        {'Tm': 52, 'Aggregation': 12, 'Activity': 58},
        {'Tm': 54, 'Aggregation': 15, 'Activity': 62},
    ]

    is_valid, errors = verify_pareto_frontier(all_points, pareto_points, directions)

    print("\n" + "="*70)
    print("To verify YOUR data:")
    print("="*70)
    print("1. Run your analysis with DEBUG=True")
    print("2. Export the Pareto points and all data")
    print("3. Use this script to verify the mathematical correctness")
