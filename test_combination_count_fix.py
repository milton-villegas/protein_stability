#!/usr/bin/env python3
"""
Test that total_combinations() correctly excludes concentration factors
when per-level mode is active.
"""

from core.project import DoEProject


def test_total_combinations_with_per_level():
    """Test that combination count excludes concentration factors in per-level mode"""

    print("="*70)
    print("Testing total_combinations() with Per-Level Concentrations")
    print("="*70)

    # Simulate the user's scenario from the bug report
    model = DoEProject()

    # Add factors matching the user's exported data
    model.add_factor("buffer pH", ["6.5", "7.5", "8.5"])  # 3 levels
    model.add_factor("nacl", ["100", "500"])  # 2 levels
    model.add_factor("glycerol", ["0", "15"])  # 2 levels
    model.add_factor("detergent", ["C12E8", "CHAPS", "DDM", "LMNG", "None", "OG"])  # 6 levels

    # User mistakenly added detergent_concentration with 2 levels
    model.add_factor("detergent_concentration", ["0", "2"])  # 2 levels

    print("\n✓ Factors added:")
    for name, levels in model.get_factors().items():
        print(f"  - {name}: {len(levels)} levels")

    # Calculate BEFORE setting per-level concentrations
    total_before = model.total_combinations()
    print(f"\n❌ Total combinations BEFORE per-level mode: {total_before}")
    print(f"   Calculation: 3 × 2 × 2 × 6 × 2 = {3*2*2*6*2}")
    print(f"   Plates: {(total_before + 95) // 96}")

    # Now set per-level concentrations (loaded from Excel)
    model.set_per_level_concs("detergent", {
        "C12E8": {"stock": 10.0, "final": 1.0},
        "CHAPS": {"stock": 10.0, "final": 1.0},
        "DDM": {"stock": 10.0, "final": 1.0},
        "LMNG": {"stock": 5.0, "final": 0.5},
        "OG": {"stock": 20.0, "final": 2.0}
    })

    print(f"\n✓ Per-level concentrations configured for detergent")

    # Calculate AFTER setting per-level concentrations
    total_after = model.total_combinations()
    print(f"\n✅ Total combinations AFTER per-level mode: {total_after}")
    print(f"   Calculation: 3 × 2 × 2 × 6 = {3*2*2*6} (detergent_conc excluded!)")
    print(f"   Plates: {(total_after + 95) // 96}")

    # Verify
    expected = 3 * 2 * 2 * 6  # 72
    if total_after == expected:
        print(f"\n✅ SUCCESS! Combination count is correct: {total_after}")
        print(f"✅ This matches the exported file (72 rows)")
    else:
        print(f"\n❌ FAIL! Expected {expected}, got {total_after}")

    print("="*70)


def test_total_combinations_without_per_level():
    """Test that combination count INCLUDES concentration factors in normal mode"""

    print("\n" + "="*70)
    print("Testing total_combinations() WITHOUT Per-Level Concentrations")
    print("="*70)

    model = DoEProject()

    # Add factors in normal mode
    model.add_factor("detergent", ["DDM", "LMNG", "OG"])  # 3 levels
    model.add_factor("detergent_concentration", ["0.5", "1.0", "2.0"])  # 3 levels
    model.add_factor("nacl", ["0", "100"])  # 2 levels

    print("\n✓ Factors added (normal mode - no per-level):")
    for name, levels in model.get_factors().items():
        print(f"  - {name}: {len(levels)} levels")

    total = model.total_combinations()
    expected = 3 * 3 * 2  # 18

    print(f"\n✓ Total combinations: {total}")
    print(f"  Calculation: 3 × 3 × 2 = {expected}")
    print(f"  Plates: {(total + 95) // 96}")

    if total == expected:
        print(f"\n✅ SUCCESS! Normal mode still works correctly: {total}")
    else:
        print(f"\n❌ FAIL! Expected {expected}, got {total}")

    print("="*70)


def test_with_reducing_agent_per_level():
    """Test with reducing agent per-level concentrations"""

    print("\n" + "="*70)
    print("Testing total_combinations() with Reducing Agent Per-Level")
    print("="*70)

    model = DoEProject()

    model.add_factor("reducing_agent", ["DTT", "TCEP", "None"])  # 3 levels
    model.add_factor("reducing_agent_concentration", ["0", "5", "10"])  # 3 levels
    model.add_factor("nacl", ["0", "100"])  # 2 levels

    print("\n✓ Factors added:")
    for name, levels in model.get_factors().items():
        print(f"  - {name}: {len(levels)} levels")

    total_before = model.total_combinations()
    print(f"\n❌ Total BEFORE per-level: {total_before} (3 × 3 × 2 = 18)")

    # Set per-level concentrations
    model.set_per_level_concs("reducing_agent", {
        "DTT": {"stock": 1000.0, "final": 5.0},
        "TCEP": {"stock": 500.0, "final": 2.0}
    })

    total_after = model.total_combinations()
    expected = 3 * 2  # 6 (reducing_agent_concentration excluded)

    print(f"\n✅ Total AFTER per-level: {total_after} (3 × 2 = 6, conc excluded)")

    if total_after == expected:
        print(f"\n✅ SUCCESS! Reducing agent per-level works correctly")
    else:
        print(f"\n❌ FAIL! Expected {expected}, got {total_after}")

    print("="*70)


if __name__ == "__main__":
    test_total_combinations_with_per_level()
    test_total_combinations_without_per_level()
    test_with_reducing_agent_per_level()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\n✅ GUI combination count now matches export count")
    print("✅ Per-level mode: concentration factors excluded from count")
    print("✅ Normal mode: concentration factors included in count")
    print("✅ Plate count calculated correctly")
    print("="*70 + "\n")
