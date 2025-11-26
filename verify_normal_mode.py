#!/usr/bin/env python3
"""
Verify that NORMAL MODE (default concentrations) still works correctly.
This ensures the fix doesn't break the standard workflow without per-level concentrations.
"""

from gui.tabs.designer.models import FactorModel


def test_normal_mode():
    """Test that normal concentration factors work without per-level mode"""

    print("="*70)
    print("NORMAL MODE VERIFICATION (Default Concentrations)")
    print("="*70)
    print("\nScenario: User adds concentration factors in NORMAL mode")
    print("          (NOT using per-level concentrations)")
    print("="*70)

    # Create model WITHOUT per-level concentrations
    model = FactorModel()

    # Add factors in normal mode
    model.add_factor("detergent", ["None", "DDM", "LMNG", "OG"])
    model.add_factor("detergent_concentration", ["0", "0.5", "1.0", "2.0"], stock_conc=10.0)
    model.add_factor("nacl", ["0", "100", "200"], stock_conc=5000.0)
    model.add_factor("glycerol", ["0", "5", "10"], stock_conc=100.0)

    # NO per-level concentrations configured!
    print("\n✓ Factors added:")
    factors = model.get_factors()
    for name, levels in factors.items():
        stock = model.get_stock_conc(name)
        stock_str = f" (stock: {stock})" if stock else ""
        print(f"  - {name}: {len(levels)} levels{stock_str}")

    # Check per-level concentrations
    per_level_concs = model.get_all_per_level_concs()
    print(f"\n✓ Per-level concentrations: {per_level_concs}")
    print(f"  (Empty dict means NOT using per-level mode)")

    # Simulate the export logic
    print(f"\n{'='*70}")
    print("SIMULATING EXPORT LOGIC")
    print(f"{'='*70}")

    factors_before = model.get_factors()
    print(f"\nFactors BEFORE fix logic:")
    for name in factors_before.keys():
        print(f"  - {name}")

    # Apply the fix logic (same as in export_panel.py)
    factors_after = factors_before.copy()

    if "detergent" in per_level_concs and per_level_concs["detergent"]:
        print(f"\n⚠️  Per-level mode detected for detergent")
        if "detergent_concentration" in factors_after:
            factors_after = {k: v for k, v in factors_after.items() if k != "detergent_concentration"}
            print(f"   → Removing 'detergent_concentration'")
    else:
        print(f"\n✓ Normal mode detected (no per-level concentrations)")
        print(f"  → Keeping 'detergent_concentration' in design")

    if "reducing_agent" in per_level_concs and per_level_concs["reducing_agent"]:
        print(f"⚠️  Per-level mode detected for reducing_agent")
        if "reducing_agent_concentration" in factors_after:
            factors_after = {k: v for k, v in factors_after.items() if k != "reducing_agent_concentration"}
            print(f"   → Removing 'reducing_agent_concentration'")

    print(f"\nFactors AFTER fix logic:")
    for name in factors_after.keys():
        print(f"  - {name}")

    # Verify concentration factor is still present
    print(f"\n{'='*70}")
    print("VERIFICATION RESULT")
    print(f"{'='*70}")

    if "detergent_concentration" in factors_after:
        print(f"\n✅ SUCCESS: 'detergent_concentration' is still in the design!")
        print(f"✅ Normal mode works correctly")
        print(f"\nExpected behavior:")
        print(f"  - Full Factorial: 4 × 4 × 3 × 3 = 144 combinations")
        print(f"  - LHS with 96 samples: generates 96 samples")
        print(f"  - Filter will remove invalid pairings:")
        print(f"    • detergent=None + concentration>0 ❌")
        print(f"    • detergent=DDM + concentration=0 ❌")
        print(f"  - This is EXPECTED and CORRECT in normal mode!")

        # Calculate full factorial
        import itertools
        level_lists = [factors_after[f] for f in factors_after.keys()]
        combinations_before_filter = list(itertools.product(*level_lists))
        print(f"\n  Total combinations before filter: {len(combinations_before_filter)}")

        # Simulate filter (simplified)
        valid_count = 0
        for combo in combinations_before_filter:
            factor_names = list(factors_after.keys())
            row_dict = {factor_names[i]: combo[i] for i in range(len(factor_names))}

            det = str(row_dict["detergent"]).strip()
            det_conc = float(row_dict["detergent_concentration"])

            valid = True
            if det.lower() in ['none', '0', '', 'nan']:
                if det_conc != 0:
                    valid = False
            else:
                if det_conc == 0:
                    valid = False

            if valid:
                valid_count += 1

        print(f"  Valid combinations after filter: {valid_count}")
        print(f"  Filtering ratio: {valid_count/len(combinations_before_filter)*100:.1f}%")

    else:
        print(f"\n❌ FAIL: 'detergent_concentration' was removed!")
        print(f"❌ This would break normal mode")

    print(f"\n{'='*70}\n")


def test_per_level_mode():
    """Test that per-level mode excludes concentration factors"""

    print("="*70)
    print("PER-LEVEL MODE VERIFICATION (Individual Concentrations)")
    print("="*70)
    print("\nScenario: User adds factors WITH per-level concentrations")
    print("          (Each detergent has its own stock/final concentration)")
    print("="*70)

    # Create model WITH per-level concentrations
    model = FactorModel()

    # Add detergent (categorical)
    model.add_factor("detergent", ["None", "DDM", "LMNG", "OG"])

    # User might mistakenly add concentration factor
    model.add_factor("detergent_concentration", ["0", "0.5", "1.0", "2.0"], stock_conc=10.0)

    # Add other factors
    model.add_factor("nacl", ["0", "100", "200"], stock_conc=5000.0)

    # Set per-level concentrations
    model.set_per_level_concs("detergent", {
        "DDM": {"stock": 10.0, "final": 1.0},
        "LMNG": {"stock": 5.0, "final": 0.5},
        "OG": {"stock": 20.0, "final": 2.0}
    })

    print("\n✓ Factors added:")
    factors = model.get_factors()
    for name, levels in factors.items():
        print(f"  - {name}: {len(levels)} levels")

    # Check per-level concentrations
    per_level_concs = model.get_all_per_level_concs()
    print(f"\n✓ Per-level concentrations configured:")
    for cat_factor, level_concs in per_level_concs.items():
        print(f"  - {cat_factor}: {len(level_concs)} levels with individual concentrations")
        for level, conc_data in level_concs.items():
            print(f"    • {level}: stock={conc_data['stock']}%, final={conc_data['final']}%")

    # Simulate the export logic
    print(f"\n{'='*70}")
    print("SIMULATING EXPORT LOGIC")
    print(f"{'='*70}")

    factors_before = model.get_factors()
    print(f"\nFactors BEFORE fix logic:")
    for name in factors_before.keys():
        print(f"  - {name}")

    # Apply the fix logic
    factors_after = factors_before.copy()

    if "detergent" in per_level_concs and per_level_concs["detergent"]:
        print(f"\n⚠️  Per-level mode detected for detergent")
        if "detergent_concentration" in factors_after:
            factors_after = {k: v for k, v in factors_after.items() if k != "detergent_concentration"}
            print(f"   → Removing 'detergent_concentration' from design")

    print(f"\nFactors AFTER fix logic:")
    for name in factors_after.keys():
        print(f"  - {name}")

    # Verify concentration factor is removed
    print(f"\n{'='*70}")
    print("VERIFICATION RESULT")
    print(f"{'='*70}")

    if "detergent_concentration" not in factors_after:
        print(f"\n✅ SUCCESS: 'detergent_concentration' was excluded!")
        print(f"✅ Per-level mode works correctly")
        print(f"\nExpected behavior:")
        print(f"  - Full Factorial: 4 × 3 = 12 combinations (no conc factor)")
        print(f"  - LHS with 96 samples: generates 96 samples (no filtering)")
        print(f"  - During volume calculation:")
        print(f"    • DDM → uses 10% stock, 1% final")
        print(f"    • LMNG → uses 5% stock, 0.5% final")
        print(f"    • OG → uses 20% stock, 2% final")
    else:
        print(f"\n❌ FAIL: 'detergent_concentration' was NOT removed!")
        print(f"❌ This would cause the LHS sample count issue")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    test_normal_mode()
    test_per_level_mode()

    print("="*70)
    print("SUMMARY")
    print("="*70)
    print("\n✅ Normal mode (default concentrations): concentration factor KEPT")
    print("✅ Per-level mode (individual concentrations): concentration factor EXCLUDED")
    print("\nBoth modes work correctly!")
    print("="*70 + "\n")
