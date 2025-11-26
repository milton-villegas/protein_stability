#!/usr/bin/env python3
"""
Debug script to understand why export is producing half the expected samples.
"""

from gui.tabs.designer.models import FactorModel
import itertools


def simulate_full_factorial_export(factors_dict, per_level_concs):
    """Simulate the full factorial export logic"""

    print("="*70)
    print("SIMULATING FULL FACTORIAL EXPORT")
    print("="*70)

    # Show initial factors
    print("\n1. Factors from model:")
    for name, levels in factors_dict.items():
        print(f"   - {name}: {len(levels)} levels → {levels}")

    # Calculate expected combinations
    expected_total = 1
    for levels in factors_dict.values():
        expected_total *= len(levels)
    print(f"\n   Expected total: {expected_total} combinations")

    # Apply the per-level fix (from export_panel.py lines 213-225)
    print("\n2. Applying per-level concentration fix:")
    factors_for_design = dict(factors_dict)

    if "detergent" in per_level_concs and per_level_concs["detergent"]:
        if "detergent_concentration" in factors_for_design:
            print(f"   ⚠️  Removing 'detergent_concentration' (per-level mode active)")
            factors_for_design.pop("detergent_concentration")
        else:
            print(f"   ✓ No 'detergent_concentration' to remove")

    if "reducing_agent" in per_level_concs and per_level_concs["reducing_agent"]:
        if "reducing_agent_concentration" in factors_for_design:
            print(f"   ⚠️  Removing 'reducing_agent_concentration' (per-level mode active)")
            factors_for_design.pop("reducing_agent_concentration")
        else:
            print(f"   ✓ No 'reducing_agent_concentration' to remove")

    print(f"\n   Factors after fix:")
    for name, levels in factors_for_design.items():
        print(f"   - {name}: {len(levels)} levels")

    # Generate full factorial
    print("\n3. Generating full factorial:")
    factor_names = list(factors_for_design.keys())
    level_lists = [factors_for_design[f] for f in factor_names]
    combinations_before_filter = list(itertools.product(*level_lists))
    print(f"   Generated: {len(combinations_before_filter)} combinations")

    # Apply filter
    print("\n4. Applying categorical-concentration filter:")
    filtered_combinations = []
    removed_count = 0
    removal_reasons = {}

    for combo in combinations_before_filter:
        row_dict = {factor_names[i]: combo[i] for i in range(len(factor_names))}
        valid = True
        reason = None

        # Check detergent-concentration pairing
        if "detergent" in row_dict and "detergent_concentration" in row_dict:
            det = str(row_dict["detergent"]).strip()
            det_conc = float(row_dict["detergent_concentration"])

            if det.lower() in ['none', '0', '', 'nan']:
                if det_conc != 0:
                    valid = False
                    reason = f"detergent=None but conc={det_conc}"
            else:
                if det_conc == 0:
                    valid = False
                    reason = f"detergent={det} but conc=0"

        # Check reducing_agent-concentration pairing
        if "reducing_agent" in row_dict and "reducing_agent_concentration" in row_dict:
            agent = str(row_dict["reducing_agent"]).strip()
            agent_conc = float(row_dict["reducing_agent_concentration"])

            if agent.lower() in ['none', '0', '', 'nan']:
                if agent_conc != 0:
                    valid = False
                    reason = f"reducing_agent=None but conc={agent_conc}"
            else:
                if agent_conc == 0:
                    valid = False
                    reason = f"reducing_agent={agent} but conc=0"

        if valid:
            filtered_combinations.append(combo)
        else:
            removed_count += 1
            if reason:
                removal_reasons[reason] = removal_reasons.get(reason, 0) + 1

    print(f"   After filter: {len(filtered_combinations)} combinations")
    print(f"   Removed: {removed_count} combinations")

    if removal_reasons:
        print(f"\n   Removal reasons:")
        for reason, count in removal_reasons.items():
            print(f"      - {reason}: {count} times")

    print(f"\n5. RESULT:")
    print(f"   Expected in export: {len(filtered_combinations)} rows")
    print(f"   User reported: 72 rows")
    print(f"   GUI showed: 144 combinations")

    if len(filtered_combinations) == 72:
        print(f"\n   ✅ This matches! The issue is understood.")
    else:
        print(f"\n   ❌ Mismatch! Need more investigation.")

    print("="*70)

    return filtered_combinations


def test_scenario_1():
    """Scenario 1: User has detergent_concentration with 2 levels + per-level concs"""

    print("\n\nSCENARIO 1: detergent_concentration factor exists (2 levels)")
    print("="*70)

    model = FactorModel()
    model.add_factor("buffer pH", ["6.5", "7.5", "8.5"])
    model.add_factor("nacl", ["100", "500"])
    model.add_factor("glycerol", ["0", "15"])
    model.add_factor("detergent", ["C12E8", "CHAPS", "DDM", "LMNG", "None", "OG"])
    model.add_factor("detergent_concentration", ["0", "1"])  # 2 levels!

    model.set_per_level_concs("detergent", {
        "C12E8": {"stock": 10.0, "final": 1.0},
        "CHAPS": {"stock": 10.0, "final": 1.0},
        "DDM": {"stock": 10.0, "final": 1.0},
        "LMNG": {"stock": 5.0, "final": 0.5},
        "OG": {"stock": 20.0, "final": 2.0}
    })

    factors = model.get_factors()
    per_level = model.get_all_per_level_concs()

    simulate_full_factorial_export(factors, per_level)


def test_scenario_2():
    """Scenario 2: User does NOT have detergent_concentration, only per-level concs"""

    print("\n\nSCENARIO 2: NO detergent_concentration factor (only per-level)")
    print("="*70)

    model = FactorModel()
    model.add_factor("buffer pH", ["6.5", "7.5", "8.5"])
    model.add_factor("nacl", ["100", "500"])
    model.add_factor("glycerol", ["0", "15"])
    model.add_factor("detergent", ["C12E8", "CHAPS", "DDM", "LMNG", "None", "OG"])
    # NO detergent_concentration factor added!

    model.set_per_level_concs("detergent", {
        "C12E8": {"stock": 10.0, "final": 1.0},
        "CHAPS": {"stock": 10.0, "final": 1.0},
        "DDM": {"stock": 10.0, "final": 1.0},
        "LMNG": {"stock": 5.0, "final": 0.5},
        "OG": {"stock": 20.0, "final": 2.0}
    })

    factors = model.get_factors()
    per_level = model.get_all_per_level_concs()

    simulate_full_factorial_export(factors, per_level)


if __name__ == "__main__":
    test_scenario_1()
    test_scenario_2()

    print("\n\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\nIf SCENARIO 1 matches (72 combinations):")
    print("  → User has detergent_concentration in their model")
    print("  → My fix correctly removes it before design generation")
    print("  → BUT GUI is counting it before removal → shows 144")
    print("  → Solution: GUI should also exclude it (my previous fix was correct!)")
    print("\nIf SCENARIO 2 matches (72 combinations):")
    print("  → User does NOT have detergent_concentration")
    print("  → Should get 72 combinations (correct)")
    print("  → GUI showing 144 is WRONG")
    print("  → Need to investigate why GUI is inflating the count")
    print("="*70 + "\n")
