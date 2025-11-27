#!/usr/bin/env python3
"""
Test that JSON project save/load works correctly
"""

from core.project import DoEProject
import pandas as pd
import os


def test_json_save_load():
    """Test saving and loading project as JSON"""

    print("="*70)
    print("Testing JSON Project Save/Load")
    print("="*70)

    # Create a project with test data
    project = DoEProject()
    project.name = "Test DoE Project"

    # Add factors
    project.add_factor("buffer pH", ["6.5", "7.5", "8.5"], stock_conc=1000.0)
    project.add_factor("nacl", ["100", "500"], stock_conc=5000.0)
    project.add_factor("detergent", ["DDM", "LMNG", "None"])

    # Add per-level concentrations
    project.set_per_level_concs("detergent", {
        "DDM": {"stock": 10.0, "final": 1.0},
        "LMNG": {"stock": 5.0, "final": 0.5}
    })

    # Add a design matrix
    project.design_matrix = pd.DataFrame({
        "buffer pH": ["6.5", "7.5", "8.5"],
        "nacl": ["100", "500", "100"],
        "detergent": ["DDM", "LMNG", "None"]
    })

    print("\n✓ Created test project:")
    print(f"  - Name: {project.name}")
    print(f"  - Factors: {len(project.get_factors())}")
    print(f"  - Stock concs: {project.get_all_stock_concs()}")
    print(f"  - Per-level concs: {list(project.get_all_per_level_concs().keys())}")
    print(f"  - Design matrix: {len(project.design_matrix)} rows")

    # Save to JSON
    test_file = "test_project.json"
    project.save(test_file)
    print(f"\n✓ Saved to: {test_file}")

    # Check file is human-readable
    with open(test_file, 'r') as f:
        content = f.read()
        print(f"\n✓ File is human-readable (first 500 chars):")
        print(content[:500] + "...")

    # Load from JSON
    loaded_project = DoEProject.load(test_file)
    print(f"\n✓ Loaded from: {test_file}")

    # Verify data
    print("\n✓ Verifying loaded data:")

    checks = []

    # Check name
    if loaded_project.name == project.name:
        print(f"  ✅ Name: {loaded_project.name}")
        checks.append(True)
    else:
        print(f"  ❌ Name mismatch: {loaded_project.name} != {project.name}")
        checks.append(False)

    # Check factors
    if loaded_project.get_factors() == project.get_factors():
        print(f"  ✅ Factors: {len(loaded_project.get_factors())} factors")
        checks.append(True)
    else:
        print(f"  ❌ Factors mismatch")
        checks.append(False)

    # Check stock concentrations
    if loaded_project.get_all_stock_concs() == project.get_all_stock_concs():
        print(f"  ✅ Stock concs: {loaded_project.get_all_stock_concs()}")
        checks.append(True)
    else:
        print(f"  ❌ Stock concs mismatch")
        checks.append(False)

    # Check per-level concentrations
    if loaded_project.get_all_per_level_concs() == project.get_all_per_level_concs():
        print(f"  ✅ Per-level concs: {list(loaded_project.get_all_per_level_concs().keys())}")
        checks.append(True)
    else:
        print(f"  ❌ Per-level concs mismatch")
        checks.append(False)

    # Check design matrix
    if loaded_project.design_matrix is not None and len(loaded_project.design_matrix) == len(project.design_matrix):
        print(f"  ✅ Design matrix: {len(loaded_project.design_matrix)} rows")
        checks.append(True)
    else:
        print(f"  ❌ Design matrix mismatch")
        checks.append(False)

    # Clean up
    os.remove(test_file)
    print(f"\n✓ Cleaned up test file")

    # Final result
    print("\n" + "="*70)
    if all(checks):
        print("✅ SUCCESS! JSON save/load working correctly!")
        print("\nBenefits:")
        print("  • Human-readable format")
        print("  • Can open in any text editor")
        print("  • Safe (no code execution)")
        print("  • Standard format (works with any tool)")
        print("  • Git-friendly (can see diffs)")
    else:
        print("❌ FAIL! Some checks failed")
    print("="*70)

    return all(checks)


if __name__ == "__main__":
    test_json_save_load()
