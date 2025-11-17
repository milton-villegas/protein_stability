# Well Mapping and Plate Organization

This guide explains how experimental designs are mapped to 96-well and 384-well plates, and how this integrates with the Opentrons robot protocol.

---

## Overview

The Designer organizes samples in **row-major order** across up to **4 × 96-well plates**, then optionally transfers to a **384-well plate** using multichannel pipettes.

**Key Concepts:**
- ✓ Row-major well filling (A1→A2→A3...→A12→B1→B2...)
- ✓ Column-major Opentrons transfer (A1→B1→C1...→H1)
- ✓ 96-well to 384-well interleaved mapping
- ✓ Support for up to 384 experimental conditions

---

## 96-Well Plate Organization

### Well Filling Order: Row-Major

Samples are assigned to wells in **row-major order** (across rows first, then down):

```
96-Well Plate Layout (12 columns × 8 rows)

       1    2    3    4    5    6    7    8    9   10   11   12
    ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
  A │  1 │  2 │  3 │  4 │  5 │  6 │  7 │  8 │  9 │ 10 │ 11 │ 12 │
    ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  B │ 13 │ 14 │ 15 │ 16 │ 17 │ 18 │ 19 │ 20 │ 21 │ 22 │ 23 │ 24 │
    ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  C │ 25 │ 26 │ 27 │ 28 │ 29 │ 30 │ 31 │ 32 │ 33 │ 34 │ 35 │ 36 │
    ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  D │ 37 │ 38 │ 39 │ 40 │ 41 │ 42 │ 43 │ 44 │ 45 │ 46 │ 47 │ 48 │
    ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  E │ 49 │ 50 │ 51 │ 52 │ 53 │ 54 │ 55 │ 56 │ 57 │ 58 │ 59 │ 60 │
    ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  F │ 61 │ 62 │ 63 │ 64 │ 65 │ 66 │ 67 │ 68 │ 69 │ 70 │ 71 │ 72 │
    ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  G │ 73 │ 74 │ 75 │ 76 │ 77 │ 78 │ 79 │ 80 │ 81 │ 82 │ 83 │ 84 │
    ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  H │ 85 │ 86 │ 87 │ 88 │ 89 │ 90 │ 91 │ 92 │ 93 │ 94 │ 95 │ 96 │
    └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

Sample 1  → A1     Sample 13 → B1     Sample 25 → C1
Sample 2  → A2     Sample 14 → B2     Sample 26 → C2
Sample 3  → A3     Sample 15 → B3     ...
...                ...
Sample 12 → A12    Sample 24 → B12    Sample 96 → H12
```

### Algorithm

```python
def _generate_well_position(idx: int) -> Tuple[int, str]:
    """Generate 96-well plate number and well position"""
    plate_num = (idx // 96) + 1
    well_idx = idx % 96

    # Row-major: fill across rows first
    row = chr(ord('A') + (well_idx // 12))  # A-H
    col = (well_idx % 12) + 1                # 1-12
    well_pos = f"{row}{col}"

    return plate_num, well_pos
```

**Example:**
- Sample 0 → Plate 1, A1
- Sample 11 → Plate 1, A12
- Sample 12 → Plate 1, B1
- Sample 96 → Plate 2, A1

---

## Multi-Plate Organization

Designs with >96 samples span multiple plates:

```
Sample Distribution Across 4 Plates (384 samples max)

Plate 1: Samples 0-95
Plate 2: Samples 96-191
Plate 3: Samples 192-287
Plate 4: Samples 288-383
```

**Excel Export includes:**
- `Plate_96` column: Which plate (1-4)
- `Well_96` column: Position on that plate (A1-H12)

---

## 96→384 Well Conversion

### Conversion Logic

The Designer maps each 96-well plate to a **6-column section** of the 384-well plate:

```
384-Well Plate Layout (24 columns × 16 rows)

Plate 1 → Columns 1-6
Plate 2 → Columns 7-12
Plate 3 → Columns 13-18
Plate 4 → Columns 19-24

       1  2  3  4  5  6 │ 7  8  9 10 11 12 │13 14 15 16 17 18 │19 20 21 22 23 24
    ┌──────────────────┬───────────────────┬───────────────────┬──────────────────┐
  A │ P1               │ P2                │ P3                │ P4               │
  B │                  │                   │                   │                  │
  C │                  │                   │                   │                  │
  D │                  │                   │                   │                  │
  E │                  │                   │                   │                  │
  F │                  │                   │                   │                  │
  G │                  │                   │                   │                  │
  H │                  │                   │                   │                  │
  I │                  │                   │                   │                  │
  J │                  │                   │                   │                  │
  K │                  │                   │                   │                  │
  L │                  │                   │                   │                  │
  M │                  │                   │                   │                  │
  N │                  │                   │                   │                  │
  O │                  │                   │                   │                  │
  P │                  │                   │                   │                  │
    └──────────────────┴───────────────────┴───────────────────┴──────────────────┘
```

### Row Interleaving

Within each 6-column section, 96-well rows map to 384-well rows with **interleaving**:

**Odd columns (1, 3, 5, 7, 9, 11):**
```
A → A  (row 0 → 0)
B → C  (row 1 → 2)
C → E  (row 2 → 4)
D → G  (row 3 → 6)
E → I  (row 4 → 8)
F → K  (row 5 → 10)
G → M  (row 6 → 12)
H → O  (row 7 → 14)
```

**Even columns (2, 4, 6, 8, 10, 12):**
```
A → B  (row 0 → 1)
B → D  (row 1 → 3)
C → F  (row 2 → 5)
D → H  (row 3 → 7)
E → J  (row 4 → 9)
F → L  (row 5 → 11)
G → N  (row 6 → 13)
H → P  (row 7 → 15)
```

### Conversion Examples

```
96-Well Position → 384-Well Position

Plate 1, A1  → A1   (plate offset: 0,  col: 1,  odd → row A)
Plate 1, A2  → B1   (plate offset: 0,  col: 1,  even → row B)
Plate 1, B1  → C1   (plate offset: 0,  col: 1,  odd → row C)
Plate 1, B2  → D1   (plate offset: 0,  col: 1,  even → row D)

Plate 1, A3  → A2   (plate offset: 0,  col: 2,  odd → row A)
Plate 1, A4  → B2   (plate offset: 0,  col: 2,  even → row B)

Plate 2, A1  → A7   (plate offset: 6,  col: 7,  odd → row A)
Plate 2, A2  → B7   (plate offset: 6,  col: 7,  even → row B)

Plate 4, H12 → P24  (plate offset: 18, col: 24, even → row P)
```

### Algorithm

```python
def _convert_96_to_384_well(plate_num: int, well_96: str) -> str:
    """Convert 96-well position to 384-well position"""
    import math

    # Parse 96-well position
    row_96 = well_96[0]      # Letter (A-H)
    col_96 = int(well_96[1:]) # Number (1-12)

    # Convert row letter to index (A=0, B=1, ..., H=7)
    row_96_index = ord(row_96) - ord('A')

    # Map to 384-well column
    # Each plate occupies 6 columns: Plate 1→1-6, Plate 2→7-12, etc.
    base_col_384 = (plate_num - 1) * 6

    # Within each plate: columns 1-2→1, 3-4→2, 5-6→3, 7-8→4, 9-10→5, 11-12→6
    col_384 = base_col_384 + math.ceil(col_96 / 2)

    # Map row based on column parity (odd/even)
    if col_96 % 2 == 1:  # Odd column
        row_384_index = row_96_index * 2      # A→A(0), B→C(2), C→E(4), ...
    else:  # Even column
        row_384_index = row_96_index * 2 + 1  # A→B(1), B→D(3), C→F(5), ...

    # Convert back to letter (A=0, B=1, ..., P=15)
    row_384 = chr(ord('A') + row_384_index)

    return f"{row_384}{col_384}"
```

---

## Opentrons Integration

### Why Row-Major for Excel but Column-Major for Robot?

**Excel Export (Row-Major):**
- Human-readable organization
- Easy to review and edit
- Natural left-to-right reading

**Opentrons CSV (Column-Major):**
- Matches multichannel pipette operation
- One column = 8 wells = one tip pickup
- Efficient liquid handling

### Opentrons Protocol Behavior

The `opentrons/protein_stability_doe.py` protocol expects **column-major organization**:

```python
def generate_well_indices(n):
    """Generate well names in column-major order for 96-well plate"""
    wells = []
    for col in range(1, 13):  # Columns 1-12
        for row in range(8):  # Rows A-H
            wells.append(f"{chr(65 + row)}{col}")
            if len(wells) >= n:
                return wells
    return wells

# Result: A1, B1, C1, D1, E1, F1, G1, H1, A2, B2, C2, ...
```

**This is DIFFERENT from the Designer's row-major Excel output!**

### Important: CSV vs Excel Well Order

⚠️ **The Designer's Excel output and Opentrons CSV have different well orders:**

**Excel (for manual review):**
- Sample 1 → A1
- Sample 2 → A2
- Sample 3 → A3

**Opentrons CSV (for robot):**
- Row 1 (Sample 1) → processed in well A1
- Row 2 (Sample 2) → processed in well B1  ❌ Different!
- Row 3 (Sample 3) → processed in well C1  ❌ Different!

**Solution:** The Opentrons protocol uses its own well indexing (column-major). The CSV file rows are matched to wells by the protocol's `generate_well_indices()` function, not by the Excel Well_96 column.

---

## Multichannel Transfer (96→384)

### Transfer Pattern

The Opentrons uses a **P300 8-channel pipette** to transfer from 96-well to 384-well:

```
96-Well Column (8 wells):        384-Well Interleaved:
   A1  ─────────────────────────→  A1
   B1  ─────────────────────────→  B1
   C1  ─────────────────────────→  ...
   D1  ─────────────────────────→  (interleaved pattern)
   E1
   F1
   G1
   H1

One multichannel pickup transfers entire column
```

### Interleaved Destination

The protocol creates an **interleaved well list** for the 384-well plate:

```python
def create_384_interleaved_wells(plate384, start_col):
    """Create interleaved destination list for 384 plate"""
    interleaved = []
    for col in range(start_col, 25):  # Columns 1-24
        col_wells = []
        # Get all 16 rows (A-P) for this column
        for row in range(16):
            row_letter = chr(ord('A') + row)
            col_wells.append(plate384[f"{row_letter}{col}"])

        # Take only rows A and B, alternating
        interleaved.append(col_wells[0])  # Row A
        interleaved.append(col_wells[1])  # Row B

    return interleaved
```

**Result:** Fills 384-well plate as A1, B1, A2, B2, A3, B3, ...

This matches the 96→384 well conversion exactly!

---

## Validation and Troubleshooting

### Check Well Assignment

To verify well assignments in your exported design:

```python
# Example: Check first 10 samples
Sample 1: Plate 1, Well A1  → 384-Well A1
Sample 2: Plate 1, Well A2  → 384-Well B1
Sample 3: Plate 1, Well A3  → 384-Well A2
Sample 4: Plate 1, Well A4  → 384-Well B2
Sample 5: Plate 1, Well A5  → 384-Well A3
...
```

### Common Issues

**Issue: "Wells don't match between Excel and robot"**
- **Cause:** Excel uses row-major, Opentrons CSV uses column-major indexing
- **Solution:** Trust the Opentrons CSV file, not the Excel Well_96 column, for robot operation

**Issue: "384-well positions seem wrong"**
- **Cause:** Expecting sequential filling (A1, A2, A3...)
- **Solution:** 384-well uses interleaved pattern (A1, B1, A2, B2...) due to multichannel transfer

**Issue: "Not all wells filled in 384-plate"**
- **Cause:** Design has <384 samples, only uses needed wells
- **Solution:** Normal behavior, unused wells remain empty

### Validation Script

Check well mapping consistency:

```python
from gui.tabs.designer_tab import DesignerTab

# Test conversion
plate_96 = 1
well_96 = "B3"

well_384 = DesignerTab._convert_96_to_384_well(plate_96, well_96)
print(f"Plate {plate_96}, {well_96} → 384-well {well_384}")
# Expected: Plate 1, B3 → 384-well C2
```

---

## Design Capacity

### Maximum Samples

| Configuration | Max Samples | Plates Required |
|---------------|-------------|-----------------|
| Single plate | 96 | 1 |
| Two plates | 192 | 2 |
| Three plates | 288 | 3 |
| Four plates (max) | 384 | 4 |

### Plate Limits by Design Type

| Design Type | Typical Size | Plates Needed |
|-------------|--------------|---------------|
| Full Factorial (2³) | 8 | 1 |
| Full Factorial (3⁴) | 81 | 1 |
| LHS (96 samples) | 96 | 1 |
| LHS (192 samples) | 192 | 2 |
| Fractional (7 factors) | 64 | 1 |
| Plackett-Burman (11 factors) | 12 | 1 |
| CCD (4 factors) | 31 | 1 |
| Box-Behnken (4 factors) | 27 | 1 |

---

## Best Practices

### 1. Randomize Run Order
Always randomize experimental run order to avoid systematic biases:
- Don't run samples in order 1, 2, 3, 4...
- Shuffle before running experiments
- Protects against time-dependent effects

### 2. Plate Organization
For multi-plate designs:
- Fill plates sequentially (complete Plate 1, then Plate 2, etc.)
- Label plates clearly (Plate 1/4, Plate 2/4, etc.)
- Keep plates in order during robot operation

### 3. 384-Well Considerations
- Verify multichannel pipette calibration
- Check that interleaved pattern matches your expectations
- Use clear 384-well plates for visual inspection

### 4. Documentation
- Save both Excel and CSV files
- Record plate layout in lab notebook
- Note which plate goes in which robot deck position

---

## Implementation Details

### Code Location

Well mapping logic is in:
- **`gui/tabs/designer_tab.py`**:
  - `_generate_well_position()` (line ~1174)
  - `_convert_96_to_384_well()` (line ~1185)

- **`opentrons/protein_stability_doe.py`**:
  - `generate_well_indices()` (line ~201)
  - `create_384_interleaved_wells()` (line ~591)

### Constants

```python
# 96-well plate
WELLS_PER_PLATE = 96
ROWS_PER_PLATE = 8  # A-H
COLS_PER_PLATE = 12  # 1-12

# 384-well plate
PLATE_384_ROWS = 16  # A-P
PLATE_384_COLS = 24  # 1-24

# Maximum capacity
MAX_TOTAL_WELLS = 384  # 4 × 96-well plates
```

---

## Related Documentation

- [Design Types Guide](DESIGN_GUIDE.md) - Choosing the right experimental design
- [README.md](../README.md) - Main application documentation
- [Opentrons Protocol](../opentrons/protein_stability_doe.py) - Robot protocol source code

---

## Questions?

**Why row-major instead of column-major like Opentrons?**
- Row-major is more intuitive for humans reading Excel files
- Opentrons CSV handles its own well indexing internally
- The two systems are compatible through proper CSV export

**Can I change the well order?**
- Not recommended - the Opentrons protocol expects this specific pattern
- Changing well order would require updating both Designer and Opentrons code

**What if I need >384 samples?**
- Split into multiple experimental batches
- Use a more efficient design (LHS, Fractional, Plackett-Burman)
- Consider screening first, then optimization with fewer factors

**How do I verify my design before running?**
- Check Excel file: review Well_96 and Well_384 columns
- Verify sample count matches expectations
- Test with small design first (8-12 samples)
