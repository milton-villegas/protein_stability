# Protein Stability DoE Toolkit

Three small tools that work together to design, execute, and analyse protein stability buffer screens:

1. **Designer (GUI)** — build full‑factorial/Custom designs and export CSV/XLSX for the robot.  
   `designer/factorial_designer_gui.pyw`
2. **Opentrons Protocol** — prepares buffers in 96‑well plates and optionally transfers to a 384‑well plate.  
   `opentrons/protein_stability_doe.py`
3. **Analysis (GUI)** — import results, run linear models, and plot main effects/interactions/residuals.  
   `analysis/doe_analysis_gui.pyw`

> Tested on Python 3.10–3.13 (Windows/macOS). Opentrons API Level: **2.20**.

---

## Repository structure

```
protein-stability-doe/
├─ designer/
│  └─ factorial_designer_gui.pyw
├─ opentrons/
│  └─ protein_stability_doe.py
├─ analysis/
│  └─ doe_analysis_gui.pyw
├─ examples/           # to be added
├─ requirements-analysis.txt
├─ requirements-designer.txt
└─ .gitignore
```

---

## Quick start

### 1) Set up a Python environment

```bash
# Option A: venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install what you need
pip install -r requirements-analysis.txt
pip install -r requirements-designer.txt
```

### 2) Designer — build an experimental design

```bash
python designer/factorial_designer_gui.pyw
```

- Add factors and levels (units supported: M, mM, µM, nM, %, w/v, v/v).
- Real‑time combination counter shows the number of conditions.
- **Export** produces:
  - `*.xlsx` (Design table)
  - `*.csv` (Opentrons volumes in **µL**, ready for the protocol)

**CSV format expected by the robot**

- **Row 1**: reagent names (headers)  
- **Row 2+**: volumes per condition (**µL**), one condition per row

Example:
```
Buffer,Glycerol,NaCl,pH Buffer
150,20,10,5
140,30,15,5
...
```

### 3) Opentrons — run the preparation

Upload `opentrons/protein_stability_doe.py` to the Opentrons App (API Level 2.20).
Provide the **CSV content** via protocol parameter.

**Deck & labware** (as used by the script):

- **Slot 1**: 96‑well plate #1 — `greiner_96_well_u_bottom_323ul`
- **Slot 2**: 384‑well plate — `corning_384_wellplate_112ul_flat`
- **Slot 3**: 96‑well plate #2 *(auto‑loaded if needed)*
- **Slot 4**: 96‑well plate #3 *(auto‑loaded if needed)*
- **Slot 5**: 24‑well reservoir — `cytiva_24_reservoir_10ml`
- **Slot 6**: 96‑well plate #4 *(auto‑loaded if needed)*
- **Slots 7–11**: `opentrons_96_tiprack_300ul` (single channel in 7; multichannel in 9 + extras in 8,10,11)

**Pipettes**

- **Left**: `p300_multi` (96→384 transfers)  
- **Right**: `p300_single` (buffer preparation)

> If you have Gen1 pipettes and/or different labware, edit the identifiers accordingly in `protein_stability_doe.py`.

The protocol:
- Reads the CSV headers (reagents) and the following rows (volumes).
- Prepares conditions across as many 96‑well plates as needed.
- Mixes before transfers and (optionally) maps columns into the 384‑well plate (A/B rows).
- Prints a summary of plate/well usage at the end of the run.

### 4) Analysis — run statistics & plots

```bash
python analysis/doe_analysis_gui.pyw
```

**Dependencies (installed via `requirements-analysis.txt`):**
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`

Features:
- Import CSV/XLSX results (expects factor columns + a **Response** column).
- Main effects and interaction plots.
- Linear model fits (similar to MATLAB `fitlm` output).
- Residual diagnostics and exportable figures.

---

## Troubleshooting

- **Missing `openpyxl` when exporting from Designer** → `pip install -r requirements-designer.txt`
- **`ModuleNotFoundError: seaborn` (Analysis GUI)** → `pip install -r requirements-analysis.txt`
- **Pipette volume assertion in Opentrons** → ensure per‑transfer volumes are ≤ pipette max; adjust volumes or split transfers in the CSV/design.
- **Gen1 vs Gen2 pipettes / alternative labware** → change the model strings in `protein_stability_doe.py` (e.g., pipette names or labware definitions).
- **CSV parsed but no rows** → verify the first line is headers and that at least one non‑empty row follows (no stray separators).

---

## Development notes

- Opentrons protocol declares **API Level 2.20** and was validated on p300 single + multi.  
- Designer/Analysis are **Tkinter** desktop apps.
- Recommended Python: **3.10–3.13**.

---

## License

This project is licensed under the MIT License.  
See the [LICENSE] file for details.
