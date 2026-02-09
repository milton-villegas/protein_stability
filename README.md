# SCOUT - Screening & Condition Optimization Utility Tool

A unified application for designing, executing, and analyzing experimental screens:

1. **Designer Tab** — build full‑factorial/Custom designs and export CSV/XLSX for the robot.
2. **Opentrons Protocol** — prepares buffers in 96‑well plates and optionally transfers to a 384‑well plate.
   `opentrons/protein_stability_doe.py`
3. **Analysis Tab** — import results, run linear models, Bayesian Optimization, and plot main effects/interactions/residuals.

> Tested on Python 3.8–3.11 (Windows/macOS/Linux). Opentrons API Level: **2.20**.

---

## Repository structure

```
protein-stability-doe/
├─ run.command                  # macOS launcher (double-click)
├─ run.bat                      # Windows launcher (double-click)
├─ run.sh                       # Linux launcher
├─ main.py                      # Main application entry point
├─ gui/
│  ├─ main_window.py           # Main GUI window
│  └─ tabs/
│     ├─ designer_tab.py       # DoE Designer interface
│     └─ analysis_tab.py       # Analysis & optimization interface
├─ core/
│  ├─ doe_designer.py          # Design generation logic
│  ├─ doe_analyzer.py          # Statistical analysis
│  ├─ optimizer.py             # Bayesian optimization
│  ├─ data_handler.py          # Data loading/preprocessing
│  ├─ exporter.py              # Results export
│  ├─ plotter.py               # Visualization
│  └─ constants.py             # Shared constants
├─ opentrons/
│  └─ protein_stability_doe.py # Robot protocol
├─ tests/                       # Unit tests (pytest)
├─ utils/                       # Shared utilities
├─ requirements.txt             # Core dependencies
├─ requirements-dev.txt         # Testing dependencies
├─ setup.py                     # Package setup
└─ .python-version              # Python version constraint
```

---

## Documentation

📚 **Guides:**
- **[Design Types Guide](docs/DESIGN_GUIDE.md)** - Choosing and using the 7 design types (Full Factorial, LHS, D-Optimal, Fractional Factorial, Plackett-Burman, CCD, Box-Behnken)
- **[Well Mapping Guide](docs/WELL_MAPPING.md)** - Understanding well organization and Opentrons compatibility
- **[Test Documentation](tests/README.md)** - Running and writing tests

---

## Quick start

### Easy Launch (Recommended)

The launcher scripts automatically set up the environment and install dependencies on first run.

| Platform | File | How to run |
|----------|------|------------|
| **macOS** | `run.command` | Double-click in Finder |
| **Windows** | `run.bat` | Double-click in Explorer |
| **Linux** | `run.sh` | Run `chmod +x run.sh && ./run.sh` |

> **Note:** Python 3.8+ must be installed. On Linux, you may also need to install tkinter (`sudo apt install python3-tk` on Ubuntu/Debian).

### Manual Setup (Alternative)

```bash
# Create virtual environment
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch
python main.py
```

The application opens with two tabs:

**Designer Tab** — Build experimental designs

- Add factors and levels (units supported: M, mM, µM, nM, %, w/v, v/v).
- Combination counter shows the number of conditions.
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

### 4) Analysis Tab — Run statistics & optimization

Switch to the **Analysis** tab in the application.

**Features:**
- Import CSV/XLSX results (expects factor columns + a **Response** column)
- Statistical modeling (Linear, Interactions, Quadratic models)
- Model comparison and automatic selection
- Main effects, interaction, and residual diagnostic plots
- Bayesian Optimization for next-batch suggestions
- Export results and publication-quality figures

---

## Troubleshooting

- **Missing dependencies** → Run `pip install -r requirements.txt`
- **GUI doesn't launch** → Ensure Python 3.8+ is installed: `python --version`
- **Tests failing** → Install dev dependencies: `pip install -r requirements-dev.txt`
- **Pipette volume assertion in Opentrons** → Ensure per‑transfer volumes are ≤ pipette max; adjust volumes or split transfers in the CSV/design
- **Gen1 vs Gen2 pipettes / alternative labware** → Change the model strings in `protein_stability_doe.py` (e.g., pipette names or labware definitions)
- **CSV parsed but no rows** → Verify the first line is headers and that at least one non‑empty row follows (no stray separators)

---

## Development notes

- **Architecture**: Modular design with separated GUI, business logic, and utilities
- **Testing**: 71 unit tests with pytest (23% coverage on core modules: 100% exporter, 99% plotter, 98% data_handler, 51% doe_analyzer)
- **GUI Framework**: Tkinter (cross-platform desktop application)
- **Statistical Engine**: statsmodels for regression analysis
- **Optimization**: Ax-Platform for Bayesian Optimization
- **Opentrons Protocol**: API Level 2.20, validated on p300 single + multi pipettes
- **Python Version**: 3.8–3.11 (specified in setup.py and .python-version)

**Running Tests:**
```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## License

This project is licensed under the MIT License.  
See the [LICENSE] file for details.
