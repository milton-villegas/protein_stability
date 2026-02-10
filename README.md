<p align="center">
  <img src="assets/scout_logo.png" alt="SCOUT Logo" width="150">
</p>

<h1 align="center">SCOUT</h1>
<p align="center"><b>Screening & Condition Optimization Utility Tool</b></p>

<p align="center">
SCOUT is in active development and feedback is appreciated.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey" alt="Platform">
  <img src="https://img.shields.io/github/last-commit/milton-villegas/SCOUT" alt="Last Commit">
  <img src="https://img.shields.io/github/issues/milton-villegas/SCOUT" alt="Issues">
</p>

<p align="center">
A Python toolkit for <b>Design of Experiments (DoE)</b>, statistical analysis, and <b>Bayesian Optimization</b><br>
for protein stability, buffer screening, crystallization, and formulation studies.
</p>

## Installation

Double-click the launcher to automatically set up the environment and install dependencies:

| Platform | File | How to run |
|----------|------|------------|
| **macOS** | `run.command` | Double-click in Finder |
| **Windows** | `run.bat` | Double-click in Explorer |
| **Linux** | `run.sh` | Run `chmod +x run.sh && ./run.sh` |

> **Note:** Python 3.8+ must be installed. On Linux, you may also need tkinter (`sudo apt install python3-tk`).

For manual installation see [Developer Information](#developer-information)

## Features

- **Design of Experiments (DoE)** — Full factorial, fractional factorial, Plackett-Burman, Box-Behnken, Central Composite, Latin Hypercube, D-Optimal
- **Statistical Analysis** — Linear models, interaction effects, quadratic models, ANOVA
- **Bayesian Optimization** — Smart suggestions for next experiments using Ax-Platform
- **Visualization** — Main effects, interaction plots, residual diagnostics, Pareto fronts
- **Opentrons Integration** — Export CSV for automated liquid handling

## Documentation

- **[Design Types Guide](docs/DESIGN_GUIDE.md)** — Choosing and using the 7 design types
- **[Well Mapping Guide](docs/WELL_MAPPING.md)** — Well organization and Opentrons compatibility

## More Information

<details>

<summary>Repository structure</summary>

```
SCOUT/
├─ run.command              # macOS launcher
├─ run.bat                  # Windows launcher
├─ run.sh                   # Linux launcher
├─ main.py                  # Application entry point
├─ gui/                     # User interface
│  ├─ main_window.py
│  └─ tabs/
│     ├─ designer_tab.py    # DoE Designer
│     └─ analysis_tab.py    # Analysis & optimization
├─ core/                    # Business logic
│  ├─ doe_designer.py       # Design generation
│  ├─ doe_analyzer.py       # Statistical analysis
│  ├─ optimizer.py          # Bayesian optimization
│  └─ ...
├─ opentrons/               # Robot protocol
│  └─ protein_stability_doe.py
├─ tests/                   # Unit tests
└─ requirements.txt         # Dependencies
```

</details>

<details>

<summary>Opentrons Protocol</summary>

Upload `opentrons/protein_stability_doe.py` to the Opentrons App (API Level 2.20).

### Deck layout

- **Slot 1**: 96-well plate #1 — `greiner_96_well_u_bottom_323ul`
- **Slot 2**: 384-well plate — `corning_384_wellplate_112ul_flat`
- **Slot 5**: 24-well reservoir — `cytiva_24_reservoir_10ml`
- **Slots 7–11**: `opentrons_96_tiprack_300ul`

### Pipettes

- **Left**: `p300_multi` (96→384 transfers)
- **Right**: `p300_single` (buffer preparation)

### CSV format

```
Buffer,Glycerol,NaCl,pH Buffer
150,20,10,5
140,30,15,5
```

</details>

<details>

<summary>Troubleshooting</summary>

- **Missing dependencies** → Run `pip install -r requirements.txt`
- **GUI doesn't launch** → Ensure Python 3.8+: `python --version`
- **Tests failing** → Install dev dependencies: `pip install -r requirements-dev.txt`
- **Pipette volume error** → Ensure volumes are ≤ pipette max

</details>

## Developer Information

<details>

<summary>Installation & Setup</summary>

### Developer installation

To develop on SCOUT please fork this repository and then install locally:

```
git clone https://github.com/YOUR_USER/SCOUT
cd SCOUT
pip install -e .
pip install -r requirements-dev.txt
```

### Running tests

```
pytest tests/ -v
```

71 unit tests covering core modules.

### Architecture

- **GUI Framework**: Tkinter (cross-platform)
- **Statistical Engine**: statsmodels
- **Optimization**: Ax-Platform for Bayesian Optimization
- **Python**: 3.8–3.11

</details>

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE.txt) for details.
