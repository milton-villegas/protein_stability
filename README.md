<p align="center">
  <img src="assets/scout_logo.png" alt="SCOUT Logo" width="150">
</p>

<h1 align="center">SCOUT</h1>
<p align="center"><b>Screening & Condition Optimization Utility Tool</b></p>

<p align="center">
  <a href="https://huggingface.co/spaces/milton-villegas/SCOUT"><b>Try it live</b></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/platform-Web%20%7C%20Desktop-lightgrey" alt="Platform">
  <img src="https://img.shields.io/github/last-commit/milton-villegas/SCOUT" alt="Last Commit">
</p>

<p align="center">
A web application for <b>Design of Experiments (DoE)</b>, statistical analysis, and <b>Bayesian Optimization</b><br>
for protein stability, buffer screening, crystallization, and formulation studies.
</p>

## Features

- **Design of Experiments** — Full factorial, fractional factorial, Plackett-Burman, Box-Behnken, Central Composite, Latin Hypercube (SMT-optimized), D-Optimal
- **Statistical Analysis** — Auto model selection, interaction effects, quadratic models, model comparison
- **Bayesian Optimization** — Multi-objective suggestions for next experiments using Ax-Platform
- **Opentron Export** — Excel (3 sheets) and CSV with volume calculations, well positions, and reagent setup
- **Visualization** — Main effects, interaction plots, residual diagnostics, Pareto fronts

## Quick Start

**Web (recommended):** Use the [live demo](https://huggingface.co/spaces/milton-villegas/SCOUT) — no installation needed.

**Local web app:**
```bash
./start-web.sh
```
Opens at http://localhost:5173 (frontend) with API at http://localhost:8000.

## Offline Desktop App

A Tkinter desktop version is also available for offline use.

| Platform | File | How to run |
|----------|------|------------|
| **macOS** | `run.command` | Double-click in Finder |
| **Windows** | `run.bat` | Double-click in Explorer |
| **Linux** | `run.sh` | Run `chmod +x run.sh && ./run.sh` |

> Requires Python 3.10+. On Linux: `sudo apt install python3-tk`.

## Documentation

- **[Design Types Guide](docs/DESIGN_GUIDE.md)** — Choosing and using the 7 design types
- **[Well Mapping Guide](docs/WELL_MAPPING.md)** — Well organization and Opentrons compatibility

## More Information

<details>
<summary>Repository structure</summary>

```
SCOUT/
├─ backend/                    # FastAPI web API
│  ├─ main.py                  # App entry point
│  ├─ routers/                 # API endpoints
│  ├─ services/                # Business logic wrappers
│  └─ schemas/                 # Request/response models
├─ frontend/                   # SvelteKit web UI
│  └─ src/
│     ├─ routes/               # Pages (design, analysis)
│     └─ lib/                  # Components, stores, API client
├─ core/                       # Shared business logic
│  ├─ doe_designer.py          # Design generation
│  ├─ doe_analyzer.py          # Statistical analysis
│  ├─ optimizer.py             # Bayesian optimization
│  ├─ volume_calculator.py     # C1V1=C2V2 calculations
│  └─ well_mapper.py           # 96/384-well plate mapping
├─ gui/                        # Tkinter desktop UI (offline)
├─ opentrons/                  # Robot protocol
├─ tests/                      # 400+ unit tests
├─ Dockerfile                  # Hugging Face Spaces deployment
├─ start-web.sh                # Local web launcher
└─ main.py                     # Desktop app entry point
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
</details>

<details>
<summary>Developer setup</summary>

```bash
git clone https://github.com/milton-villegas/SCOUT
cd SCOUT
pip install -e .
pip install -r requirements-dev.txt
pytest tests/ -v
```

### Stack

- **Backend**: FastAPI + uvicorn
- **Frontend**: SvelteKit + Tailwind + DaisyUI
- **Analysis**: statsmodels, Ax-Platform, SMT
- **Python**: 3.10+
</details>

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE.txt) for details.
