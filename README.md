# PS-XGB-RLINE: A Physics-Structured AERMOD-RLINE Surrogate for Fast Near-Road Dispersion Modeling

A physics-structured XGBoost surrogate model that replaces computationally expensive AERMOD-RLINE numerical integrations with a fast line-source decomposition strategy for high-resolution traffic-related pollution simulations.

---

## Overview

High-resolution dispersion modeling is essential for capturing near-road air pollution gradients, but the regulatory AERMOD-RLINE system can be computationally expensive for regional-scale applications. PS-XGB-RLINE addresses this bottleneck by integrating a physics-based partitioning framework with an XGBoost surrogate model.

### Pollutant Applicability

The provided examples focus on NOx. Because the surrogate is designed to learn physical dispersion relationships rather than atmospheric chemical reactions, the workflow can also be adapted to pollutants whose near-road concentrations are primarily governed by dispersion processes, such as particulate matter (PM), carbon dioxide (CO2), and nitrogen oxides (NOx), by adjusting the emission inputs.

Key features:

- Wind-direction-aware coordinate rotation
- Road-orientation-relative wind encoding per source segment
- Separate models for downwind (`x >= 0`) and upwind (`x < 0`) regions
- Six atmospheric stability classes (VS / S / N1 / N2 / U / VU)
- Vectorized batch inference with configurable memory footprint

---

## Repository Structure

```text
.
|-- data_gen.py          # Generate AERMOD input files for training data collection
|-- training.py          # Train XGBoost surrogate models from AERMOD output
|-- mode_inference.py    # Road-network time-series inference using the surrogate
|-- models/              # Example trained model files for demonstration and review
|   |-- README_models.md
|   |-- model_z=0.05/
|   |-- model_z=0.5/
|   `-- model_z=1/
|-- requirements.txt
`-- README.md
```

---

## Code, Model, and Data Availability

This repository provides the source code, core algorithms, data-processing scripts, and example trained XGBoost surrogate model files for demonstrating the PS-XGB-RLINE workflow described in the associated paper.

The public model files are provided for academic review, reproducibility checks, and non-commercial demonstration of the inference pipeline. Additional trained models, extended model sets, deployment-specific model weights, and models trained after publication are not included in this public repository and may be released separately under different terms.

Commercial use of the code, trained models, model weights, or derived model products is not permitted without prior written permission from the author.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Example trained models

Example trained XGBoost models are provided under the `models/` directory to demonstrate the workflow and support non-commercial academic review.

The repository may include example model sets for selected surface roughness lengths:

```text
models/
  model_z=0.05/   # z0 = 0.05 m
  model_z=0.5/    # z0 = 0.5 m
  model_z=1/      # z0 = 1 m
```

Each model set is organized by atmospheric stability class and wind-region branch. See `models/README_models.md` for details.

### 3. Configure paths

Edit the user configuration block at the top of each script:

| Script | Variables to set |
| --- | --- |
| `data_gen.py` | `SFC_FILE`, `PFL_FILE`, `OUTPUT_BASE`, `AERMOD_EXE` |
| `training.py` | `DATA_PATH`, `MODEL_SAVE_PATH` |
| `mode_inference.py` | `ROAD_SHP`, `EMISSION_CSV`, `MET_SFC`, `MODEL_DIR` |

### 4. Run

```bash
# Optional: re-generate training data with AERMOD
python data_gen.py

# Optional: re-train surrogate models
python training.py

# Run dispersion inference
python mode_inference.py
```

---

## Script Descriptions

### `data_gen.py` - Training Data Generation

Generates AERMOD `.inp`, `.sfc`, and `.pfl` input files for a unit line source across a sweep of wind directions. After running AERMOD on the generated inputs, the output files can be used by `training.py`.

Key steps:

1. Define a receptor grid centered on a unit line source.
2. Rotate the grid to align with each target wind direction.
3. Filter receptors located on the source body.
4. Write one AERMOD input folder per wind direction.

### `training.py` - Surrogate Model Training

Reads AERMOD concentration output and matched meteorological conditions, then trains XGBoost regressors in the wind-aligned coordinate frame.

Features used:

| Feature | Description |
| --- | --- |
| `x_rot` | Along-wind distance from source (m) |
| `y_rot` | Cross-wind distance from source (m) |
| `wind_sin`, `wind_cos` | Wind direction relative to road axis |
| `H` | Sensible heat flux (W/m2) |
| `L` | Obukhov length (m) |
| `WSPD` | Wind speed (m/s) |
| `MixHGT_C` | Mixing height (m), used for unstable/neutral classes |

### `mode_inference.py` - Road Network Inference

Applies the surrogate to a real road network with hourly emissions and meteorology.

Pipeline:

1. Merge road shapefile with hourly NOx emission CSV.
2. Convert WGS-84 coordinates to a local projected coordinate frame.
3. Decompose polylines into midpoint source segments.
4. Generate near-road and background receptor grids.
5. Load stability-class XGBoost models.
6. Batch-infer concentration at all receptors for all hours.

---

## Input Data Format

### Road shapefile (`roads.shp`)

Standard GeoDataFrame with at least:

- `NAME_1`: road identifier
- `geometry`: LineString or MultiLineString in WGS-84

### Hourly emission CSV

```text
NAME,data_time,nox,length
Road_A,2021-01-01 00:00:00,12.5,350
...
```

- `nox`: hourly NOx emission input for the road segment
- `length`: road length (m)

### Meteorological file

Standard AERMET surface file (`.sfc`). Required variables include:

- `H`
- `MixHGT_C`
- `L`
- `WSPD`
- `WDIR`

---

## Requirements

See `requirements.txt`. Main dependencies:

```text
numpy  pandas  geopandas  shapely  pyproj
xgboost  scikit-learn  matplotlib  seaborn  scipy
```

---

## Citation

If you use this work, please cite the associated paper:

Ma, J., Wang, C., and Xiang, S. Fast near-road pollutant dispersion modeling: A physics-structured AERMOD-RLINE surrogate. Atmospheric Environment, 375, 122009.

---

## License and Usage Restrictions

This repository is provided for academic research, peer review, and non-commercial demonstration purposes only.

Commercial use is not permitted without prior written permission from the author. This includes, but is not limited to, commercial deployment, paid consulting, integration into commercial software, and use of the trained models or model weights in commercial products or services.

The example trained model files are provided only for demonstrating and reproducing the workflow described in the associated paper. Additional trained models, deployment-specific models, and future commercial versions are not included in this public repository.

No open-source license is granted for commercial use. All rights not expressly granted here are reserved by the author.
