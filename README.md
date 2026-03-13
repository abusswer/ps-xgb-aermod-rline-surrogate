# Fast Near-Road Pollutant Dispersion Modeling: A Physics-Structured AERMOD-RLINE Surrogate

A fast XGBoost surrogate that replaces AERMOD-RLINE batch runs for road-traffic pollutant dispersion modelling over real road networks with time-varying emissions and meteorology.

---

## Overview

Traditional Gaussian dispersion models (AERMOD-RLINE) are accurate but slow—running a full year of hourly simulations for a dense road network can take days. This project trains an XGBoost surrogate on pre-computed AERMOD unit-line-source output, then uses it to infer concentrations at thousands of receptor points in seconds per hour.

**Key features**

- Wind-direction-aware coordinate rotation (no wind-direction binning)
- Road-orientation-relative wind encoding per source segment
- Separate models for downwind (x ≥ 0) and upwind (x < 0) regions
- Six atmospheric stability classes (VS / S / N1 / N2 / U / VU)
- Vectorised batch inference with configurable memory footprint

---

## Repository Structure

```
├── data_gen.py          # Generate AERMOD input files for training data collection
├── training.py          # Train XGBoost surrogate from AERMOD output
├── mode_inference.py    # Road-network time-series inference using the surrogate
├── models/              # Pre-trained model files (download separately — see below)
│   └── README_models.md
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download pre-trained models

Pre-trained models (~48 MB per file, 12 files total) are too large for GitHub.  
Download from the **[Releases page](../../releases)** and place all `.json` files into the `models/` directory.

Expected files:
```
models/
  model_RLINE_remet_multidir_stable_2000_x0_M.json
  model_RLINE_remet_multidir_stable_2000_x-1_M.json
  model_RLINE_remet_multidir_verystable_2000_x0_M.json
  model_RLINE_remet_multidir_verystable_2000_x-1_M.json
  model_RLINE_remet_multidir_unstable_2000_x0_M.json
  model_RLINE_remet_multidir_unstable_2000_x-1_M.json
  model_RLINE_remet_multidir_veryunstable_2000_x0_M.json
  model_RLINE_remet_multidir_veryunstable_2000_x-1_M.json
  model_RLINE_remet_multidir_neutral1_x0_M.json
  model_RLINE_remet_multidir_neutral1_x-1_M.json
  model_RLINE_remet_multidir_neutral2_x0_M.json
  model_RLINE_remet_multidir_neutral2_x-1_M.json
```

### 3. Configure paths

Edit the **User Configuration** block at the top of each script:

| Script | Variables to set |
|--------|-----------------|
| `data_gen.py` | `SFC_FILE`, `PFL_FILE`, `OUTPUT_BASE`, `AERMOD_EXE` |
| `training.py` | `DATA_PATH`, `MODEL_SAVE_PATH` |
| `mode_inference.py` | `ROAD_SHP`, `EMISSION_CSV`, `MET_SFC`, `MODEL_DIR` |

### 4. Run

```bash
# (Optional) Re-generate training data with AERMOD
python data_gen.py

# (Optional) Re-train surrogate models
python training.py

# Run dispersion inference
python mode_inference.py
```

---

## Script Descriptions

### `data_gen.py` — Training Data Generation

Generates AERMOD `.inp` / `.sfc` / `.pfl` input files for a **unit line source** across a sweep of wind directions (0°–90° in 4° steps).  
After running AERMOD on the generated inputs, the output `.txt` files feed into `training.py`.

Key steps:
1. Define a receptor grid centred on a unit line source (x: −100 → 2000 m, y: ±100 m)
2. Rotate the grid to align with each target wind direction
3. Filter receptors located on the source body
4. Write one AERMOD input folder per wind direction

### `training.py` — Surrogate Model Training

Reads AERMOD concentration output and matched meteorological conditions, then trains XGBoost regressors in the wind-aligned coordinate frame.

Features used:
| Feature | Description |
|---------|-------------|
| `x_rot` | Along-wind distance from source (m) |
| `y_rot` | Cross-wind distance from source (m) |
| `wind_sin`, `wind_cos` | Wind direction relative to road axis |
| `H` | Sensible heat flux (W/m²) |
| `L` | Obukhov length (m) |
| `WSPD` | Wind speed (m/s) |
| `MixHGT_C` | Mixing height (m) — unstable/neutral classes only |

### `mode_inference.py` — Road Network Inference

Applies the surrogate to a real road network with hourly emissions and meteorology.

Pipeline:
1. Merge road shapefile with hourly NOx emission CSV
2. Convert WGS-84 coordinates → local UTM frame
3. Decompose polylines into 10-m midpoint segments
4. Generate near-road + background receptor grid
5. Load stability-class XGBoost models
6. Batch-infer concentration at all receptors for all hours

---

## Input Data Format

### Road shapefile (`roads.shp`)
Standard GeoDataFrame with at least:
- `NAME_1`: road identifier
- `geometry`: LineString or MultiLineString in WGS-84

### Hourly emission CSV
```
NAME,data_time,nox,length
Road_A,2021-01-01 00:00:00,12.5,350
...
```
- `nox`: hourly NOx emission (g/s per road segment)
- `length`: road length (m)

### Meteorological file (AERMOD `.sfc` format)
Standard AERMET surface file. Required columns: `H`, `MixHGT_C`, `L`, `WSPD`, `WDIR`.

---

## Requirements

See `requirements.txt`. Main dependencies:

```
numpy  pandas  geopandas  shapely  pyproj
xgboost  scikit-learn  matplotlib  seaborn  scipy
```

---

## Citation

If you use this work, please cite the associated paper (details to be added upon publication).

---

## License

MIT License. See `LICENSE` for details.
