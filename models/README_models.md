# Example Trained Models

This directory contains example trained XGBoost surrogate model files for academic review, reproducibility checks, and non-commercial demonstration of the PS-XGB-RLINE workflow.

The public model files are not intended to represent all trained models, deployment-specific model weights, future retrained models, or commercial model versions. Additional model sets may be released separately under different terms.

Commercial use of these model files, model weights, or derived model products is not permitted without prior written permission from the author.

## Directory Layout

The example model files are organized by surface roughness length:

| Directory | Surface roughness length |
| --- | --- |
| `model_z=0.05/` | `z0 = 0.05 m` |
| `model_z=0.5/` | `z0 = 0.5 m` |
| `model_z=1/` | `z0 = 1 m` |

Each directory contains XGBoost JSON model files for atmospheric stability classes and wind-region branches.

## Expected Files Per Model Set

The suffix indicates the surface roughness group:

- `L` for `z0 = 0.05 m`
- `M` for `z0 = 0.5 m`
- `H` for `z0 = 1 m`

Example for the `z0 = 0.5 m` model set:

| File | Description |
| --- | --- |
| `model_RLINE_remet_multidir_stable_2000_x0_M.json` | Stable, downwind (`x >= 0`) |
| `model_RLINE_remet_multidir_stable_2000_x-1_M.json` | Stable, upwind (`x < 0`) |
| `model_RLINE_remet_multidir_verystable_2000_x0_M.json` | Very stable, downwind |
| `model_RLINE_remet_multidir_verystable_2000_x-1_M.json` | Very stable, upwind |
| `model_RLINE_remet_multidir_unstable_2000_x0_M.json` | Unstable, downwind |
| `model_RLINE_remet_multidir_unstable_2000_x-1_M.json` | Unstable, upwind |
| `model_RLINE_remet_multidir_veryunstable_2000_x0_M.json` | Very unstable, downwind |
| `model_RLINE_remet_multidir_veryunstable_2000_x-1_M.json` | Very unstable, upwind |
| `model_RLINE_remet_multidir_neutral1_x0_M.json` | Neutral type 1, downwind |
| `model_RLINE_remet_multidir_neutral1_x-1_M.json` | Neutral type 1, upwind |
| `model_RLINE_remet_multidir_neutral2_x0_M.json` | Neutral type 2, downwind |
| `model_RLINE_remet_multidir_neutral2_x-1_M.json` | Neutral type 2, upwind |

## Stability Class Definitions

| Class | Obukhov length `L` (m) | Description |
| --- | --- | --- |
| VS | `0 < L <= 200` | Very stable |
| S | `200 < L < 1000` | Stable |
| N1 | `L >= 1000` | Neutral type 1 |
| N2 | `L <= -1000` | Neutral type 2 |
| U | `-1000 < L <= -200` | Unstable |
| VU | `-200 < L < 0` | Very unstable |

## Model Input Features

| Feature | Unit | Notes |
| --- | --- | --- |
| `x_rot` | m | Along-wind distance from source to receptor |
| `y_rot` | m | Cross-wind distance |
| `wind_sin` | dimensionless | Sine of relative wind direction to road |
| `wind_cos` | dimensionless | Cosine of relative wind direction to road |
| `H` | W/m2 | Sensible heat flux |
| `MixHGT_C` | m | Mixing height, used by selected stability classes |
| `L` | m | Obukhov length |
| `WSPD` | m/s | Wind speed |
