# Pre-trained Models

The trained XGBoost model files (`.json`) are included in this repository.

Alternatively, you can re-train the models from scratch using `training.py` after generating the AERMOD training data with `data_gen.py`.

## Expected Files

*Note: The following example uses models trained with surface roughness length **z₀ = 0.5 m (denoted as M)**.*

Place all downloaded `.json` files directly in this `models/` directory:

| File | Description |
|------|-------------|
| `model_RLINE_remet_multidir_stable_2000_x0_M.json`        | Stable — downwind (x ≥ 0) |
| `model_RLINE_remet_multidir_stable_2000_x-1_M.json`       | Stable — upwind  (x < 0)  |
| `model_RLINE_remet_multidir_verystable_2000_x0_M.json`    | Very stable — downwind    |
| `model_RLINE_remet_multidir_verystable_2000_x-1_M.json`   | Very stable — upwind      |
| `model_RLINE_remet_multidir_unstable_2000_x0_M.json`      | Unstable — downwind       |
| `model_RLINE_remet_multidir_unstable_2000_x-1_M.json`     | Unstable — upwind         |
| `model_RLINE_remet_multidir_veryunstable_2000_x0_M.json`  | Very unstable — downwind  |
| `model_RLINE_remet_multidir_veryunstable_2000_x-1_M.json` | Very unstable — upwind    |
| `model_RLINE_remet_multidir_neutral1_x0_M.json`           | Neutral type-1 — downwind |
| `model_RLINE_remet_multidir_neutral1_x-1_M.json`          | Neutral type-1 — upwind   |
| `model_RLINE_remet_multidir_neutral2_x0_M.json`           | Neutral type-2 — downwind |
| `model_RLINE_remet_multidir_neutral2_x-1_M.json`          | Neutral type-2 — upwind   |

## Stability Class Definitions

| Class | Obukhov Length L (m) | Description |
|-------|----------------------|-------------|
| VS    | 0 < L ≤ 200          | Very stable |
| S     | 200 < L < 1000       | Stable      |
| N1    | L ≥ 1000             | Neutral type-1 |
| N2    | L ≤ −1000            | Neutral type-2 |
| U     | −1000 < L ≤ −200     | Unstable    |
| VU    | −200 < L < 0         | Very unstable |

## Model Input Features

| Feature | Unit | Notes |
|---------|------|-------|
| `x_rot` | m | Along-wind distance (receptor − source) |
| `y_rot` | m | Cross-wind distance |
| `wind_sin` | — | sin(relative wind direction to road) |
| `wind_cos` | — | cos(relative wind direction to road) |
| `H` | W/m² | Sensible heat flux |
| `MixHGT_C` | m | Mixing height (N2/U/VU models only) |
| `L` | m | Obukhov length |
| `WSPD` | m/s | Wind speed |
