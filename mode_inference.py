# =============================================================================
# mode_inference.py
# XGBoost Surrogate Model for Road-Traffic pollutant Dispersion Inference
#
# Replaces AERMOD / RLINE batch runs with a fast surrogate model trained
# on unit-line-source AERMOD output.  Supports time-series inference over
# a real road network with time-varying emissions and meteorology.
#
# Pipeline overview:
#   1. Load road network + hourly emission data (GeoDataFrame merge)
#   2. Convert road geometries to local UTM coordinate frame
#   3. Decompose each road into 10-m line-source segments (midpoints)
#   4. Generate receptor grid (near-road offset + background)
#   5. Load pre-trained XGBoost models (one per stability class, pos/neg x)
#   6. Run time-series inference: predict pollutant concentration at all receptors
#      for every hour using predict_time_series_xgb
#
# Usage:
#   1. Set all path variables in the "User Configuration" block below.
#   2. Ensure model .json files are placed in the models/ directory.
#   3. Run the script.
# =============================================================================

# ---- User Configuration (edit before running) ------------------------------
ROAD_SHP    = r"YOUR_PATH\roads.shp"
EMISSION_CSV= r"YOUR_PATH\hourly_emission.csv"
MET_SFC     = r"YOUR_PATH\met_file.SFC"
MODEL_DIR   = r"models"   # directory containing the .json model files
# ----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
import math
import geopandas as gpd
import os
import shutil
from scipy.interpolate import griddata
from pyproj import Proj, transform
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from shapely.validation import make_valid
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler


# ============================================================
# Step 1: Merge road network geometry with hourly emission data
# ============================================================
e_road_gdf = gpd.read_file(ROAD_SHP)
e_h_df     = pd.read_csv(EMISSION_CSV)

e_h_df = e_h_df.rename(columns={'NAME': 'NAME_1'})

# Non-spatial join on road name
merged = pd.merge(e_h_df, e_road_gdf[['NAME_1', 'geometry']],
                  on='NAME_1', how='left')
merged_gdf = gpd.GeoDataFrame(merged, geometry='geometry', crs=e_road_gdf.crs)

unmatched = merged_gdf[merged_gdf['geometry'].isna()]
if not unmatched.empty:
    print("Warning: the following roads could not be matched to a geometry:")
    print(unmatched['NAME_1'].unique())


# ============================================================
# Step 2: Convert WGS-84 coordinates to local UTM frame
# ============================================================
unique_roads = (merged_gdf
                .drop_duplicates(subset='NAME_1')[['NAME_1', 'geometry']]
                .reset_index(drop=True))

wgs84  = Proj(proj="latlong", datum="WGS84")
utm51n = Proj(proj="utm", zone=51, datum="WGS84", hemisphere="north")

def convert_to_utm(lon, lat):
    """Convert WGS-84 lon/lat to UTM Zone 51N (x, y) in metres."""
    utm_x, utm_y = transform(wgs84, utm51n, lon, lat)
    return utm_x, utm_y

sources = []
for _, row in unique_roads.iterrows():
    geom   = row['geometry']
    coords = []
    if geom.geom_type == 'LineString':
        for lon, lat in geom.coords:
            coords.append(convert_to_utm(lon, lat))
    elif geom.geom_type == 'MultiLineString':
        for line in geom.geoms:
            for lon, lat in line.coords:
                coords.append(convert_to_utm(lon, lat))
    sources.append(coords)

unique_roads['utm_coords'] = sources

# Shift to local origin (min x, min y = 0, 0)
all_points = [pt for line in sources for pt in line]
min_x = min(p[0] for p in all_points)
min_y = min(p[1] for p in all_points)
unique_roads['new_coords'] = [
    [(x - min_x, y - min_y) for (x, y) in line]
    for line in unique_roads['utm_coords']
]

# Map local coordinates back to the full merged GeoDataFrame
merged_gdf = merged_gdf.merge(
    unique_roads[['NAME_1', 'new_coords']], on='NAME_1', how='left'
)

# Parse datetime and derive helper columns
merged_gdf["data_time"] = pd.to_datetime(merged_gdf["data_time"])
merged_gdf["day"]       = merged_gdf["data_time"].dt.date
merged_gdf["hour"]      = merged_gdf["data_time"].dt.hour
merged_gdf["index"]     = merged_gdf["NAME_1"].astype("category").cat.codes

# Emission in g/s/m2  (NOx per unit area per second)
merged_gdf["nox_g_m_s2"] = merged_gdf["nox"] / (merged_gdf["length"] * 7 * 1000 * 3600)

merged_gdf = merged_gdf.sort_values('data_time')
first_time    = merged_gdf['data_time'].iloc[0]
first_hour_df = merged_gdf[merged_gdf['data_time'] == first_time].copy()


# ============================================================
# Step 3: Receptor grid generation
# ============================================================

def make_rectangular_buffer(line: LineString, half_width: float) -> Polygon:
    """
    Build a rectangular road buffer polygon (no semi-circle end caps).

    Args:
        line:       Shapely LineString in local coordinates
        half_width: half-width of the buffer in metres

    Returns:
        Shapely Polygon or None if the input is degenerate
    """
    coords = np.array(line.coords)
    if len(coords) < 2:
        return None

    left_points, right_points = [], []
    for i in range(len(coords) - 1):
        (x1, y1), (x2, y2) = coords[i], coords[i + 1]
        dx, dy = x2 - x1, y2 - y1
        seg_len = np.hypot(dx, dy)
        if seg_len == 0:
            continue
        nx, ny = -dy / seg_len, dx / seg_len
        left_points.append((x1 + nx * half_width, y1 + ny * half_width))
        right_points.append((x1 - nx * half_width, y1 - ny * half_width))
        if i == len(coords) - 2:
            left_points.append((x2 + nx * half_width, y2 + ny * half_width))
            right_points.append((x2 - nx * half_width, y2 - ny * half_width))

    right_points.reverse()
    return Polygon(left_points + right_points)


def generate_receptors_custom_offset(
    e_road_df,
    offset_rule={6.5: 2, 15: 4, 30: 7, 50: 12},
    background_spacing=50,
    buffer_extra=30,
    width_col='width',
    global_extent=None
):
    """
    Generate receptor points for a road network:
      - Near-road receptors at user-defined lateral offsets from each segment
      - Background receptors on a regular grid outside the road buffers

    Args:
        e_road_df:          GeoDataFrame with 'new_coords' and width_col columns
        offset_rule:        dict {lateral_offset_from_buffer (m): along-road spacing (m)}
        background_spacing: grid spacing (m) for background receptors
        buffer_extra:       extra buffer distance beyond road half-width (m)
        width_col:          column name for road width
        global_extent:      (xmin, xmax, ymin, ymax) for background grid;
                            auto-computed from road coordinates if None

    Returns:
        GeoDataFrame of receptor points
    """
    receptor_list = []

    # --- Build rectangular road buffers ---
    if width_col not in e_road_df.columns:
        e_road_df[width_col] = 7.0  # default road width

    buffers = []
    for _, row in e_road_df.iterrows():
        half_width = 0.5 * row[width_col] + buffer_extra
        try:
            coords = row['new_coords']
            line   = LineString(coords)
            poly   = make_rectangular_buffer(line, half_width)
            if poly and not poly.is_empty:
                if not poly.is_valid:
                    poly = make_valid(poly)
                buffers.append(poly)
        except Exception as e:
            print(f"Skip invalid geometry for road {row.get('index', '?')}: {e}")

    if not buffers:
        raise ValueError("No valid buffer polygons generated.")
    buffer_union = unary_union(buffers)

    # --- Near-road offset receptors ---
    for _, row in e_road_df.iterrows():
        name        = row.get('index', 'road')
        coords      = row['new_coords']
        base_offset = 0.5 * row[width_col] + buffer_extra

        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            dx, dy  = x2 - x1, y2 - y1
            seg_len = np.hypot(dx, dy)
            if seg_len == 0:
                continue
            ux, uy = dx / seg_len, dy / seg_len
            nx, ny = -uy, ux

            for offset, spacing in offset_rule.items():
                true_offset = base_offset + offset
                n_points    = max(1, int(seg_len / spacing) + 1)
                t           = np.linspace(0, 1, n_points)
                xs          = x1 + t * dx
                ys          = y1 + t * dy

                for side, sgn in [('left', 1), ('right', -1)]:
                    x_offset = xs + sgn * true_offset * nx
                    y_offset = ys + sgn * true_offset * ny
                    for x, y in zip(x_offset, y_offset):
                        receptor_list.append({
                            'NAME_1':             name,
                            'segment_id':         f"{name}__{i}",
                            'x': x, 'y': y,
                            'offset_from_buffer': offset,
                            'true_offset':        true_offset,
                            'type': 'road_near',
                            'side': side
                        })

    # --- Background grid receptors ---
    if global_extent is None:
        all_x = [x for coords in e_road_df['new_coords'] for x, _ in coords]
        all_y = [y for coords in e_road_df['new_coords'] for _, y in coords]
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
    else:
        xmin, xmax, ymin, ymax = global_extent

    gx = np.arange(xmin, xmax + background_spacing, background_spacing)
    gy = np.arange(ymin, ymax + background_spacing, background_spacing)
    grid_x, grid_y = np.meshgrid(gx, gy)

    for x, y in zip(grid_x.flatten(), grid_y.flatten()):
        receptor_list.append({
            'NAME_1': 'background', 'segment_id': 'grid',
            'x': x, 'y': y,
            'offset_from_buffer': None, 'true_offset': None,
            'type': 'background'
        })

    receptors_df  = pd.DataFrame(receptor_list)
    receptors_gdf = gpd.GeoDataFrame(
        receptors_df,
        geometry=gpd.points_from_xy(receptors_df["x"], receptors_df["y"])
    )

    # Remove background points inside road buffer
    receptors_gdf["in_buffer"] = receptors_gdf.geometry.intersects(buffer_union)
    receptors_gdf = receptors_gdf[~receptors_gdf["in_buffer"]].copy()

    print(f"Buffer count: {len(buffers)}")
    print(f"Union type:   {buffer_union.geom_type}")

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    for line in e_road_df['new_coords']:
        xs, ys = zip(*line)
        ax.plot(xs, ys, color='black', linewidth=1)

    if buffer_union.geom_type == "MultiPolygon":
        buffer_gdf = gpd.GeoDataFrame(geometry=list(buffer_union.geoms))
    else:
        buffer_gdf = gpd.GeoDataFrame(geometry=[buffer_union])
    buffer_gdf.plot(ax=ax, color="r", alpha=0.6, edgecolor="none")

    near_points = receptors_gdf[receptors_gdf["type"] == "road_near"]
    bg_points   = receptors_gdf[receptors_gdf["type"] == "background"]
    ax.scatter(near_points["x"], near_points["y"], s=2, c="blue",
               label="Near-road receptors")
    ax.scatter(bg_points["x"],   bg_points["y"],   s=2, c="green",
               alpha=0.6, label="Background receptors")
    ax.set_aspect("equal", "box")
    ax.legend()
    plt.xlim(1000, 1400)
    plt.ylim(600,  900)
    plt.title("Receptor layout (rectangular buffer + local coordinates)")
    plt.show()

    return receptors_gdf


receptors = generate_receptors_custom_offset(
    first_hour_df,
    offset_rule={3.5: 40, 8.5: 40},
    background_spacing=50,
    buffer_extra=3,
    global_extent=None
)
print(f"Total receptors: {len(receptors)}")

# Remove duplicate (x, y) receptor positions
receptors_unique = receptors.drop_duplicates(subset=['x', 'y']).reset_index(drop=True)
print(f"Before dedup: {len(receptors)}, after dedup: {len(receptors_unique)}")


# ============================================================
# Step 4: Decompose each road polyline into 10-m segment midpoints
# ============================================================

def split_polyline_by_interval_with_angle(coords, interval=10):
    """
    Split a polyline into segments of fixed arc length.

    For each segment, return its midpoint (x, y) and the road bearing
    in degrees (0-180 range, direction-agnostic).

    Args:
        coords:   list of (x, y) tuples defining the polyline
        interval: target arc-length of each sub-segment (m)

    Returns:
        list of (x_mid, y_mid, angle_deg) tuples
    """
    if len(coords) < 2:
        return []

    seg_lengths = [np.hypot(coords[i+1][0] - coords[i][0],
                            coords[i+1][1] - coords[i][1])
                   for i in range(len(coords) - 1)]
    cum_lengths = np.cumsum([0] + seg_lengths)
    total_len   = cum_lengths[-1]
    n_segments  = int(np.ceil(total_len / interval))

    mids = []
    for i in range(n_segments):
        start_d = i * interval
        end_d   = min((i + 1) * interval, total_len)
        mid_d   = (start_d + end_d) / 2

        x_mid = np.interp(mid_d, cum_lengths, [p[0] for p in coords])
        y_mid = np.interp(mid_d, cum_lengths, [p[1] for p in coords])

        # Road bearing from the enclosing original segment
        seg_idx = np.searchsorted(cum_lengths, mid_d) - 1
        seg_idx = min(seg_idx, len(coords) - 2)
        dx = coords[seg_idx + 1][0] - coords[seg_idx][0]
        dy = coords[seg_idx + 1][1] - coords[seg_idx][1]

        angle_deg = np.degrees(np.arctan2(dy, dx))
        angle_deg = (270 - angle_deg) % 360
        angle_deg = angle_deg % 180  # direction-agnostic (0-180)

        mids.append((x_mid, y_mid, angle_deg))
    return mids


# Build base midpoints for the first time step
interval   = 10
first_day  = merged_gdf["data_time"].unique()[0]
first_gdf  = merged_gdf[merged_gdf["data_time"] == first_day]

base_midpoints = []
for _, row in first_gdf.iterrows():
    coords = row['new_coords']
    if not coords or len(coords) < 2:
        continue
    road_id = row['index']
    for j, (xm, ym, angle_deg) in enumerate(
            split_polyline_by_interval_with_angle(coords, interval)):
        base_midpoints.append({
            'road_id':    road_id,
            'segment_id': f"{road_id}_{j}",
            'xm': xm, 'ym': ym,
            'angle_deg':  angle_deg,
            'interval':   interval
        })

base_midpoints_df = pd.DataFrame(base_midpoints)

# Tile midpoints across all time steps and join emission rates
unique_data_times = merged_gdf["data_time"].unique()

midpoints_df = base_midpoints_df.loc[
    base_midpoints_df.index.repeat(len(unique_data_times))
].copy()
midpoints_df["data_time"] = np.tile(unique_data_times, len(base_midpoints_df))

emission_df  = merged_gdf[["index", "data_time", "nox_g_m_s2"]].copy()
midpoints_df = midpoints_df.merge(
    emission_df,
    left_on=["road_id", "data_time"],
    right_on=["index",  "data_time"],
    how="left"
)
midpoints_df.rename(columns={"nox_g_m_s2": "emission"}, inplace=True)
midpoints_df.drop(columns="index", inplace=True)


# ============================================================
# Step 5: Load meteorological data and assign stability class
# ============================================================
col_names = [
    "Year", "Month", "Day", "JulianDay", "Hour",
    "H", "USTAR", "WSTAR", "ThetaGrad", "MixHGT_C", "MixHGT_M", "L",
    "Z0", "B0", "Albedo",
    "WSPD", "WDIR", "AnemoHt", "Temp", "MeasHt",
    "PrecipType", "PrecipAmt", "RH", "Pressure", "CloudCover",
    "WindFlag", "CloudTempFlag"
]

def read_sfc(path):
    """Read an AERMOD surface (.sfc) meteorological file."""
    return pd.read_csv(path, delim_whitespace=True,
                       names=col_names, skiprows=1, comment="#")

sfc = read_sfc(MET_SFC)

years  = sfc["Year"].astype(int) % 100
months = sfc["Month"].astype(int)
days   = sfc["Day"].astype(int)
hours  = sfc["Hour"].astype(int)
sfc["Date"] = (years * 1000000 + months * 10000 + days * 100 + hours).astype(int)

L    = sfc["L"]
WSPD = sfc["WSPD"]
MixHGT = sfc["MixHGT_C"]

sfc["Stab_Class"] = "UNK"
sfc.loc[(L > 0)    & (L <= 200),                               "Stab_Class"] = "VS"   # very stable
sfc.loc[(L > 200)  & (L < 1000),                               "Stab_Class"] = "S"    # stable
sfc.loc[(L >= 1000) & (WSPD != 999) & (L != -99999),           "Stab_Class"] = "N1"   # neutral type 1
sfc.loc[(L <= -1000) & (WSPD != 999) & (L != -99999)
         & (MixHGT != -999),                                   "Stab_Class"] = "N2"   # neutral type 2
sfc.loc[(L > -1000) & (L <= -200) & (MixHGT != -999)
         & (WSPD != 999),                                      "Stab_Class"] = "U"    # unstable
sfc.loc[(L > -200)  & (L < 0)    & (MixHGT != -999)
         & (WSPD != 999),                                      "Stab_Class"] = "VU"   # very unstable

met = sfc[["H", "MixHGT_C", "L", "WSPD", "WDIR", "Date", "Stab_Class"]]


# ============================================================
# Step 6: Load pre-trained XGBoost models
# ============================================================
def load_model(path):
    m = xgb.XGBRegressor(n_jobs=1)
    m.load_model(path)
    return m

print("Loading models...")
stable_x0        = load_model(os.path.join(MODEL_DIR, "model_RLINE_remet_multidir_stable_2000_x0_M.json"))
stable_x_1       = load_model(os.path.join(MODEL_DIR, "model_RLINE_remet_multidir_stable_2000_x-1_M.json"))
verystable_x0    = load_model(os.path.join(MODEL_DIR, "model_RLINE_remet_multidir_verystable_2000_x0_M.json"))
verystable_x_1   = load_model(os.path.join(MODEL_DIR, "model_RLINE_remet_multidir_verystable_2000_x-1_M.json"))
unstable_x0      = load_model(os.path.join(MODEL_DIR, "model_RLINE_remet_multidir_unstable_2000_x0_M.json"))
unstable_x_1     = load_model(os.path.join(MODEL_DIR, "model_RLINE_remet_multidir_unstable_2000_x-1_M.json"))
veryunstable_x0  = load_model(os.path.join(MODEL_DIR, "model_RLINE_remet_multidir_veryunstable_2000_x0_M.json"))
veryunstable_x_1 = load_model(os.path.join(MODEL_DIR, "model_RLINE_remet_multidir_veryunstable_2000_x-1_M.json"))
neutral1_x0      = load_model(os.path.join(MODEL_DIR, "model_RLINE_remet_multidir_neutral1_x0_M.json"))
neutral1_x_1     = load_model(os.path.join(MODEL_DIR, "model_RLINE_remet_multidir_neutral1_x-1_M.json"))
neutral2_x0      = load_model(os.path.join(MODEL_DIR, "model_RLINE_remet_multidir_neutral2_x0_M.json"))
neutral2_x_1     = load_model(os.path.join(MODEL_DIR, "model_RLINE_remet_multidir_neutral2_x-1_M.json"))
print("Models loaded.")


# ============================================================
# Step 7: Time-series inference function
# ============================================================

def predict_time_series_xgb(
    models: dict,
    receptors_x: np.ndarray,
    receptors_y: np.ndarray,
    sources: np.ndarray,   # shape = (T, N_sources, 4): sx, sy, strength, road_angle_deg
    met: pd.DataFrame,
    x_range0=(0, 1000.0),
    x_range1=(-100, 0.0),
    y_range=(-50.0, 50.0),
    batch_size=200000
):
    """
    XGBoost surrogate time-series inference .

    For each time step t and stability class:
      1. Rotate all receptor and source coordinates into the wind-aligned frame.
      2. Compute (x_hat, y_hat) = rotated receptor position - rotated source position.
      3. Mask pairs within the downwind (x_range0) and upwind (x_range1) ranges.
      4. Build feature matrix X and call the corresponding XGBoost model.
      5. Accumulate concentration contributions at each receptor.

    Wind-direction encoding uses each source's relative wind direction
    (global_wind_deg - road_angle) % 360, so the model can account for
    the angle between wind and road orientation.

    Args:
        models:       dict keyed by stability class, each with 'pos' and 'neg' sub-models
        receptors_x:  1-D array of receptor x-coordinates (m)
        receptors_y:  1-D array of receptor y-coordinates (m)
        sources:      3-D array, shape (T, N_sources, 4)
                        columns: [source_x, source_y, emission_strength, road_angle_deg]
        met:          DataFrame with columns [H, MixHGT_C, L, WSPD, WDIR, Date, Stab_Class]
        x_range0:     (x_min, x_max) downwind distance range (m)  for positive-x model
        x_range1:     (x_min, x_max) upwind  distance range (m)   for negative-x model
        y_range:      (y_min, y_max) crosswind range (m)
        batch_size:   max rows per XGBoost prediction call (memory vs. speed trade-off)

    Returns:
        pd.DataFrame with columns [Date, Receptor_ID, Receptor_X, Receptor_Y, Conc]
    """
    # Stability classes that lack a mixing-height (HC) feature
    no_HC_classes = ["VS", "S", "N1"]

    rx = np.asarray(receptors_x, float)
    ry = np.asarray(receptors_y, float)
    n_receptors = rx.size
    indices     = np.arange(n_receptors)

    results = []

    for t, (_, row) in enumerate(met.iterrows()):
        date       = row["Date"]
        stab_class = row["Stab_Class"]

        if stab_class not in models:
            print(f"Warning: no model for stability class '{stab_class}', skipping {date}")
            continue

        wspd      = float(row["WSPD"])
        xr0, xr1 = x_range0, x_range1   # uniform range for all classes

        # Current-hour source parameters
        src      = sources[t]
        sx, sy   = src[:, 0], src[:, 1]
        strength = src[:, 2]
        road_angle = src[:, 3]  # road bearing in degrees (0-180)

        model_pos = models[stab_class]["pos"]
        model_neg = models[stab_class]["neg"]

        L_val  = float(row["L"])
        H_val  = float(row["H"])
        H_C    = float(row["MixHGT_C"])
        wind_deg = float(row["WDIR"])

        # Global rotation: align x-axis with mean wind direction
        theta  = np.deg2rad(270 - wind_deg)
        cos_t  = np.cos(theta)
        sin_t  = np.sin(theta)

        rx_rot =  rx * cos_t + ry * sin_t
        ry_rot = -rx * sin_t + ry * cos_t
        sx_rot =  sx * cos_t + sy * sin_t
        sy_rot = -sx * sin_t + sy * cos_t

        # Relative positions: receptor minus source (N_receptors x N_sources)
        x_hat = rx_rot[:, None] - sx_rot[None, :]
        y_hat = ry_rot[:, None] - sy_rot[None, :]

        total_conc = np.zeros(n_receptors, float)

        # Per-source relative wind direction (for road-angle-aware encoding)
        rel_wind_deg  = (wind_deg - road_angle) % 360
        wind_sin_src  = np.sin(np.deg2rad(rel_wind_deg))  # (N_sources,)
        wind_cos_src  = np.cos(np.deg2rad(rel_wind_deg))

        # --- Downwind region (positive x) ---
        mask_pos = (
            (x_hat >= xr0[0]) & (x_hat <= xr0[1]) &
            (y_hat >= y_range[0]) & (y_hat <= y_range[1])
        )
        r_idx, s_idx = np.where(mask_pos)

        if r_idx.size > 0:
            wind_sin_vals = wind_sin_src[s_idx]
            wind_cos_vals = wind_cos_src[s_idx]

            if stab_class in no_HC_classes:
                X = np.column_stack((
                    x_hat[mask_pos], y_hat[mask_pos],
                    wind_sin_vals, wind_cos_vals,
                    np.full(r_idx.size, H_val),
                    np.full(r_idx.size, L_val),
                    np.full(r_idx.size, wspd)
                ))
            else:
                X = np.column_stack((
                    x_hat[mask_pos], y_hat[mask_pos],
                    wind_sin_vals, wind_cos_vals,
                    np.full(r_idx.size, H_val),
                    np.full(r_idx.size, H_C),
                    np.full(r_idx.size, L_val),
                    np.full(r_idx.size, wspd)
                ))

            preds = np.zeros(r_idx.size)
            for start in range(0, r_idx.size, batch_size):
                preds[start:start+batch_size] = model_pos.predict(
                    X[start:start+batch_size])
            preds  = np.clip(preds, 0, None)
            contrib = preds * strength[s_idx] / 1e-6
            np.add.at(total_conc, r_idx, contrib)

        # --- Upwind region (negative x) ---
        mask_neg = (
            (x_hat >= xr1[0]) & (x_hat <= xr1[1]) &
            (y_hat >= y_range[0]) & (y_hat <= y_range[1])
        )
        r_idx, s_idx = np.where(mask_neg)

        if r_idx.size > 0:
            wind_sin_vals = wind_sin_src[s_idx]
            wind_cos_vals = wind_cos_src[s_idx]

            if stab_class in no_HC_classes:
                X = np.column_stack((
                    x_hat[mask_neg], y_hat[mask_neg],
                    wind_sin_vals, wind_cos_vals,
                    np.full(r_idx.size, H_val),
                    np.full(r_idx.size, L_val),
                    np.full(r_idx.size, wspd)
                ))
            else:
                X = np.column_stack((
                    x_hat[mask_neg], y_hat[mask_neg],
                    wind_sin_vals, wind_cos_vals,
                    np.full(r_idx.size, H_val),
                    np.full(r_idx.size, H_C),
                    np.full(r_idx.size, L_val),
                    np.full(r_idx.size, wspd)
                ))

            preds = np.zeros(r_idx.size)
            for start in range(0, r_idx.size, batch_size):
                preds[start:start+batch_size] = model_neg.predict(
                    X[start:start+batch_size])
            preds   = np.clip(preds, 0, None)
            contrib = preds * strength[s_idx] / 1e-6
            np.add.at(total_conc, r_idx, contrib)

        results.append(pd.DataFrame({
            "Date":       date,
            "Receptor_ID": indices,
            "Receptor_X":  np.round(rx, 1),
            "Receptor_Y":  np.round(ry, 1),
            "Conc":        total_conc
        }))

    return pd.concat(results, ignore_index=True)


# ============================================================
# Step 8: Prepare inputs and run inference
# ============================================================
print("Preparing source data...")
pollution_sources = [
    (x, y, strength, road_angle)
    for x, y, strength, road_angle in zip(
        midpoints_df["xm"], midpoints_df["ym"],
        midpoints_df["emission"], midpoints_df["angle_deg"]
    )
]
sources     = np.array(pollution_sources)
sources_re  = sources.reshape(len(met), len(base_midpoints_df), 4)   # adjust 2911 to actual segment count
print("Source data ready.")

x_rp = receptors_unique['x'].round(1).to_numpy()
y_rp = receptors_unique['y'].round(1).to_numpy()

models = {
    "VS": {"pos": verystable_x0,   "neg": verystable_x_1},
    "S":  {"pos": stable_x0,       "neg": stable_x_1},
    "N1": {"pos": neutral1_x0,     "neg": neutral1_x_1},
    "N2": {"pos": neutral2_x0,     "neg": neutral2_x_1},
    "U":  {"pos": unstable_x0,     "neg": unstable_x_1},
    "VU": {"pos": veryunstable_x0, "neg": veryunstable_x_1},
}

# Run full time-series inference
time_series_conc = predict_time_series_xgb(
    models=models,
    receptors_x=x_rp,
    receptors_y=y_rp,
    sources=sources_re,
    met=met,
    x_range0=(0, 1000.0),
    x_range1=(-100, 0.0),
    y_range=(-100.0, 100.0),
)
