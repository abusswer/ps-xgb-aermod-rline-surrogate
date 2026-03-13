# =============================================================================
# training.py  (original: "uint line sources training.py")
# XGBoost Surrogate Model Training for RLINE / AERMOD Unit Line Source
#
# Workflow:
#   1. Read AERMOD output (.txt) and meteorological (.sfc) files from
#      per-stability-class subfolders.
#   2. Match concentrations to meteorological conditions by timestamp.
#   3. Rotate (x, y) coordinates to wind-aligned frame.
#   4. Train an XGBoost regression model for the downwind region (x >= 0). 
#      (one sub-model as example)
#   5. Evaluate and save the model.
#
# Usage: set DATA_PATH and MODEL_SAVE_PATH at the top, then run.
# =============================================================================

# ---- User Configuration (edit these paths before running) ------------------
DATA_PATH       = r"YOUR_PATH\stable"          # root folder with subfolders
MODEL_SAVE_PATH = r"model_RLINE_remet_multidir_stable_2000_x0_M.json"
# ----------------------------------------------------------------------------

import numpy as np
import os
import re
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ============================================================
# Data I/O helpers
# ============================================================

def read_aermod_data_numpy(filename):
    """
    Efficiently read an AERMOD PLOT output file using numpy.

    Returns a structured numpy array with fields:
        X (float), Y (float), AVERAGE_CONC (float), DATE (int)
    """
    data = np.loadtxt(
        filename,
        comments='*',          # skip comment lines
        usecols=(0, 1, 2, 8),  # X, Y, AVERAGE CONC, DATE columns
        dtype={
            'names':   ['X', 'Y', 'AVERAGE_CONC', 'DATE'],
            'formats': [float, float, float, int]
        }
    )
    return data


def read_met_data_numpy(filename):
    """
    Read a meteorological surface (.sfc) file using numpy.

    Selected columns (0-indexed): [0,1,2,4,5,9,11,15,16]
        0-3  -> Year, Month, Day, Hour  (used to build DATE key)
        4    -> H (sensible heat flux)
        5    -> MixHGT_C (mixing height)
        9    -> L (Obukhov length)
        11   -> WSPD (wind speed)
        15   -> WDIR (wind direction)

    Returns:
        2-D numpy array, each row = one hour, last column = DATE (int)
    """
    selected_cols = [0, 1, 2, 4, 5, 9, 11, 15, 16]
    data = np.loadtxt(filename, skiprows=1, usecols=selected_cols)

    years  = data[:, 0].astype(int) % 100
    months = data[:, 1].astype(int)
    days   = data[:, 2].astype(int)
    hours  = data[:, 3].astype(int)

    datetime_col = (years * 1000000 + months * 10000 +
                    days   * 100    + hours).astype(int)

    data_selected = data[:, 4:]                              # drop raw date cols
    data_selected = np.column_stack((data_selected, datetime_col))
    return data_selected


def read_all_data(base_path, max_folders=None):
    """
    Read all subfolders under base_path.
    Each subfolder must contain exactly one .txt and one .sfc file.
    Meteorological rows are matched to concentration rows by DATE.

    Args:
        base_path:   root directory containing stability-class subfolders
        max_folders: limit on number of subfolders to read (None = all)

    Returns:
        all_txt (structured ndarray), all_sfc (2-D ndarray)
    """
    all_txt, all_sfc = [], []
    folder_count = 0

    for subfolder in os.listdir(base_path):
        subpath = os.path.join(base_path, subfolder)
        if not os.path.isdir(subpath):
            continue

        print(f"Reading subfolder: {subfolder}")

        txt_files = [f for f in os.listdir(subpath) if f.endswith(".txt")]
        sfc_files = [f for f in os.listdir(subpath) if f.endswith(".sfc")]

        if len(txt_files) != 1 or len(sfc_files) != 1:
            raise ValueError(
                f"Subfolder '{subfolder}' must contain exactly 1 .txt and 1 .sfc file."
            )

        txt_data = read_aermod_data_numpy(os.path.join(subpath, txt_files[0]))
        sfc_data = read_met_data_numpy(os.path.join(subpath, sfc_files[0]))

        # Build date-keyed dict for fast lookup
        sfc_date_dict = {int(row[-1]): row for row in sfc_data}
        matched_sfc   = np.array([sfc_date_dict[d] for d in txt_data['DATE']])

        all_txt.append(txt_data)
        all_sfc.append(matched_sfc)

        folder_count += 1
        print(f"Done: {folder_count} subfolder(s)")

        if max_folders is not None and folder_count >= max_folders:
            print(f"Reached max_folders={max_folders}, stopping.")
            break

    all_txt = np.concatenate(all_txt, axis=0) if all_txt else None
    all_sfc = np.vstack(all_sfc)              if all_sfc else None

    print(f"\nTotal subfolders read: {folder_count}")
    return all_txt, all_sfc


# ============================================================
# Load data
# ============================================================
all_txt, all_sfc = read_all_data(DATA_PATH)

y = all_txt['AVERAGE_CONC']
X = np.column_stack((
    all_txt['X'],        # downwind distance x
    all_txt['Y'],        # crosswind distance y
    all_sfc[:, 0:5]     # selected met features: H, MixHGT_C, L, WSPD, WDIR
))


# ============================================================
# Coordinate rotation to wind-aligned frame
# ============================================================

def convert_to_rotated_xy(input_data):
    """
    Rotate receptor (x, y) into the wind-aligned coordinate frame.

    Wind direction convention:
        0   deg -> wind blows toward -y axis
        90  deg -> wind blows toward -x axis
        180 deg -> wind blows toward +y axis
        270 deg -> wind blows toward +x axis

    Rotation formula:
        theta  = 270 - wind_dir   (converts to standard math angle)
        x_rot  =  x * cos(theta) + y * sin(theta)
        y_rot  = -x * sin(theta) + y * cos(theta)

    Input columns:  [x, y, H, MixHGT_C, L, WSPD, wind_dir]
    Output columns: [x_rot, y_rot, sin(wind_dir), cos(wind_dir),
                     H, L, WSPD]          <- for stable / no-HC models
    """
    x      = input_data[:, 0]
    y      = input_data[:, 1]
    angles = input_data[:, -1]  # wind direction in last column

    sin_angles = np.sin(np.radians(angles))
    cos_angles = np.cos(np.radians(angles))

    theta  = np.radians(270 - angles)
    x_rot  =  x * np.cos(theta) + y * np.sin(theta)
    y_rot  = -x * np.sin(theta) + y * np.cos(theta)

    # Features: x_rot, y_rot, sin, cos, H, L, WSPD
    rotated_data = np.column_stack((
        x_rot, y_rot,
        sin_angles, cos_angles,
        input_data[:, 2],        # H
        input_data[:, 4:6]       # L, WSPD
    ))
    return rotated_data


X_new = convert_to_rotated_xy(X)

# Keep only downwind region (x_rot >= 0) for the positive-x model
idx       = np.nonzero(X_new[:, 0] >= 0)[0]
X_filtered = X_new[idx]
y_filtered = y[idx]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_filtered, test_size=0.2, random_state=42
)
print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")


# ============================================================
# XGBoost model training
# ============================================================
xgb_params = {
    'n_estimators':        1000,
    'learning_rate':       0.15,
    'max_depth':           5,
    'subsample':           0.8,
    'colsample_bytree':    0.8,
    'random_state':        42,
    'early_stopping_rounds': 100,
    'eval_metric':         'rmse'
}

print("\n=== Training: multi-direction stable model (x >= 0) ===")
model_RLINE = xgb.XGBRegressor(**xgb_params)
model_RLINE.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)


# ============================================================
# Model evaluation
# ============================================================

def evaluate_model(model, X_test, y_test, label):
    """Compute and print RMSE, MAE, and R² for a fitted model."""
    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    mae    = mean_absolute_error(y_test, y_pred)
    print(f"{label} - RMSE: {rmse:.6f}, MAE: {mae:.6f}, R2: {r2:.4f}")
    return y_pred, rmse, r2


print("\n=== Model Evaluation ===")
y_pred, rmse, r2 = evaluate_model(model_RLINE, X_test, y_test,
                                   "Stable multi-direction (x>=0)")


# ============================================================
# Feature importance plot
# ============================================================
feature_names = ['x_rot', 'y_rot', 'wind_sin', 'wind_cos', 'H', 'L', 'WSPD']

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
importance_ = model_RLINE.feature_importances_
plt.barh(range(len(importance_)), importance_)
plt.yticks(range(len(feature_names)), feature_names)
plt.title('Stable multi-direction model - Feature Importance')
plt.xlabel('Importance score')
plt.tight_layout()
plt.show()


# ============================================================
# Save model
# ============================================================
model_RLINE.save_model(MODEL_SAVE_PATH)
print(f"\nModel saved to: {MODEL_SAVE_PATH}")


# ============================================================
# Scatter plot: predicted vs. actual
# ============================================================
plt.figure(figsize=(10, 8))

plt.scatter(y_test, y_pred, alpha=0.6, s=2, c='blue', edgecolors='none')

min_val = min(np.min(y_test), np.min(y_pred))
max_val = max(np.max(y_test), np.max(y_pred))
x_line  = np.linspace(min_val, max_val, 500)

plt.plot(x_line, x_line, 'r--', linewidth=2, label='Ideal fit (y = x)')

offset = 0.1
plt.plot(x_line, x_line + offset, color='orange', linestyle='--', linewidth=1)
plt.plot(x_line, x_line - offset, color='orange', linestyle='--', linewidth=1)
plt.fill_between(x_line, x_line - offset, x_line + offset,
                 color='orange', alpha=0.15, label='+/-0.1 error band')

plt.text(0.95, 0.05, f'R2 = {r2:.4f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.title('Stable Model: Actual vs. Predicted', fontsize=16)
plt.xlabel('Actual', fontsize=12)
plt.ylabel('Predicted', fontsize=12)
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Residual plot
# ============================================================
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.6, s=2, c='green', edgecolors='none')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.title('Residual Plot: Predicted vs. Residual', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Residual (actual - predicted)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
