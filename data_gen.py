# =============================================================================
# data_gen.py
# AERMOD / RLINE Input File Generator
#
# Generates AERMOD input files for each wind direction by:
#   1. Defining receptor grids along and around a unit line source
#   2. Rotating receptor coordinates to match meteorological wind direction
#   3. Writing .inp, .sfc, and .pfl files into per-wind-direction subfolders
#   4. Copying aermod.exe and batch-running simulations
#
# Usage: configure the path variables at the top of this file, then run.
# =============================================================================

# ---- User Configuration (edit these paths before running) ------------------
SFC_FILE  = r"YOUR_PATH\verystable_representative.sfc"
PFL_FILE  = r"YOUR_PATH\verystable_representative.pfl"
OUTPUT_BASE = r"YOUR_OUTPUT_BASE_PATH\verystable"
AERMOD_EXE  = r"YOUR_PATH\aermod.exe"
# ----------------------------------------------------------------------------

import time
import numpy as np
from shapely.geometry import box
from shapely.affinity import translate
from pyproj import Transformer
import os
from shapely import wkt
import shutil
import pandas as pd
import geopandas as gpd
import math
import subprocess

# ============================================================
# AERMOD Input File Template
# ============================================================
TEMPLATE = """CO STARTING
   TITLEONE  An Example Transportation Project
   MODELOPT  BETA ALPHA FLAT CONC 
   RUNORNOT  RUN
   AVERTIME  {AVERTIME}
   URBANOPT  {URBANOPT}
   FLAGPOLE  {FLAGPOLE}
   POLLUTID  {POLLUTID}
CO FINISHED

** AREAPOLY Source        xini yini (z)
** RLINE Source         x1 y1 x2 y2 z
** RLINEXT Source      x1 y1 z1 x2 y2 z2
** VOLUME Source      x y (z)
SO STARTING
{LINK_LOCATION}

** AREAPOLY Source        g/s/m2  height  VertixNO  Szinit
** RLINE Source         g/s/m2  height  width  Szinit
** RLINEXT Source      g/s/m  height  width  Szinit
** VOLUME Source      g/s  height  radii  Szinit
** Parameters:        ------  ------  -------   -------   -----  ------
{SRCPARAM}

** HourEmission:
{HROFDY}

** For RLINEXT RBARRIER (optional)
{RBARRIER}

** For AREAPOLY Source Only      x1 y1 x2 y2 x3 y3 ...
{LINKCOORD}

SO URBANSRC ALL
SO SRCGROUP ALL
SO FINISHED

RE STARTING
{RECEPTORCOORD}
RE FINISHED

ME STARTING
   SURFFILE  {file_sfc}
   PROFFILE  {file_pfl}
   SURFDATA     58367   2023
   UAIRDATA     00058367   2023
   PROFBASE     3.048   METERS
ME FINISHED

OU STARTING
   RECTABLE  ALLAVE  FIRST
   POSTFILE 1 ALL PLOT  hourly_nox_post1.txt
   SUMMFILE  AERTEST.SUM
OU FINISHED
"""

# ============================================================
# Receptor Grid Definition
# Symmetric x-range: negative (upwind) and positive (downwind)
# Each x-position has multiple y-segments with varying spacing
# (finer spacing near the source center, coarser farther away)
# ============================================================

z_value = 1.5  # Fixed receptor height (m)

# Dict: x_position -> list of (y_min, y_max, y_spacing) segments
x_y_segments = {
    # --- Negative x (upwind) ---
    -0.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -1:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -1.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -2:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -2.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -3:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -3.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -4:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -4.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -5:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -5.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -6:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -6.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -7:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -7.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -8:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -8.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -9:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -9.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -10:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2), (-10, 10, 1), (10, 20, 2), (20, 50, 5), (50, 100, 10)],
    -11:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 1), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    -12:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 1), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    -13:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 1), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    -14:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 1), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    -15:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 1), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    -17.5: [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 1), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    -20:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 1), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    -22.5: [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 1), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    -25:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    -30:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    -35:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    -40:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    -50:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    -60:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    -70:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    -80:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    -90:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    -100:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],

    # --- Positive x (downwind) ---
    0:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    0.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    1:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    1.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    2:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    2.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    3:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    3.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    4:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    4.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    5:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    5.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    6:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    6.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    7:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    7.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    8:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    8.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    9:    [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    9.5:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    10:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 1), (-10, 10, 0.5), (10, 20, 1), (20, 50, 5), (50, 100, 10)],
    11:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 1), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    12:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 1), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    13:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 1), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    14:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 1), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    15:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 1), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    17.5: [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 1), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    20:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 1), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    22.5: [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 1), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    25:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    30:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    35:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    40:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    50:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    60:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    70:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    80:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    90:   [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    100:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    120:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    140:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    160:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    180:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    200:  [(-100, -50, 10), (-50, -20, 5), (-20, -10, 2.5), (-10, 10, 2), (10, 20, 2.5), (20, 50, 5), (50, 100, 10)],
    250:  [(-100, -50, 10), (-50, -20, 10), (-20, -10, 5), (-10, 10, 4), (10, 20, 5), (20, 50, 10), (50, 100, 10)],
    300:  [(-100, -50, 10), (-50, -20, 10), (-20, -10, 5), (-10, 10, 4), (10, 20, 5), (20, 50, 10), (50, 100, 10)],
    350:  [(-100, -50, 10), (-50, -20, 10), (-20, -10, 5), (-10, 10, 4), (10, 20, 5), (20, 50, 10), (50, 100, 10)],
    400:  [(-100, -50, 10), (-50, -20, 10), (-20, -10, 5), (-10, 10, 4), (10, 20, 5), (20, 50, 10), (50, 100, 10)],
    450:  [(-100, -50, 10), (-50, -20, 10), (-20, -10, 5), (-10, 10, 4), (10, 20, 5), (20, 50, 10), (50, 100, 10)],
    500:  [(-100, -50, 10), (-50, -20, 10), (-20, -10, 5), (-10, 10, 4), (10, 20, 5), (20, 50, 10), (50, 100, 10)],
    600:  [(-100, -50, 10), (-50, -20, 10), (-20, -10, 5), (-10, 10, 4), (10, 20, 5), (20, 50, 10), (50, 100, 10)],
    700:  [(-100, -50, 10), (-50, -20, 10), (-20, -10, 5), (-10, 10, 4), (10, 20, 5), (20, 50, 10), (50, 100, 10)],
    800:  [(-100, -50, 10), (-50, -20, 10), (-20, -10, 5), (-10, 10, 4), (10, 20, 5), (20, 50, 10), (50, 100, 10)],
    900:  [(-100, -50, 10), (-50, -20, 10), (-20, -10, 5), (-10, 10, 4), (10, 20, 5), (20, 50, 10), (50, 100, 10)],
    1000: [(-100, -50, 10), (-50, -20, 10), (-20, -10, 5), (-10, 10, 4), (10, 20, 5), (20, 50, 10), (50, 100, 10)],
    1200: [(-100, -50, 10), (-50, -20, 10), (-20, -10, 5), (-10, 10, 4), (10, 20, 5), (20, 50, 10), (50, 100, 10)],
    1400: [(-100, -50, 10), (-50, -20, 10), (-20, -10, 5), (-10, 10, 4), (10, 20, 5), (20, 50, 10), (50, 100, 10)],
    1600: [(-100, -50, 10), (-50, -20, 10), (-20, -10, 5), (-10, 10, 4), (10, 20, 5), (20, 50, 10), (50, 100, 10)],
    1800: [(-100, -50, 10), (-50, -20, 10), (-20, -10, 5), (-10, 10, 4), (10, 20, 5), (20, 50, 10), (50, 100, 10)],
    2000: [(-100, -50, 10), (-50, -20, 10), (-20, -10, 5), (-10, 10, 4), (10, 20, 5), (20, 50, 10), (50, 100, 10)],
}

# ============================================================
# Generate receptor point strings
# ============================================================
points = []
for x, y_segments in x_y_segments.items():
    for (y_min, y_max, y_spacing) in y_segments:
        y = y_min
        while y <= y_max + 1e-6:  # small tolerance to avoid float precision issues
            points.append(f"RE DISCCART {x:.1f} {y:.1f} {z_value}")
            y += y_spacing

RLINE_RECEPTORCOORD = "\n".join(points)

print(f"Total receptor points generated: {len(points)}")
print(f"x-range: [{min(x_y_segments.keys())}, {max(x_y_segments.keys())}]")


# ============================================================
# Deduplicate receptor points by (x, y, z) coordinate
# ============================================================
def deduplicate_points(points):
    """
    Remove duplicate receptor points by their (x, y, z) coordinate.
    Preserves the original order.

    Args:
        points: list of strings in 'RE DISCCART x y z' format

    Returns:
        list of unique point strings
    """
    seen = set()
    unique_points = []
    for p in points:
        _, _, x, y, z = p.split()
        key = (float(x), float(y), float(z))
        if key not in seen:
            seen.add(key)
            unique_points.append(p)
    return unique_points


points_dedup = deduplicate_points(points)
print(f"Before dedup: {len(points)}, after dedup: {len(points_dedup)}")


# ============================================================
# Visualize receptor layout (2D scatter)
# ============================================================
import matplotlib.pyplot as plt

def plot_receptors(points):
    """
    Visualize receptor positions as a 2D scatter plot.

    Args:
        points: list of strings in 'RE DISCCART x y z' format
    """
    xs, ys = [], []
    for p in points:
        _, _, x, y, z = p.split()
        xs.append(float(x))
        ys.append(float(y))

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, c=xs, cmap="tab10", s=30, alpha=0.8, edgecolors="k")
    plt.xlabel("X coordinate", fontsize=12)
    plt.ylabel("Y coordinate", fontsize=12)
    plt.title("Receptor Distribution", fontsize=16)
    plt.grid(alpha=0.3)
    plt.show()


plot_receptors(points_dedup)


# ============================================================
# Rotate receptor coordinates by wind direction angle
# ============================================================
def rotate_discart(points, angle_deg):
    """
    Rotate 'RE DISCCART x y z' receptor points by angle_deg degrees
    (counter-clockwise positive).

    Args:
        points: list of 'RE DISCCART x y z' strings
        angle_deg: rotation angle in degrees (CCW positive)

    Returns:
        list of rotated point strings
    """
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    rotated_points = []
    for line in points:
        parts = line.split()
        if len(parts) != 5:
            continue  # skip malformed lines
        _, tag, x_str, y_str, z_str = parts
        x, y, z = float(x_str), float(y_str), float(z_str)
        x_new = x * cos_t - y * sin_t
        y_new = x * sin_t + y * cos_t
        rotated_points.append(f"RE {tag} {x_new:.1f} {y_new:.1f} {z:.1f}")

    return rotated_points


rotated_points = rotate_discart(points_dedup, 270 - 0)
plot_receptors(rotated_points)


# ============================================================
# Filter out receptor points located on the line source body
# ============================================================
def filter_points(rotated_points, L, W):
    """
    Remove receptors inside the exclusion zone of the line source.

    Exclusion zone:
        X in [-(W+6)/2, (W+6)/2]
        Y in [-L/2, L/2]

    Args:
        rotated_points: list of 'RE DISCCART x y z' strings
        L: line source length (m)
        W: line source width (m)

    Returns:
        list of filtered point strings
    """
    exclude_y_min = -L / 2
    exclude_y_max =  L / 2
    exclude_x_min = -(W + 6) / 2
    exclude_x_max =  (W + 6) / 2

    filtered_points = []
    for line in rotated_points:
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            x = float(parts[-3])
            y = float(parts[-2])
        except ValueError:
            continue
        if (exclude_x_min <= x <= exclude_x_max) and (exclude_y_min <= y <= exclude_y_max):
            continue  # inside exclusion zone -> skip
        filtered_points.append(line)

    return filtered_points


L = 10  # line source length (m)
W = 7   # line source width (m)
points_filtered = filter_points(rotated_points, L, W)
plot_receptors(points_filtered)


# ============================================================
# Read meteorological surface (.sfc) file
# ============================================================
col_names = [
    "Year", "Month", "Day", "JulianDay", "Hour",
    "H", "USTAR", "WSTAR", "ThetaGrad", "MixHGT_C", "MixHGT_M", "L",
    "Z0", "B0", "Albedo",
    "WSPD", "WDIR", "AnemoHt", "Temp", "MeasHt",
    "PrecipType", "PrecipAmt", "RH", "Pressure", "CloudCover",
    "WindFlag", "CloudTempFlag"
]
col_names_p = [
    "Year_p", "Month_p", "Day_p", "Hour_p", "MeasHt_p", "Top flag_p",
    "WDIR_p", "WSPD_p", "Temp_p",
    "Sigma_theta", "Sigma_w", "sources"
]

df1    = pd.read_csv(SFC_FILE, delim_whitespace=True,
                     names=col_names, skiprows=1, comment="#")
df_all = pd.concat([df1], ignore_index=True)

dfp1     = pd.read_csv(PFL_FILE, delim_whitespace=True,
                       names=col_names_p, comment="#")
df_all_p = pd.concat([dfp1], ignore_index=True)
df_all_p = df_all_p.iloc[:, :-1]  # drop last 'sources' column

# ============================================================
# Simulation parameters
# ============================================================
AVERTIME          = '1'
URBANOPT          = '200000'
FLAGPOLE          = "1.5"
POLLUTID          = "NOx"
RLINE_RBARRIER    = ''
RLINE_LINKCOORD   = "** no nodes coord needed"
RLINE_hrofay      = ''
RLINE_LINKLOC     = 'SO LOCATION 1__0 RLINE 0 -5 0 5 0'
RLINE_SRCPARAMEM  = 'SO SRCPARAM 1__0 1.0e-06 1.3 7 2'
header_line       = ("   32.031N    96.399          UA_ID: 13957     SF_ID: 53912"
                     "     OS_ID:              VERSION: 22112"
                     " THRESH_1MIN =  0.50 m/s; ADJ_U*  CCVR_Sub TEMP_Sub")

# ============================================================
# Main loop: generate one input folder per wind direction
# ============================================================
for WDIR in np.append(np.arange(0, 90, 4), 90):
    rotated_points  = rotate_discart(points_dedup, 270 - WDIR)
    points_filtered = filter_points(rotated_points, L=10 + 2, W=7)
    RLINE_RECEPTORCOORD = "\n".join(points_filtered)

    content = TEMPLATE.format(
        AVERTIME=AVERTIME, URBANOPT=URBANOPT, FLAGPOLE=FLAGPOLE,
        POLLUTID=POLLUTID, LINK_LOCATION=RLINE_LINKLOC,
        SRCPARAM=RLINE_SRCPARAMEM, HROFDY=RLINE_hrofay,
        RBARRIER=RLINE_RBARRIER, LINKCOORD=RLINE_LINKCOORD,
        RECEPTORCOORD=RLINE_RECEPTORCOORD,
        file_sfc='wr_dynamic_x.SFC', file_pfl='wr_dynamic_x.PFL'
    )

    subfolder   = f"WDIR_{WDIR:.1f}"
    output_path = os.path.join(OUTPUT_BASE, subfolder)
    os.makedirs(output_path, exist_ok=True)

    inp_path = os.path.join(output_path, "aermod.inp")
    with open(inp_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"WDIR={WDIR:.1f} -> saved: {inp_path}")

    # Override wind direction in met data for this run
    df_all.loc[:, "WDIR"]   = WDIR
    df_all_p.loc[:, "WDIR_p"] = WDIR

    path_sfc = os.path.join(output_path, "wr_dynamic_x.sfc")
    path_pfl = os.path.join(output_path, "wr_dynamic_x.pfl")

    with open(path_sfc, 'w', encoding='utf-8') as f:
        f.write(header_line + '\n')
        f.write(df_all.to_string(index=False, header=False))
    print(f"Saved: {path_sfc}")

    with open(path_pfl, 'w', encoding='utf-8') as f:
        f.write(df_all_p.to_string(index=False, header=False))
    print(f"Saved: {path_pfl}")


# ============================================================
# Copy aermod.exe to each wind-direction subfolder
# ============================================================
for root, dirs, files in os.walk(OUTPUT_BASE):
    for folder in dirs:
        target_dir  = os.path.join(root, folder)
        target_path = os.path.join(target_dir, os.path.basename(AERMOD_EXE))
        try:
            shutil.copy2(AERMOD_EXE, target_path)
            print(f"Copied exe to: {target_path}")
        except Exception as e:
            print(f"Copy failed: {target_path}, reason: {e}")
    break  # only process first-level subfolders

print("\nAll copy tasks completed.")


# ============================================================
# Batch-run aermod.exe in all subfolders (parallel mode)
# ============================================================
WAIT_FOR_FINISH = False  # True = sequential, False = parallel
exe_name = "aermod.exe"

for root, dirs, files in os.walk(OUTPUT_BASE):
    for folder in dirs:
        exe_path = os.path.join(root, folder, exe_name)
        print(f"Checking: {exe_path}")
        if os.path.exists(exe_path):
            print(f"Running: {exe_path}")
            try:
                if WAIT_FOR_FINISH:
                    subprocess.run(exe_path, cwd=os.path.dirname(exe_path),
                                   shell=True, check=True)
                else:
                    subprocess.Popen(exe_path, cwd=os.path.dirname(exe_path),
                                     shell=True)
            except Exception as e:
                print(f"Run failed: {e}")
        else:
            print(f"Not found: {exe_path}")

print("\nAll simulation jobs dispatched.")
