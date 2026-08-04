"""Live surface winds over the continental US -- a real forecast, animated.

This pulls the *actual* hourly 10 m wind forecast (speed + direction) on a grid
of points spanning the lower 48 from the free Open-Meteo API, converts each
(speed, direction) into an eastward/northward vector, and draws it as an *animated*
quiver field -- one keyframe per forecast hour, so every arrow swings to follow
the local wind and recolors with its speed as the day unfolds.

Then it releases a cloud of massless "air parcels" and advects them through the
**time-varying** forecast field (bilinear in space, linear in time between
hourly steps), so the dots stream across the country exactly the way tomorrow's
weather will push the air. Parcels are seeded from a reservoir wider than the
map and simply blow off the edge when they reach it (no teleporting back), while
fresh ones keep drifting in -- the moving parts inside a real weather map.

Data: Open-Meteo forecast API (no key required), 10 m winds on a lon/lat grid.
    Cached to examples/data/us_surface_wind.csv on first run.
"""

RENDERING_TIME = "VERY_LONG"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
import json
import urllib.request
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import numpy as np
import pandas as pd

THEME = "Arctic Signal"
DATA = "examples/data/us_surface_wind.csv"

# Grid over the continental US (degrees).
LON = np.linspace(-123.0, -69.0, 15)
LAT = np.linspace(26.0, 48.0, 9)
FORECAST_DAYS = 2


def fetch_wind_grid():
    """Download the hourly wind forecast for every grid point and cache a CSV.

    Long-format columns: hour, time, lat, lon, speed_kmh, dir_deg
    (dir is the meteorological 'coming from' bearing, 0=N, 90=E).
    """
    LO, LA = np.meshgrid(LON, LAT)
    lat_list = ",".join(f"{v:.4f}" for v in LA.ravel())
    lon_list = ",".join(f"{v:.4f}" for v in LO.ravel())
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat_list}&longitude={lon_list}"
        "&hourly=wind_speed_10m,wind_direction_10m"
        f"&forecast_days={FORECAST_DAYS}&timezone=GMT"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = json.loads(resp.read().decode())

    rows = []
    for pt in payload:
        lat, lon = pt["latitude"], pt["longitude"]
        h = pt["hourly"]
        for k, t in enumerate(h["time"]):
            rows.append((k, t, lat, lon,
                         h["wind_speed_10m"][k], h["wind_direction_10m"][k]))
    df = pd.DataFrame(rows, columns=["hour", "time", "lat", "lon",
                                     "speed_kmh", "dir_deg"])
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    df.to_csv(DATA, index=False)
    return df


def load_field():
    """Return (lons, lats, U, V) where U,V are (n_hours, ny, nx) in km/h.

    Snaps the API's nearest-grid-point coordinates back onto our requested
    LON/LAT axes so the data lands on a clean regular grid.
    """
    df = pd.read_csv(DATA) if os.path.exists(DATA) else fetch_wind_grid()
    df["lon"] = LON[np.abs(df["lon"].to_numpy()[:, None] - LON).argmin(1)]
    df["lat"] = LAT[np.abs(df["lat"].to_numpy()[:, None] - LAT).argmin(1)]

    n_hours = df["hour"].max() + 1
    nx, ny = len(LON), len(LAT)
    U = np.zeros((n_hours, ny, nx))
    V = np.zeros((n_hours, ny, nx))
    ix = {v: i for i, v in enumerate(LON)}
    iy = {v: j for j, v in enumerate(LAT)}
    for _, r in df.iterrows():
        d = np.deg2rad(r["dir_deg"])
        s = r["speed_kmh"]
        # Meteorological 'from' bearing -> vector the wind blows toward.
        u = -s * np.sin(d)
        v = -s * np.cos(d)
        U[int(r["hour"]), iy[r["lat"]], ix[r["lon"]]] = u
        V[int(r["hour"]), iy[r["lat"]], ix[r["lon"]]] = v
    return LON, LAT, U, V


def sample(field, lons, lats, tf, x, y):
    """Bilinear-in-space, linear-in-time sample of a (nt, ny, nx) field."""
    nt = field.shape[0]
    it = int(np.clip(np.floor(tf), 0, nt - 2))
    ft = tf - it
    nx, ny = len(lons), len(lats)
    ix = np.clip(np.searchsorted(lons, x) - 1, 0, nx - 2)
    iy = np.clip(np.searchsorted(lats, y) - 1, 0, ny - 2)
    # Clamp weights to the cell so points outside the grid take the (bounded)
    # edge wind instead of a runaway linear extrapolation.
    tx = np.clip((x - lons[ix]) / (lons[ix + 1] - lons[ix]), 0.0, 1.0)
    ty = np.clip((y - lats[iy]) / (lats[iy + 1] - lats[iy]), 0.0, 1.0)

    def bil(F):
        return ((F[iy, ix] * (1 - tx) + F[iy, ix + 1] * tx) * (1 - ty) +
                (F[iy + 1, ix] * (1 - tx) + F[iy + 1, ix + 1] * tx) * ty)

    return bil(field[it]) * (1 - ft) + bil(field[it + 1]) * ft


lons, lats, U, V = load_field()
n_hours = U.shape[0]
LON_G, LAT_G = np.meshgrid(lons, lats)

# --- Advect air parcels through the evolving forecast field ---
rng = np.random.default_rng(11)
n_parcels = 200
n_frames = 150
# Map animation frames across the whole forecast window.
tf_of_frame = np.linspace(0, n_hours - 1, n_frames)
# Degrees moved per km (rough mid-latitude conversion) times an exaggeration so
# the drift reads clearly on screen.
DEG_PER_KM_LON = 1.0 / 88.0
DEG_PER_KM_LAT = 1.0 / 111.0
GAIN = 2.6
dt = (FORECAST_DAYS * 24.0) / n_frames   # hours per frame

# Seed from a reservoir wider than the map so parcels keep drifting into view as
# others blow off the edge -- a parcel that leaves the grid is gone for good (no
# teleporting back, which would streak across the comp between keyframes).
pad_x = 0.4 * (lons[-1] - lons[0])
pad_y = 0.4 * (lats[-1] - lats[0])
px = rng.uniform(lons[0] - pad_x, lons[-1] + pad_x, n_parcels)
py = rng.uniform(lats[0] - pad_y, lats[-1] + pad_y, n_parcels)

x_frames = np.zeros((n_frames, n_parcels))
y_frames = np.zeros((n_frames, n_parcels))
spd_frames = np.zeros((n_frames, n_parcels))
for f in range(n_frames):
    x_frames[f], y_frames[f] = px, py
    u = sample(U, lons, lats, tf_of_frame[f], px, py)
    v = sample(V, lons, lats, tf_of_frame[f], px, py)
    spd_frames[f] = np.hypot(u, v)
    px = px + GAIN * dt * u * DEG_PER_KM_LON
    py = py + GAIN * dt * v * DEG_PER_KM_LAT

wind = (
    AEGraph(theme=THEME, comp_name="Live Surface Winds", fps=60,
            drop_shadow=True, cinematic_effects=True,
            xaxis_location="bottom", yaxis_location="left", plot_frame=True)
    .set_title("Surface Winds over the US")
    .set_subtitle("a real Open-Meteo forecast: air parcels drift with the wind")
    .set_xlabel("Longitude")
    .set_ylabel("Latitude")
    .grid()
)

# The field is dynamic, so animate the arrows themselves: each one swings to
# follow the local wind and recolors with its speed, hour by hour. One keyframe
# per forecast hour, spread across the same time span as the drifting parcels.
U_arrows = U.reshape(n_hours, -1)
V_arrows = V.reshape(n_hours, -1)
spd_arrows = np.hypot(U_arrows, V_arrows)
arrow_times = np.linspace(0.0, (n_frames - 1) * 0.06, n_hours)

wind.quiver_evolving(
    LON_G, LAT_G, U_arrows, V_arrows,
    scale=0.024, scale_mode="comp", normalize=True,         # ~4% of plot width
    c_frames=spd_arrows, gradient=([70, 120, 175], [240, 250, 170]),  # calm -> windy
    width=3.2, headwidth=3.2, headlength=11.0,
    alpha=0.55, frame_times=arrow_times.tolist(),
    hold_keyframes=False, fade_in=1.0,
)

wind.scatter_evolving(
    x_frames, y_frames,
    c_frames=spd_frames,
    gradient=([130, 215, 255], [255, 240, 150]),
    radius=5,
    frame_duration=0.06,
    hold_keyframes=False,
    alpha=0.95,
    fade_in=0.4,
    drop_shadow=True,
)

wind.set_xlim(lons[0] - 1, lons[-1] + 1).set_ylim(lats[0] - 1, lats[-1] + 1)
wind.set_xticks().set_yticks()

wind.render()
