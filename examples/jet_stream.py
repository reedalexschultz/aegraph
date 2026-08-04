"""The jet stream: 250 hPa winds ripping west-to-east across North America.

Same idea as the surface-wind map, but up at ~10 km altitude (the 250 hPa
pressure level) where the polar jet lives. The quiver shows the high-altitude
wind field; the speeds up here routinely top 150-300 km/h, so the color scale
runs hot. A swarm of tracers gets dropped into the field and is carried along by
the **time-varying** forecast, ribboning eastward and bunching up wherever the
jet core accelerates -- the river of air that steers every mid-latitude storm.

Data: Open-Meteo forecast API (no key), wind_speed/direction_250hPa on a grid.
    Cached to examples/data/jet_stream_250hpa.csv on first run.
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

THEME = "Tidal Glass"
DATA = "examples/data/jet_stream_250hpa.csv"

# Broad North-American window so the whole jet meander is visible.
LON = np.linspace(-140.0, -55.0, 16)
LAT = np.linspace(25.0, 60.0, 9)
FORECAST_DAYS = 2


def fetch_wind_grid():
    LO, LA = np.meshgrid(LON, LAT)
    lat_list = ",".join(f"{v:.4f}" for v in LA.ravel())
    lon_list = ",".join(f"{v:.4f}" for v in LO.ravel())
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat_list}&longitude={lon_list}"
        "&hourly=wind_speed_250hPa,wind_direction_250hPa"
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
                         h["wind_speed_250hPa"][k], h["wind_direction_250hPa"][k]))
    df = pd.DataFrame(rows, columns=["hour", "time", "lat", "lon",
                                     "speed_kmh", "dir_deg"])
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    df.to_csv(DATA, index=False)
    return df


def load_field():
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
        U[int(r["hour"]), iy[r["lat"]], ix[r["lon"]]] = -s * np.sin(d)
        V[int(r["hour"]), iy[r["lat"]], ix[r["lon"]]] = -s * np.cos(d)
    return LON, LAT, U, V


def sample(field, lons, lats, tf, x, y):
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

# --- Advect tracers through the evolving jet ---
rng = np.random.default_rng(5)
n_parcels = 440
n_frames = 150
tf_of_frame = np.linspace(0, n_hours - 1, n_frames)
DEG_PER_KM_LON = 1.0 / 78.0     # ~cos(43 deg) * 111 km per deg lon
DEG_PER_KM_LAT = 1.0 / 111.0
GAIN = 1.1
dt = (FORECAST_DAYS * 24.0) / n_frames

# The jet blows west -> east, so stock a deep reservoir to the west and let it
# drain across the map. Parcels that exit the east edge are gone for good (no
# teleport-back, which would streak between keyframes).
span = lons[-1] - lons[0]
px = rng.uniform(lons[0] - 0.7 * span, lons[-1] + 0.05 * span, n_parcels)
py = rng.uniform(lats[0] - 0.15 * (lats[-1] - lats[0]),
                 lats[-1] + 0.15 * (lats[-1] - lats[0]), n_parcels)

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

jet = (
    AEGraph(theme=THEME, comp_name="The Jet Stream", fps=60,
            drop_shadow=True, cinematic_effects=True,
            xaxis_location="bottom", yaxis_location="left", plot_frame=True)
    .set_title("The Jet Stream")
    .set_subtitle("250 hPa winds steering the weather across North America")
    .set_xlabel("Longitude")
    .set_ylabel("Latitude")
    .grid()
)

# Animate the arrows along with the forecast: each one tracks the local jet
# direction and recolors with speed, one keyframe per forecast hour.
U_arrows = U.reshape(n_hours, -1)
V_arrows = V.reshape(n_hours, -1)
spd_arrows = np.hypot(U_arrows, V_arrows)
arrow_times = np.linspace(0.0, (n_frames - 1) * 0.06, n_hours)

jet.quiver_evolving(
    LON_G, LAT_G, U_arrows, V_arrows,
    scale=0.035, scale_mode="comp", normalize=True,       # ~3.5% of plot width
    c_frames=spd_arrows, gradient=([60, 70, 150], [255, 95, 160]),  # slow -> screaming
    width=3.2, headwidth=3.2, headlength=11.0,
    alpha=0.55, frame_times=arrow_times.tolist(),
    hold_keyframes=False, fade_in=1.0,
)

jet.scatter_evolving(
    x_frames, y_frames,
    c_frames=spd_frames,
    gradient=([120, 180, 255], [255, 230, 130]),
    radius=5,
    frame_duration=0.06,
    hold_keyframes=False,
    alpha=0.95,
    fade_in=0.4,
    drop_shadow=True,
)

jet.set_xlim(lons[0] - 1, lons[-1] + 1).set_ylim(lats[0] - 1, lats[-1] + 1)
jet.set_xticks().set_yticks()

jet.render()
