"""Hurricane Katrina: a real storm track driving a moving cyclone wind field.

The eye positions and peak wind speeds are the *actual* best-track record for
Hurricane Katrina (AL122005) from NOAA's HURDAT2 database. We follow the storm
across the Gulf of Mexico to its Louisiana landfall.

At every instant we wrap a parametric cyclone around the real eye location -- a
modified-Rankine tangential profile (winds peak at the radius of maximum wind,
then decay outward) plus a touch of inflow, scaled by the true max-wind value at
that time. The quiver shows a snapshot of that field; a cloud of air parcels is
then advected through the **moving, intensifying** field, spiraling cyclonically
(counter-clockwise) into the eye as it tracks toward the coast.

Data: NOAA HURDAT2 best-track (Hurricane Katrina, 2005).
    Cached to examples/data/hurricane_katrina_track.csv on first run.
"""

RENDERING_TIME = "VERY_LONG"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
import urllib.request
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import numpy as np
import pandas as pd

THEME = "Blood Moon"
DATA = "examples/data/hurricane_katrina_track.csv"
HURDAT_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt"
STORM_ID = "AL122005"      # Hurricane Katrina


def fetch_track():
    """Pull Katrina's best-track block out of HURDAT2 and cache it as a CSV."""
    with urllib.request.urlopen(HURDAT_URL, timeout=60) as resp:
        lines = resp.read().decode().splitlines()

    rows, grab = [], False
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if parts[0].startswith("AL") and len(parts[0]) == 8 and len(parts) <= 4:
            grab = parts[0] == STORM_ID
            continue
        if grab:
            date, hhmm, _, status, lat_s, lon_s, vmax = parts[:7]
            lat = float(lat_s[:-1]) * (1 if lat_s[-1] == "N" else -1)
            lon = float(lon_s[:-1]) * (-1 if lon_s[-1] == "W" else 1)
            rows.append((date, hhmm, status, lat, lon, int(vmax)))
    df = pd.DataFrame(rows, columns=["date", "hhmm", "status", "lat", "lon", "vmax_kt"])
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    df.to_csv(DATA, index=False)
    return df


df = pd.read_csv(DATA) if os.path.exists(DATA) else fetch_track()
df["hhmm"] = df["hhmm"].astype(str).str.zfill(4)
df["t"] = pd.to_datetime(df["date"].astype(str) + df["hhmm"], format="%Y%m%d%H%M")

# Focus on the Gulf approach + Louisiana landfall (the dramatic part).
gulf = df[(df["lon"] <= -82) & (df["lat"] <= 30.5) & (df["lat"] >= 23.5)].copy()
gulf = gulf.reset_index(drop=True)
hours = (gulf["t"] - gulf["t"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
track_lon = gulf["lon"].to_numpy()
track_lat = gulf["lat"].to_numpy()
track_vmax = gulf["vmax_kt"].to_numpy(dtype=float)
T_END = hours[-1]


def eye_at(t):
    """Interpolated eye lon/lat and peak wind (kt) at time t (hours)."""
    lon = np.interp(t, hours, track_lon)
    lat = np.interp(t, hours, track_lat)
    vmax = np.interp(t, hours, track_vmax)
    return lon, lat, vmax


KT_TO_KMH = 1.852
RMAX_KM = 45.0          # radius of maximum wind
ALPHA = 0.55            # outer decay exponent of the modified-Rankine profile
INFLOW = np.deg2rad(22) # boundary-layer inflow angle


def wind_field(px, py, eye_lon, eye_lat, vmax_kt):
    """Cyclone wind (east, north) in km/h at points (px, py) about the eye."""
    coslat = np.cos(np.deg2rad(eye_lat))
    dx_km = (px - eye_lon) * 111.0 * coslat
    dy_km = (py - eye_lat) * 111.0
    r = np.hypot(dx_km, dy_km) + 1e-6

    vt = np.where(r <= RMAX_KM,
                  vmax_kt * (r / RMAX_KM),
                  vmax_kt * (RMAX_KM / r) ** ALPHA) * KT_TO_KMH
    # Counter-clockwise tangential unit vector + a little inward inflow.
    tang = np.stack([-dy_km / r, dx_km / r])
    radial_in = np.stack([-dx_km / r, -dy_km / r])
    vec = vt * (np.cos(INFLOW) * tang + np.sin(INFLOW) * radial_in)
    return vec[0], vec[1]


# --- Quiver grid (the arrows live on this fixed grid and animate in place) ---
gx = np.linspace(track_lon.min() - 4, track_lon.max() + 4, 19)
gy = np.linspace(track_lat.min() - 3, track_lat.max() + 3, 13)
GX, GY = np.meshgrid(gx, gy)
GXr, GYr = GX.ravel(), GY.ravel()

# --- Advect air parcels through the moving, intensifying cyclone ---
rng = np.random.default_rng(8)
n_parcels = 200
n_frames = 150
tgrid = np.linspace(0, T_END, n_frames)
DEG_PER_KM_LAT = 1.0 / 111.0
GAIN = 0.9                       # exaggerate drift for on-screen clarity

# Seed parcels broadly across the basin so the traveling vortex sweeps them up.
px = rng.uniform(gx[0], gx[-1], n_parcels)
py = rng.uniform(gy[0], gy[-1], n_parcels)

x_frames = np.zeros((n_frames, n_parcels))
y_frames = np.zeros((n_frames, n_parcels))
spd_frames = np.zeros((n_frames, n_parcels))
for f in range(n_frames):
    x_frames[f], y_frames[f] = px, py
    elon, elat, evmax = eye_at(tgrid[f])
    u, v = wind_field(px, py, elon, elat, evmax)     # km/h east, north
    spd_frames[f] = np.hypot(u, v)
    dt = (tgrid[1] - tgrid[0])
    coslat = np.cos(np.deg2rad(py))
    px = px + GAIN * dt * u / (111.0 * coslat)
    py = py + GAIN * dt * v * DEG_PER_KM_LAT

storm = (
    AEGraph(theme=THEME, comp_name="Hurricane Katrina Wind Field", fps=60,
            drop_shadow=True, cinematic_effects=True,
            xaxis_location="bottom", yaxis_location="left", plot_frame=True)
    .set_title("Hurricane Katrina")
    .set_subtitle("real NOAA track driving a moving cyclone wind field")
    .set_xlabel("Longitude")
    .set_ylabel("Latitude")
    .grid()
)

# Faint guide line: the real best-track path the eye follows.
storm.plot(track_lon, track_lat, color=[150, 110, 120], linewidth=2, animate=0.01)

# Animated quiver: sample the moving cyclone on the fixed grid over time, so the
# whole rotating field visibly translates and intensifies as the eye tracks in.
n_arrow_frames = 60
arrow_t = np.linspace(0, T_END, n_arrow_frames)
Uq = np.zeros((n_arrow_frames, GXr.size))
Vq = np.zeros((n_arrow_frames, GXr.size))
Sq = np.zeros((n_arrow_frames, GXr.size))
for k, t in enumerate(arrow_t):
    elon, elat, evmax = eye_at(t)
    Uq[k], Vq[k] = wind_field(GXr, GYr, elon, elat, evmax)
    Sq[k] = np.hypot(Uq[k], Vq[k])
arrow_times = np.linspace(0.0, (n_frames - 1) * 0.06, n_arrow_frames)

storm.quiver_evolving(
    GXr, GYr, Uq, Vq,
    scale=0.032, scale_mode="comp", normalize=True,        # ~3.2% of plot width
    c_frames=Sq, gradient=([90, 70, 130], [255, 120, 80]),  # calm -> ferocious
    width=3.2, headwidth=3.2, headlength=11.0,
    alpha=0.45, frame_times=arrow_times.tolist(),
    hold_keyframes=False, fade_in=1.0,
)

# a quiver doesn't make sense here because the wind field is not static. the quiver only includes the initial frame

storm.scatter_evolving(
    x_frames, y_frames,
    c_frames=spd_frames,
    gradient=([120, 180, 255], [255, 230, 140]),
    radius=5,
    frame_duration=0.06,
    hold_keyframes=False,
    alpha=0.95,
    fade_in=0.4,
    drop_shadow=True,
)

storm.set_xlim(gx[0], gx[-1]).set_ylim(gy[0], gy[-1])
storm.set_xticks().set_yticks()

storm.render()
