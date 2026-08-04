"""Pulse of the planet: a month of major earthquakes flashing across the globe.

Every magnitude-5.0+ quake from one month of the USGS catalog, placed at its
real longitude/latitude. Each one stays dark until the day it struck, then
flashes -- big quakes flash brighter, color encodes depth (shallow = hot red,
deep = cool blue) -- and fades. The Ring of Fire lights up like a heartbeat.

Data: USGS Earthquake Catalog (M5.0+, one month).
    examples/data/earthquakes.csv
"""

RENDERING_TIME = "MODERATE"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import numpy as np
import pandas as pd

THEME = "Magma"

df = pd.read_csv("examples/data/earthquakes.csv")
df = df[df["mag"] >= 5.0].copy()
df["day"] = (pd.to_datetime(df["time"]) - pd.to_datetime(df["time"]).min()).dt.days

lon = df["longitude"].to_numpy(dtype=float)
lat = df["latitude"].to_numpy(dtype=float)
mag = df["mag"].to_numpy(dtype=float)
depth = df["depth"].to_numpy(dtype=float)
day = df["day"].to_numpy(dtype=int)

n_quakes = len(df)
n_frames = int(day.max()) + 1
peak = 5.0 + (mag - 5.0) * 7.0          # flash radius scales with magnitude
tau = 2.2                                # fade time constant (frames)

# Pulse: 0 before the quake's day, jumps to peak, then exponential decay.
f = np.arange(n_frames)[:, None]         # (n_frames, 1)
struck = f >= day[None, :]
radius_frames = np.where(struck, peak[None, :] * np.exp(-(f - day[None, :]) / tau), 0.0)

# Color by depth (shallow = hot, deep = cool), constant per quake over time.
depth_frames = np.tile(depth, (n_frames, 1))

quakes = (
    AEGraph(theme=THEME, comp_name="Pulse of the Planet", fps=60,
            drop_shadow=True, cinematic_effects=True,
            xaxis_location="bottom", yaxis_location="left", plot_frame=True)
    .set_title("Pulse of the Planet")
    .set_subtitle("a month of magnitude 5.0+ earthquakes (USGS)")
    .set_xlabel("Longitude")
    .set_ylabel("Latitude")
    .grid()
)

quakes.scatter_evolving(
    np.tile(lon, (n_frames, 1)),
    np.tile(lat, (n_frames, 1)),
    radius_frames=radius_frames,
    c_frames=depth_frames,
    gradient=([255, 110, 60], [70, 130, 255]),   # shallow -> deep
    frame_duration=0.3,
    hold_keyframes=False,
    alpha=0.9,
    fade_in=0.0,
    drop_shadow=True,
)

quakes.set_xlim(-180, 180).set_ylim(-90, 90)
quakes.set_xticks(positions=[-180, -120, -60, 0, 60, 120, 180])
quakes.set_yticks(positions=[-90, -60, -30, 0, 30, 60, 90])

quakes.render()
