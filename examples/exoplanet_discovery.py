"""A galaxy of new worlds: every confirmed exoplanet lighting up the year we
found it, color-coded by *how* we found it.

Each dot is a real planet, placed by its orbital period (x) and mass (y) on a
log-log map. Dots stay dark until their discovery year, then pop into existence.
Watch the slow radial-velocity trickle of the late 1990s give way to the Kepler
transit *explosion* of the 2010s -- the moment humanity went from knowing a
handful of other worlds to thousands.

Data: NASA Exoplanet Archive (Planetary Systems table, default parameter set).
    examples/data/exoplanets.csv
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

THEME = "Spectral Dusk"
rng = np.random.default_rng(2026)

YEAR_MIN, YEAR_MAX = 1995, 2024

# Discovery method -> (display label, color, max points to sample).
METHODS = {
    "Transit":          ("Transit",          [255, 205, 70],  70),
    "Radial Velocity":  ("Radial velocity",  [80, 210, 235],  60),
    "Microlensing":     ("Microlensing",     [230, 110, 220], 30),
    "Imaging":          ("Direct imaging",   [140, 230, 130], 30),
    "Transit Timing Variations": ("Transit timing", [245, 140, 60], 25),
}
OTHER = ("Other methods", [170, 180, 195], 18)

# --- Load and clean ---
df = pd.read_csv("examples/data/exoplanets.csv")
df = df.dropna(subset=["disc_year", "pl_orbper", "pl_bmasse"])
df = df[(df["disc_year"] >= YEAR_MIN) & (df["disc_year"] <= YEAR_MAX)]
df = df[(df["pl_orbper"] > 0.2) & (df["pl_orbper"] <= 1e5)]
df = df[(df["pl_bmasse"] > 0.2) & (df["pl_bmasse"] <= 30000)]
df["group"] = df["discoverymethod"].where(df["discoverymethod"].isin(METHODS), "Other")

years = np.arange(YEAR_MIN, YEAR_MAX + 1)
n_frames = len(years)


def mass_to_radius(mass):
    """Bubble radius (px) from planet mass (Earth masses), log-compressed."""
    return np.clip(2.5 + 3.0 * np.log10(mass), 3.0, 18.0)


def sample_group(name, cap):
    sub = df[df["group"] == name]
    if len(sub) > cap:
        sub = sub.sample(cap, random_state=rng.integers(1 << 30))
    return sub


# --- Build the night sky ---
sky = (
    AEGraph(theme=THEME, comp_name="A Galaxy of New Worlds", fps=60,
            drop_shadow=True, cinematic_effects=True,
            xaxis_location="bottom", yaxis_location="left",
            plot_frame=True)
    .set_title("A Galaxy of New Worlds")
    .set_subtitle("every confirmed exoplanet, by year & method of discovery (1995\u20132024)")
    .set_xlabel("Orbital period (days)")
    .set_ylabel("Mass (Earth masses)")
    .set_xscale("log")
    .set_yscale("log")
    .grid()
)

# Reference lines for Earth and Jupiter mass so the scale lands emotionally.
xref = np.array([0.3, 1e5])
sky.plot(xref, np.array([1.0, 1.0]), color=[110, 120, 140],
         linestyle="dashed", linewidth=2, animate=0.01)
sky.plot(xref, np.array([317.8, 317.8]), color=[110, 120, 140],
         linestyle="dashed", linewidth=2, animate=0.01)
sky.annotate("Earth", x=0.4, y=1.25, fontsize=18)
sky.annotate("Jupiter", x=0.4, y=400, fontsize=18)

for name, (label, color, cap) in {**METHODS, "Other": OTHER}.items():
    sub = sample_group(name, cap if name != "Other" else OTHER[2])
    if sub.empty:
        continue
    period = sub["pl_orbper"].to_numpy(dtype=float)
    mass = sub["pl_bmasse"].to_numpy(dtype=float)
    disc = sub["disc_year"].to_numpy(dtype=float)
    size = mass_to_radius(mass)

    # Positions are fixed in time; tile them across frames.
    x_frames = np.tile(period, (n_frames, 1))
    y_frames = np.tile(mass, (n_frames, 1))

    # Radius is 0 until the discovery year, then the planet's size.
    visible = (years[:, None] >= disc[None, :]).astype(float)
    radius_frames = visible * size[None, :]

    sky.scatter_evolving(
        x_frames, y_frames,
        radius_frames=radius_frames,
        color=color,
        label=label,
        frame_duration=0.36,        # ~11 s sweep
        hold_keyframes=False,
        alpha=0.9,
        fade_in=0.0,
        drop_shadow=True,
    )

sky.set_xlim(0.3, 1e5).set_ylim(0.3, 30000)
sky.set_xticks().set_yticks()
sky.add_legend(legend_pos="bottom_right")

# A discovery-year ticker, counting up in sync with the planet sweep.
sky.evolving_text(
    years,
    location=(0, 18000),
    frame_duration=0.36,
    hold_keyframes=False,
    fontsize=80,
    horizontal_alignment="left",
    vertical_alignment="top",
    fade_in=0.0,
)

sky.render()
