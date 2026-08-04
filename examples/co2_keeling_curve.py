"""The Keeling Curve: atmospheric CO2 climbing relentlessly since 1959.

This is arguably the single most important graph of the modern era -- Charles
David Keeling's continuous measurement of CO2 at Mauna Loa, Hawaii. The line
animates in across ~7 seconds, sweeping from 315 ppm in 1959 past 420 ppm today,
with the pre-industrial baseline (~280 ppm) drawn as a dashed reference so you
feel exactly how far, and how fast, we have departed from it.

Data: NOAA GML / Scripps, annual mean CO2 (ppm), Mauna Loa Observatory.
    examples/data/co2_mauna_loa.csv
"""

RENDERING_TIME = "VERY_SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import numpy as np
import pandas as pd

THEME = "Petroleum Glow"
PRE_INDUSTRIAL = 280.0   # ppm, ~1750 baseline

df = pd.read_csv("examples/data/co2_mauna_loa.csv")
year = df["year"].to_numpy(dtype=float)
co2 = df["co2_ppm"].to_numpy(dtype=float)

baseline = np.full_like(year, PRE_INDUSTRIAL)
pct_above = (co2[-1] - PRE_INDUSTRIAL) / PRE_INDUSTRIAL * 100.0

keeling = (
    AEGraph(theme=THEME, comp_name="The Keeling Curve", fps=60,
            drop_shadow=True, cinematic_effects=True,
            xaxis_location="bottom", yaxis_location="left",
            plot_frame=True)
    .set_title("The Keeling Curve")
    .set_subtitle("Atmospheric CO\u2082 at Mauna Loa, 1959 \u2192 2025")
    .set_xlabel("Year")
    .set_ylabel("CO\u2082 (parts per million)")
    .grid()
)

# Pre-industrial reference line (drawn instantly, sits underneath).
keeling.plot(year, baseline, color=[120, 130, 150], linestyle="dashed",
             linewidth=3, animate=0.01)

# The main event: CO2 sweeping upward.
keeling.plot(year, co2, color=[255, 95, 70], linewidth=7,
             animate=7.0, drop_shadow=True)

# keeling.annotate(f"Pre-industrial \u2248 {PRE_INDUSTRIAL:.0f} ppm",
#                  x=year[0] + 6, y=PRE_INDUSTRIAL - 6, fontsize=22)
# keeling.annotate(f"+{pct_above:.0f}% since pre-industrial",
#                  x=year[-1] - 30, y=co2[-1] - 18, fontsize=26)

keeling.set_xlim(year.min(), year.max())
keeling.set_ylim(270, co2.max() + 12)
keeling.set_xticks().set_yticks()

keeling.render()
