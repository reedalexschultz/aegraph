"""Moore's Law: half a century of exponential transistor growth.

Every dot is a real microprocessor, plotted by its release year and transistor
count on a logarithmic axis -- where pure exponential growth becomes a straight
line. The dots reveal chronologically while the dashed guide shows the famous
"doubling every two years." From 2,300 transistors (Intel 4004, 1971) to over
50 billion, this is the engine of the digital age.

Data: Karl Rupp, "Microprocessor Trend Data" (transistor counts).
    examples/data/moores_law.csv
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

THEME = "Retro Terminal"

df = pd.read_csv("examples/data/moores_law.csv").sort_values("year")
year = df["year"].to_numpy(dtype=float)
trans = df["transistors"].to_numpy(dtype=float)

# Moore's Law guide: doubling every 2 years, anchored at the Intel 4004.
gx = np.linspace(year.min(), year.max(), 50)
gy = 2300.0 * 2.0 ** ((gx - 1971.0) / 2.0)

moore = (
    AEGraph(theme=THEME, comp_name="Moores Law", fps=60,
            drop_shadow=True, cinematic_effects=True,
            xaxis_location="bottom", yaxis_location="left", plot_frame=True)
    .set_title("Moores Law")
    .set_subtitle("transistors per microprocessor, 1971 \u2192 2022")
    .set_xlabel("Year")
    .set_ylabel("Transistors (log scale)")
    .set_yscale("log")
    .grid()
)

# Dashed doubling guide underneath, drawn instantly.
moore.plot(gx, gy, color=[120, 130, 120], linestyle="dashed", linewidth=3, animate=0.01)
# The chips themselves, revealing year by year.
moore.scatter(year, trans, color=[120, 240, 150], radius=7, animate=6.0, drop_shadow=True)

moore.annotate("doubling every ~2 years", x=1986, y=4e8, fontsize=22)
moore.annotate("Intel 4004 (2,300)", x=1972, y=900, fontsize=16)

moore.set_xlim(1970, 2024).set_ylim(1e3, 1e11)
moore.set_xticks().set_yticks()

moore.render()
