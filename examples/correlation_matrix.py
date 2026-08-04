"""Static heatmap: a correlation matrix of synthetic features (imshow-style).

A classic data-science heatmap: every cell (i, j) is the Pearson correlation
between feature i and feature j, drawn as a colored square. We use
``origin="upper"`` so the matrix reads top-to-bottom like a printed table,
``gap``/``edge_color`` for crisp tiles, and custom tick labels for the feature
names.

The nuanced, real-matrix cousin of the gallery ``heatmap`` demo.
"""

RENDERING_TIME = "VERY_SHORT"  # Likert render-time estimate: VERY_SHORT < SHORT < MODERATE < LONG < VERY_LONG
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env.user"))
sys.path.append(os.environ["AEGRAPH_PATH"])

from aegraph import AEGraph
import numpy as np

THEME = "Tidal Glass"

features = ["price", "size", "rooms", "age", "dist_cbd", "crime", "income"]
n = len(features)
rng = np.random.default_rng(7)

# Build a plausible correlated dataset, then take its correlation matrix.
m = 600
size = rng.normal(0, 1, m)
rooms = 0.8 * size + 0.4 * rng.normal(0, 1, m)
income = rng.normal(0, 1, m)
crime = -0.6 * income + 0.5 * rng.normal(0, 1, m)
dist_cbd = 0.5 * crime - 0.3 * income + 0.6 * rng.normal(0, 1, m)
age = 0.4 * dist_cbd + 0.6 * rng.normal(0, 1, m)
price = 1.0 * size + 0.6 * income - 0.5 * crime - 0.4 * dist_cbd + 0.4 * rng.normal(0, 1, m)
data = np.vstack([price, size, rooms, age, dist_cbd, crime, income])
corr = np.corrcoef(data)          # shape (n, n), values in [-1, 1]

# --- Build the heatmap ---
mat = (
    AEGraph(theme=THEME, comp_name="Correlation Matrix", drop_shadow=False,
            cinematic_effects=True, plot_frame=True,
            xaxis_location="bottom", yaxis_location="left", font_scale=0.85)
    .set_title("Feature Correlation Matrix")
    .set_subtitle("Pearson correlation between features")
)
mat.heatmap(
    corr,
    x=np.arange(n), y=np.arange(n),
    origin="upper",                          # row 0 at the top, like a table
    gradient=([70, 90, 200], [255, 110, 110]),   # diverging-ish: indigo -> coral
    vmin=-1.0, vmax=1.0,
    gap=0.06,
    edge_color=THEME and [234, 242, 246],    # bg-colored grout between tiles
    edge_width=2.0,
    colorbar_label="correlation",
    animate=1.6,
    reveal="diagonal"
)
# Label the axes with the feature names (y is flipped for origin="upper").
mat.set_xticks(positions=list(range(n)), labels=features)
mat.set_yticks(positions=list(range(n)), labels=list(reversed(features)))

mat.render()
