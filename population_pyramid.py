# I used this in a YouTube video here: https://www.youtube.com/shorts/CglFZ1yDYfA

from aegraph import AEGraph
import numpy as np
import pandas as pd
from pathlib import Path

file = Path("United States of America-2024.csv")
df = pd.read_csv(file)
ages = df["Age"].astype(str).tolist()
male = df["M"].astype(float).values
female = df["F"].astype(float).values
y_positions = np.arange(len(ages))
(
AEGraph(
    width=800,
    height=1500,
    compwidth=1500,
    compheight=1920,
    drop_shadow=True,
    comp_name="USA",
    fps=60,
    show_all_points=True,
    animate_opacity=False,
    animate_axes=False,
    bg_color="none",
    ui_color="black",
    font_scale = 1.7,
).add_population_pyramid(
    ages=ages,
    male=male,
    female=female,
    mode="percent",
    animate_downward=True,
    animate=1,
    bar_duration=0.5,
    drop_shadow=True,
    label_male=False,
    label_female=False,
    color_female=[224, 76, 157],
    color_male=[73, 118, 222],
    show_grid=False,
)
.set_xlabel("United States")
# .set_ylabel("Age Range")
.set_yticks(positions=y_positions, labels=ages)
.set_xticks([-6,-4,-2,0,2,4,6])  # defaults to absolute percent labels
.set_xlim(-6,6)
.render()
)