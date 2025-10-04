"""Generates two somewhat random datasets to demonstrate the scatterplot abilit"""

from aegraph import AEGraph
import numpy as np
import pandas as pd
import random

num_points = 25
linspace = np.linspace(0, 1, num_points)
x_points = linspace + np.random.normal(0, 0.08, size=num_points)
y_points = x_points + np.random.normal(0, 0.08, size=num_points)
x_points_blue = linspace * (1/2) + np.random.normal(0, 0.08, size=num_points)
y_points_blue = linspace + np.random.normal(0, 0.08, size=num_points)
scatter_plot_test = (
    AEGraph(width=1280, height=720, compwidth=1920, compheight=1080, drop_shadow=True, cinematic_effects=True,comp_name="Scatterplot Test", fps=60)
    .grid()
    .set_xlabel("x")
    .set_ylabel("y")
    .set_title("Demo Scatterplot Relation")
    .grid(show=True, color="gray", alpha=0.3)
    .add_legend()
)
scatter_plot_test.scatter(
    x_points, 
    y_points, 
    color="p_red", 
    label="Test Points", 
    radius=10, 
    alpha=0.8, 
    animate=3, 
    bar_anim_times=1.0,
    drop_shadow=True)
scatter_plot_test.scatter(
    x_points_blue, 
    y_points_blue, 
    color="p_blue", 
    label="Blue Test Points", 
    radius=10, 
    alpha=0.8, 
    animate=3, 
    bar_anim_times=1.0,
    drop_shadow=True)

scatter_plot_test.set_yticks().set_xticks().set_ylim(0,1).set_xlim(0,1)
scatter_plot_test.render()