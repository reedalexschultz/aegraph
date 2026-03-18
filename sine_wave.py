"""Generates sine wave"""

from aegraph import AEGraph
import numpy as np

num_points = 500
linspace = np.linspace(0, 10, num_points)
x_points = linspace
y_points = np.sin(x_points)
sine_wave = (
    AEGraph(width=1280, height=720, compwidth=1920, compheight=1080, drop_shadow=True, wiggle=True, cinematic_effects=True,comp_name="Scatterplot Test", fps=60)
    .grid()
    .set_xlabel("x")
    .set_ylabel("y")
    .set_title("Sine Wave Graph")
    .grid(show=True, color="gray", alpha=0.3)
    .add_legend()
)
sine_wave.plot(
    x_points, 
    y_points, 
    color="p_red", 
    label="Sine wave", 
    radius=10, 
    alpha=0.8, 
    animate=3, 
    bar_anim_times=1.0,
    drop_shadow=True)

sine_wave.set_yticks().set_xticks()
sine_wave.set_ylim(-2,2)
sine_wave.render()