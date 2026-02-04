# Testing aegraph3d
# For better visual output, right click on all of the point layers at once and then go transform --> auto orient --> orient towards camera

import numpy as np
from aegraph3d import AEGraph3D

x = np.linspace(-2,2,16)
(x,y) = np.meshgrid(x, x)
z = (np.sin(x) * np.cos(y))* 20

graph = (
    AEGraph3D()
    .scatter3d(x, y, z, color="red", radius=5,z_time_expr=True)
)

graph.xlim = (-4, 4)
graph.ylim = (-4, 4)
graph.zlim = (-3, 3)
graph.render()