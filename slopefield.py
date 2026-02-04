from aegraph import AEGraph
from aegraph import COLOR_NAMES
from random import choice
import numpy as np

def f(x, y):
    return x+y + np.cos(np.pi * x)

# domain, range
X_MIN, X_MAX = -4, 4
Y_MIN, Y_MAX = -2, 2

NX, NY = 15, 10     # number of slope segments
SEG_LEN = 30        # length of each slope segment
SEG_WIDTH = 0.5     # width of each slope segment

COMP_DIMENSIONS = (1920, 1080)
GRAPH_DIMENSIONS = (1280, 720)

ANIMATE_HORIZ = True
ANIMATE_VERT = True

TIME_SLOPES_GENERATE = 1 # how long the animation of all of the slopes in tandem from left to right takes

g = (
    AEGraph(
        width=GRAPH_DIMENSIONS[0],
        height=GRAPH_DIMENSIONS[1],
        compwidth=COMP_DIMENSIONS[0],
        compheight=COMP_DIMENSIONS[1],
        drop_shadow=True,
        cinematic_effects=True,
        comp_name="SlopeFields",
        wiggle=True,
        fps=60,
        bg_stroke_width = 2,
        show_all_points = True,
    )
    .grid(show=True, color="gray", alpha=0.25)
    .set_xlabel("x")
    .set_ylabel("y")
    .set_title("Slope Field: dy/dx = x+y + cos(pi*x)")
    .set_xlim(X_MIN, X_MAX)
    .set_ylim(Y_MIN, Y_MAX)
)

X_SCALE = g.width / (X_MAX - X_MIN)
Y_SCALE = g.height / (Y_MAX - Y_MIN)

# generates slope field
xs = np.linspace(X_MIN, X_MAX, NX)
ys = np.linspace(Y_MIN, Y_MAX, NY)

for idx0, x in enumerate(xs):
    for idx1, y in enumerate(ys):
        slope = f(x, y)
        # base direction in data space

        dx_data = 1.0
        dy_data = slope

        # convert to pixel space
        dx_px = dx_data * X_SCALE
        dy_px = dy_data * Y_SCALE

        L_px = np.sqrt(dx_px**2 + dy_px**2)
        if L_px == 0:
            continue

        # normalize in pixel space
        dx_px /= L_px
        dy_px /= L_px

        # convert back to data space
        dx = dx_px / X_SCALE
        dy = dy_px / Y_SCALE

        dx *= SEG_LEN / 2
        dy *= SEG_LEN / 2

        x_seg = [x - dx, x, x + dx]
        y_seg = [y - dy, y, y + dy]

        delay = 0
        if ANIMATE_HORIZ:
            delay += idx0 * (TIME_SLOPES_GENERATE / len(xs))
                            
        if ANIMATE_VERT:
            delay += idx1 * (TIME_SLOPES_GENERATE / len(ys))


        g.plot(
            x_seg,
            y_seg,
            color="gray",
            alpha=0.7,
            linewidth=SEG_WIDTH,
            animate=1,
            delay = delay
        )

def lerp_color(c0, c1, t):
    return [
        float(c0[0] + t * (c1[0] - c0[0])),
        float(c0[1] + t * (c1[1] - c0[1])),
        float(c0[2] + t * (c1[2] - c0[2]))
    ]

dt = 0.03 # step size
STEPS = 500 # num of points on the line (if it goes beyond the window, they aren't plotted)

y0_vals = list(np.arange(-2, 2, 0.1))       # list of initial y conditions to solve for
SOLUTIONS_ANIM_TIME = 4 # how many seconds should it take to start animating all solutions in order
initial_x = np.zeros_like(y0_vals)           # list of initial x conditions to solve for
N = len(y0_vals)

INCLUDE_OOB = False # includes out of bounds points on the plot. if false, it stops integrating once the points leave the window in the y direction

LINE_WIDTH = 1
COLOR_START = choice(list(COLOR_NAMES.values()))
COLOR_END   = choice(list(COLOR_NAMES.values()))    # makes a gradient of two random colors

# solves each differential equation with initial conditions listed above
# plots each line, colored to match the gradient between COLOR_START and COLOR_END
for idx, y0 in enumerate(y0_vals):

    (x0, y0) = (initial_x[idx], y0) # initial x and y

    # backward integration
    x = x0
    y = y0
    xs_back = []
    ys_back = []

    for _ in range(STEPS):

        if not (X_MIN <= x <= X_MAX):
            break

        xs_back.append(x)
        ys_back.append(y)

        y -= dt * f(x, y)   # backward Euler
        x -= dt

        if not INCLUDE_OOB and not (Y_MIN <= y <= Y_MAX):
            break

    # reverse so time flows left → right
    xs_back.reverse()
    ys_back.reverse()

    # forward integration
    x = x0
    y = y0

    xs_fwd = []
    ys_fwd = []

    for _ in range(STEPS):
        
        if not (X_MIN <= x <= X_MAX):
            break

        xs_fwd.append(x)
        ys_fwd.append(y)

        y += dt * f(x, y)
        x += dt

        if not INCLUDE_OOB and not (Y_MIN <= y <= Y_MAX):
            break

    xs_curve = xs_back[:-1] + xs_fwd
    ys_curve = ys_back[:-1] + ys_fwd

    if len(xs_curve) < 5:
        continue

    if N != 1:
        t = idx / (N - 1)   # normalize idx → [0,1]
    else:
        t = 0
    color = lerp_color(COLOR_START, COLOR_END, t)

    g.plot(
        xs_curve,
        ys_curve,
        color=color,
        radius=4,
        alpha=0.9, 
        animate=2,
        delay = 0.75 + (idx * (SOLUTIONS_ANIM_TIME / len(y0_vals))),
        drop_shadow = True,
        linewidth = LINE_WIDTH,
    )

# finalize and render
g.set_xticks().set_yticks()
g.render()
