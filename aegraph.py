import datetime
import numpy as np
from typing import List, Tuple, Optional, Union
import math
try:
    import pandas as pd
except ImportError:
    pd = None

# Color dictionary (from your original)
COLOR_NAMES = {
    "black":      [0.0, 0.0, 0.0],
    "white":      [1.0, 1.0, 1.0],
    "red":        [1.0, 0.0, 0.0],
    "green":      [0.0, 1.0, 0.0],
    "blue":       [0.0, 0.0, 1.0],
    "yellow":     [1.0, 1.0, 0.0],
    "cyan":       [0.0, 1.0, 1.0],
    "magenta":    [1.0, 0.0, 1.0],
    "orange":     [1.0, 0.5, 0.0],
    "purple":     [0.5, 0.0, 0.5],
    "gray":       [0.5, 0.5, 0.5],
    "lightgray":  [0.8, 0.8, 0.8],
    "brown":      [0.6, 0.4, 0.2],
    "lime":       [0.75, 1.0, 0.0],
    "navy":       [0.0, 0.0, 0.5],
    "teal":       [0.0, 0.5, 0.5],
    "gold":       [1.0, 0.84, 0.0],
    # ... pastel colors ...
    "p_red":      [1.0, 0.6, 0.6],
    "p_green":    [0.6, 1.0, 0.6],
    "p_blue":     [0.6, 0.8, 1.0],
    "p_yellow":   [1.0, 1.0, 0.6],
    "p_cyan":     [0.6, 1.0, 1.0],
    "p_magenta":  [1.0, 0.6, 1.0],
    "p_orange":   [1.0, 0.8, 0.6],
    "p_purple":   [0.8, 0.6, 1.0],
    "p_pink":     [1.0, 0.8, 0.9],
    "p_gray":     [0.8, 0.8, 0.8],
    "p_brown":    [0.8, 0.7, 0.6],
    "p_lime":     [0.8, 1.0, 0.6],
    "p_navy":     [0.6, 0.7, 0.9],
    "p_teal":     [0.6, 0.9, 0.9],
    "p_gold":     [1.0, 0.9, 0.6],
}

DEFAULT_DROP_SHADOW = {
    'direction': 135,
    'distance': 5,
    'softness': 10,
    'color': 'black',
    'opacity': 0.3,
}


def color_to_js(color: Union[str, List[float], Tuple[float, float, float]]) -> str:
    """Convert color name or RGB list/tuple to JS array string."""
    if isinstance(color, str):
        rgb = COLOR_NAMES.get(color.lower())
        if rgb is None:
            raise ValueError(f"Unknown color name: {color}")
    elif isinstance(color, (list, tuple)) and len(color) == 3:
        rgb = color
    else:
        raise ValueError("Color must be a name or 3-value RGB list/tuple.")
    return f"[{rgb[0]}, {rgb[1]}, {rgb[2]}]"


def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


class AEGraph:
    """
    AEGraph: A matplotlib-inspired graphing API for Adobe After Effects.
    
    This class provides a simple interface to create animated graphs in After Effects
    using Python. It supports line plots, scatter plots, axes, ticks, grid, and labels.
    
    Key Features:
    - Line plots with customizable colors, linewidth, and animation
    - Scatter plots with customizable colors, radius, and animation
    - Automatic axes positioning and styling
    - Custom tick marks and labels
    - Grid lines with customizable appearance
    - Title and axis labels
    - Automatic coordinate system handling for After Effects
    - Optional cinematic adjustment layer with CC Vignette effects
    - Optional wiggle adjustment layer with Turbulent Displace for organic movement
    - Easy ease animation system for smooth, professional keyframe interpolation
    
    Coordinate System:
    - Data coordinates are automatically converted to After Effects coordinates
    - Line plots and axes use shape layer coordinates (centered at comp center)
    - Scatter points use layer positioning (absolute comp coordinates)
    - All elements are automatically aligned and positioned correctly
    
    Example Usage:
        from aegraph import AEGraph
        import numpy as np
        
        # Create data
        t = np.linspace(0, 2*np.pi, 50)
        y = np.sin(t)
        
        # Create graph with drop shadows, cinematic effects, wiggle, and easy ease
        plot = (AEGraph(width=1920, height=1080, drop_shadow=True, cinematic_effects=True, wiggle=True, easy_ease=True, ease_speed=00, ease_influence=33)
                .plot(t, y, color="blue", animate=2.0, drop_shadow=True)
                .scatter(t[::5], y[::5], color="red", radius=8, drop_shadow=True)
                .set_title("Sine Wave")
                .set_xticks([0, np.pi, 2*np.pi], ["0", "π", "2π"])
                .set_yticks([-1, 0, 1])
                .grid(show=True, color="gray", alpha=0.3))
        
        # Save and render
        plot.save("my_graph.jsx")
        plot.render()
    """
    def __init__(self, width=1920, height=1080, comp_name="AEGraph_Comp", bg_color="white", comp_width=None, comp_height=None, compwidth=None, compheight=None, position=None, show_all_points=False, drop_shadow=False, cinematic_effects=False, wiggle=False, easy_ease=True, ease_speed=00, ease_influence=33, fps=24):
        """
        Initialize a new AEGraph instance.
        Args:
            width (int): Graph logical width (default: 1920)
            height (int): Graph logical height (default: 1080)
            comp_name (str): Name of the AE composition (default: "AEGraph_Comp")
            bg_color (str or list): Background color name or RGB list (default: "white")
            comp_width/compheight/compwidth/compheight (int, optional): AE composition width/height (default: width/height)
            position (tuple, optional): (x, y) center of graph in comp coordinates (default: comp center)
            show_all_points (bool): Whether to plot all points or only those within bounds (default: False)
            drop_shadow (bool): Whether to add drop shadow to graph elements (default: False)
            cinematic_effects (bool): Whether to add cinematic adjustment layer with vignette effects (default: False)
            wiggle (bool): Whether to add wiggle adjustment layer with Turbulent Displace (default: False)
            easy_ease (bool): Whether to apply easy ease to all animation keyframes (default: True)
            ease_speed (int): Easy ease speed percentage (default: 00)
            ease_influence (int): Easy ease influence percentage (default: 33)
            fps (int): Frame rate for the After Effects composition (default: 24)
        """
        self.width = width
        self.height = height
        # Priority: compwidth/compheight > comp_width/comp_height > width/height
        if compwidth is not None:
            self.comp_width = compwidth
        elif comp_width is not None:
            self.comp_width = comp_width
        else:
            self.comp_width = width
        if compheight is not None:
            self.comp_height = compheight
        elif comp_height is not None:
            self.comp_height = comp_height
        else:
            self.comp_height = height
        self.comp_name = comp_name
        self.bg_color = bg_color
        self.position = position  # (x, y) in comp coordinates, or None for center
        self.show_all_points = show_all_points
        self.drop_shadow = drop_shadow  # Global drop shadow setting
        self.elements = []  # List of plot elements (dicts)
        self.title = None
        self.xlabel = None
        self.ylabel = None
        self.legend = []
        self.xlim = None
        self.ylim = None
        self.xticks = None  # X-axis tick positions and labels
        self.yticks = None  # Y-axis tick positions and labels
        self.show_grid = False   # Whether to show grid
        self.grid_color = "gray"  # Grid color
        self.grid_alpha = 0.3     # Grid opacity
        self.show_tick_labels = True  # New feature: control tick label visibility
        self.cinematic_effects = cinematic_effects  # Whether to add cinematic adjustment layer
        self.wiggle = wiggle  # Whether to add wiggle adjustment layer with Turbulent Displace
        self.easy_ease = easy_ease  # Whether to apply easy ease to all animation keyframes
        self.ease_speed = ease_speed  # Easy ease speed percentage
        self.ease_influence = ease_influence  # Easy ease influence percentage
        self.fps = fps  # Frame rate for the After Effects composition

    def _filter_points(self, x, y):
        """
        Filter points based on show_all_points setting and current limits.
        Returns filtered x, y arrays.
        """
        if self.show_all_points:
            return x, y
        
        # Get current bounds - use actual graph limits, not data min/max
        if self.xlim:
            xmin, xmax = self.xlim
        else:
            # If no xlim set, don't filter on x
            xmin, xmax = float('-inf'), float('inf')
            
        if self.ylim:
            ymin, ymax = self.ylim
        else:
            # If no ylim set, don't filter on y
            ymin, ymax = float('-inf'), float('inf')
        
        # Filter points within bounds
        filtered_x, filtered_y = [], []
        for xi, yi in zip(x, y):
            if xmin <= xi <= xmax and ymin <= yi <= ymax:
                filtered_x.append(xi)
                filtered_y.append(yi)
        
        return filtered_x, filtered_y

    def plot(self, x, y, color="blue", label=None, linewidth=4, animate=1.0, drop_shadow=False, **kwargs):
        """
        Add a line plot to the graph.
        x, y: Data points (list, tuple, numpy array, or pandas Series).
        color: Color name or RGB list.
        label: Legend label.
        linewidth: Stroke width.
        animate: Animation duration in seconds.
        drop_shadow: Whether to add drop shadow effect (default: False).
        """
        # Support pandas Series
        if pd is not None:
            if isinstance(x, pd.Series):
                x = x.values
            if isinstance(y, pd.Series):
                y = y.values
        
        # Filter points if needed
        x, y = self._filter_points(x, y)
        
        self.elements.append({
            "type": "line",
            "x": list(x),
            "y": list(y),
            "color": color,
            "label": label,
            "linewidth": linewidth,
            "animate": animate,
            "drop_shadow": drop_shadow,
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def scatter(self, x, y, color="red", label=None, radius=8, animate=1.0, drop_shadow=False, bar_anim_times=None, **kwargs):
        """
        Add a scatter plot to the graph.
        x, y: Data points (list, tuple, numpy array, or pandas Series).
        color: Color name or RGB list.
        label: Legend label.
        radius: Point radius.
        animate: Total animation duration in seconds (points animate sequentially).
        drop_shadow: Whether to add drop shadow effect (default: False).
        bar_anim_times: Optional list of per-point animation durations (overrides sequential timing).
        """
        # Support pandas Series
        if pd is not None:
            if isinstance(x, pd.Series):
                x = x.values
            if isinstance(y, pd.Series):
                y = y.values
        
        # Filter points if needed
        x, y = self._filter_points(x, y)
        
        self.elements.append({
            "type": "scatter",
            "x": list(x),
            "y": list(y),
            "color": color,
            "label": label,
            "radius": radius,
            "animate": animate,
            "drop_shadow": drop_shadow,
            "bar_anim_times": bar_anim_times,
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def histogram(self, data, bins=10, color="p_blue", label=None, alpha=0.8, animate=1.0, drop_shadow=False, bar_anim_times=None, density=False, **kwargs):
        """
        Add a histogram to the graph.
        data: 1D array-like data to bin.
        bins: Number of bins or bin edges.
        color: Bar color.
        label: Legend label.
        alpha: Bar opacity (0-1).
        animate: Total animation duration in seconds (bars animate sequentially).
        drop_shadow: Whether to add drop shadow effect.
        bar_anim_times: Optional list of per-bar animation durations (overrides sequential timing).
        density: If True, normalize heights so the area under the histogram is 1 (PDF style); if False, heights are counts.
        """
        # Support pandas Series
        if pd is not None and isinstance(data, pd.Series):
            data = data.values
        data = np.asarray(data)
        counts, bin_edges = np.histogram(data, bins=bins, density=False)
        bin_left = bin_edges[:-1]
        bin_right = bin_edges[1:]
        bin_centers = (bin_left + bin_right) / 2
        heights = counts
        if density:
            bin_widths = bin_right - bin_left
            total_area = np.sum(counts * bin_widths)
            if total_area > 0:
                heights = counts / total_area
            else:
                heights = counts
        self.elements.append({
            "type": "histogram",
            "bin_left": list(bin_left),
            "bin_right": list(bin_right),
            "bin_centers": list(bin_centers),
            "heights": list(heights),
            "color": color,
            "label": label,
            "alpha": alpha,
            "animate": animate,
            "drop_shadow": drop_shadow,
            "bar_anim_times": bar_anim_times,
            "density": density,
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def bar_graph(self, x_values, heights, bar_width=None, color="p_blue", label=None, alpha=0.8, animate=1.0, drop_shadow=False, bar_anim_times=None, **kwargs):
        """
        Add a bar graph to the plot using x values and corresponding heights.
        x_values: 1D array-like x positions for bars.
        heights: 1D array-like heights for bars (must match length of x_values).
        bar_width: Width of each bar in data coordinates. If None, auto-calculated from spacing.
        color: Bar color.
        label: Legend label.
        alpha: Bar opacity (0-1).
        animate: Total animation duration in seconds (bars animate sequentially).
        drop_shadow: Whether to add drop shadow effect.
        bar_anim_times: Optional list of per-bar animation durations (overrides sequential timing).
        """
        # Support pandas Series
        if pd is not None:
            if isinstance(x_values, pd.Series):
                x_values = x_values.values
            if isinstance(heights, pd.Series):
                heights = heights.values
        
        x_values = np.asarray(x_values)
        heights = np.asarray(heights)
        
        if len(x_values) != len(heights):
            raise ValueError("x_values and heights must have the same length")
        
        # Auto-calculate bar width if not specified
        if bar_width is None:
            if len(x_values) > 1:
                # Use 80% of the minimum spacing between consecutive x values
                spacings = np.diff(np.sort(x_values))
                min_spacing = np.min(spacings[spacings > 0]) if len(spacings) > 0 and np.any(spacings > 0) else 1.0
                bar_width = 0.8 * min_spacing
            else:
                bar_width = 1.0  # Default for single bar
        
        # Calculate bin edges for each bar (left and right edges)
        half_width = bar_width / 2
        bin_left = x_values - half_width
        bin_right = x_values + half_width
        bin_centers = x_values.copy()  # x_values are already the centers
        
        self.elements.append({
            "type": "bar_graph",
            "bin_left": list(bin_left),
            "bin_right": list(bin_right),
            "bin_centers": list(bin_centers),
            "heights": list(heights),
            "color": color,
            "label": label,
            "alpha": alpha,
            "animate": animate,
            "drop_shadow": drop_shadow,
            "bar_anim_times": bar_anim_times,
            **kwargs
        })
        if label:
            self.legend.append(label)
        return self

    def gradient(self, color_start, color_end):
        """
        Apply a color gradient to the most recently added histogram or bar_graph.
        Supports named colors, 0–1 floats, or 0–255 ints.
        """
        if not self.elements:
            raise ValueError("No elements to apply gradient to. Add a histogram or bar_graph first.")

        # Get the last element
        last_elem = self.elements[-1]
        if last_elem["type"] not in ["histogram", "bar_graph"]:
            raise ValueError("Gradient can only be applied to histogram or bar_graph elements.")

        def normalize_color(c):
            if isinstance(c, str):
                rgb = COLOR_NAMES.get(c.lower())
                if rgb is None:
                    raise ValueError(f"Unknown color name: {c}")
                return rgb
            elif isinstance(c, (list, tuple)) and len(c) == 3:
                c = list(c)
                if max(c) > 1:  # assume 0–255 ints
                    return [v/255.0 for v in c]
                return c
            else:
                raise ValueError("Color must be a name or 3-value RGB list/tuple.")

        start_rgb = normalize_color(color_start)
        end_rgb = normalize_color(color_end)

        # Calculate number of bars
        n_bars = len(last_elem["heights"])
        if n_bars <= 1:
            last_elem["gradient_colors"] = [start_rgb]
        else:
            gradient_colors = []
            for i in range(n_bars):
                t = i / (n_bars - 1)  # interpolation factor
                interpolated_rgb = [
                    start_rgb[0] + t * (end_rgb[0] - start_rgb[0]),
                    start_rgb[1] + t * (end_rgb[1] - start_rgb[1]),
                    start_rgb[2] + t * (end_rgb[2] - start_rgb[2])
                ]
                gradient_colors.append(interpolated_rgb)
            last_elem["gradient_colors"] = gradient_colors

        return self


    def set_title(self, title: str):
        """Set the plot title."""
        self.title = title
        return self

    def set_xlabel(self, label: str):
        """Set the x-axis label."""
        self.xlabel = label
        return self

    def set_ylabel(self, label: str):
        """Set the y-axis label."""
        self.ylabel = label
        return self

    def set_xlim(self, xmin, xmax):
        """Set the x-axis limits."""
        self.xlim = (xmin, xmax)
        return self

    def set_ylim(self, ymin, ymax):
        """Set the y-axis limits."""
        self.ylim = (ymin, ymax)
        return self

    def add_legend(self):
        """Add a legend to the plot (auto from labels)."""
        # Legend is auto-tracked from labels
        return self

    def set_xticks(self, positions=None, labels=None, nticks=7):
        """
        Set X-axis tick positions and labels.
        """
        if positions is None:
            # Collect x-like data from all elements
            all_x = []
            for elem in self.elements:
                if elem["type"] in ["line", "scatter"]:
                    all_x.extend(elem.get("x", []))
                elif elem["type"] in ["histogram", "bar_graph"]:
                    all_x.extend(elem.get("bin_left", []))
                    all_x.extend(elem.get("bin_right", []))
                    all_x.extend(elem.get("bin_centers", []))
            if not all_x:
                all_x = [0, 1]  # fallback if nothing plotted

            if self.xlim:
                xmin, xmax = self.xlim
            else:
                xmin, xmax = min(all_x), max(all_x)

            padding = 0.1
            x_range = xmax - xmin
            xmin -= x_range * padding
            xmax += x_range * padding
            positions = self._nice_ticks(xmin, xmax, nticks)

        if labels is None:
            labels = [str(pos) for pos in positions]

        self.xticks = list(zip(positions, labels))
        return self

    def set_yticks(self, positions=None, labels=None, nticks=7):
        """
        Set Y-axis tick positions and labels.
        """
        if positions is None:
            # Collect y-like data from all elements
            all_y = []
            for elem in self.elements:
                if elem["type"] in ["line", "scatter"]:
                    all_y.extend(elem.get("y", []))
                elif elem["type"] in ["histogram", "bar_graph"]:
                    all_y.extend(elem.get("heights", []))
            if not all_y:
                all_y = [0, 1]  # fallback if nothing plotted

            if self.ylim:
                ymin, ymax = self.ylim
            else:
                ymin, ymax = min(all_y), max(all_y)

            padding = 0.1
            y_range = ymax - ymin
            ymin -= y_range * padding
            ymax += y_range * padding
            positions = self._nice_ticks(ymin, ymax, nticks)

        if labels is None:
            labels = [str(pos) for pos in positions]

        self.yticks = list(zip(positions, labels))
        return self

    def grid(self, show=True, color="gray", alpha=0.3):
        """
        Enable or disable grid lines.
        
        Args:
            show (bool): Whether to show grid (default: True)
            color (str or list): Grid color (default: "gray")
            alpha (float): Grid opacity 0-1 (default: 0.3)
        """
        self.show_grid = show
        self.grid_color = color
        self.grid_alpha = alpha
        return self

    def set_tick_labels(self, show: bool = True):
        """Enable or disable tick labels entirely."""
        self.show_tick_labels = show
        return self

    def reset_comp(self):
        """
        Delete all layers in the active composition to start fresh.
        Returns self for method chaining.
        """
        script = []
        script.append("app.beginUndoGroup(\"Reset Composition\");\n")
        script.append(
            """
var comp = app.project.activeItem;
if (comp && comp instanceof CompItem) {
    while (comp.numLayers > 0) {
        comp.layer(1).remove();
    }
}
"""
        )
        script.append("app.endUndoGroup();\n")
        # Execute the reset script immediately
        import subprocess
        import tempfile
        import os
        
        reset_jsx = "".join(script)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsx', delete=False) as f:
            f.write(reset_jsx)
            temp_file = f.name
        
        try:
            apple_script = f'''
            tell application "Adobe After Effects 2025"
                DoScriptFile "{temp_file}"
            end tell
            '''
            subprocess.run(["osascript", "-e", apple_script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            os.unlink(temp_file)
        
        return self

    def _graph_to_comp(self, x, y):
        """
        Map graph logical coordinates (centered at 0,0, width/height) to comp pixel coordinates (centered in comp).
        """
        # Only translate, do not scale
        comp_cx = self.comp_width / 2
        comp_cy = self.comp_height / 2
        comp_x = x + comp_cx
        comp_y = y + comp_cy
        return comp_x, comp_y

    def _data_to_canvas(self, x, y, xmin=None, xmax=None, ymin=None, ymax=None):
        """
        Convert data coordinates to graph logical pixel coordinates (width/height), then to comp pixel coordinates.
        """
        # Use provided limits or global limits
        if xmin is None or xmax is None:
            if self.xlim:
                xmin, xmax = self.xlim
            else:
                xmin, xmax = min(x), max(x)
        if ymin is None or ymax is None:
            if self.ylim:
                ymin, ymax = self.ylim
            else:
                ymin, ymax = min(y), max(y)
        # No padding
        xrange = xmax - xmin
        if xrange == 0:
            xrange = 1e-8
        yrange = ymax - ymin
        if yrange == 0:
            yrange = 1e-8
        # Map data to graph logical coordinates (origin at top-left)
        px = [self.width * (xi - xmin) / xrange for xi in x]
        py = [self.height - self.height * (yi - ymin) / yrange for yi in y]
        # Shift to graph center (origin at 0,0)
        px = [p - self.width/2 for p in px]
        py = [p - self.height/2 for p in py]
        # Map to comp coordinates
        comp_px, comp_py = zip(*[self._graph_to_comp(xg, yg) for xg, yg in zip(px, py)])
        return list(comp_px), list(comp_py)

    def _data_to_shape(self, x, y, xmin, xmax, ymin, ymax):
        """
        Convert data (x, y) to graph-local coordinates (centered at 0,0, width/height).
        """
        sx = self.width * (x - xmin) / (xmax - xmin) - self.width/2
        sy = self.height - self.height * (y - ymin) / (ymax - ymin) - self.height/2
        return sx, sy

    def _nice_ticks(self, vmin, vmax, nticks=7):
        """Generate nice tick positions between vmin and vmax."""
        if vmin == vmax:
            return [vmin]
        raw_step = (vmax - vmin) / max(1, nticks-1)
        mag = 10 ** math.floor(math.log10(abs(raw_step)))
        norm = raw_step / mag
        if norm < 1.5:
            step = 1 * mag
        elif norm < 3:
            step = 2 * mag
        elif norm < 7:
            step = 5 * mag
        else:
            step = 10 * mag
        tick_start = math.ceil(vmin / step) * step
        tick_end = math.floor(vmax / step) * step
        ticks = []
        v = tick_start
        while v <= tick_end + 1e-8:
            ticks.append(round(v, 10))
            v += step
        return ticks

    def _generate_drop_shadow_jsx(self, layer_name: str, index: str = "") -> str:
        """Generate JSX for drop shadow effect."""
        if not self.drop_shadow:
            return ""
        
        jsx = []
        jsx.append(f"var fx{index} = {layer_name}.property('Effects').addProperty('ADBE Drop Shadow');\n")
        jsx.append(f"fx{index}.property('Direction').setValue({DEFAULT_DROP_SHADOW['direction']});\n")
        jsx.append(f"fx{index}.property('Distance').setValue({DEFAULT_DROP_SHADOW['distance']});\n")
        jsx.append(f"fx{index}.property('Softness').setValue({DEFAULT_DROP_SHADOW['softness']});\n")
        jsx.append(f"fx{index}.property('Shadow Color').setValue({color_to_js(DEFAULT_DROP_SHADOW['color'])});\n")
        jsx.append(f"fx{index}.property('Opacity').setValue({255 * DEFAULT_DROP_SHADOW['opacity']});\n")
        
        return "".join(jsx)

    def _generate_axes_jsx(self, script, center_x, center_y, xmin_pad, xmax_pad, ymin_pad, ymax_pad, ANIM_DURATION):
        """Helper function to generate axes and ticks JSX code."""
        # Axes: y-axis at x=0 if in range, else at left; x-axis at y=0 if in range, else at bottom
        y_axis_x = 0 if xmin_pad <= 0 <= xmax_pad else xmin_pad
        x_axis_y = 0 if ymin_pad <= 0 <= ymax_pad else ymin_pad
        # Clamp axis reference values to visible range for tick placement
        y_axis_x = min(max(y_axis_x, xmin_pad), xmax_pad)
        x_axis_y = min(max(x_axis_y, ymin_pad), ymax_pad)
        # Axes endpoints in data coordinates
        x_axis_start = (xmin_pad, x_axis_y)
        x_axis_end = (xmax_pad, x_axis_y)
        y_axis_start = (y_axis_x, ymin_pad)
        y_axis_end = (y_axis_x, ymax_pad)
        # Convert to shape coordinates
        x0s, y0s = self._data_to_shape(*x_axis_start, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
        x1s, y1s = self._data_to_shape(*x_axis_end, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
        yx0s, yy0s = self._data_to_shape(*y_axis_start, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
        yx1s, yy1s = self._data_to_shape(*y_axis_end, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
        # X axis
        script.append("var axesLayer = comp.layers.addShape();\n")
        script.append("axesLayer.name = \"Axes\";\n")
        script.append(f"axesLayer.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
        script.append(f"axesLayer.parent = PlotAnchor;\n")
        script.append("var axesContents = axesLayer.property('ADBE Root Vectors Group');\n")
        script.append("var axesPathGroup = axesContents.addProperty('ADBE Vector Shape - Group');\n")
        script.append("var axesPath = axesPathGroup.property('ADBE Vector Shape');\n")
        script.append("var axesShape = new Shape();\n")
        script.append(f"axesShape.vertices = [[{x0s}, {y0s}], [{x1s}, {y1s}]];\n")
        script.append("axesShape.closed = false;\n")
        script.append("axesPath.setValue(axesShape);\n")
        script.append("var axesStroke = axesContents.addProperty('ADBE Vector Graphic - Stroke');\n")
        script.append("axesStroke.property('ADBE Vector Stroke Color').setValue([0, 0, 0]);\n")
        script.append("axesStroke.property('ADBE Vector Stroke Width').setValue(3);\n")
        script.append("axesStroke.property('ADBE Vector Stroke Opacity').setValue(100);\n")
        # Animate axes with Trim Paths
        script.append("var axesTrim = axesContents.addProperty('ADBE Vector Filter - Trim');\n")
        script.append(f"var axesTrimEnd = axesTrim.property('ADBE Vector Trim End');\n")
        script.append(f"axesTrimEnd.setValueAtTime(0, 0);\n")
        script.append(f"axesTrimEnd.setValueAtTime({ANIM_DURATION}, 100);\n")
        # Apply easy ease to axes trim path keyframes
        if self.easy_ease:
            script.append(f"applyEasyEase(axesTrimEnd, {self.ease_speed}, {self.ease_influence});\n")
        # Y axis
        script.append("var yAxisLayer = comp.layers.addShape();\n")
        script.append("yAxisLayer.name = \"Y-Axis\";\n")
        script.append(f"yAxisLayer.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
        script.append(f"yAxisLayer.parent = PlotAnchor;\n")
        script.append("var yAxisContents = yAxisLayer.property('ADBE Root Vectors Group');\n")
        script.append("var yAxisPathGroup = yAxisContents.addProperty('ADBE Vector Shape - Group');\n")
        script.append("var yAxisPath = yAxisPathGroup.property('ADBE Vector Shape');\n")
        script.append("var yAxisShape = new Shape();\n")
        script.append(f"yAxisShape.vertices = [[{yx0s}, {yy0s}], [{yx1s}, {yy1s}]];\n")
        script.append("yAxisShape.closed = false;\n")
        script.append("yAxisPath.setValue(yAxisShape);\n")
        script.append("var yAxisStroke = yAxisContents.addProperty('ADBE Vector Graphic - Stroke');\n")
        script.append("yAxisStroke.property('ADBE Vector Stroke Color').setValue([0, 0, 0]);\n")
        script.append("yAxisStroke.property('ADBE Vector Stroke Width').setValue(3);\n")
        script.append("yAxisStroke.property('ADBE Vector Stroke Opacity').setValue(100);\n")
        # Animate y axis with Trim Paths
        script.append("var yAxisTrim = yAxisContents.addProperty('ADBE Vector Filter - Trim');\n")
        script.append(f"var yAxisTrimEnd = yAxisTrim.property('ADBE Vector Trim End');\n")
        script.append(f"yAxisTrimEnd.setValueAtTime(0, 0);\n")
        script.append(f"yAxisTrimEnd.setValueAtTime({ANIM_DURATION}, 100);\n")
        # Apply easy ease to Y axis trim path keyframes
        if self.easy_ease:
            script.append(f"applyEasyEase(yAxisTrimEnd, {self.ease_speed}, {self.ease_influence});\n")
        
        # Add drop shadow to axes if specified
        if self.drop_shadow:
            script.append(self._generate_drop_shadow_jsx("axesLayer", "Axes"))
            script.append(self._generate_drop_shadow_jsx("yAxisLayer", "YAxis"))

        # --- Refactored Tick Mark and Label Placement (graph-local coordinates) ---
        # X Ticks
        if self.xticks:
            for idx, (pos, label) in enumerate(self.xticks):
                pos_var = str(pos).replace('-', 'm').replace('.', '_')
                # Get shape coordinates for tick mark
                tick_xs, tick_ys0 = self._data_to_shape(pos, x_axis_y-8*(ymax_pad-ymin_pad)/self.height, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                _, tick_ys1 = self._data_to_shape(pos, x_axis_y+8*(ymax_pad-ymin_pad)/self.height, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                # Draw tick mark as vertical line in shape coordinates
                script.append(f"var xtickLayer{pos_var} = comp.layers.addShape();\n")
                script.append(f"xtickLayer{pos_var}.name = \"XTick_{pos}\";\n")
                script.append(f"xtickLayer{pos_var}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                script.append(f"xtickLayer{pos_var}.parent = PlotAnchor;\n")
                script.append(f"var xtickContents{pos_var} = xtickLayer{pos_var}.property('ADBE Root Vectors Group');\n")
                script.append(f"var xtickPathGroup{pos_var} = xtickContents{pos_var}.addProperty('ADBE Vector Shape - Group');\n")
                script.append(f"var xtickPath{pos_var} = xtickPathGroup{pos_var}.property('ADBE Vector Shape');\n")
                script.append(f"var xtickShape{pos_var} = new Shape();\n")
                script.append(f"xtickShape{pos_var}.vertices = [[{tick_xs}, {tick_ys0}], [{tick_xs}, {tick_ys1}]];\n")
                script.append(f"xtickShape{pos_var}.closed = false;\n")
                script.append(f"xtickPath{pos_var}.setValue(xtickShape{pos_var});\n")
                script.append(f"var xtickStroke{pos_var} = xtickContents{pos_var}.addProperty('ADBE Vector Graphic - Stroke');\n")
                script.append(f"xtickStroke{pos_var}.property('ADBE Vector Stroke Color').setValue([0, 0, 0]);\n")
                script.append(f"xtickStroke{pos_var}.property('ADBE Vector Stroke Width').setValue(2);\n")
                # Fade-in animation for tick mark
                script.append(f"xtickStroke{pos_var}.property('ADBE Vector Stroke Opacity').setValueAtTime(0, 0);\n")
                script.append(f"xtickStroke{pos_var}.property('ADBE Vector Stroke Opacity').setValueAtTime({ANIM_DURATION * (0.7 + 0.3 * idx / 10)}, 100);\n")
                # Apply easy ease to X tick stroke opacity keyframes
                if self.easy_ease:
                    script.append(f"applyEasyEase(xtickStroke{pos_var}.property('ADBE Vector Stroke Opacity'), {self.ease_speed}, {self.ease_influence});\n")
                # Tick label: just below the tick, in graph-local coordinates
                if self.show_tick_labels:
                    lx, ly = self._data_to_shape(pos, x_axis_y, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    ly += 30
                    script.append(f"var xtickLabel{pos_var} = comp.layers.addText(\"{label}\");\n")
                    script.append(f"xtickLabel{pos_var}.property('Transform').property('Position').setValue([{center_x + lx}, {center_y + ly}]);\n")
                    script.append(f"xtickLabel{pos_var}.parent = PlotAnchor;\n")
                    script.append(f"var xtickLabelProp{pos_var} = xtickLabel{pos_var}.property('Source Text');\n")
                    script.append(f"var xtickLabelDoc{pos_var} = xtickLabelProp{pos_var}.value;\n")
                    script.append(f"xtickLabelDoc{pos_var}.fontSize = 16;\n")
                    script.append(f"xtickLabelDoc{pos_var}.fillColor = [0, 0, 0];\n")
                    script.append(f"xtickLabelDoc{pos_var}.justification = ParagraphJustification.CENTER_JUSTIFY;\n")
                    script.append(f"xtickLabelProp{pos_var}.setValue(xtickLabelDoc{pos_var});\n")
                    # Fade-in animation for label
                    script.append(f"xtickLabel{pos_var}.property('Transform').property('Opacity').setValueAtTime(0, 0);\n")
                    script.append(f"xtickLabel{pos_var}.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION * (0.8 + 0.2 * idx / 10)}, 100);\n")
                    # Apply easy ease to X tick label opacity keyframes
                    if self.easy_ease:
                        script.append(f"applyEasyEase(xtickLabel{pos_var}.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n")
        # Y Ticks
        if self.yticks:
            for idx, (pos, label) in enumerate(self.yticks):
                pos_var = str(pos).replace('-', 'm').replace('.', '_')
                # Get shape coordinates for tick mark
                tick_xs0, tick_ys = self._data_to_shape(y_axis_x-8*(xmax_pad-xmin_pad)/self.width, pos, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                tick_xs1, _ = self._data_to_shape(y_axis_x+8*(xmax_pad-xmin_pad)/self.width, pos, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                # Draw tick mark as horizontal line in shape coordinates
                script.append(f"var ytickLayer{pos_var} = comp.layers.addShape();\n")
                script.append(f"ytickLayer{pos_var}.name = \"YTick_{pos}\";\n")
                script.append(f"ytickLayer{pos_var}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                script.append(f"ytickLayer{pos_var}.parent = PlotAnchor;\n")
                script.append(f"var ytickContents{pos_var} = ytickLayer{pos_var}.property('ADBE Root Vectors Group');\n")
                script.append(f"var ytickPathGroup{pos_var} = ytickContents{pos_var}.addProperty('ADBE Vector Shape - Group');\n")
                script.append(f"var ytickPath{pos_var} = ytickPathGroup{pos_var}.property('ADBE Vector Shape');\n")
                script.append(f"var ytickShape{pos_var} = new Shape();\n")
                script.append(f"ytickShape{pos_var}.vertices = [[{tick_xs0}, {tick_ys}], [{tick_xs1}, {tick_ys}]];\n")
                script.append(f"ytickShape{pos_var}.closed = false;\n")
                script.append(f"ytickPath{pos_var}.setValue(ytickShape{pos_var});\n")
                script.append(f"var ytickStroke{pos_var} = ytickContents{pos_var}.addProperty('ADBE Vector Graphic - Stroke');\n")
                script.append(f"ytickStroke{pos_var}.property('ADBE Vector Stroke Color').setValue([0, 0, 0]);\n")
                script.append(f"ytickStroke{pos_var}.property('ADBE Vector Stroke Width').setValue(2);\n")
                # Fade-in animation for tick mark
                script.append(f"ytickStroke{pos_var}.property('ADBE Vector Stroke Opacity').setValueAtTime(0, 0);\n")
                script.append(f"ytickStroke{pos_var}.property('ADBE Vector Stroke Opacity').setValueAtTime({ANIM_DURATION * (0.7 + 0.3 * idx / 10)}, 100);\n")
                # Apply easy ease to Y tick stroke opacity keyframes
                if self.easy_ease:
                    script.append(f"applyEasyEase(ytickStroke{pos_var}.property('ADBE Vector Stroke Opacity'), {self.ease_speed}, {self.ease_influence});\n")
                # Tick label: just left of the tick, in graph-local coordinates
                if self.show_tick_labels:
                    lx, ly = self._data_to_shape(y_axis_x, pos, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    lx -= 30
                    script.append(f"var ytickLabel{pos_var} = comp.layers.addText(\"{label}\");\n")
                    script.append(f"ytickLabel{pos_var}.property('Transform').property('Position').setValue([{center_x + lx}, {center_y + ly}]);\n")
                    script.append(f"ytickLabel{pos_var}.parent = PlotAnchor;\n")
                    script.append(f"var ytickLabelProp{pos_var} = ytickLabel{pos_var}.property('Source Text');\n")
                    script.append(f"var ytickLabelDoc{pos_var} = ytickLabelProp{pos_var}.value;\n")
                    script.append(f"ytickLabelDoc{pos_var}.fontSize = 16;\n")
                    script.append(f"ytickLabelDoc{pos_var}.fillColor = [0, 0, 0];\n")
                    script.append(f"ytickLabelDoc{pos_var}.justification = ParagraphJustification.CENTER_JUSTIFY;\n")
                    script.append(f"ytickLabelProp{pos_var}.setValue(ytickLabelDoc{pos_var});\n")
                    # Fade-in animation for label
                    script.append(f"ytickLabel{pos_var}.property('Transform').property('Opacity').setValueAtTime(0, 0);\n")
                    script.append(f"ytickLabel{pos_var}.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION * (0.8 + 0.2 * idx / 10)}, 100);\n")
                    # Apply easy ease to Y tick label opacity keyframes
                    if self.easy_ease:
                        script.append(f"applyEasyEase(ytickLabel{pos_var}.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n")

    def _generate_jsx(self) -> str:
        ANIM_DURATION = 3 * (1/2)  # seconds for default animation
        script = []
        script.append("app.beginUndoGroup(\"AEGraph Render\");\n")
        
        # Add Easy Ease Helper Function
        if self.easy_ease:
            script.append("""
// --- Helper Function: Apply Easy Ease to All Keyframes of a Property ---
function applyEasyEase(prop, easeSpeed, easeInfluence) {
    if (!prop || !prop.numKeys || prop.numKeys < 2) return;
    
    // Determine the number of dimensions
    var dim = prop.value.length ? prop.value.length : 1;
    
    // Create KeyframeEase array per dimension
    var easeInArray = [];
    var easeOutArray = [];
    for (var d = 0; d < dim; d++) {
        easeInArray.push(new KeyframeEase(easeSpeed, easeInfluence));
        easeOutArray.push(new KeyframeEase(easeSpeed, easeInfluence));
    }
    
    // Apply to all keyframes
    for (var k = 1; k <= prop.numKeys; k++) {
        prop.setTemporalEaseAtKey(k, easeInArray, easeOutArray);
    }
}

""")
        
        script.append(
            f"""
var comp = app.project.activeItem;
if (!comp || !(comp instanceof CompItem) || comp.name != '{self.comp_name}') {{
    comp = app.project.items.addComp('{self.comp_name}', {self.comp_width}, {self.comp_height}, 1, 10, {self.fps});
    comp.openInViewer();
}}
"""
        )
        # Remove comp background color setting - let it be transparent
        # script.append(f"comp.bgColor = {color_to_js(self.bg_color)};\n")

        # --- PlotAnchor Null Layer ---
        # Always place anchor at comp center by default, or user-specified position
        anchor_x = self.position[0] if self.position is not None else self.comp_width / 2
        anchor_y = self.position[1] if self.position is not None else self.comp_height / 2
        script.append(f"var PlotAnchor = comp.layers.addNull();\n")
        script.append(f"PlotAnchor.name = 'PlotAnchor';\n")
        script.append(f"PlotAnchor.property('Transform').property('Position').setValue([{anchor_x}, {anchor_y}]);\n")
        script.append(f"PlotAnchor.moveToBeginning();\n")

        # All graph elements (background, axes, grid, etc.) are drawn at the comp center
        center_x = self.comp_width / 2
        center_y = self.comp_height / 2
        script.append(f"var bgLayer = comp.layers.addShape();\n")
        script.append(f"bgLayer.name = 'GraphBG';\n")
        script.append(f"var bgContents = bgLayer.property('ADBE Root Vectors Group');\n")
        script.append(f"var bgRect = bgContents.addProperty('ADBE Vector Shape - Rect');\n")
        script.append(f"bgRect.property('ADBE Vector Rect Size').setValue([{self.width}, {self.height}]);\n")
        script.append(f"bgRect.property('ADBE Vector Rect Position').setValue([0, 0]);\n")
        script.append(f"var bgFill = bgContents.addProperty('ADBE Vector Graphic - Fill');\n")
        script.append(f"bgFill.property('ADBE Vector Fill Color').setValue({color_to_js(self.bg_color)});\n")
        script.append(f"bgLayer.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
        script.append(f"bgLayer.parent = PlotAnchor;\n")

        # Title (relative to graph area)
        if self.title:
            script.append(f"var titleLayer = comp.layers.addText(\"{self.title}\");\n")
            script.append(f"titleLayer.property('Transform').property('Position').setValue([{center_x}, {center_y - self.height/2 + 80}]);\n")
            script.append(f"titleLayer.parent = PlotAnchor;\n")
            script.append("var titleProp = titleLayer.property('Source Text');\n")
            script.append("var titleDoc = titleProp.value;\n")
            script.append("titleDoc.fontSize = 48;\n")
            script.append("titleDoc.fillColor = [0, 0, 0];\n")
            script.append("titleDoc.justification = ParagraphJustification.CENTER_JUSTIFY;\n")
            script.append("titleProp.setValue(titleDoc);\n")
            
            # Add drop shadow to title if specified
            if self.drop_shadow:
                script.append(self._generate_drop_shadow_jsx("titleLayer", "Title"))

        # Axis labels (relative to graph area)
        if self.xlabel:
            script.append(f"var xlabelLayer = comp.layers.addText(\"{self.xlabel}\");\n")
            # Position x-label below the x-axis ticks (if any) or below the axis
            if self.xticks:
                # If there are ticks, position below the tick labels
                script.append(f"xlabelLayer.property('Transform').property('Position').setValue([{center_x}, {center_y + self.height/2 + 60}]);\n")
            else:
                # If no ticks, position below the axis
                script.append(f"xlabelLayer.property('Transform').property('Position').setValue([{center_x}, {center_y + self.height/2 + 40}]);\n")
            script.append(f"xlabelLayer.parent = PlotAnchor;\n")
            script.append("var xlabelProp = xlabelLayer.property('Source Text');\n")
            script.append("var xlabelDoc = xlabelProp.value;\n")
            script.append("xlabelDoc.fontSize = 24;\n")
            script.append("xlabelDoc.fillColor = [0, 0, 0];\n")
            script.append("xlabelDoc.justification = ParagraphJustification.CENTER_JUSTIFY;\n")
            script.append("xlabelProp.setValue(xlabelDoc);\n")
            
            # Add drop shadow to x-label if specified
            if self.drop_shadow:
                script.append(self._generate_drop_shadow_jsx("xlabelLayer", "XLabel"))
        if self.ylabel:
            script.append(f"var ylabelLayer = comp.layers.addText(\"{self.ylabel}\");\n")
            # Position y-label to the left of the y-axis ticks (if any) or to the left of the axis
            if self.yticks:
                # If there are ticks, position to the left of the tick labels
                script.append(f"ylabelLayer.property('Transform').property('Position').setValue([{center_x - self.width/2 - 60}, {center_y}]);\n")
            else:
                # If no ticks, position to the left of the axis
                script.append(f"ylabelLayer.property('Transform').property('Position').setValue([{center_x - self.width/2 - 20}, {center_y}]);\n")
            script.append(f"ylabelLayer.parent = PlotAnchor;\n")
            script.append("ylabelLayer.property('Transform').property('Rotation').setValue(-90);\n")
            script.append("var ylabelProp = ylabelLayer.property('Source Text');\n")
            script.append("var ylabelDoc = ylabelProp.value;\n")
            script.append("ylabelDoc.fontSize = 24;\n")
            script.append("ylabelDoc.fillColor = [0, 0, 0];\n")
            script.append("ylabelDoc.justification = ParagraphJustification.CENTER_JUSTIFY;\n")
            script.append("ylabelProp.setValue(ylabelDoc);\n")
            
            # Add drop shadow to y-label if specified
            if self.drop_shadow:
                script.append(self._generate_drop_shadow_jsx("ylabelLayer", "YLabel"))

        # In _generate_jsx, calculate global data limits with NO padding
        all_x = []
        all_y = []
        for elem in self.elements:
            if elem["type"] in ["line", "scatter"]:
                all_x.extend(elem["x"])
                all_y.extend(elem["y"])
            elif elem["type"] in ["histogram", "bar_graph"]:
                # Use bin edges for x, heights for y
                all_x.extend(elem["bin_centers"])
                all_x.extend(elem["bin_left"])
                all_x.extend(elem["bin_right"])
                all_y.extend(elem["heights"])
                all_y.append(0)  # include baseline
        if self.xlim:
            xmin, xmax = self.xlim
        else:
            xmin, xmax = min(all_x), max(all_x)
        if self.ylim:
            ymin, ymax = self.ylim
        else:
            ymin, ymax = min(all_y), max(all_y)
        # No padding
        xmin_pad = xmin
        xmax_pad = xmax
        ymin_pad = ymin
        ymax_pad = ymax
        # Check what plot types we have to decide on axes placement
        has_scatter = any(elem["type"] == "scatter" for elem in self.elements)
        has_bars_or_hist = any(elem["type"] in ["histogram", "bar_graph"] for elem in self.elements)
        
        # For grid lines, fade in with opacity (GENERATED FIRST so grid appears behind plot elements)
        if self.show_grid:
            grid_color_js = color_to_js(self.grid_color)
            for idx, i in enumerate(self._nice_ticks(xmin_pad, xmax_pad)):
                xg, yg0 = self._data_to_shape(i, ymin_pad, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                xg, yg1 = self._data_to_shape(i, ymax_pad, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                grid_var = str(i).replace('-', 'm').replace('.', '_');
                script.append(f"var gridV{grid_var} = comp.layers.addShape();\n")
                script.append(f"gridV{grid_var}.name = \"Grid_V_{i}\";\n")
                script.append(f"gridV{grid_var}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                script.append(f"gridV{grid_var}.parent = PlotAnchor;\n")
                script.append(f"var gridVContents{grid_var} = gridV{grid_var}.property('ADBE Root Vectors Group');\n")
                script.append(f"var gridVPathGroup{grid_var} = gridVContents{grid_var}.addProperty('ADBE Vector Shape - Group');\n")
                script.append(f"var gridVPath{grid_var} = gridVPathGroup{grid_var}.property('ADBE Vector Shape');\n")
                script.append(f"var gridVShape{grid_var} = new Shape();\n")
                script.append(f"gridVShape{grid_var}.vertices = [[{xg}, {yg0}], [{xg}, {yg1}]];\n")
                script.append(f"gridVShape{grid_var}.closed = false;\n")
                script.append(f"gridVPath{grid_var}.setValue(gridVShape{grid_var});\n")
                script.append(f"var gridVStroke{grid_var} = gridVContents{grid_var}.addProperty('ADBE Vector Graphic - Stroke');\n")
                script.append(f"gridVStroke{grid_var}.property('ADBE Vector Stroke Color').setValue({grid_color_js});\n")
                script.append(f"gridVStroke{grid_var}.property('ADBE Vector Stroke Width').setValue(1);\n")
                script.append(f"gridVStroke{grid_var}.property('ADBE Vector Stroke Opacity').setValue(0);\n")
                # Animate opacity in
                script.append(f"gridVStroke{grid_var}.property('ADBE Vector Stroke Opacity').setValueAtTime(0, 0);\n")
                script.append(f"gridVStroke{grid_var}.property('ADBE Vector Stroke Opacity').setValueAtTime({ANIM_DURATION * (0.5 + 0.5 * idx / 10)}, {int(self.grid_alpha * 100)});\n")
                # Apply easy ease to grid V opacity keyframes
                if self.easy_ease:
                    script.append(f"applyEasyEase(gridVStroke{grid_var}.property('ADBE Vector Stroke Opacity'), {self.ease_speed}, {self.ease_influence});\n")
            for idx, i in enumerate(self._nice_ticks(ymin_pad, ymax_pad)):
                xg0, yg = self._data_to_shape(xmin_pad, i, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                xg1, yg = self._data_to_shape(xmax_pad, i, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                grid_var = str(i).replace('-', 'm').replace('.', '_');
                script.append(f"var gridH{grid_var} = comp.layers.addShape();\n")
                script.append(f"gridH{grid_var}.name = \"Grid_H_{i}\";\n")
                script.append(f"gridH{grid_var}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                script.append(f"gridH{grid_var}.parent = PlotAnchor;\n")
                script.append(f"var gridHContents{grid_var} = gridH{grid_var}.property('ADBE Root Vectors Group');\n")
                script.append(f"var gridHPathGroup{grid_var} = gridHContents{grid_var}.addProperty('ADBE Vector Shape - Group');\n")
                script.append(f"var gridHPath{grid_var} = gridHPathGroup{grid_var}.property('ADBE Vector Shape');\n")
                script.append(f"var gridHShape{grid_var} = new Shape();\n")
                script.append(f"gridHShape{grid_var}.vertices = [[{xg0}, {yg}], [{xg1}, {yg}]];\n")
                script.append(f"gridHShape{grid_var}.closed = false;\n")
                script.append(f"gridHPath{grid_var}.setValue(gridHShape{grid_var});\n")
                script.append(f"var gridHStroke{grid_var} = gridHContents{grid_var}.addProperty('ADBE Vector Graphic - Stroke');\n")
                script.append(f"gridHStroke{grid_var}.property('ADBE Vector Stroke Color').setValue({grid_color_js});\n")
                script.append(f"gridHStroke{grid_var}.property('ADBE Vector Stroke Width').setValue(1);\n")
                script.append(f"gridHStroke{grid_var}.property('ADBE Vector Stroke Opacity').setValue(0);\n")
                # Animate opacity in
                script.append(f"gridHStroke{grid_var}.property('ADBE Vector Stroke Opacity').setValueAtTime(0, 0);\n")
                script.append(f"gridHStroke{grid_var}.property('ADBE Vector Stroke Opacity').setValueAtTime({ANIM_DURATION * (0.5 + 0.5 * idx / 10)}, {int(self.grid_alpha * 100)});\n")
                # Apply easy ease to grid H opacity keyframes
                if self.easy_ease:
                    script.append(f"applyEasyEase(gridHStroke{grid_var}.property('ADBE Vector Stroke Opacity'), {self.ease_speed}, {self.ease_influence});\n")
        
        # Generate axes BEFORE plot elements ONLY for scatter plots (so axes appear behind points)
        # For histograms and bar graphs, axes will be generated AFTER plot elements (so they appear on top)
        if has_scatter and not has_bars_or_hist:
            self._generate_axes_jsx(script, center_x, center_y, xmin_pad, xmax_pad, ymin_pad, ymax_pad, ANIM_DURATION)

        # Plot elements
        for i, elem in enumerate(self.elements):
            # Use the same data limits for all mapping
            if elem["type"] == "line":
                px, py = elem["x"], elem["y"]
                shape_px, shape_py = zip(*[self._data_to_shape(x, y, xmin_pad, xmax_pad, ymin_pad, ymax_pad) for x, y in zip(px, py)])
                points_js = ",".join(f"[{x},{y}]" for x, y in zip(shape_px, shape_py))
                color_js = color_to_js(elem["color"])
                script.append(f"var lineLayer{i} = comp.layers.addShape();\n")
                script.append(f"lineLayer{i}.name = \"Line_{i}\";\n")
                script.append(f"lineLayer{i}.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
                script.append(f"lineLayer{i}.parent = PlotAnchor;\n")
                script.append(f"var contents{i} = lineLayer{i}.property('ADBE Root Vectors Group');\n")
                script.append(f"var pathGroup{i} = contents{i}.addProperty('ADBE Vector Shape - Group');\n")
                script.append(f"var path{i} = pathGroup{i}.property('ADBE Vector Shape');\n")
                script.append(f"var shape{i} = new Shape();\n")
                script.append(f"shape{i}.vertices = [{points_js}];\n")
                script.append(f"shape{i}.closed = false;\n")
                script.append(f"path{i}.setValue(shape{i});\n")
                script.append(f"var stroke{i} = contents{i}.addProperty('ADBE Vector Graphic - Stroke');\n")
                script.append(f"stroke{i}.property('ADBE Vector Stroke Color').setValue({color_js});\n")
                script.append(f"stroke{i}.property('ADBE Vector Stroke Width').setValue({elem['linewidth']});\n")
                # Animation using Trim Paths
                if elem["animate"] and elem["animate"] > 0:
                    script.append(f"var trim{i} = contents{i}.addProperty('ADBE Vector Filter - Trim');\n")
                    script.append(f"var endProp{i} = trim{i}.property('ADBE Vector Trim End');\n")
                    script.append(f"endProp{i}.setValueAtTime(0, 0);\n")
                    script.append(f"endProp{i}.setValueAtTime({elem['animate']}, 100);\n")
                    # Apply easy ease to trim path keyframes
                    if self.easy_ease:
                        script.append(f"applyEasyEase(endProp{i}, {self.ease_speed}, {self.ease_influence});\n")
                # Add drop shadow if specified
                if elem.get("drop_shadow", False):
                    script.append(self._generate_drop_shadow_jsx(f"lineLayer{i}", f"{i}"))
            elif elem["type"] == "scatter":
                px, py = elem["x"], elem["y"]
                color_js = color_to_js(elem["color"])
                n_points = len(px)
                bar_anim_times = elem.get("bar_anim_times")
                total_anim = elem["animate"] if elem["animate"] else 1.0
                
                # Accept a single float for bar_anim_times and apply to all points
                if bar_anim_times is None:
                    bar_anim_times = [total_anim / n_points] * n_points
                elif isinstance(bar_anim_times, (int, float)):
                    bar_anim_times = [float(bar_anim_times)] * n_points
                # Defensive: if bar_anim_times is not a list of correct length, fallback to default
                if not hasattr(bar_anim_times, '__iter__') or len(bar_anim_times) != n_points:
                    bar_anim_times = [total_anim / n_points] * n_points
                
                # Overlapping animation: distribute start times evenly within total_anim
                start_times = np.linspace(0, total_anim - bar_anim_times[0], n_points)
                
                scatter_px, scatter_py = zip(*[self._data_to_shape(x, y, xmin_pad, xmax_pad, ymin_pad, ymax_pad) for x, y in zip(px, py)])
                for j, (sx, sy) in enumerate(zip(scatter_px, scatter_py)):
                    script.append(f"var scatterLayer{i}_{j} = comp.layers.addShape();\n")
                    script.append(f"scatterLayer{i}_{j}.name = \"Scatter_{i}_{j}\";\n")
                    script.append(f"scatterLayer{i}_{j}.property('Transform').property('Position').setValue([{center_x + sx}, {center_y + sy}]);\n")
                    script.append(f"scatterLayer{i}_{j}.parent = PlotAnchor;\n")
                    script.append(f"var contents{i}_{j} = scatterLayer{i}_{j}.property('ADBE Root Vectors Group');\n")
                    script.append(f"var ellipseGroup{i}_{j} = contents{i}_{j}.addProperty('ADBE Vector Shape - Ellipse');\n")
                    script.append(f"ellipseGroup{i}_{j}.property('ADBE Vector Ellipse Size').setValue([{elem['radius']*2}, {elem['radius']*2}]);\n")
                    script.append(f"var fill{i}_{j} = contents{i}_{j}.addProperty('ADBE Vector Graphic - Fill');\n")
                    script.append(f"fill{i}_{j}.property('ADBE Vector Fill Color').setValue({color_js});\n")
                    
                    # Sequential animation for scatter points
                    if elem["animate"] and elem["animate"] > 0:
                        anim_time = bar_anim_times[j]
                        start_time = start_times[j]
                        script.append(f"var scale{i}_{j} = scatterLayer{i}_{j}.property('Transform').property('Scale');\n")
                        script.append(f"scale{i}_{j}.setValueAtTime({start_time}, [0, 0, 100]);\n")
                        script.append(f"scale{i}_{j}.setValueAtTime({start_time + anim_time}, [100, 100, 100]);\n")
                        # Apply easy ease to scale keyframes
                        if self.easy_ease:
                            script.append(f"applyEasyEase(scale{i}_{j}, {self.ease_speed}, {self.ease_influence});\n")
                    
                    # Add drop shadow if specified
                    if elem.get("drop_shadow", False):
                        script.append(self._generate_drop_shadow_jsx(f"scatterLayer{i}_{j}", f"{i}_{j}"))
            elif elem["type"] == "histogram":
                alpha = elem.get("alpha", 0.8)
                bin_left = elem["bin_left"]
                bin_right = elem["bin_right"]
                heights = elem["heights"]
                n_bars = len(bin_left)
                bar_anim_times = elem.get("bar_anim_times")
                total_anim = elem["animate"] if elem["animate"] else 1.0
                # Accept a single float for bar_anim_times and apply to all bars
                if bar_anim_times is None:
                    bar_anim_times = [total_anim / n_bars] * n_bars
                elif isinstance(bar_anim_times, (int, float)):
                    bar_anim_times = [float(bar_anim_times)] * n_bars
                # Defensive: if bar_anim_times is not a list of correct length, fallback to default
                if not hasattr(bar_anim_times, '__iter__') or len(bar_anim_times) != n_bars:
                    bar_anim_times = [total_anim / n_bars] * n_bars
                # Overlapping animation: distribute start times evenly within total_anim
                start_times = np.linspace(0, total_anim - bar_anim_times[0], n_bars)
                
                # Check if gradient colors are defined
                gradient_colors = elem.get("gradient_colors")
                if not gradient_colors:
                    # Use single color for all bars
                    color_js = color_to_js(elem["color"])
                
                for j, (left, right, height) in enumerate(zip(bin_left, bin_right, heights)):
                    # Use gradient color for this bar if available
                    if gradient_colors and j < len(gradient_colors):
                        bar_color_js = f"[{gradient_colors[j][0]}, {gradient_colors[j][1]}, {gradient_colors[j][2]}]"
                    else:
                        bar_color_js = color_js
                    
                    # Rectangle corners in data coordinates
                    x0, y0 = left, 0  # bottom left (axis)
                    x1, y1 = right, height  # top right
                    # Convert to shape coordinates
                    sx0, sy0 = self._data_to_shape(x0, y0, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    sx1, sy1 = self._data_to_shape(x1, y1, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    bar_width = abs(sx1 - sx0)
                    bar_height = abs(sy1 - sy0)
                    anchor_x = (sx0 + sx1) / 2
                    anchor_y = sy0  # axis (bottom)
                    position_x = center_x + anchor_x
                    position_y = center_y + anchor_y
                    script.append(f"var histLayer{i}_{j} = comp.layers.addShape();\n")
                    script.append(f"histLayer{i}_{j}.name = 'Histogram_{i}_{j}';\n")
                    script.append(f"histLayer{i}_{j}.property('Transform').property('Position').setValue([{position_x}, {position_y}]);\n")
                    script.append(f"histLayer{i}_{j}.parent = PlotAnchor;\n")
                    script.append(f"var histContents{i}_{j} = histLayer{i}_{j}.property('ADBE Root Vectors Group');\n")
                    script.append(f"var histRect{i}_{j} = histContents{i}_{j}.addProperty('ADBE Vector Shape - Rect');\n")
                    script.append(f"histRect{i}_{j}.property('ADBE Vector Rect Size').setValue([{bar_width}, {bar_height}]);\n")
                    script.append(f"histRect{i}_{j}.property('ADBE Vector Rect Position').setValue([0, -{bar_height/2}]);\n")
                    script.append(f"var histFill{i}_{j} = histContents{i}_{j}.addProperty('ADBE Vector Graphic - Fill');\n")
                    script.append(f"histFill{i}_{j}.property('ADBE Vector Fill Color').setValue({bar_color_js});\n")
                    script.append(f"histFill{i}_{j}.property('ADBE Vector Fill Opacity').setValue({int(alpha*100)});\n")
                    # Animate bar height (scale Y from axis)
                    anim_time = bar_anim_times[j]
                    start_time = start_times[j]
                    script.append(f"var histScale{i}_{j} = histLayer{i}_{j}.property('Transform').property('Scale');\n")
                    script.append(f"histScale{i}_{j}.setValueAtTime({start_time}, [100, 0, 100]);\n")
                    script.append(f"histScale{i}_{j}.setValueAtTime({start_time + anim_time}, [100, 100, 100]);\n")
                    if self.easy_ease:
                        script.append(f"applyEasyEase(histScale{i}_{j}, {self.ease_speed}, {self.ease_influence});\n")
                    if elem.get("drop_shadow", False):
                        script.append(self._generate_drop_shadow_jsx(f"histLayer{i}_{j}", f"{i}_{j}"))
            elif elem["type"] == "bar_graph":
                alpha = elem.get("alpha", 0.8)
                bin_left = elem["bin_left"]
                bin_right = elem["bin_right"]
                heights = elem["heights"]
                n_bars = len(bin_left)
                bar_anim_times = elem.get("bar_anim_times")
                total_anim = elem["animate"] if elem["animate"] else 1.0
                # Accept a single float for bar_anim_times and apply to all bars
                if bar_anim_times is None:
                    bar_anim_times = [total_anim / n_bars] * n_bars
                elif isinstance(bar_anim_times, (int, float)):
                    bar_anim_times = [float(bar_anim_times)] * n_bars
                # Defensive: if bar_anim_times is not a list of correct length, fallback to default
                if not hasattr(bar_anim_times, '__iter__') or len(bar_anim_times) != n_bars:
                    bar_anim_times = [total_anim / n_bars] * n_bars
                # Overlapping animation: distribute start times evenly within total_anim
                start_times = np.linspace(0, total_anim - bar_anim_times[0], n_bars)
                
                # Check if gradient colors are defined
                gradient_colors = elem.get("gradient_colors")
                if not gradient_colors:
                    # Use single color for all bars
                    color_js = color_to_js(elem["color"])
                
                for j, (left, right, height) in enumerate(zip(bin_left, bin_right, heights)):
                    # Use gradient color for this bar if available
                    if gradient_colors and j < len(gradient_colors):
                        bar_color_js = f"[{gradient_colors[j][0]}, {gradient_colors[j][1]}, {gradient_colors[j][2]}]"
                    else:
                        bar_color_js = color_js
                    
                    # Rectangle corners in data coordinates
                    x0, y0 = left, 0  # bottom left (axis)
                    x1, y1 = right, height  # top right
                    # Convert to shape coordinates
                    sx0, sy0 = self._data_to_shape(x0, y0, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    sx1, sy1 = self._data_to_shape(x1, y1, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                    bar_width = abs(sx1 - sx0)
                    bar_height = abs(sy1 - sy0)
                    anchor_x = (sx0 + sx1) / 2
                    anchor_y = sy0  # axis (bottom)
                    position_x = center_x + anchor_x
                    position_y = center_y + anchor_y
                    script.append(f"var barLayer{i}_{j} = comp.layers.addShape();\n")
                    script.append(f"barLayer{i}_{j}.name = 'Bar_{i}_{j}';\n")
                    script.append(f"barLayer{i}_{j}.property('Transform').property('Position').setValue([{position_x}, {position_y}]);\n")
                    script.append(f"barLayer{i}_{j}.parent = PlotAnchor;\n")
                    script.append(f"var barContents{i}_{j} = barLayer{i}_{j}.property('ADBE Root Vectors Group');\n")
                    script.append(f"var barRect{i}_{j} = barContents{i}_{j}.addProperty('ADBE Vector Shape - Rect');\n")
                    script.append(f"barRect{i}_{j}.property('ADBE Vector Rect Size').setValue([{bar_width}, {bar_height}]);\n")
                    script.append(f"barRect{i}_{j}.property('ADBE Vector Rect Position').setValue([0, -{bar_height/2}]);\n")
                    script.append(f"var barFill{i}_{j} = barContents{i}_{j}.addProperty('ADBE Vector Graphic - Fill');\n")
                    script.append(f"barFill{i}_{j}.property('ADBE Vector Fill Color').setValue({bar_color_js});\n")
                    script.append(f"barFill{i}_{j}.property('ADBE Vector Fill Opacity').setValue({int(alpha*100)});\n")
                    # Animate bar height (scale Y from axis)
                    anim_time = bar_anim_times[j]
                    start_time = start_times[j]
                    script.append(f"var barScale{i}_{j} = barLayer{i}_{j}.property('Transform').property('Scale');\n")
                    script.append(f"barScale{i}_{j}.setValueAtTime({start_time}, [100, 0, 100]);\n")
                    script.append(f"barScale{i}_{j}.setValueAtTime({start_time + anim_time}, [100, 100, 100]);\n")
                    if self.easy_ease:
                        script.append(f"applyEasyEase(barScale{i}_{j}, {self.ease_speed}, {self.ease_influence});\n")
                    if elem.get("drop_shadow", False):
                        script.append(self._generate_drop_shadow_jsx(f"barLayer{i}_{j}", f"{i}_{j}"))

        # Generate axes AFTER plot elements for histograms and bar graphs (so they appear on top)
        # Mixed plots get axes on top to accommodate bar/histogram rendering
        if has_bars_or_hist or not has_scatter:
            self._generate_axes_jsx(script, center_x, center_y, xmin_pad, xmax_pad, ymin_pad, ymax_pad, ANIM_DURATION)

        # Animate title and axis labels (only if they exist)
        if self.title:
            script.append(f"titleLayer.property('Transform').property('Opacity').setValueAtTime(0, 0);\n")
            script.append(f"titleLayer.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION * 1.1}, 100);\n")
            # Apply easy ease to title opacity keyframes
            if self.easy_ease:
                script.append(f"applyEasyEase(titleLayer.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n")
        if self.xlabel:
            script.append(f"xlabelLayer.property('Transform').property('Opacity').setValueAtTime(0, 0);\n")
            script.append(f"xlabelLayer.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION * 1.2}, 100);\n")
            # Apply easy ease to xlabel opacity keyframes
            if self.easy_ease:
                script.append(f"applyEasyEase(xlabelLayer.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n")
        if self.ylabel:
            script.append(f"ylabelLayer.property('Transform').property('Opacity').setValueAtTime(0, 0);\n")
            script.append(f"ylabelLayer.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION * 1.2}, 100);\n")
            # Apply easy ease to ylabel opacity keyframes
            if self.easy_ease:
                script.append(f"applyEasyEase(ylabelLayer.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n")

        # --- DYNAMIC LEGEND PLACEMENT (relative to graph area) ---
        legend_entries = []
        legend_colors = []
        for elem in self.elements:
            if elem.get('label'):
                legend_entries.append(elem['label'])
                legend_colors.append(color_to_js(elem['color']))
        legend_loc = 'topright'  # default
        margin = 80
        legend_width = 300
        legend_height = 40 + 30 * len(legend_entries)
        # Corners: (x, y) in graph-local coordinates (centered at 0,0)
        corners = {
            'topright': (self.width/2 - margin - legend_width/2, -self.height/2 + margin + legend_height/2),
            'topleft': (-self.width/2 + margin + legend_width/2, -self.height/2 + margin + legend_height/2),
            'bottomright': (self.width/2 - margin - legend_width/2, self.height/2 - margin - legend_height/2),
            'bottomleft': (-self.width/2 + margin + legend_width/2, self.height/2 - margin - legend_height/2),
        }
        # Heuristic: if any data point is in the legend area, move legend
        data_px, data_py = [], []
        for elem in self.elements:
            if elem["type"] in ["line", "scatter"]:
                px, py = elem["x"], elem["y"]
            elif elem["type"] in ["histogram", "bar_graph"]:
                px = elem["bin_centers"]
                py = elem["heights"]
            else:
                continue

            shape_px, shape_py = zip(*[self._data_to_shape(x, y, xmin_pad, xmax_pad, ymin_pad, ymax_pad)
                                    for x, y in zip(px, py)])
            data_px.extend(shape_px)
            data_py.extend(shape_py)

        for loc, (lx, ly) in corners.items():
            overlap = False
            for x, y in zip(data_px, data_py):
                if lx - legend_width/2 < x < lx + legend_width/2 and ly - legend_height/2 < y < ly + legend_height/2:
                    overlap = True
                    break
            if not overlap:
                legend_loc = loc
                break
        legend_x, legend_y = corners[legend_loc]
        if legend_entries:
            script.append(f"var legendGroup = [];\n")
            for i, (label, color) in enumerate(zip(legend_entries, legend_colors)):
                y_offset = legend_y + 20 + i * 30
                # Color swatch
                script.append(f"var legendSwatch{i} = comp.layers.addShape();\n")
                script.append(f"legendSwatch{i}.property('Transform').property('Position').setValue([{center_x + legend_x - legend_width/2 + 30}, {center_y + y_offset}]);\n")
                script.append(f"legendSwatch{i}.parent = PlotAnchor;\n")
                script.append(f"var legendSwatchContents{i} = legendSwatch{i}.property('ADBE Root Vectors Group');\n")
                script.append(f"var legendSwatchRect{i} = legendSwatchContents{i}.addProperty('ADBE Vector Shape - Rect');\n")
                script.append(f"legendSwatchRect{i}.property('ADBE Vector Rect Size').setValue([24, 24]);\n")
                script.append(f"var legendSwatchFill{i} = legendSwatchContents{i}.addProperty('ADBE Vector Graphic - Fill');\n")
                script.append(f"legendSwatchFill{i}.property('ADBE Vector Fill Color').setValue({color});\n")
                # Fade-in animation for swatch
                script.append(f"legendSwatch{i}.property('Transform').property('Opacity').setValueAtTime(0, 0);\n")
                script.append(f"legendSwatch{i}.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION}, 100);\n")
                # Apply easy ease to legend swatch opacity keyframes
                if self.easy_ease:
                    script.append(f"applyEasyEase(legendSwatch{i}.property('Transform').property('Opacity'), {self.ease_speed}, {self.ease_influence});\n")
                
                # Add drop shadow to legend swatch if specified
                if self.drop_shadow:
                    script.append(self._generate_drop_shadow_jsx(f"legendSwatch{i}", f"LegendSwatch{i}"))
                
                # Label
                script.append(f"var legendLabel{i} = comp.layers.addText(\"{label}\");\n")
                script.append(f"legendLabel{i}.property('Transform').property('Position').setValue([{center_x + legend_x - legend_width/2 + 70}, {center_y + y_offset + 2}]);\n")
                script.append(f"legendLabel{i}.parent = PlotAnchor;\n")
                script.append(f"var legendLabelProp{i} = legendLabel{i}.property('Source Text');\n")
                script.append(f"var legendLabelDoc{i} = legendLabelProp{i}.value;\n")
                script.append(f"legendLabelDoc{i}.fontSize = 24;\n")
                script.append(f"legendLabelDoc{i}.fillColor = [0, 0, 0];\n")
                script.append(f"legendLabelDoc{i}.justification = ParagraphJustification.LEFT_JUSTIFY;\n")
                script.append(f"legendLabelProp{i}.setValue(legendLabelDoc{i});\n")
                # Fade-in animation for label
                script.append(f"legendLabel{i}.property('Transform').property('Opacity').setValueAtTime(0, 0);\n")
                script.append(f"legendLabel{i}.property('Transform').property('Opacity').setValueAtTime({ANIM_DURATION}, 100);\n")
                # Apply easy ease to legend label opacity keyframes
                if self.drop_shadow:
                    script.append(self._generate_drop_shadow_jsx(f"legendLabel{i}", f"LegendLabel{i}"))



                # Title (relative to graph area)
        if self.title:
            script.append(f"var titleLayer = comp.layers.addText(\"{self.title}\");\n")
            script.append(f"titleLayer.property('Transform').property('Position').setValue([{center_x}, {center_y - self.height/2 + 80}]);\n")
            script.append(f"titleLayer.parent = PlotAnchor;\n")
            script.append("var titleProp = titleLayer.property('Source Text');\n")
            script.append("var titleDoc = titleProp.value;\n")
            script.append("titleDoc.fontSize = 48;\n")
            script.append("titleDoc.fillColor = [0, 0, 0];\n")
            script.append("titleDoc.justification = ParagraphJustification.CENTER_JUSTIFY;\n")
            script.append("titleProp.setValue(titleDoc);\n")
            
            # Add drop shadow to title if specified
            if self.drop_shadow:
                script.append(self._generate_drop_shadow_jsx("titleLayer", "Title"))

        # Axis labels (relative to graph area)
        if self.xlabel:
            script.append(f"var xlabelLayer = comp.layers.addText(\"{self.xlabel}\");\n")
            # Position x-label below the x-axis ticks (if any) or below the axis
            if self.xticks:
                # If there are ticks, position below the tick labels
                script.append(f"xlabelLayer.property('Transform').property('Position').setValue([{center_x}, {center_y + self.height/2 + 60}]);\n")
            else:
                # If no ticks, position below the axis
                script.append(f"xlabelLayer.property('Transform').property('Position').setValue([{center_x}, {center_y + self.height/2 + 40}]);\n")
            script.append(f"xlabelLayer.parent = PlotAnchor;\n")
            script.append("var xlabelProp = xlabelLayer.property('Source Text');\n")
            script.append("var xlabelDoc = xlabelProp.value;\n")
            script.append("xlabelDoc.fontSize = 24;\n")
            script.append("xlabelDoc.fillColor = [0, 0, 0];\n")
            script.append("xlabelDoc.justification = ParagraphJustification.CENTER_JUSTIFY;\n")
            script.append("xlabelProp.setValue(xlabelDoc);\n")
            
            # Add drop shadow to x-label if specified
            if self.drop_shadow:
                script.append(self._generate_drop_shadow_jsx("xlabelLayer", "XLabel"))
        if self.ylabel:
            script.append(f"var ylabelLayer = comp.layers.addText(\"{self.ylabel}\");\n")
            # Position y-label to the left of the y-axis ticks (if any) or to the left of the axis
            if self.yticks:
                # If there are ticks, position to the left of the tick labels
                script.append(f"ylabelLayer.property('Transform').property('Position').setValue([{center_x - self.width/2 - 60}, {center_y}]);\n")
            else:
                # If no ticks, position to the left of the axis
                script.append(f"ylabelLayer.property('Transform').property('Position').setValue([{center_x - self.width/2 - 20}, {center_y}]);\n")
            script.append(f"ylabelLayer.parent = PlotAnchor;\n")
            script.append("ylabelLayer.property('Transform').property('Rotation').setValue(-90);\n")
            script.append("var ylabelProp = ylabelLayer.property('Source Text');\n")
            script.append("var ylabelDoc = ylabelProp.value;\n")
            script.append("ylabelDoc.fontSize = 24;\n")
            script.append("ylabelDoc.fillColor = [0, 0, 0];\n")
            script.append("ylabelDoc.justification = ParagraphJustification.CENTER_JUSTIFY;\n")
            script.append("ylabelProp.setValue(ylabelDoc);\n")
            
            # Add drop shadow to y-label if specified
            if self.drop_shadow:
                script.append(self._generate_drop_shadow_jsx("ylabelLayer", "YLabel"))

        # --- CINEMATIC ADJUSTMENT LAYER (generated last to appear on top) ---
        if self.cinematic_effects:
            script.append(f"var adj = comp.layers.addSolid([1,1,1], \"CinematicAdjustment\", {self.comp_width}, {self.comp_height}, 1.0);\n")
            script.append(f"adj.adjustmentLayer = true;\n")
            script.append(f"adj.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
            script.append(f"adj.parent = PlotAnchor;\n")
            script.append(f"adj.moveToBeginning();\n")
            # Add CC Vignette effect
            script.append(f"var vignette = adj.property(\"Effects\").addProperty(\"CC Vignette\");\n")
            script.append(f"if (vignette != null) {{\n")
            script.append(f"    vignette.property(\"Amount\").setValue(50);     // Darken edges\n")
            script.append(f"    vignette.property(\"Angle of View\").setValue(50);   // Vignette spread\n")
            script.append(f"    vignette.property(\"Center\").setValue([{self.comp_width/2}, {self.comp_height/2}]);\n")
            script.append(f"    vignette.property(\"Pin Highlights\").setValue(0);   // Preserve highlights\n")
            script.append(f"}} else {{\n")
            script.append(f"    // Fallback to CS Vignette if CC Vignette not available\n")
            script.append(f"    var csVignette = adj.property(\"Effects\").addProperty(\"CS Vignette\");\n")
            script.append(f"    if (csVignette != null) {{\n")
            script.append(f"        csVignette.property(\"Amount\").setValue(50);     // Darken edges\n")
            script.append(f"        csVignette.property(\"Angle of View\").setValue(50);   // Vignette spread\n")
            script.append(f"        csVignette.property(\"Center\").setValue([{self.comp_width/2}, {self.comp_height/2}]);\n")
            script.append(f"        csVignette.property(\"Pin Highlights\").setValue(0);   // Preserve highlights\n")
            script.append(f"    }}\n")
            script.append(f"}}\n")
            
            # Add Sharpen effect
            script.append(f"var sharpen = adj.property(\"Effects\").addProperty(\"Sharpen\");\n")
            script.append(f"if (sharpen != null) {{\n")
            script.append(f"    sharpen.property(\"Sharpen Amount\").setValue(87);     // Sharpening intensity\n")
            script.append(f"}}\n")
            
            # Add Noise effect
            script.append(f"var noise = adj.property(\"Effects\").addProperty(\"Noise\");\n")
            script.append(f"if (noise != null) {{\n")
            script.append(f"    noise.property(\"Amount of Noise\").setValue(11);     // Noise intensity\n")
            script.append(f"    noise.property(\"Noise Type\").setValue(1);     // 0=Uniform, 1=Squared\n")
            script.append(f"    noise.property(\"Clipping\").setValue(1);     // 0=Clip, 1=Wrap\n")
            script.append(f"}}\n")

        # --- WIGGLE ADJUSTMENT LAYER (generated last for organic movement) ---
        if self.wiggle:
            script.append(f"var wiggleAdj = comp.layers.addSolid([1,1,1], \"WiggleAdjustment\", {self.comp_width}, {self.comp_height}, 1.0);\n")
            script.append(f"wiggleAdj.adjustmentLayer = true;\n")
            script.append(f"wiggleAdj.property('Transform').property('Position').setValue([{center_x}, {center_y}]);\n")
            script.append(f"wiggleAdj.parent = PlotAnchor;\n")
            script.append(f"wiggleAdj.moveToBeginning();\n")
            # Add Turbulent Displace effect
            script.append(f"var turbulent = wiggleAdj.property(\"Effects\").addProperty(\"Turbulent Displace\");\n")
            script.append(f"if (turbulent != null) {{\n")
            script.append(f"    turbulent.property(\"Displacement\").setValue(1);     // Displacement type\n")
            script.append(f"    turbulent.property(\"Amount\").setValue(3);     // Displacement amount\n")
            script.append(f"    turbulent.property(\"Size\").setValue(19);     // Turbulence size\n")
            script.append(f"    turbulent.property(\"Offset (Turbulence)\").setValue([{self.comp_width/2}, {self.comp_height/2}]);     // Center offset\n")
            script.append(f"    turbulent.property(\"Complexity\").setValue(10);     // Complexity\n")
            script.append(f"    turbulent.property(\"Evolution\").setValueAtTime(0, 0);     // Start evolution\n")
            script.append(f"    turbulent.property(\"Evolution\").setValueAtTime(0.4, -267);     // Keyframe 1\n")
            script.append(f"    turbulent.property(\"Evolution\").setValueAtTime(0.8, 76);     // Keyframe 2\n")
            script.append(f"    turbulent.property(\"Evolution\").setValueAtTime(1.2, -143);     // Keyframe 3\n")
            script.append(f"    turbulent.property(\"Evolution\").setValueAtTime(1.6, -313);     // Keyframe 4\n")
            script.append(f"    turbulent.property(\"Evolution\").setValueAtTime(2.0, 0);     // End evolution\n")
            script.append(f"    // Set all keyframes to hold interpolation\n")
            script.append(f"    turbulent.property(\"Evolution\").setInterpolationTypeAtKey(1, KeyframeInterpolationType.HOLD);\n")
            script.append(f"    turbulent.property(\"Evolution\").setInterpolationTypeAtKey(2, KeyframeInterpolationType.HOLD);\n")
            script.append(f"    turbulent.property(\"Evolution\").setInterpolationTypeAtKey(3, KeyframeInterpolationType.HOLD);\n")
            script.append(f"    turbulent.property(\"Evolution\").setInterpolationTypeAtKey(4, KeyframeInterpolationType.HOLD);\n")
            script.append(f"    turbulent.property(\"Evolution\").setInterpolationTypeAtKey(5, KeyframeInterpolationType.HOLD);\n")
            script.append(f"    turbulent.property(\"Evolution\").expression = \"loopOut('cycle')\";     // Loop animation\n")
            script.append(f"    turbulent.property(\"Pinning\").setValue(3);     // Pinning mode\n")
            script.append(f"    turbulent.property(\"Resize Layer\").setValue(0);     // Don't resize layer\n")
            script.append(f"    turbulent.property(\"Antialiasing for Best Quality\").setValue(1);     // Best quality\n")
            script.append(f"}}\n")

        script.append("app.endUndoGroup();\n")
        return "".join(script)

    def save(self, filename: str = "", folder_path: str = "./"):
        """
        Write the JSX script to a file for After Effects.
        """
        if filename == "":
            filename = f"AEGraph_{get_time()}.jsx"
        jsx = self._generate_jsx()
        with open(f"{folder_path.rstrip('/')}/{filename}", "w") as f:
            f.write(jsx)
        print(f"JSX script saved to {folder_path.rstrip('/')}/{filename}")
        return self

    def render(self, ae_version="Adobe After Effects 2025", folder_path: str = "./"):
        """
        Write and run the JSX script in After Effects (macOS only, requires AE scripting enabled).
        Now prints diagnostics and uses absolute path.
        """
        import os
        import subprocess
        filename = f"AEGraph_{get_time()}.jsx"
        self.save(filename, folder_path)
        abs_path = os.path.abspath(os.path.join(folder_path, filename))
        print(f"[AEGraph] Attempting to run JSX script: {abs_path}")
        apple_script = f'''
        tell application \"{ae_version}\"
            activate
            DoScriptFile \"{abs_path}\"
        end tell
        '''
        print("[AEGraph] AppleScript command:")
        print(apple_script)
        try:
            result = subprocess.run(["osascript", "-e", apple_script], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[AEGraph] Error running osascript: {result.stderr}")
            else:
                print(f"[AEGraph] Script sent to After Effects: {filename}")
        except Exception as e:
            print(f"[AEGraph] Exception during render: {e}")
        return self