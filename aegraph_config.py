"""Central default configuration for AEGraph.

This module is the single source of truth for AEGraph defaults. Any value that
is not explicitly passed to ``AEGraph(...)`` or its plotting methods falls back
to the values defined here. Edit the field defaults below to change behavior
globally, or mutate the module-level ``config`` object at runtime, e.g.::

    import aegraph_config
    aegraph_config.config.comp_width = 3840
    aegraph_config.config.graph_scale = 0.8
    aegraph_config.config.gradient_high = "p_orange"

You can also load a named theme from ``themes.json``::

    aegraph_config.apply_theme("Slate Report")

All AEGraph calls made afterward will pick up the new defaults.
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

from dotenv import load_dotenv

# Load per-user / per-machine settings from .env.user (next to this file) so any
# value tied to a specific person or computer (AE version, paths, etc.) lives
# outside the tracked source. Missing file is fine: every consumer falls back to
# the hardcoded defaults below.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.user"))


class _Unset:
    """Sentinel marking "argument not provided" (distinct from None/0/False).

    Using a dedicated sentinel lets AEGraph tell apart "caller passed None"
    (a meaningful value for things like xlim) from "caller passed nothing,
    so use the config default".
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "UNSET"

    def __bool__(self):
        return False


UNSET = _Unset()


# Default directory for generated .jsx files. Lives inside the aegraph package
# directory (next to this file) so scripts don't litter the current working
# directory. Override via ``aegraph_config.config.jsx_output_dir = "..."``.
_PKG_JSX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jsx_output")
# Honor JSX_OUTPUT_DIR from .env.user when set; otherwise use the package dir.
_DEFAULT_JSX_DIR = os.getenv("JSX_OUTPUT_DIR") or _PKG_JSX_DIR


@dataclass
class AEGraphConfig:
    """Mutable, global default settings consulted by AEGraph.

    Attributes:
        comp_width / comp_height: Default After Effects composition size.
        graph_scale: Graph (plot area) size as a fraction of the comp size
            when no explicit graph width/height is given.
        auto_limits: When True and no xlim/ylim is set, pad data limits.
        lim_factor: Total span multiplier for auto limits. 1.2 means the data
            range is expanded to 1.2x (10% breathing room on each side).
        auto_axis_titles: When True, derive axis labels from pandas column
            names (Series.name) when no explicit label is set.
        auto_title: When True, derive the plot title from the axis labels.
        auto_title_template: Format string for the auto title; receives
            ``x`` and ``y`` (the effective axis labels).
        scatter_animate: Default total scatter animation duration (seconds).
        scatter_point_anim_times: Default per-point animation duration
            (seconds) for scatter plots.
        text_animate: Default slide-in animation duration (seconds) for title,
            subtitle, axis labels, legend entries, and annotations.
        gradient_low / gradient_high: Default endpoint colors for scatter
            gradients (color names from COLOR_NAMES or RGB triplets).
        is_dark: Theme category flag set by ``apply_theme()`` (1 for dark,
            0 for light, None when no theme has been applied).
    """

    # After Effects application name used by render()/reset(). Defaults to the
    # AE_VERSION value in .env.user, falling back to the latest known release.
    ae_version: str = field(
        default_factory=lambda: os.getenv("AE_VERSION", "Adobe After Effects 2026")
    )

    # Composition + graph sizing
    comp_width: int = 1920*2
    comp_height: int = 1080*2

    graph_scale: float = 0.75

    # Default length (seconds) of a newly-created After Effects composition.
    # Renders auto-extend the comp beyond this when evolving plots or animated
    # view bounds need more time, so this is just the floor for short graphs.
    comp_duration: float = 30.0

    # Output: directory where ``save()``/``render()`` write generated .jsx files
    # when no explicit ``folder_path`` is passed. Defaults to a ``jsx_output``
    # folder inside the aegraph package directory. Set to "." to use the current
    # working directory, or any path you like.
    jsx_output_dir: str = _DEFAULT_JSX_DIR

    # Auto axis limits
    auto_limits: bool = True
    lim_factor: float = 1.2

    # Auto text (labels + title)
    auto_axis_titles: bool = True
    auto_title: bool = True
    auto_title_template: str = "{x} vs {y}"

    # Scatter animation defaults
    scatter_animate: float = 1.0
    scatter_point_anim_times: float = 0.5

    # Text slide-in animation duration (seconds) for title, subtitle, axis
    # labels, legend entries, and annotations. Overridable per-graph via the
    # AEGraph constructor's ``text_animate`` argument.
    text_animate: float = 1.5

    # Scatter marker outline defaults. The outline color is derived from each
    # marker fill color at render time.
    scatter_outline: bool = True
    scatter_outline_width: float = 1.0

    # When the graph uses an animated view window (`set_view_keyframes`),
    # whether scatter markers shrink away as soon as their data point falls
    # outside the *current* view window, instead of remaining visible past
    # the plot frame's edge. Has no effect without an animated view. Override
    # per-call via `scatter(..., clip_to_view=False)`.
    scatter_clip_to_view: bool = True

    # Line plot animation default (total duration in seconds).
    line_animate: float = 2.0

    # Default keyframe interpolation for the time-evolving plots
    # (``plot_evolving`` / ``scatter_evolving``). When True, keyframes are HOLD
    # (the graph snaps from one dataset to the next); when False they are LINEAR
    # (the graph morphs smoothly between datasets, no bezier easing).
    hold_keyframes: bool = False

    # Default seconds between consecutive frames for the evolving plots when no
    # explicit ``frame_times`` is supplied.
    frame_duration: float = 0.5

    # Per-bar animation duration default for histogram, bar_graph, and barh
    # (seconds). When the caller omits ``bar_anim_times``, each bar takes
    # this many seconds (bars overlap to fit within ``animate``). Pass
    # ``bar_anim_times=None`` explicitly to fall back to the legacy sequential
    # behavior where each bar gets ``animate / n_bars`` seconds.
    bar_anim_times: float = 0.5

    # Easy ease defaults. ``easy_ease`` applies eased interpolation to each
    # animation keyframe; ``ease_speed``/``ease_influence`` are the AE keyframe
    # ease percentages (speed 0, influence 33 == the classic "Easy Ease").
    easy_ease: bool = True
    ease_speed: float = 0
    ease_influence: float = 33

    # Meta easy ease defaults. Where ``easy_ease`` smooths an individual
    # element's animation, ``meta_easy_ease`` eases the *sequence* of element
    # entrances in a multi-element plot (scatter, histogram, bar_graph, barh,
    # population pyramids, quiver) so the staggered in-points follow an
    # easy-ease curve instead of marching in at a constant rate. The
    # speed/influence percentages mirror ``ease_speed``/``ease_influence``.
    meta_easy_ease: bool = True
    meta_ease_speed: float = 0
    meta_ease_influence: float = 33

    # Maximum point radius (px) when scatter `radius_size="dynamic_range"`.
    # Per-point radii are scaled proportionally so max(radii) equals this value.
    scatter_max_radius: float = 15.0

    # Scatter gradient endpoints
    gradient_low: str = "p_blue"
    gradient_high: str = "p_yellow"

    # Colorbar (cmap) drawn on the right when a scatter has a gradient
    show_cmap: bool = True
    cmap_steps: int = 64        # number of stacked color slices in the gradient strip
    cmap_width: int = 50        # strip width in graph-shape pixels
    cmap_gap: int = 60          # gap from plot's right edge to strip's left edge
    cmap_tick_count: int = 6    # target number of nice ticks along the strip
    # When a value-based gradient's data is all integers, the colorbar can
    # render as discrete color bands (one per integer level) instead of a
    # smooth gradient. Auto-detection only kicks in when the number of integer
    # levels is at most this cap (keeps the band stack readable); larger ranges
    # fall back to a smooth gradient unless ``discrete=True`` is forced.
    cmap_discrete_max_levels: int = 20

    # Visual theme defaults. These map onto ``AEGraph`` constructor args of the
    # same name. ``label_color`` and ``subtitle_color`` of ``None`` fall back to
    # ``ui_color`` at render time. Themes from ``themes.json`` overwrite these.
    full_bg: bool = True
    bg_color: Union[str, List[float], None] = "white"
    ui_color: Union[str, List[float]] = "black"
    grid_color: Union[str, List[float]] = "gray"
    grid_linewidth: float = 1.0
    label_color: Union[str, List[float], None] = None
    subtitle_color: Union[str, List[float], None] = None
    distress_texture: Optional[int] = None
    object_color_1: Optional[Union[str, List[float]]] = None  # alias for gradient_high
    object_color_2: Optional[Union[str, List[float]]] = None  # alias for gradient_low
    is_dark: Optional[int] = None  # 1 for dark themes, 0 for light themes

    # Per-role font defaults. Values are After Effects *PostScript* font names
    # (the string AE expects for ``TextDocument.font``), e.g. "Helvetica-Bold"
    # or "AvenirNext-DemiBold". Themes from ``themes.json`` overwrite these via
    # their ``fonts`` block; an AEGraph(...) call can override per role (or all
    # at once with ``font=``). The defaults below are the Helvetica family so a
    # bare config (no theme) renders with fonts that ship on macOS / AE.
    font_title: str = "Helvetica-Bold"
    font_subtitle: str = "Helvetica"
    font_label: str = "Helvetica-Bold"
    font_tick: str = "Helvetica"
    font_legend: str = "Helvetica"
    font_body: str = "Helvetica"

    # Global text-size multiplier. Applied to every font size in the chart
    # (ticks, axis labels, title, subtitle, legend, annotations, evolving text).
    # 1.0 = default sizes; 1.5 = 50% larger everywhere; 0.75 = 25% smaller.
    # Override per-chart by passing font_scale= to AEGraph(), or set globally:
    #   aegraph_config.config.font_scale = 1.4
    font_scale: float = 2

    # Global scatter radius multiplier. All scatter/scatter_evolving radii are
    # multiplied by this value after any dynamic_range rescaling. Useful when
    # changing comp resolution -- e.g. set to 2.0 when doubling to 4K so dots
    # stay the same visual size on screen.
    #   aegraph_config.config.scatter_radius_scale = 2.0
    scatter_radius_scale: float = 2.0

    # Global dash size scale factor. Applied to all dashed/dotted/dashdot line
    # styles (plot lines, grid lines). 1.0 = default lengths; 2.0 = longer
    # dashes with wider gaps; 0.5 = shorter/tighter dashes.
    # Override per-chart element by passing dash_size= to .plot() / .grid().
    #   aegraph_config.config.dash_size = 2.0
    dash_size: float = 2.0

    # ------------------------------------------------------------------
    # Experimental "film style" post-processing pass. Set ``film_style=True``
    # on an AEGraph/AEFigure/subplots(...) call to layer a documentary-style
    # treatment on top of the render: a paper-texture background, a
    # front-heavy push-in zoom, a looping light-leak overlay, a global
    # adjustment stack (temporal posterize + exposure flicker, vignette, soft
    # edge blur), and Roughen Edges + Multiply blending on every chart/text
    # layer. Every knob below can be overridden globally here, or per-graph /
    # per-figure via ``.film_style_parameters(...)`` (see ``AEGraph`` /
    # ``AEFigure``) -- pass the field name *without* the ``film_style_``
    # prefix, e.g. ``g.film_style_parameters(roughen_border=1.0)``.
    # ------------------------------------------------------------------
    film_style: bool = False

    # --- Per-element pass: Roughen Edges -------------------------------
    # ``element_roughen_kinds`` controls Roughen Edges per element kind.
    # Omit a kind to use ``roughen_border``; ``True`` also uses
    # ``roughen_border``; ``False`` or ``0`` turns roughen off; a positive
    # number sets a custom border for just that kind. Recognized kinds:
    # "scatter", "line", "bar" (incl. histogram/stacked/barh), "pie",
    # "heatmap", "quiver", "annotation", "evolving_text", "grid", "tick",
    # "errorbar", "refline", "band", "colorbar", "text" (titles/labels/
    # legend/ticks), and "other" (anything unmatched). Scatter is excluded
    # from Roughen Edges by default -- the jitter reads as "warped" on
    # small dots.
    film_style_element_roughen: bool = True
    film_style_element_roughen_kinds: dict = field(default_factory=lambda: {"scatter": False})
    film_style_roughen_edge_type: int = 1          # 0=Corner, 1=Bezier, 2=Bezier Corner Peaks
    film_style_roughen_edge_color: List[float] = field(
        default_factory=lambda: [0.6, 0.2, 0.0, 1.0]
    )
    film_style_roughen_border: float = 1.75
    film_style_roughen_sharpness: float = 1.24
    film_style_roughen_scale: float = 100.0

    # --- Per-element pass: Gaussian Blur (off by default -- it muddies
    # small text/markers) and Multiply blending ------------------------
    film_style_element_blur: bool = False
    film_style_element_blur_kinds: dict = field(default_factory=dict)
    film_style_element_blur_amount: float = 2.5
    film_style_element_multiply: bool = True
    film_style_element_multiply_kinds: dict = field(default_factory=dict)

    # --- Paper texture background. File lives in ``film_elements/`` next
    # to ``aegraph.py`` (sibling of ``distress_textures/``). -------------
    film_style_paper: bool = True
    film_style_paper_file: str = "Texturelabs_Paper_378XL.jpg"
    film_style_paper_tint_black: List[float] = field(
        default_factory=lambda: [0.48663830757141, 0.51876533031464, 0.44195976853371, 1.0]
    )
    film_style_paper_tint_white: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 0.0]
    )

    # --- Looping light-leak video overlay (also in ``film_elements/``). -
    film_style_light_leak: bool = True
    film_style_light_leak_file: str = "Overlay Mode Screan_01.1.mp4"
    film_style_light_leak_opacity: float = 40.0
    film_style_light_leak_blend: str = "OVERLAY"

    # --- Global adjustment layer: temporal posterize + exposure wiggle --
    # ``temporal=False`` disables the whole layer; ``posterize_time=False``
    # keeps exposure flicker but skips Posterize Time.
    film_style_temporal: bool = True
    film_style_posterize_time: bool = True
    film_style_posterize_fps: float = 12.0
    film_style_exposure_wiggle: str = "wiggle(24, 0.05)"

    # --- Global adjustment layer: Lumetri vignette ----------------------
    film_style_vignette: bool = True
    film_style_vignette_amount: float = -0.82
    film_style_vignette_midpoint: float = 50.0
    film_style_vignette_roundness: float = 0.0
    film_style_vignette_feather: float = 50.0

    # --- Global adjustment layer: soft edge blur ------------------------
    # An inverted, feathered rectangular mask keeps the blur confined to a
    # thin band near the true frame edge -- the center stays sharp.
    # ``margin``/``feather`` are fractions of the comp's shorter side.
    film_style_edge_blur: bool = True
    film_style_edge_blur_amount: float = 25.0
    film_style_edge_blur_margin: float = 0.04
    film_style_edge_blur_feather: float = 0.06

    # --- Front-heavy push-in zoom applied to the plot anchor. Scale goes
    # from ``film_style_zoom_start`` to ``film_style_zoom_end`` (fractions
    # of 100%) over ``film_style_zoom_duration`` seconds, eased so most of
    # the motion happens early and settles into a long, gentle deceleration.
    film_style_zoom: bool = True
    film_style_zoom_start: float = 0.87
    film_style_zoom_end: float = 1.0
    # Multiplier on the push-in distance (``zoom_end - zoom_start``), applied
    # around the fixed ``zoom_end`` (so the graph still settles at its intended
    # framing). 1.0 = as configured; 2.0 = twice the push-in; 0.0 = no zoom.
    # Handy for dialing the effect up/down without recomputing zoom_start.
    film_style_zoom_amount: float = 1.0
    film_style_zoom_duration: float = 5.05
    film_style_zoom_ease_out_influence: float = 20.0
    film_style_zoom_ease_in_influence: float = 85.0


config = AEGraphConfig()


# Theme applied at import so any ``AEGraph(...)`` created without an explicit
# ``theme=`` inherits it. Set to a theme name from ``themes.json`` (or to
# ``None`` to keep the bare dataclass defaults above).
DEFAULT_THEME: Optional[str] = "Slate Report"


def _themes_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "themes.json")


def _load_themes() -> List[dict]:
    """Load and return the list of theme dicts from ``themes.json``."""
    with open(_themes_path(), "r") as f:
        data = json.load(f)
    return list(data.get("themes", []))


def available_themes() -> List[str]:
    """Return the list of theme names defined in ``themes.json``."""
    return [t["name"] for t in _load_themes()]


def get_theme(name: str) -> dict:
    """Look up a theme dict by name (case-insensitive). Raises if not found."""
    target = (name or "").strip().lower()
    for t in _load_themes():
        if str(t.get("name", "")).strip().lower() == target:
            return t
    names = ", ".join(available_themes())
    raise ValueError(f"Unknown theme {name!r}. Available themes: {names}")


def apply_theme(name: str) -> dict:
    """Apply a named theme to the module-level ``config``.

    Updates ``bg_color``, ``ui_color``, ``grid_color``, ``label_color``,
    ``subtitle_color``, ``distress_texture``, and the gradient endpoints
    (``gradient_low``/``gradient_high``) so any subsequent ``AEGraph(...)``
    instance and ``scatter(...)`` call inherits the theme. Color values are
    raw RGB triplets (0-255) compatible with AEGraph's ``color_to_js`` helper.
    Returns the resolved theme dict.
    """
    theme = get_theme(name)
    config.bg_color = theme["bg"]
    config.ui_color = theme["ui"]
    config.grid_color = theme["grid"]
    config.label_color = theme.get("label")
    config.subtitle_color = theme.get("subtitle")
    config.distress_texture = theme.get("distress_texture")
    config.object_color_1 = theme.get("object_color_1")
    config.object_color_2 = theme.get("object_color_2")
    config.is_dark = theme.get("is_dark")
    if theme.get("object_color_2") is not None:
        config.gradient_low = theme["object_color_2"]
    if theme.get("object_color_1") is not None:
        config.gradient_high = theme["object_color_1"]
    # Per-role fonts. A theme's ``fonts`` block is a dict keyed by role
    # (title/subtitle/label/tick/legend/body); any missing role falls back to
    # the theme's ``body`` font, then to the existing config default.
    fonts = theme.get("fonts") or {}
    _body = fonts.get("body")
    for role in ("title", "subtitle", "label", "tick", "legend", "body"):
        val = fonts.get(role, _body)
        if val:
            setattr(config, f"font_{role}", val)
    config.theme = theme.get("name")
    return theme


def reset_theme() -> None:
    """Reset theme-affecting config fields to the default state.

    When ``DEFAULT_THEME`` is set, this restores that theme; otherwise it
    restores the bare dataclass defaults.
    """
    defaults = AEGraphConfig()
    config.bg_color = defaults.bg_color
    config.ui_color = defaults.ui_color
    config.grid_color = defaults.grid_color
    config.label_color = defaults.label_color
    config.subtitle_color = defaults.subtitle_color
    config.distress_texture = defaults.distress_texture
    config.object_color_1 = defaults.object_color_1
    config.object_color_2 = defaults.object_color_2
    config.is_dark = defaults.is_dark
    config.gradient_low = defaults.gradient_low
    config.gradient_high = defaults.gradient_high
    config.font_title = defaults.font_title
    config.font_subtitle = defaults.font_subtitle
    config.font_label = defaults.font_label
    config.font_tick = defaults.font_tick
    config.font_legend = defaults.font_legend
    config.font_body = defaults.font_body
    config.theme = defaults.theme
    if DEFAULT_THEME is not None:
        apply_theme(DEFAULT_THEME)


# Names of every theme defined in ``themes.json``, snapshotted at import time.
# Useful for IDE autocomplete, validation, or just printing the catalog::
#
#     import aegraph_config
#     for name in aegraph_config.THEMES:
#         print(name)
#
# If you edit ``themes.json`` at runtime, call ``available_themes()`` instead
# for a fresh list.
THEMES: List[str] = available_themes()


# Apply the default theme so AEGraph instances created without an explicit
# ``theme=`` argument inherit it. Guarded so a missing/renamed theme can't break
# import.
if DEFAULT_THEME is not None:
    try:
        apply_theme(DEFAULT_THEME)
    except ValueError:
        pass
