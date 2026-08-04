#!/usr/bin/env python3
"""Build & render every time-evolving AEGraph example in one go.

Each "key" maps to a standalone file in ``examples/`` that uses the
time-evolution API (``plot_evolving`` / ``scatter_evolving``). This runner
executes them for you, either rendering each in After Effects or — with the
``save`` flag — writing only the ``.jsx`` files (no After Effects needed) into
``./animated_example_jsx/``.

Usage::

    # Render every evolving example in After Effects, one at a time:
    python scripts/generate_animated_examples.py

    # Write the .jsx files into ./animated_example_jsx/ instead of rendering:
    python scripts/generate_animated_examples.py save

    # Only build specific ones (render or save):
    python scripts/generate_animated_examples.py lorenz kepler
    python scripts/generate_animated_examples.py save gapminder gas-diffusion

    # List the available keys and exit:
    python scripts/generate_animated_examples.py --list
"""

import os
import runpy
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# Examples resolve the package via the AEGRAPH_PATH env var (set in .env.user);
# default it to the repo root so the runner works out of the box.
os.environ.setdefault("AEGRAPH_PATH", REPO_ROOT)

import aegraph  # noqa: E402  (import after sys.path / env setup)

EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")

# key -> example filename (in examples/). Keys double as the .jsx stem in save mode.
EXAMPLES = {

}


def _run_example(path):
    """Execute an example file as if it were ``python examples/foo.py``."""
    runpy.run_path(path, run_name="__main__")


def main(argv):
    if "--list" in argv or "-l" in argv:
        print("Available keys:")
        for key, fname in EXAMPLES.items():
            print(f"  {key:20s} -> examples/{fname}")
        return

    save_only = False
    keys = []
    for arg in argv:
        if arg == "save":
            save_only = True
        else:
            keys.append(arg)

    if not keys:
        keys = list(EXAMPLES)

    unknown = [k for k in keys if k not in EXAMPLES]
    if unknown:
        raise SystemExit(
            f"Unknown key(s): {', '.join(unknown)}\n"
            f"Available: {', '.join(EXAMPLES)}\n"
            f"(run with --list for the file mapping)"
        )

    out_dir = os.path.join(os.getcwd(), "animated_example_jsx")
    if save_only:
        os.makedirs(out_dir, exist_ok=True)
        # Redirect every render() call in the examples to a named save() so we
        # can generate the .jsx files without launching After Effects.
        current = {"stem": None}
        original_render = aegraph.AEGraph.render

        def render_to_save(self, *args, **kwargs):
            return self.save(f"{current['stem']}.jsx", out_dir)

        aegraph.AEGraph.render = render_to_save

    try:
        for key in keys:
            path = os.path.join(EXAMPLES_DIR, EXAMPLES[key])
            if save_only:
                current["stem"] = key
                print(f"[animated] saving {key} -> {EXAMPLES[key]}")
            else:
                print(f"[animated] rendering {key} -> {EXAMPLES[key]}")
            _run_example(path)
    finally:
        if save_only:
            aegraph.AEGraph.render = original_render

    if save_only:
        print(f"[animated] wrote {len(keys)} .jsx file(s) to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
