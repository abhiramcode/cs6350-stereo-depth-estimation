"""Visualization package for stereo depth estimation.

Provides utilities for rendering side-by-side comparison grids, 3-column
assignment grids (Estimated Depth, Left View, Right View), error heatmaps,
and qualitative failure diagnostics.
"""

from src.visualization.grids import (
    render_3column_grid,
    render_diagnostic_grid,
    render_multimethod_grid,
    compute_error_heatmap,
    colorize_disparity,
    colorize_depth,
)

__all__ = [
    "render_3column_grid",
    "render_diagnostic_grid",
    "render_multimethod_grid",
    "compute_error_heatmap",
    "colorize_disparity",
    "colorize_depth",
]

