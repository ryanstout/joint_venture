"""Utility functions for STL splitting."""

import numpy as np


def axis_index(axis: str) -> int:
    """Convert axis name to index."""
    return {"x": 0, "y": 1, "z": 2}[axis.lower()]


def axis_name(index: int) -> str:
    """Convert axis index to name."""
    return ["X", "Y", "Z"][index]


def create_plane_normal(axis: int) -> np.ndarray:
    """Create a unit normal vector for the given axis."""
    normal = np.zeros(3)
    normal[axis] = 1.0
    return normal


def create_plane_origin(axis: int, position: float) -> np.ndarray:
    """Create an origin point for a plane at the given position on the axis."""
    origin = np.zeros(3)
    origin[axis] = position
    return origin


def format_dimensions(dims: np.ndarray) -> str:
    """Format dimensions as a readable string."""
    return f"{dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm"
