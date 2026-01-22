"""Mesh loading and analysis functions."""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import trimesh

from .utils import format_dimensions


class MeshAnalysis(NamedTuple):
    """Analysis results for a loaded mesh."""
    
    mesh: trimesh.Trimesh
    bounds: np.ndarray  # Shape (2, 3): [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    extents: np.ndarray  # Shape (3,): [width, depth, height]
    volume: float
    center: np.ndarray  # Shape (3,): center of bounding box


def load_mesh(path: Path | str) -> trimesh.Trimesh:
    """
    Load an STL file and return a trimesh object.
    
    Args:
        path: Path to the STL file
        
    Returns:
        A trimesh.Trimesh object
        
    Raises:
        ValueError: If the file cannot be loaded or is not a valid mesh
    """
    path = Path(path)
    
    if not path.exists():
        raise ValueError(f"File not found: {path}")
    
    if path.suffix.lower() != ".stl":
        raise ValueError(f"Expected .stl file, got: {path.suffix}")
    
    mesh = trimesh.load_mesh(str(path), force="mesh")
    
    if not isinstance(mesh, trimesh.Trimesh):
        # If it's a Scene with multiple meshes, combine them
        if isinstance(mesh, trimesh.Scene):
            meshes = list(mesh.geometry.values())
            if not meshes:
                raise ValueError(f"No geometry found in file: {path}")
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise ValueError(f"Could not load mesh from file: {path}")
    
    if mesh.is_empty:
        raise ValueError(f"Mesh is empty: {path}")
    
    return mesh


def analyze_mesh(mesh: trimesh.Trimesh) -> MeshAnalysis:
    """
    Analyze a mesh and return its properties.
    
    Args:
        mesh: A trimesh object
        
    Returns:
        MeshAnalysis containing bounds, extents, volume, and center
    """
    bounds = mesh.bounds  # Shape (2, 3)
    extents = mesh.extents  # Shape (3,)
    
    # Calculate volume (may not be accurate for non-watertight meshes)
    try:
        volume = mesh.volume if mesh.is_watertight else mesh.convex_hull.volume
    except Exception:
        # Fallback to bounding box volume estimate
        volume = np.prod(extents)
    
    center = mesh.bounds.mean(axis=0)
    
    return MeshAnalysis(
        mesh=mesh,
        bounds=bounds,
        extents=extents,
        volume=volume,
        center=center,
    )


def fits_in_build_volume(
    extents: np.ndarray, 
    build_volume: tuple[float, float, float],
    margin: float = 0.0,
) -> bool:
    """
    Check if mesh extents fit within the build volume.
    
    Args:
        extents: Mesh extents (width, depth, height)
        build_volume: Maximum build dimensions (x, y, z)
        margin: Safety margin to subtract from build volume
        
    Returns:
        True if the mesh fits, False otherwise
    """
    available = np.array(build_volume) - margin * 2
    return all(extents <= available)


def get_required_splits(
    extents: np.ndarray,
    build_volume: tuple[float, float, float],
    margin: float = 0.0,
) -> list[int]:
    """
    Calculate the number of splits needed along each axis.
    
    Args:
        extents: Mesh extents (width, depth, height)
        build_volume: Maximum build dimensions (x, y, z)
        margin: Safety margin to subtract from build volume
        
    Returns:
        List of [splits_x, splits_y, splits_z]
    """
    available = np.array(build_volume) - margin * 2
    splits = np.ceil(extents / available).astype(int)
    return splits.tolist()


def print_mesh_info(analysis: MeshAnalysis, build_volume: tuple[float, float, float], margin: float = 0.0) -> None:
    """Print information about the mesh."""
    print(f"\nMesh Analysis:")
    print(f"  Dimensions: {format_dimensions(analysis.extents)}")
    print(f"  Volume: {analysis.volume:.1f} mm³")
    print(f"  Watertight: {analysis.mesh.is_watertight}")
    print(f"  Triangles: {len(analysis.mesh.faces)}")
    
    fits = fits_in_build_volume(analysis.extents, build_volume, margin)
    if fits:
        print(f"\n  ✓ Mesh fits within build volume!")
    else:
        splits = get_required_splits(analysis.extents, build_volume, margin)
        total_parts = splits[0] * splits[1] * splits[2]
        print(f"\n  ✗ Mesh exceeds build volume")
        print(f"  Required splits: {splits[0]} x {splits[1]} x {splits[2]} = {total_parts} parts")
