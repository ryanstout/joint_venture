"""Core mesh splitting functionality."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import trimesh

from .dovetail import DovetailConfig, add_dovetails_to_cut
from .mesh_loader import MeshAnalysis, analyze_mesh, get_required_splits
from .utils import axis_name, create_plane_normal, create_plane_origin


@dataclass
class CutPlane:
    """Represents a cutting plane."""
    
    axis: int  # 0=X, 1=Y, 2=Z
    position: float  # Position along the axis
    
    @property
    def origin(self) -> np.ndarray:
        """Get the plane origin point."""
        return create_plane_origin(self.axis, self.position)
    
    @property
    def normal(self) -> np.ndarray:
        """Get the plane normal vector."""
        return create_plane_normal(self.axis)
    
    def __repr__(self) -> str:
        return f"CutPlane({axis_name(self.axis)}={self.position:.2f})"


@dataclass
class SplitResult:
    """Result of splitting a mesh."""
    
    parts: list[trimesh.Trimesh]
    cut_planes: list[CutPlane]
    
    @property
    def num_parts(self) -> int:
        return len(self.parts)


def calculate_cut_positions(
    mesh: trimesh.Trimesh,
    axis: int,
    num_cuts: int,
    samples: int = 50,
) -> list[float]:
    """
    Calculate optimal cut positions along an axis.
    
    Tries to find positions that:
    1. Divide the mesh into roughly equal volume parts
    2. Minimize cross-section area at cut locations (safer cuts)
    
    Args:
        mesh: The mesh to analyze
        axis: Axis index (0=X, 1=Y, 2=Z)
        num_cuts: Number of cuts to make (results in num_cuts + 1 parts)
        samples: Number of positions to sample for analysis
        
    Returns:
        List of cut positions along the axis
    """
    if num_cuts <= 0:
        return []
    
    bounds = mesh.bounds
    min_pos = bounds[0][axis]
    max_pos = bounds[1][axis]
    extent = max_pos - min_pos
    
    # Simple approach: evenly spaced cuts
    # This works well for most cases and is predictable
    positions = []
    step = extent / (num_cuts + 1)
    
    for i in range(1, num_cuts + 1):
        pos = min_pos + step * i
        positions.append(pos)
    
    return positions


def calculate_cut_planes(
    analysis: MeshAnalysis,
    build_volume: tuple[float, float, float],
    margin: float = 0.0,
) -> list[CutPlane]:
    """
    Calculate all cut planes needed to split the mesh.
    
    Args:
        analysis: Mesh analysis results
        build_volume: Maximum build dimensions (x, y, z)
        margin: Safety margin
        
    Returns:
        List of CutPlane objects
    """
    splits = get_required_splits(analysis.extents, build_volume, margin)
    planes = []
    
    for axis in range(3):
        num_parts = splits[axis]
        if num_parts > 1:
            # Number of cuts = number of parts - 1
            num_cuts = num_parts - 1
            positions = calculate_cut_positions(
                analysis.mesh, 
                axis, 
                num_cuts,
            )
            for pos in positions:
                planes.append(CutPlane(axis=axis, position=pos))
    
    return planes


def slice_mesh_at_plane(
    mesh: trimesh.Trimesh,
    plane: CutPlane,
    cap: bool = True,
) -> tuple[trimesh.Trimesh | None, trimesh.Trimesh | None]:
    """
    Slice a mesh at a plane, returning two halves.
    
    Args:
        mesh: The mesh to slice
        plane: The cutting plane
        cap: Whether to cap the cut surfaces (make watertight)
        
    Returns:
        Tuple of (below_mesh, above_mesh), either can be None if empty
    """
    try:
        # Use trimesh's slice_plane to cut the mesh
        # Returns the portion of the mesh on the positive side of the plane
        above = mesh.slice_plane(
            plane_origin=plane.origin,
            plane_normal=plane.normal,
            cap=cap,
        )
        
        # Get the other half by flipping the normal
        below = mesh.slice_plane(
            plane_origin=plane.origin,
            plane_normal=-plane.normal,
            cap=cap,
        )
        
        # Check if results are valid
        above = above if above is not None and not above.is_empty else None
        below = below if below is not None and not below.is_empty else None
        
        return below, above
        
    except Exception as e:
        print(f"  Warning: Failed to slice at {plane}: {e}")
        return None, None


def get_connected_components(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    """
    Split a mesh into its connected components.
    
    Args:
        mesh: The mesh to analyze
        
    Returns:
        List of connected component meshes
    """
    try:
        components = mesh.split(only_watertight=False)
        if isinstance(components, list):
            return components
        return [components]
    except Exception:
        return [mesh]


def filter_small_components(
    meshes: list[trimesh.Trimesh],
    min_volume_ratio: float = 0.01,
) -> list[trimesh.Trimesh]:
    """
    Filter out very small mesh fragments.
    
    Args:
        meshes: List of mesh parts
        min_volume_ratio: Minimum volume as ratio of largest part
        
    Returns:
        Filtered list with small parts removed
    """
    if not meshes:
        return meshes
    
    # Calculate volumes (use convex hull for non-watertight)
    volumes = []
    for m in meshes:
        try:
            vol = m.volume if m.is_watertight else m.convex_hull.volume
        except Exception:
            vol = np.prod(m.extents)
        volumes.append(vol)
    
    max_volume = max(volumes)
    min_volume = max_volume * min_volume_ratio
    
    filtered = [m for m, v in zip(meshes, volumes) if v >= min_volume]
    
    removed = len(meshes) - len(filtered)
    if removed > 0:
        print(f"  Removed {removed} small fragments")
    
    return filtered


def split_mesh(
    mesh: trimesh.Trimesh,
    planes: list[CutPlane],
    validate_connectivity: bool = True,
    min_volume_ratio: float = 0.01,
    dovetail_config: DovetailConfig | None = None,
) -> list[trimesh.Trimesh]:
    """
    Split a mesh using the given cut planes.
    
    Dovetails are added AFTER all cuts are made, ensuring each final
    cut surface gets the correct number of dovetails.
    
    Args:
        mesh: The mesh to split
        planes: List of cutting planes
        validate_connectivity: Whether to check for disconnected parts
        min_volume_ratio: Minimum volume ratio to keep a part
        dovetail_config: Optional dovetail configuration for joints
        
    Returns:
        List of mesh parts
    """
    if not planes:
        return [mesh]
    
    parts = [mesh]
    
    # Phase 1: Make all cuts WITHOUT dovetails
    for i, plane in enumerate(planes):
        print(f"  Applying cut {i + 1}/{len(planes)}: {plane}")
        
        new_parts = []
        for part in parts:
            # Check if this part needs to be cut by this plane
            bounds = part.bounds
            if bounds[0][plane.axis] < plane.position < bounds[1][plane.axis]:
                # Part spans the cut plane, slice it (no dovetails yet)
                below, above = slice_mesh_at_plane(part, plane, cap=True)
                
                if below is not None:
                    new_parts.append(below)
                if above is not None:
                    new_parts.append(above)
                
                # If slicing failed, keep the original part
                if below is None and above is None:
                    new_parts.append(part)
            else:
                # Part doesn't span this plane, keep as is
                new_parts.append(part)
        
        parts = new_parts
    
    # Validate connectivity and filter small parts BEFORE adding dovetails
    if validate_connectivity:
        all_parts = []
        for part in parts:
            components = get_connected_components(part)
            all_parts.extend(components)
        parts = all_parts
    
    # Filter out tiny fragments
    parts = filter_small_components(parts, min_volume_ratio)
    
    # Phase 2: Add dovetails between adjacent parts
    if dovetail_config is not None and dovetail_config.count > 0:
        print(f"\n  Adding dovetails between adjacent parts...")
        parts = add_dovetails_to_parts(parts, planes, dovetail_config)
    
    return parts


def add_dovetails_to_parts(
    parts: list[trimesh.Trimesh],
    planes: list[CutPlane],
    config: DovetailConfig,
) -> list[trimesh.Trimesh]:
    """
    Add dovetails between adjacent parts that share a cut surface.
    
    For each cut plane, find pairs of parts where one part's max bound
    is at the cut position and another part's min bound is at the cut position.
    Add dovetails between these pairs.
    
    Args:
        parts: List of mesh parts after cutting
        planes: List of cut planes used
        config: Dovetail configuration
        
    Returns:
        List of parts with dovetails added
    """
    # Tolerance for detecting parts at a cut plane
    tolerance = 0.5
    
    # Work with a mutable list
    parts = list(parts)
    
    for plane in planes:
        axis = plane.axis
        position = plane.position
        
        # Find parts that are "below" the cut (their max is at the cut position)
        # and parts that are "above" the cut (their min is at the cut position)
        below_indices = []
        above_indices = []
        
        for i, part in enumerate(parts):
            bounds = part.bounds
            # Check if part's max bound is at the cut position (below/negative side)
            if abs(bounds[1][axis] - position) < tolerance:
                below_indices.append(i)
            # Check if part's min bound is at the cut position (above/positive side)
            elif abs(bounds[0][axis] - position) < tolerance:
                above_indices.append(i)
        
        # For each below part, find an adjacent above part and add dovetails
        for below_idx in below_indices:
            below_part = parts[below_idx]
            below_bounds = below_part.bounds
            
            # Find an above part that overlaps with this below part in the other axes
            best_above_idx = None
            best_overlap = 0
            
            for above_idx in above_indices:
                above_part = parts[above_idx]
                above_bounds = above_part.bounds
                
                # Calculate overlap in non-cut axes
                overlap = 1.0
                for other_axis in range(3):
                    if other_axis == axis:
                        continue
                    overlap_min = max(below_bounds[0][other_axis], above_bounds[0][other_axis])
                    overlap_max = min(below_bounds[1][other_axis], above_bounds[1][other_axis])
                    if overlap_max > overlap_min:
                        overlap *= (overlap_max - overlap_min)
                    else:
                        overlap = 0
                        break
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_above_idx = above_idx
            
            if best_above_idx is not None and best_overlap > 0:
                above_part = parts[best_above_idx]
                
                # Add dovetails between these two parts
                try:
                    new_below, new_above = add_dovetails_to_cut(
                        mesh_below=below_part,
                        mesh_above=above_part,
                        cut_axis=axis,
                        cut_position=position,
                        config=config,
                    )
                    parts[below_idx] = new_below
                    parts[best_above_idx] = new_above
                except Exception as e:
                    print(f"    Warning: Failed to add dovetails at {plane}: {e}")
    
    return parts


def split_stl(
    input_path: Path | str,
    build_volume: tuple[float, float, float],
    margin: float = 0.0,
    min_volume_ratio: float = 0.01,
    dovetail_config: DovetailConfig | None = None,
) -> SplitResult:
    """
    Main function to split an STL file into printable parts.
    
    Args:
        input_path: Path to input STL file
        build_volume: Maximum build dimensions (x, y, z) in mm
        margin: Safety margin in mm
        min_volume_ratio: Minimum volume ratio to keep a part
        dovetail_config: Optional dovetail configuration for joints
        
    Returns:
        SplitResult containing the parts and cut planes used
    """
    from .mesh_loader import load_mesh
    
    print(f"\nLoading mesh: {input_path}")
    mesh = load_mesh(input_path)
    analysis = analyze_mesh(mesh)
    
    print(f"  Dimensions: {analysis.extents[0]:.1f} x {analysis.extents[1]:.1f} x {analysis.extents[2]:.1f} mm")
    print(f"  Triangles: {len(mesh.faces)}")
    
    # Calculate cut planes
    print("\nCalculating cut planes...")
    planes = calculate_cut_planes(analysis, build_volume, margin)
    
    if not planes:
        print("  No cuts needed - mesh fits in build volume!")
        return SplitResult(parts=[mesh], cut_planes=[])
    
    print(f"  Need {len(planes)} cuts to fit build volume")
    
    if dovetail_config is not None and dovetail_config.count > 0:
        print(f"  Dovetails: {dovetail_config.count} per cut, {dovetail_config.tolerance}mm tolerance")
    
    # Perform the splits
    print("\nSplitting mesh...")
    parts = split_mesh(
        mesh, 
        planes, 
        validate_connectivity=True,
        min_volume_ratio=min_volume_ratio,
        dovetail_config=dovetail_config,
    )
    
    print(f"\nGenerated {len(parts)} parts")
    
    return SplitResult(parts=parts, cut_planes=planes)


def export_parts(
    parts: list[trimesh.Trimesh],
    output_dir: Path | str,
    base_name: str = "part",
) -> list[Path]:
    """
    Export mesh parts to STL files.
    
    Args:
        parts: List of mesh parts
        output_dir: Directory to save files
        base_name: Base name for output files
        
    Returns:
        List of paths to exported files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    for i, part in enumerate(parts):
        filename = f"{base_name}_{i + 1:03d}.stl"
        filepath = output_dir / filename
        
        part.export(str(filepath), file_type="stl")
        
        # Get part info
        try:
            volume = part.volume if part.is_watertight else part.convex_hull.volume
        except Exception:
            volume = np.prod(part.extents)
        
        print(f"  Saved: {filename} ({part.extents[0]:.1f} x {part.extents[1]:.1f} x {part.extents[2]:.1f} mm, {volume:.0f} mm³)")
        paths.append(filepath)
    
    return paths
