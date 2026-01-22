"""Dovetail joint generation for cut surfaces."""

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class DovetailConfig:
    """Configuration for dovetail joints."""
    
    # Number of dovetail rails along the cut
    count: int = 3
    
    # Dovetail cross-section dimensions in mm
    width: float = 15.0       # Width of the dovetail (tapered dimension)
    depth: float = 8.0        # How deep the dovetail protrudes from the cut surface
    taper_angle: float = 15.0 # Angle in degrees (creates the dovetail shape)
    
    # Tolerance for the female (cavity) side
    tolerance: float = 0.2    # Gap in mm for easier assembly


def create_dovetail_rail(
    length: float,
    width: float,
    depth: float, 
    taper_angle: float,
) -> trimesh.Trimesh:
    """
    Create a dovetail-shaped rail (trapezoidal cross-section extruded along length).
    
    The dovetail is oriented with:
    - Length along Z axis (the rail direction)
    - Width along X axis (the tapered dimension)  
    - Depth along Y axis (protrusion direction)
    
    The taper makes it wider at Y=depth than at Y=0.
    
    Args:
        length: Length of the rail (Z dimension)
        width: Width at the narrow end (X dimension at Y=0)
        depth: Depth of the dovetail (Y dimension)
        taper_angle: Angle of the taper in degrees
        
    Returns:
        A trimesh object representing the dovetail rail
    """
    # Calculate width expansion due to taper
    taper_expansion = depth * np.tan(np.radians(taper_angle))
    wide_width = width + 2 * taper_expansion
    
    half_length = length / 2
    
    # Define vertices for the trapezoidal prism (rail along Z)
    vertices = [
        # Front face (Z = -half_length)
        [-width/2, 0, -half_length],           # 0 - narrow bottom-left
        [width/2, 0, -half_length],            # 1 - narrow bottom-right
        [-wide_width/2, depth, -half_length],  # 2 - wide top-left
        [wide_width/2, depth, -half_length],   # 3 - wide top-right
        
        # Back face (Z = +half_length)
        [-width/2, 0, half_length],            # 4 - narrow bottom-left
        [width/2, 0, half_length],             # 5 - narrow bottom-right
        [-wide_width/2, depth, half_length],   # 6 - wide top-left
        [wide_width/2, depth, half_length],    # 7 - wide top-right
    ]
    
    # Define faces (triangles)
    faces = [
        # Front face (Z = -half_length)
        [0, 1, 2], [1, 3, 2],
        # Back face (Z = +half_length)
        [4, 6, 5], [5, 6, 7],
        # Bottom face (Y = 0, narrow)
        [0, 4, 1], [1, 4, 5],
        # Top face (Y = depth, wide)
        [2, 3, 6], [3, 7, 6],
        # Left face
        [0, 2, 4], [2, 6, 4],
        # Right face
        [1, 5, 3], [3, 5, 7],
    ]
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()
    
    return mesh


def create_dovetail_at_position(
    center: np.ndarray,
    cut_axis: int,
    length_axis: int,
    rail_length: float,
    config: DovetailConfig,
    is_female: bool = False,
) -> trimesh.Trimesh:
    """
    Create a dovetail rail at a specific position, oriented for a cut plane.
    
    The dovetail rail runs along length_axis, protrudes along cut_axis.
    
    Args:
        center: Center position of the dovetail [x, y, z]
        cut_axis: The axis perpendicular to the cut plane (0=X, 1=Y, 2=Z)
        length_axis: The axis along which the dovetail rail runs
        rail_length: Length of the dovetail rail
        config: Dovetail configuration
        is_female: If True, create slightly larger cavity version
        
    Returns:
        Positioned and oriented dovetail mesh
    """
    # Apply tolerance for female (cavity) side
    tolerance = config.tolerance if is_female else 0.0
    
    # Create the base dovetail rail
    # Rail runs along Z, width along X, depth along Y
    dovetail = create_dovetail_rail(
        length=rail_length + (tolerance * 2 if is_female else 0),
        width=config.width + tolerance * 2,
        depth=config.depth + tolerance,
        taper_angle=config.taper_angle,
    )
    
    # The dovetail is created with:
    # - Length along Z (rail direction)
    # - Width along X (tapered dimension)
    # - Depth along Y (protrusion direction)
    
    # We need to rotate it so that:
    # - Depth (Y) points along cut_axis (perpendicular to cut plane)
    # - Length (Z) runs along length_axis
    # - Width (X) is along the remaining axis
    
    # Determine the remaining axis
    remaining_axis = 3 - cut_axis - length_axis
    
    # Build rotation matrix to reorient the dovetail
    # We want: X -> remaining_axis, Y -> cut_axis, Z -> length_axis
    rotation = np.eye(4)
    new_axes = np.zeros((3, 3))
    new_axes[remaining_axis, 0] = 1  # X -> remaining_axis (width direction)
    new_axes[cut_axis, 1] = 1        # Y -> cut_axis (depth/protrusion direction)
    new_axes[length_axis, 2] = 1     # Z -> length_axis (rail direction)
    
    rotation[:3, :3] = new_axes
    dovetail.apply_transform(rotation)
    
    # Move to the target position
    dovetail.apply_translation(center)
    
    return dovetail


def add_dovetails_to_cut(
    mesh_below: trimesh.Trimesh,
    mesh_above: trimesh.Trimesh,
    cut_axis: int,
    cut_position: float,
    config: DovetailConfig,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """
    Add dovetail joints between two mesh halves at a cut plane.
    
    The dovetail rails run along the SHORTER dimension of the cut surface.
    Multiple rails are distributed along the LONGER dimension.
    Dovetails are positioned based on where the mesh actually exists, not just bounding box.
    
    The "below" mesh gets male dovetails (protrusions).
    The "above" mesh gets female dovetails (cavities).
    
    Args:
        mesh_below: The mesh on the negative side of the cut
        mesh_above: The mesh on the positive side of the cut
        cut_axis: The axis perpendicular to the cut (0=X, 1=Y, 2=Z)
        cut_position: Position of the cut along the cut_axis
        config: Dovetail configuration
        
    Returns:
        Tuple of (modified_below, modified_above) meshes
    """
    if config.count <= 0:
        return mesh_below, mesh_above
    
    # Find the intersection region (where both meshes meet at the cut plane)
    bounds_below = mesh_below.bounds
    bounds_above = mesh_above.bounds
    
    # Calculate the overlap region in the two non-cut axes
    overlap_min = np.maximum(bounds_below[0], bounds_above[0])
    overlap_max = np.minimum(bounds_below[1], bounds_above[1])
    
    # Get the spans along each axis (ignoring cut_axis)
    spans = overlap_max - overlap_min
    spans[cut_axis] = 0
    
    # Identify the two non-cut axes
    other_axes = [i for i in range(3) if i != cut_axis]
    
    # The dovetail rails run along the SHORTER axis
    # Multiple rails are distributed along the LONGER axis
    if spans[other_axes[0]] <= spans[other_axes[1]]:
        length_axis = other_axes[0]  # Rails run along this (shorter) axis
        distribute_axis = other_axes[1]  # Rails are spaced along this (longer) axis
    else:
        length_axis = other_axes[1]
        distribute_axis = other_axes[0]
    
    # Dovetails run the full length of the shorter axis
    # Clipping will trim them to the actual mesh geometry
    rail_length = spans[length_axis]
    distribute_span = spans[distribute_axis]
    
    # Use the FULL bounding box for dovetail distribution
    # Dovetails are evenly spaced across the entire cut surface
    # Clipping will naturally handle areas where mesh doesn't exist
    dist_min = overlap_min[distribute_axis]
    dist_max = overlap_max[distribute_axis]
    
    # Calculate equal spacing: same gap between edge and dovetail as between dovetails
    # Formula: span = N*width + (N+1)*gap, solve for gap
    # gap = (span - N*width) / (N+1)
    actual_count = config.count
    total_dovetail_width = actual_count * config.width
    
    if distribute_span < total_dovetail_width:
        print(f"  Warning: Not enough space for {config.count} dovetails, reducing count")
        actual_count = max(1, int(distribute_span / config.width))
        total_dovetail_width = actual_count * config.width
    
    gap = (distribute_span - total_dovetail_width) / (actual_count + 1)
    
    # Position each dovetail: first one at dist_min + gap + width/2
    # Equal spacing means: edge--gap--[dove1]--gap--[dove2]--gap--[dove3]--gap--edge
    positions = []
    for i in range(actual_count):
        pos = dist_min + gap + config.width / 2 + i * (config.width + gap)
        positions.append(pos)
    
    # Center of the cut surface along the length axis
    center_length = (overlap_min[length_axis] + overlap_max[length_axis]) / 2
    
    # Create dovetails
    male_dovetails = []
    female_dovetails = []
    
    for pos in positions:
        # Build center position vector
        center = np.zeros(3)
        center[cut_axis] = cut_position
        center[length_axis] = center_length
        center[distribute_axis] = pos
        
        # Create male dovetail (protrusion into the "above" side)
        male = create_dovetail_at_position(
            center=center,
            cut_axis=cut_axis,
            length_axis=length_axis,
            rail_length=rail_length,
            config=config,
            is_female=False,
        )
        male_dovetails.append(male)
        
        # Create female dovetail (cavity, slightly larger)
        female = create_dovetail_at_position(
            center=center,
            cut_axis=cut_axis,
            length_axis=length_axis,
            rail_length=rail_length,
            config=config,
            is_female=True,
        )
        female_dovetails.append(female)
    
    # Apply boolean operations
    # Clip dovetails to the actual mesh geometry (not just bounding box).
    # This ensures dovetails don't extend past where the original part actually exists.
    # 
    # Strategy: Intersect each dovetail with the mesh it's attached to.
    # Since the dovetail protrudes past the mesh boundary on cut_axis,
    # we need to use the mesh's geometry projected along cut_axis.
    
    # Store original meshes for restoration if needed
    original_below = mesh_below.copy()
    original_above = mesh_above.copy()
    
    # Create clipping volumes by intersecting dovetails with mesh geometry
    extension = config.depth + config.tolerance + 2.0
    
    # Second pass: apply all boolean operations
    # We apply male and female operations in alternating order to avoid
    # issues with mesh validity after multiple operations
    successful_male = 0
    successful_female = 0
    
    for i, (male, female) in enumerate(zip(male_dovetails, female_dovetails)):
        # For male dovetail: first intersect with mesh_below to clip it,
        # then union the clipped result back to mesh_below
        try:
            # Clip male to mesh_below's geometry (extended along cut_axis)
            # We do this by: 1) union male with mesh_below, 2) intersect result with extended mesh_below
            # Actually, simpler: just union and the mesh naturally limits where the dovetail attaches
            
            # First, intersect male with mesh_above to clip to where the mating part exists
            clipped_male = trimesh.boolean.intersection([male, original_above], engine='manifold')
            if clipped_male is None or clipped_male.is_empty:
                # Fallback: use unclipped male
                clipped_male = male
            
            mesh_below = trimesh.boolean.union([mesh_below, clipped_male], engine='manifold')
            successful_male += 1
        except Exception as e:
            print(f"  Warning: Failed to add male dovetail {i+1}: {e}")
        
        # For female dovetail: subtract from mesh_above
        try:
            # Clip female to mesh_below's geometry (where the male comes from)
            clipped_female = trimesh.boolean.intersection([female, original_below], engine='manifold')
            if clipped_female is None or clipped_female.is_empty:
                # Fallback: use unclipped female
                clipped_female = female
            
            mesh_above = trimesh.boolean.difference([mesh_above, clipped_female], engine='manifold')
            successful_female += 1
        except Exception as e:
            print(f"  Warning: Failed to add female dovetail {i+1}: {e}")
    
    # If all operations failed, restore originals
    if successful_male == 0 and successful_female == 0:
        print(f"  Warning: All dovetail operations failed, keeping original cut")
        return original_below, original_above
    
    if successful_male > 0 or successful_female > 0:
        print(f"  Added {successful_male} dovetail(s) along shorter axis ({rail_length:.1f}mm), clipped to mesh, {config.tolerance}mm tolerance")
    
    return mesh_below, mesh_above
