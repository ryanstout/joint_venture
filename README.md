# STL Splitter

Split large STL files into multiple printable parts that fit your 3D printer's build volume.

## Features

- Automatically calculates optimal cut planes based on build volume
- Caps cut surfaces to maintain watertight meshes
- **Dovetail joints** for easy assembly of split parts
- Filters out tiny fragments from cuts
- Validates mesh connectivity after splitting
- Default settings optimized for Bambu X1C (256 x 256 x 256 mm)

## Installation

```bash
# Using uv (recommended)
uv sync

# Install as CLI tool
uv pip install -e .
```

## Usage

```bash
# Basic usage - splits model to fit Bambu X1C build volume
stl-split large_model.stl

# Specify output directory
stl-split large_model.stl --output-dir ./parts

# Custom build volume (in mm)
stl-split large_model.stl --build-volume 200 200 200

# Adjust safety margin (default: 5mm)
stl-split large_model.stl --margin 10

# Add dovetail joints for assembly (2 dovetails per cut)
stl-split large_model.stl --dovetails 2

# Dovetails with custom tolerance (for looser/tighter fit)
stl-split large_model.stl --dovetails 3 --dovetail-tolerance 0.3

# Just analyze the mesh without splitting
stl-split large_model.stl --info-only
```

## Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output-dir` | `-o` | Directory for output parts | `./parts` |
| `--build-volume` | `-b` | Build volume X Y Z in mm | `256 256 256` |
| `--margin` | `-m` | Safety margin in mm | `5.0` |
| `--prefix` | `-p` | Prefix for output filenames | Input filename |
| `--info-only` | `-i` | Only show mesh info | - |
| `--min-volume-ratio` | - | Min volume ratio to keep parts | `0.01` |

### Dovetail Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--dovetails` | `-d` | Number of dovetail rails per cut (0 = disabled) | `0` |
| `--dovetail-tolerance` | `-t` | Gap between male/female parts in mm | `0.2` |
| `--dovetail-depth` | - | Depth of dovetail protrusion in mm | `8.0` |
| `--dovetail-width` | - | Width of dovetail cross-section in mm | `15.0` |

## How It Works

1. **Load & Analyze**: Reads the STL file and calculates its bounding box
2. **Plan Cuts**: Determines how many cuts are needed along each axis (X, Y, Z) to fit within the build volume
3. **Find Cut Planes**: Calculates evenly-spaced cut positions along each axis
4. **Slice Mesh**: Uses trimesh's plane slicing to cut the mesh, automatically capping cut surfaces
5. **Validate**: Checks for disconnected components and filters out tiny fragments
6. **Add Dovetails** (optional): Creates interlocking dovetail joints between each pair of adjacent parts
7. **Export**: Saves each part as a separate STL file

## Dovetails

Dovetail joints are rails that run along each cut surface, allowing parts to slide together and interlock. The "male" side (protrusion) is added to one part, and the "female" side (cavity with tolerance) is subtracted from the mating part.

- **Rails run along the shorter dimension**: Dovetails extend along the shorter axis of the cut surface
- **Multiple rails distributed along longer dimension**: Use 2-4 rails per cut for larger surfaces
- **Tolerance**: Add 0.2-0.4mm for a snug fit, more for looser fit
- **Clipped to mesh geometry**: Rails are automatically clipped to the actual mesh shape, so they don't extend past where the original part existed

## Example

A model that is 500 x 300 x 200 mm with a 250mm build volume:
- X axis: 500 / 250 = 2 parts needed (1 cut)
- Y axis: 300 / 250 = 2 parts needed (1 cut)  
- Z axis: 200 / 250 = 1 part needed (no cuts)

Result: 2 x 2 x 1 = 4 parts

## Requirements

- Python 3.12+
- trimesh
- numpy
- scipy
- shapely
- click
- manifold3d
