"""Command-line interface for Joint Venture."""

from pathlib import Path

import click

from .dovetail import DovetailConfig
from .mesh_loader import analyze_mesh, fits_in_build_volume, load_mesh, print_mesh_info
from .splitter import export_parts, split_stl


# Default Bambu X1C build volume in mm
DEFAULT_BUILD_VOLUME = (256, 256, 256)


@click.command()
@click.argument(
    "input_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory for split parts. Defaults to 'parts' in input file directory.",
)
@click.option(
    "--build-volume", "-b",
    type=float,
    nargs=3,
    default=DEFAULT_BUILD_VOLUME,
    metavar="X Y Z",
    help=f"Build volume dimensions in mm. Default: {DEFAULT_BUILD_VOLUME[0]} {DEFAULT_BUILD_VOLUME[1]} {DEFAULT_BUILD_VOLUME[2]}",
)
@click.option(
    "--margin", "-m",
    type=float,
    default=5.0,
    help="Safety margin in mm subtracted from each build dimension. Default: 5.0",
)
@click.option(
    "--prefix", "-p",
    type=str,
    default=None,
    help="Prefix for output filenames. Defaults to input filename.",
)
@click.option(
    "--info-only", "-i",
    is_flag=True,
    help="Only show mesh info, don't split.",
)
@click.option(
    "--min-volume-ratio",
    type=float,
    default=0.01,
    help="Minimum volume ratio to keep a part (filters tiny fragments). Default: 0.01",
)
@click.option(
    "--dovetails", "-d",
    type=int,
    default=0,
    help="Number of dovetail joints per cut (0 to disable). Default: 0",
)
@click.option(
    "--dovetail-tolerance", "-t",
    type=float,
    default=0.2,
    help="Tolerance (gap) for dovetail joints in mm. Default: 0.2",
)
@click.option(
    "--dovetail-depth",
    type=float,
    default=8.0,
    help="Depth of dovetail protrusion in mm. Default: 8.0",
)
@click.option(
    "--dovetail-width",
    type=float,
    default=15.0,
    help="Width of dovetail at narrow end in mm. Default: 15.0",
)
def main(
    input_file: Path,
    output_dir: Path | None,
    build_volume: tuple[float, float, float],
    margin: float,
    prefix: str | None,
    info_only: bool,
    min_volume_ratio: float,
    dovetails: int,
    dovetail_tolerance: float,
    dovetail_depth: float,
    dovetail_width: float,
) -> None:
    """
    Split a large STL file into multiple parts that fit your 3D printer's build volume.
    
    INPUT_FILE: Path to the STL file to split.
    
    \b
    Examples:
      joint-venture model.stl
      joint-venture model.stl --output-dir ./parts
      joint-venture model.stl --build-volume 250 250 250 --margin 10
      joint-venture model.stl --dovetails 3 --dovetail-tolerance 0.3
      joint-venture model.stl --info-only
    """
    click.echo(f"\n{'=' * 60}")
    click.echo("Joint Venture - Split large models for 3D printing")
    click.echo(f"{'=' * 60}")
    
    # Show build volume info
    effective_volume = tuple(v - margin * 2 for v in build_volume)
    click.echo(f"\nBuild volume: {build_volume[0]} x {build_volume[1]} x {build_volume[2]} mm")
    click.echo(f"Safety margin: {margin} mm")
    click.echo(f"Effective volume: {effective_volume[0]:.0f} x {effective_volume[1]:.0f} x {effective_volume[2]:.0f} mm")
    
    # Show dovetail info if enabled
    if dovetails > 0:
        click.echo(f"\nDovetails: {dovetails} rail(s) per cut")
        click.echo(f"  Tolerance: {dovetail_tolerance} mm")
        click.echo(f"  Cross-section: {dovetail_width}mm wide x {dovetail_depth}mm deep")
    
    # Load and analyze mesh
    try:
        mesh = load_mesh(input_file)
        analysis = analyze_mesh(mesh)
    except ValueError as e:
        click.echo(f"\nError: {e}", err=True)
        raise SystemExit(1)
    
    # Print mesh info
    print_mesh_info(analysis, build_volume, margin)
    
    if info_only:
        click.echo("\n(Info only mode - not splitting)")
        return
    
    # Check if splitting is needed
    if fits_in_build_volume(analysis.extents, build_volume, margin):
        click.echo("\nMesh already fits in build volume. No splitting needed!")
        return
    
    # Set up output
    if output_dir is None:
        output_dir = input_file.parent / "parts"
    
    if prefix is None:
        prefix = input_file.stem
    
    # Create dovetail config if enabled
    dovetail_config = None
    if dovetails > 0:
        dovetail_config = DovetailConfig(
            count=dovetails,
            tolerance=dovetail_tolerance,
            depth=dovetail_depth,
            width=dovetail_width,
        )
    
    # Split the mesh
    try:
        result = split_stl(
            input_file,
            build_volume=build_volume,
            margin=margin,
            min_volume_ratio=min_volume_ratio,
            dovetail_config=dovetail_config,
        )
    except Exception as e:
        click.echo(f"\nError during splitting: {e}", err=True)
        raise SystemExit(1)
    
    if not result.parts:
        click.echo("\nError: No parts were generated!", err=True)
        raise SystemExit(1)
    
    # Export parts
    click.echo(f"\nExporting {result.num_parts} parts to: {output_dir}")
    paths = export_parts(result.parts, output_dir, prefix)
    
    click.echo(f"\n{'=' * 60}")
    click.echo(f"Done! Generated {len(paths)} parts in {output_dir}")
    click.echo(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
