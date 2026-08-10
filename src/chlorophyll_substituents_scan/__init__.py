__authors__ = ["Giovanni Consoli", "Marco Sandrin", "James W. Murray"]
__contact__ = "gconsoli@ic.ac.uk"
__copyright__ = "Copyright 2025, Imperial College London"
__credits__ = ["Giovanni Consoli", "Marco Sandrin", "James W. Murray"]
__date__ = "2025/03/10"
__deprecated__ = False
__email__ = "gconsoli@ic.ac.uk"
__license__ = "MIT"
__maintainer__ = "Giovanni Consoli"
__status__ = "Production"
__version__ = "0.2.0"


import argparse

from .analyzer import (
    DEFAULT_MAX_DISTANCE,
    DEFAULT_STEP,
    GEOMETRIES,
    Analyzer,
    get_geometry,
    validate_ref_substituent,
)

__all__ = ["Analyzer", "get_geometry", "main", "validate_ref_substituent"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--structure", type=str, required=True, help="Path to the structure file"
    )
    parser.add_argument(
        "-m",
        "--map",
        type=str,
        required=True,
        help="Path to the electron density map file",
    )
    parser.add_argument(
        "-l",
        "--locres",
        type=str,
        required=False,
        help="Path to the local resolution map file",
    )
    parser.add_argument(
        "-o", "--outdir", type=str, required=True, help="Output directory for results"
    )
    parser.add_argument(
        "-r", "--reference", type=str, required=True, help="Reference substituent"
    )
    parser.add_argument(
        "-g",
        "--geometry",
        type=str,
        default="cone",
        choices=sorted(GEOMETRIES),
        help="Scan geometry: 'cone' (2-D, default) or 'hemisphere' (3-D)",
    )
    parser.add_argument(
        "-d",
        "--max-distance",
        type=float,
        default=None,
        help=(
            "Maximum scan distance (angstrom). "
            f"Default: {DEFAULT_MAX_DISTANCE} for both geometries."
        ),
    )
    parser.add_argument(
        "-t",
        "--step",
        type=float,
        default=None,
        help=f"Spacing between scan distances (angstrom). Default: {DEFAULT_STEP}.",
    )
    parser.add_argument(
        "-p",
        "--save-pdb",
        action="store_true",
        help=(
            "Write the per-chlorophyll PDB files to "
            "'pdb_intensity/' and 'pdb_zscores/'. Off by default."
        ),
    )
    args = parser.parse_args()

    analyzer = Analyzer(
        structure=args.structure,
        density_map=args.map,
        outdir=args.outdir,
        reference=args.reference,
        locres=args.locres,
        geometry=args.geometry,
        max_distance=args.max_distance,
        step=args.step,
        save_pdb=args.save_pdb,
    )
    analyzer.run()
