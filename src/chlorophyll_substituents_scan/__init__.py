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
__version__ = "0.1.0"


import argparse

from .analyzer import GEOMETRIES, Analyzer, get_geometry, validate_ref_substituent

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
    args = parser.parse_args()

    analyzer = Analyzer(
        structure=args.structure,
        density_map=args.map,
        outdir=args.outdir,
        reference=args.reference,
        locres=args.locres,
        geometry=args.geometry,
    )
    analyzer.run()
