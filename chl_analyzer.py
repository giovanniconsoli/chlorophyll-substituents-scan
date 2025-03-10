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
__version__ = "0.0.1"


import pickle
import argparse
from collections import namedtuple
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd


ATOMS = ["CMB", "CAB", "CMC", "CAC", "CMD"]
REFS = [
    ["C1B", "C2B", "CMB", "C3B"],  # C2
    ["C2B", "C3B", "CAB", "C4B"],  # C3
    ["C1C", "C2C", "CMC", "C3C"],  # C7
    ["C2C", "C3C", "CAC", "C4C"],  # C8
    ["C1D", "C2D", "CMD", "C3D"],  # C12 (always methyl)
]
SCAN_DISTANCES = np.arange(0, 2.6, 0.1)
SCAN_ANGLES = np.arange(0, 360, 5)
SUBSTITUENTS = ["C2", "C3", "C7", "C8", "C12"]

Cone = namedtuple("Cone", ["atoms", "length", "angle", "group"])


def main():
    args = parse_arguments()

    structure_file = args.structure
    eden_map = gemmi.read_ccp4_map(args.map, setup=True)
    loc_res_map = gemmi.read_ccp4_map(args.locres, setup=True) if args.locres else None
    out_dir = Path(args.outdir)
    ref_substituent = args.reference

    structure = gemmi.read_structure(structure_file)
    validate_ref_substituent(ref_substituent)

    chlorophylls = analyze_chlorophylls(structure, eden_map, loc_res_map)
    results_df = get_df(chlorophylls)
    stats_df = get_statistics_df(results_df)
    zscores_df = get_zscores_df(results_df, stats_df, ref_substituent)

    create_output_directories(out_dir)

    base_filename = Path(structure_file).stem
    save_dataframes(out_dir, base_filename, results_df, stats_df, zscores_df)
    save_cone_pdb(chlorophylls, zscores_df, out_dir)


def parse_arguments() -> argparse.Namespace:
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
    return parser.parse_args()


def validate_ref_substituent(ref: str) -> None:
    if ref not in SUBSTITUENTS:
        raise ValueError(f"Reference substituent {ref} not in substituents list.")


def analyze_chlorophylls(
    structure: gemmi.Structure, emap: gemmi.Ccp4Map, loc_res_map: gemmi.Ccp4Map | None
) -> list[dict]:
    chlorophylls = get_chlorophylls(structure)

    for chl in chlorophylls:
        chl["loc_res_Mg"] = get_mg_res(chl["chl_structure"], loc_res_map)
        chl["scan_distances"] = SCAN_DISTANCES
        chl["scan_angles"] = SCAN_ANGLES
        for distance in SCAN_DISTANCES:
            for cone in get_cones(distance):
                ref_atoms = [
                    chl["chl_structure"].find_atom(atom, "*") for atom in cone.atoms
                ]
                vec0 = new_position(cone.angle, cone.length, ref_atoms[:3])
                scan_amps, map_positions = calculate_scan_amps(
                    ref_atoms[:3], vec0, emap
                )

                for atom, substituent in zip(ATOMS, SUBSTITUENTS):
                    attr_types = [
                        f"scan_amp_{substituent}",
                        f"map_position_{substituent}",
                    ]
                    values = [scan_amps, map_positions]
                    if atom in cone.atoms:
                        for attr, value in zip(attr_types, values):
                            if attr in chl.keys():
                                chl[attr] = np.vstack((chl[attr], value))
                            else:
                                chl[attr] = value

        for substituent in SUBSTITUENTS:
            chl[f"map_position_{substituent}"] = chl[
                f"map_position_{substituent}"
            ].reshape(len(SCAN_DISTANCES), len(SCAN_ANGLES), 3)

    return chlorophylls


def get_chlorophylls(structure: gemmi.Structure) -> list[dict]:
    sel = gemmi.Selection("(CL0,CHL,CLA,F6C,CL7,G9R,PHO)")
    chlorophylls = []
    for model in sel.models(structure):
        for chain in sel.chains(model):
            for chl in sel.residues(chain):
                if all([chl.find_atom(at, "*") for ref in REFS for at in ref]):
                    chl_dict = {
                        "chl_id": f"{chain.name}{chl.seqid.num}",
                        "chain": chain,
                        "chl_type": chl.name,
                        "chl_structure": chl,
                    }
                    chlorophylls.append(chl_dict)
                else:
                    print(
                        f"Chlorophyll {chain.name}{chl.seqid.num} is missing one or more reference atoms."
                    )
    return chlorophylls


def get_mg_res(
    chl_structure: gemmi.Residue,
    loc_res_map: gemmi.Ccp4Map | None,
) -> float:
    mg_name = "MG"
    mg_res = (
        loc_res_map.grid.tricubic_interpolation(chl_structure.sole_atom(mg_name).pos)
        if mg_name in chl_structure and loc_res_map
        else 0
    )
    return mg_res


def get_cones(distance: float) -> list[Cone]:
    cones = []
    for ref, sub in zip(REFS, SUBSTITUENTS):
        cone_aperture = 120
        cone = Cone(ref, distance, cone_aperture, sub)
        cones.append(cone)
    return cones


def new_position(
    bond_angle: int, bond_length: float, three_atoms: list[gemmi.Atom]
) -> gemmi.Vec3:
    ap1, ap2, ap3 = [a.pos for a in three_atoms]
    av1, av2, av3 = [gemmi.Vec3(*[a for a in ap.pos]) for ap in three_atoms]
    avector = normalise(av3 - av2)
    theta = 180 - bond_angle
    pvector = normalise(perpendicular_vector(avector))
    new_pos = gemmi.Position(
        *(
            av3
            + avector * bond_length * np.cos(np.radians(theta))
            + pvector * bond_length * np.sin(np.radians(theta))
        )
    )
    new_vec = gemmi.Vec3(*[a for a in new_pos])
    dihedral = np.degrees(gemmi.calculate_dihedral(ap1, ap2, ap3, new_pos))
    return increment_torsion(new_vec, avector, av3, -dihedral)


def normalise(vector: gemmi.Vec3) -> gemmi.Vec3:
    return vector / vector.length()


def perpendicular_vector(vector: gemmi.Vec3) -> gemmi.Vec3:
    # get a perpendicular vector, swap y,x, change a sign and zero z, dot product = 0
    # later we measure the dihedral and adjust position to dihedral of 0
    x, y, z = vector
    return gemmi.Vec3(y, -x, 0)


def increment_torsion(
    v: gemmi.Vec3, k: gemmi.Vec3, apoint: gemmi.Vec3, theta: float
) -> gemmi.Vec3:
    # v vector, k axis, theta angle, apoint - point in axis
    # Rodrigues rotation formula https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    # gv1.cross(gv2) = gv1 X gv2
    v = v - apoint  # move point close to origin
    cos_theta = np.cos(np.radians(theta))
    sin_theta = np.sin(np.radians(theta))
    vrot = (
        v * cos_theta + (k.cross(v)) * sin_theta + k * k.dot(v) * (1 - cos_theta)
    )  # rotate around axis
    vrot = vrot + apoint  # move back
    return vrot


def calculate_scan_amps(
    three_atoms: list[gemmi.Atom],
    vec0: gemmi.Vec3,
    emap: gemmi.Ccp4Map,
) -> tuple[np.ndarray, np.ndarray]:
    _, av2, av3 = [gemmi.Vec3(*[a for a in ap.pos]) for ap in three_atoms]
    avector = normalise(av3 - av2)
    scan_amps = []
    map_positions = []
    theta: int
    for n, theta in enumerate(SCAN_ANGLES):
        p = increment_torsion(vec0, avector, av3, theta)
        map_pos = gemmi.Position(*p)
        eden = emap.grid.tricubic_interpolation(map_pos)
        scan_amps.append(eden)
        map_positions.append([pos for pos in map_pos])

    return np.array(scan_amps), np.array(map_positions)


def get_df(chlorophylls: list[dict]) -> pd.DataFrame:
    chlorophylls_dict = {"loc_res_Mg": {}}
    for substituent in SUBSTITUENTS:
        chlorophylls_dict[substituent] = {}
    for chl in chlorophylls:
        chlorophylls_dict["loc_res_Mg"][chl["chl_id"]] = chl["loc_res_Mg"]
        for substituent in SUBSTITUENTS:
            chlorophylls_dict[substituent][chl["chl_id"]] = chl[
                f"scan_amp_{substituent}"
            ]
    return pd.DataFrame(chlorophylls_dict)


def get_statistics_df(df: pd.DataFrame) -> pd.DataFrame:
    stats_df = pd.DataFrame(
        index=[
            "Average",
            "Standard Deviation",
            "Average + 3 StdDev",
            "Average - 3 StdDev",
        ]
    )

    num_chl = df.shape[0]
    for column in SUBSTITUENTS:
        avg = df[column].mean()
        std_dev = np.zeros((len(SCAN_DISTANCES), len(SCAN_ANGLES)))
        for chl in df[column]:
            std_dev += (chl - avg) ** 2
        std_dev = np.sqrt(std_dev / num_chl)
        stats_df[column] = [
            avg,
            std_dev,
            avg + 3 * std_dev,
            avg - 3 * std_dev,
        ]

    return stats_df


def get_zscores_df(
    df: pd.DataFrame, stats_df: pd.DataFrame, ref_substituent: str
) -> pd.DataFrame:
    zscores_df = pd.DataFrame(index=df.index, columns=SUBSTITUENTS)

    for row_idx, row in df.iterrows():
        for substituent in SUBSTITUENTS:
            z = (
                row[substituent] - stats_df.loc["Average", ref_substituent]
            ) / stats_df.loc["Standard Deviation", ref_substituent]
            zscores_df.loc[row_idx, substituent] = z

    return zscores_df


def create_output_directories(out_dir: Path) -> None:
    for sub_dir in ["pdb_intensity", "pdb_zscores"]:
        path = out_dir / sub_dir
        path.mkdir(parents=True, exist_ok=True)


def save_dataframes(
    out_dir: Path,
    base_filename: str,
    results_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    zscores_df: pd.DataFrame,
) -> None:
    structure_data_filename = out_dir / (base_filename + "_conedata.pickle")
    stats_filename = out_dir / (base_filename + "_stats.pickle")
    zscores_filename = out_dir / (base_filename + "_zscores.pickle")

    save_pickle(results_df, structure_data_filename)
    save_pickle(stats_df, stats_filename)
    save_pickle(zscores_df, zscores_filename)


def save_pickle(data: pd.DataFrame, filename: Path) -> None:
    with open(filename, "wb") as out:
        pickle.dump(data, out, protocol=pickle.HIGHEST_PROTOCOL)


def save_cone_pdb(
    chlorophylls: list[dict], zscores: pd.DataFrame, out_dir: Path
) -> None:
    theta: int
    for chl in chlorophylls:
        for atom, substituent in zip(ATOMS, SUBSTITUENTS):
            pdb_lines = []
            pdb_zsc_lines = []
            for ii, _ in enumerate(SCAN_DISTANCES):
                for jj, theta in enumerate(SCAN_ANGLES):
                    pdb_line = mock_pdb(
                        jj,
                        theta,
                        *chl[f"map_position_{substituent}"][ii, jj, :],
                        chl[f"scan_amp_{substituent}"][ii, jj] * 100,
                    )
                    pdb_zsc_line = mock_pdb(
                        jj,
                        theta,
                        *chl[f"map_position_{substituent}"][ii, jj, :],
                        zscores.loc[chl["chl_id"], substituent][ii, jj],
                    )
                    pdb_lines.append(pdb_line)
                    pdb_zsc_lines.append(pdb_zsc_line)

            pdb_filepaths = get_pdb_filepaths(atom, chl, out_dir, substituent)
            for path, content in zip(pdb_filepaths, [pdb_lines, pdb_zsc_lines]):
                with open(path, "w", newline="\n") as file:
                    file.writelines(content)


def mock_pdb(n: int, resi: int, x: float, y: float, z: float, temp: float) -> str:
    return pdb_string(
        "ATOM",  # atom
        n,
        "CA",  # name
        "",  # alt_loc
        "UNK",  # res_name
        "A",  # chain
        resi,
        "",  # ins
        x,
        y,
        z,
        1.00,  # occ
        temp,
    )


def pdb_string(
    atom: str,
    serial: int,
    name: str,
    alt_loc: str,
    res_name: str,
    chain: str,
    resi: int,
    ins: str,
    x: float,
    y: float,
    z: float,
    occ: float,
    temp: float,
) -> str:
    # https://cupnet.net/pdb-format/
    return (
        f"{atom:6s}{serial:5d} {name:^4s}{alt_loc:1s}{res_name:3s} {chain:1s}{resi:4d}{ins:1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{temp:6.2f}\n"
    )


def get_pdb_filepaths(
    atom: str, chl: dict, out_dir: Path, substituent: str
) -> list[Path]:
    filename = f"cone_{chl['chl_id']}_{chl['chl_structure'].name}_{atom}_{substituent}"
    return [
        out_dir / "pdb_intensity" / (filename + ".pdb"),
        out_dir / "pdb_zscores" / (filename + ".pdb"),
    ]


if __name__ == "__main__":
    main()
