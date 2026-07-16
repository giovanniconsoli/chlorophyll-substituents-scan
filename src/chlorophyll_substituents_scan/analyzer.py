"""Cone-scan analysis of chlorophyll substituents.

Each substituent position (:data:`SUBSTITUENTS`) is scanned by sweeping a probe
through the CryoEM density map around the bond axis of its reference atoms.
Z-scores against a reference substituent then expose the positions whose density
departs from it.

:class:`Analyzer` runs the analysis and writes pickled dataframes plus
PDB files.
"""

import pickle
import warnings
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
CONE_APERTURE = 120
SCAN_DISTANCES = np.arange(0, 2.6, 0.1)
SCAN_ANGLES = np.arange(0, 360, 5)
SUBSTITUENTS = ["C2", "C3", "C7", "C8", "C12"]


class Analyzer:
    """Cone-scan analysis of chlorophyll substituents.

    A single fixed aperture (120°) is scanned over a ``distance x angle`` grid,
    so each substituent's ESP is a 2-D array.

    Parameters
    ----------
    structure : str
        Path to the structure file (``.pdb``, ``.cif``).
    density_map : str
        Path to the electron density (CryoEM) map file (``.map``, ``.mrc``).
    outdir : str
        Output directory for the results.
    reference : str
        Reference substituent to take z-scores against, one of
        :data:`SUBSTITUENTS`.
    locres : str or None, optional
        Path to a local resolution map file. When omitted, the per-chlorophyll
        Mg local resolution is reported as ``0``.

    Attributes
    ----------
    chlorophylls : list[dict] or None
        Per-chlorophyll scan records, or ``None`` before :meth:`run`.
    results_df : pandas.DataFrame or None
        Per-substituent scan grids, or ``None`` before :meth:`run`.
    stats_df : pandas.DataFrame or None
        Per-substituent grid statistics, or ``None`` before :meth:`run`.
    zscores_df : pandas.DataFrame or None
        Per-substituent z-score grids, or ``None`` before :meth:`run`.

    Raises
    ------
    ValueError
        If ``reference`` is not a known substituent.

    Examples
    --------
    >>> analyzer = Analyzer("model.pdb", "map.map", "out/", "C12")
    >>> results_df, stats_df, zscores_df = analyzer.run()
    """

    def __init__(
        self,
        structure: str,
        density_map: str,
        outdir: str,
        reference: str,
        locres: str | None = None,
    ) -> None:
        validate_ref_substituent(reference)

        self.structure_file = structure
        self.map_file = density_map
        self.locres_file = locres
        self.out_dir = Path(outdir)
        self.ref_substituent = reference

        self.chlorophylls: list[dict] | None = None
        self.results_df: pd.DataFrame | None = None
        self.stats_df: pd.DataFrame | None = None
        self.zscores_df: pd.DataFrame | None = None

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Scan every chlorophyll and write the results to ``outdir``.

        Writes three pickled dataframes, plus one PDB file per chlorophyll and
        substituent under ``pdb_intensity/`` (scan amplitudes) and ``pdb_zscores/``
        (z-scores).

        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
            The ``(results_df, stats_df, zscores_df)`` dataframes, also stored
            as attributes on the instance.

        Raises
        ------
        ValueError
            If the structure holds fewer than two complete chlorophylls, which
            leaves the per-substituent spread undefined.
        """
        self._create_output_directories()

        eden_map = gemmi.read_ccp4_map(self.map_file, setup=True)
        loc_res_map = (
            gemmi.read_ccp4_map(self.locres_file, setup=True)
            if self.locres_file
            else None
        )
        structure = gemmi.read_structure(self.structure_file)

        chlorophylls = self._analyze_chlorophylls(structure, eden_map, loc_res_map)
        results_df = self._get_df(chlorophylls)
        stats_df = self._get_statistics_df(results_df)
        zscores_df = self._get_zscores_df(results_df, stats_df)

        self.chlorophylls = chlorophylls
        self.results_df = results_df
        self.stats_df = stats_df
        self.zscores_df = zscores_df

        base_filename = Path(self.structure_file).stem
        self._save_dataframes(base_filename, results_df, stats_df, zscores_df)
        self._save_cone_pdb(chlorophylls, zscores_df)

        return results_df, stats_df, zscores_df

    def _analyze_chlorophylls(
        self,
        structure: gemmi.Structure,
        emap: gemmi.Ccp4Map,
        loc_res_map: gemmi.Ccp4Map | None,
    ) -> list[dict]:
        chlorophylls = _get_chlorophylls(structure)

        n_dist = len(SCAN_DISTANCES)
        n_ang = len(SCAN_ANGLES)

        for chl in chlorophylls:
            chl["loc_res_Mg"] = _get_mg_res(chl["chl_structure"], loc_res_map)
            chl["scan_distances"] = SCAN_DISTANCES
            chl["scan_angles"] = SCAN_ANGLES

            for ref, substituent in zip(REFS, SUBSTITUENTS, strict=True):
                # The first three reference atoms define the local scan frame.
                ref_atoms = [
                    chl["chl_structure"].find_atom(atom, "*") for atom in ref[:3]
                ]

                amps = np.empty((n_dist, n_ang))
                positions = np.empty((n_dist, n_ang, 3))
                for i_d, distance in enumerate(SCAN_DISTANCES):
                    vec0 = _new_position(CONE_APERTURE, distance, ref_atoms)
                    scan_amps, map_positions = _calculate_scan_amps(
                        ref_atoms, vec0, emap
                    )
                    amps[i_d] = scan_amps
                    positions[i_d] = map_positions

                chl[f"scan_amp_{substituent}"] = amps
                chl[f"map_position_{substituent}"] = positions

        return chlorophylls

    def _get_df(self, chlorophylls: list[dict]) -> pd.DataFrame:
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

    def _get_statistics_df(self, df: pd.DataFrame) -> pd.DataFrame:
        num_chl = df.shape[0]
        if num_chl < 2:
            raise ValueError(
                f"Need at least 2 chlorophylls to compute statistics, found "
                f"{num_chl}. Each substituent is scored against the spread "
                f"across all chlorophylls in the structure, which is undefined "
                f"for a single one."
            )

        stats_df = pd.DataFrame(
            index=[
                "Average",
                "Standard Deviation",
                "Average + 3 StdDev",
                "Average - 3 StdDev",
            ]
        )

        for column in SUBSTITUENTS:
            grids = np.stack(df[column].to_numpy())
            avg = grids.mean(axis=0)
            std_dev = grids.std(axis=0, ddof=1)
            stats_df[column] = [
                avg,
                std_dev,
                avg + 3 * std_dev,
                avg - 3 * std_dev,
            ]

        return stats_df

    def _get_zscores_df(self, df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
        ref_substituent = self.ref_substituent
        zscores_df = pd.DataFrame(index=df.index, columns=SUBSTITUENTS)

        for row_idx, row in df.iterrows():
            for substituent in SUBSTITUENTS:
                z = (
                    row[substituent] - stats_df.loc["Average", ref_substituent]
                ) / stats_df.loc["Standard Deviation", ref_substituent]
                zscores_df.loc[row_idx, substituent] = z

        return zscores_df

    def _create_output_directories(self) -> None:
        for sub_dir in ["pdb_intensity", "pdb_zscores"]:
            path = self.out_dir / sub_dir
            path.mkdir(parents=True, exist_ok=True)

    def _save_dataframes(
        self,
        base_filename: str,
        results_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        zscores_df: pd.DataFrame,
    ) -> None:
        structure_data_filename = self.out_dir / (base_filename + "_conedata.pickle")
        stats_filename = self.out_dir / (base_filename + "_stats.pickle")
        zscores_filename = self.out_dir / (base_filename + "_zscores.pickle")

        _save_pickle(results_df, structure_data_filename)
        _save_pickle(stats_df, stats_filename)
        _save_pickle(zscores_df, zscores_filename)

    def _save_cone_pdb(
        self, chlorophylls: list[dict], zscores_df: pd.DataFrame
    ) -> None:
        for chl in chlorophylls:
            for atom, substituent in zip(ATOMS, SUBSTITUENTS, strict=True):
                amp = chl[f"scan_amp_{substituent}"]
                pos = chl[f"map_position_{substituent}"]
                zsc = zscores_df.loc[chl["chl_id"], substituent]

                pdb_lines = []
                pdb_zsc_lines = []
                serial = 0
                for i_d in range(len(SCAN_DISTANCES)):
                    for i_a, theta in enumerate(SCAN_ANGLES):
                        angle = int(theta)
                        x, y, z = pos[i_d, i_a]
                        # The PDB serial field is 5 characters wide; wrap rather
                        # than overflow it into the neighbouring column.
                        pdb_lines.append(
                            _mock_pdb(
                                serial % 100000,
                                angle,
                                x,
                                y,
                                z,
                                float(amp[i_d, i_a]) * 100,
                            )
                        )
                        pdb_zsc_lines.append(
                            _mock_pdb(
                                serial % 100000,
                                angle,
                                x,
                                y,
                                z,
                                float(zsc[i_d, i_a]),
                            )
                        )
                        serial += 1

                pdb_filepaths = _get_pdb_filepaths(atom, chl, self.out_dir, substituent)
                for path, content in zip(
                    pdb_filepaths, [pdb_lines, pdb_zsc_lines], strict=True
                ):
                    with open(path, "w", newline="\n") as file:
                        file.writelines(content)


def validate_ref_substituent(ref: str) -> None:
    """Check that ``ref`` names a scannable substituent.

    Parameters
    ----------
    ref : str
        Candidate reference substituent.

    Raises
    ------
    ValueError
        If ``ref`` is not in :data:`SUBSTITUENTS`.
    """
    if ref not in SUBSTITUENTS:
        raise ValueError(f"Reference substituent {ref} not in substituents list.")


def _get_chlorophylls(structure: gemmi.Structure) -> list[dict]:
    sel = gemmi.Selection("(CL0,CHL,CLA,F6C,CL7,G9R,PHO)")
    chlorophylls = []
    for model in sel.models(structure):
        for chain in sel.chains(model):
            for chl in sel.residues(chain):
                if all(chl.find_atom(at, "*") for ref in REFS for at in ref):
                    chl_dict = {
                        "chl_id": f"{chain.name}{chl.seqid.num}",
                        "chain": chain,
                        "chl_type": chl.name,
                        "chl_structure": chl,
                    }
                    chlorophylls.append(chl_dict)
                else:
                    warnings.warn(
                        f"Chlorophyll {chain.name}{chl.seqid.num} is missing one or "
                        "more reference atoms.",
                        stacklevel=2,
                    )
    return chlorophylls


def _get_mg_res(
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


def _new_position(
    bond_angle: float, bond_length: float, three_atoms: list[gemmi.Atom]
) -> gemmi.Vec3:
    ap1, ap2, ap3 = [a.pos for a in three_atoms]
    av2, av3 = gemmi.Vec3(*ap2), gemmi.Vec3(*ap3)
    avector = _normalize(av3 - av2)
    theta = 180 - bond_angle
    pvector = _normalize(_perpendicular_vector(avector))
    new_pos = gemmi.Position(
        *(
            av3
            + avector * bond_length * np.cos(np.radians(theta))
            + pvector * bond_length * np.sin(np.radians(theta))
        )
    )
    new_vec = gemmi.Vec3(*new_pos)
    dihedral = np.degrees(gemmi.calculate_dihedral(ap1, ap2, ap3, new_pos))
    return _increment_torsion(new_vec, avector, av3, -dihedral)


def _normalize(vector: gemmi.Vec3) -> gemmi.Vec3:
    return vector / vector.length()


def _perpendicular_vector(vector: gemmi.Vec3) -> gemmi.Vec3:
    # get a perpendicular vector, swap y,x, change a sign and zero z, dot product = 0
    # later we measure the dihedral and adjust position to dihedral of 0
    return gemmi.Vec3(vector.y, -vector.x, 0)


def _increment_torsion(
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


def _calculate_scan_amps(
    three_atoms: list[gemmi.Atom],
    vec0: gemmi.Vec3,
    emap: gemmi.Ccp4Map,
) -> tuple[np.ndarray, np.ndarray]:
    _, atom2, atom3 = three_atoms
    av2, av3 = gemmi.Vec3(*atom2.pos), gemmi.Vec3(*atom3.pos)
    avector = _normalize(av3 - av2)
    scan_amps = []
    map_positions = []
    for theta in SCAN_ANGLES:
        p = _increment_torsion(vec0, avector, av3, theta)
        map_pos = gemmi.Position(*p)
        eden = emap.grid.tricubic_interpolation(map_pos)
        scan_amps.append(eden)
        map_positions.append(map_pos.tolist())

    return np.array(scan_amps), np.array(map_positions)


def _save_pickle(data: pd.DataFrame, filename: Path) -> None:
    with open(filename, "wb") as out:
        pickle.dump(data, out, protocol=pickle.HIGHEST_PROTOCOL)


def _mock_pdb(n: int, resi: int, x: float, y: float, z: float, temp: float) -> str:
    return _pdb_string(
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


def _pdb_string(
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
        f"{atom:6s}{serial:5d} {name:^4s}{alt_loc:1s}{res_name:3s} "
        f"{chain:1s}{resi:4d}{ins:1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{temp:6.2f}\n"
    )


def _get_pdb_filepaths(
    atom: str, chl: dict, out_dir: Path, substituent: str
) -> list[Path]:
    filename = f"cone_{chl['chl_id']}_{chl['chl_structure'].name}_{atom}_{substituent}"
    return [
        out_dir / "pdb_intensity" / (filename + ".pdb"),
        out_dir / "pdb_zscores" / (filename + ".pdb"),
    ]
