"""Cone- and hemisphere-scan analysis of chlorophyll substituents.

Each substituent position (:data:`SUBSTITUENTS`) is scanned by sweeping a probe
through the CryoEM density map around the bond axis of its reference atoms.
Z-scores against a reference substituent then expose the positions whose density
departs from it.

:class:`Analyzer` runs the analysis and writes pickled dataframes, plus
PDB files when ``save_pdb`` is set.
"""

import pickle
import warnings
from pathlib import Path
from typing import NamedTuple

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
SUBSTITUENTS = ["C2", "C3", "C7", "C8", "C12"]


# Both geometries share the distance and angle grids; only the apertures differ.
# +step/2 keeps the endpoint despite float drift; round clears arange noise.
DEFAULT_MAX_DISTANCE = 2.5
DEFAULT_STEP = 0.1
DEFAULT_DISTANCES = np.round(
    np.arange(0.0, DEFAULT_MAX_DISTANCE + DEFAULT_STEP / 2, DEFAULT_STEP), 3
)
# 72 angles at a 5 degree step.
DEFAULT_ANGLE_STEP = 5.0
DEFAULT_ANGLES = np.arange(0.0, 360.0, DEFAULT_ANGLE_STEP)


# A geometry is fully described by the aperture(s), distances and angles it
# samples. The cone is the special case of a single aperture; the hemisphere
# sweeps a range of apertures and therefore gains an extra array axis.
class Geometry(NamedTuple):
    name: str
    apertures: np.ndarray
    distances: np.ndarray
    angles: np.ndarray


CONE_GEOMETRY = Geometry(
    name="cone",
    apertures=np.array([120.0]),
    distances=DEFAULT_DISTANCES,
    angles=DEFAULT_ANGLES,
)
HEMISPHERE_GEOMETRY = Geometry(
    name="hemisphere",
    apertures=np.arange(90, 181, 10, dtype=float),
    distances=DEFAULT_DISTANCES,
    angles=DEFAULT_ANGLES,
)
GEOMETRIES = {"cone": CONE_GEOMETRY, "hemisphere": HEMISPHERE_GEOMETRY}


def get_geometry(
    geometry: "str | Geometry",
    max_distance: float | None = None,
    step: float | None = None,
) -> Geometry:
    """Resolve a geometry name to its :data:`Geometry`.

    Parameters
    ----------
    geometry : str or Geometry
        A key of :data:`GEOMETRIES` (case-insensitive), or a ``Geometry``.
    max_distance : float or None, optional
        Maximum distance of the scan grid, in ångström. Defaults to the
        selected geometry's built-in maximum.
    step : float or None, optional
        Spacing between successive scan distances, in ångström. Defaults to the
        selected geometry's built-in step.

    Returns
    -------
    Geometry
        The resolved geometry, with ``distances`` rebuilt when an override is set.

    Raises
    ------
    ValueError
        If ``geometry`` is unknown, or an override is not positive.
    """
    if isinstance(geometry, Geometry):
        base = geometry
    else:
        key = str(geometry).lower()
        if key not in GEOMETRIES:
            raise ValueError(
                f"Unknown geometry {geometry!r}. Choose from {sorted(GEOMETRIES)}."
            )
        base = GEOMETRIES[key]

    if max_distance is None and step is None:
        return base
    stop = base.distances[-1] if max_distance is None else max_distance
    if step is None:
        step = base.distances[1] - base.distances[0]
    if stop <= 0 or step <= 0:
        raise ValueError(f"max_distance and step must be positive, got {stop}, {step}.")
    # +step/2 keeps the endpoint despite float drift; round clears arange noise.
    return base._replace(distances=np.round(np.arange(0.0, stop + step / 2, step), 3))


class Analyzer:
    """Cone- or hemisphere-scan analysis of chlorophyll substituents.

    The scan geometry is selected with ``geometry``. Both geometries share the
    same algorithm; only the grid that is sampled changes, since a cone is just
    the hemisphere restricted to a single aperture:

    ``"cone"`` (default)
        A single fixed aperture (120°) scanned over a ``distance x angle`` grid,
        so each substituent's ESP is a **2-D** array.
    ``"hemisphere"``
        A range of apertures (90°…180°) scanned, adding an extra axis, so each
        substituent's ESP is a **3-D** ``aperture x distance x angle`` array.

    The downstream statistics, z-scores and PDB export are rank-agnostic
    and work with either.

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
    geometry : str or Geometry, optional
        Scan geometry, ``"cone"`` (default) or ``"hemisphere"``.
    max_distance : float or None, optional
        Maximum distance of the scan grid, in ångström. When ``None`` (default)
        the selected geometry keeps its built-in maximum. See :func:`get_geometry`.
    step : float or None, optional
        Spacing between successive scan distances, in ångström. When ``None``
        (default) the selected geometry keeps its built-in step. See
        :func:`get_geometry`.
    save_pdb : bool, optional
        Write the per-chlorophyll PDB files. ``False`` by default,
        since the export is two files per chlorophyll and substituent and is
        only needed for visual inspection; the dataframes are always written
        either way.

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
        If ``reference`` is not a known substituent, or ``geometry`` is not a
        known geometry.

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
        geometry: "str | Geometry" = "cone",
        max_distance: float | None = None,
        step: float | None = None,
        save_pdb: bool = False,
    ) -> None:
        validate_ref_substituent(reference)

        self.structure_file = structure
        self.map_file = density_map
        self.locres_file = locres
        self.out_dir = Path(outdir)
        self.ref_substituent = reference
        self.geometry = get_geometry(geometry, max_distance, step)
        self.save_pdb = save_pdb

        self.chlorophylls: list[dict] | None = None
        self.results_df: pd.DataFrame | None = None
        self.stats_df: pd.DataFrame | None = None
        self.zscores_df: pd.DataFrame | None = None

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Scan every chlorophyll and write the results to ``outdir``.

        Writes three pickled dataframes. When ``save_pdb`` is set, also writes
        one PDB file per chlorophyll and substituent under ``pdb_intensity/``
        (scan amplitudes) and ``pdb_zscores/`` (z-scores).

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
        results_df.attrs.update(
            geometry=self.geometry.name,
            apertures=self.geometry.apertures.tolist(),
            distances=self.geometry.distances.tolist(),
            angles=self.geometry.angles.tolist(),
        )
        stats_df = self._get_statistics_df(results_df)
        zscores_df = self._get_zscores_df(results_df, stats_df)

        self.chlorophylls = chlorophylls
        self.results_df = results_df
        self.stats_df = stats_df
        self.zscores_df = zscores_df

        base_filename = Path(self.structure_file).stem
        self._save_dataframes(base_filename, results_df, stats_df, zscores_df)
        if self.save_pdb:
            self._save_scan_pdb(chlorophylls, zscores_df)

        return results_df, stats_df, zscores_df

    def _analyze_chlorophylls(
        self,
        structure: gemmi.Structure,
        emap: gemmi.Ccp4Map,
        loc_res_map: gemmi.Ccp4Map | None,
    ) -> list[dict]:
        geometry = self.geometry
        chlorophylls = _get_chlorophylls(structure)

        n_ap = len(geometry.apertures)
        n_dist = len(geometry.distances)
        n_ang = len(geometry.angles)

        for chl in chlorophylls:
            chl["loc_res_Mg"] = _get_mg_res(chl["chl_structure"], loc_res_map)
            chl["geometry"] = geometry.name
            chl["scan_apertures"] = geometry.apertures
            chl["scan_distances"] = geometry.distances
            chl["scan_angles"] = geometry.angles

            for ref, substituent in zip(REFS, SUBSTITUENTS, strict=True):
                # The first three reference atoms define the local scan frame.
                ref_atoms = [
                    chl["chl_structure"].find_atom(atom, "*") for atom in ref[:3]
                ]

                amps = np.empty((n_ap, n_dist, n_ang))
                positions = np.empty((n_ap, n_dist, n_ang, 3))
                for i_ap, aperture in enumerate(geometry.apertures):
                    for i_d, distance in enumerate(geometry.distances):
                        vec0 = _new_position(aperture, distance, ref_atoms)
                        scan_amps, map_positions = _calculate_scan_amps(
                            ref_atoms, vec0, emap, geometry.angles
                        )
                        amps[i_ap, i_d] = scan_amps
                        positions[i_ap, i_d] = map_positions

                # A cone has a single aperture: drop that axis so the arrays are
                # 2-D (distance x angle).
                if n_ap == 1:
                    amps = amps[0]
                    positions = positions[0]

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
            # Stacking the per-chlorophyll grids along a new leading axis keeps
            # this shape-agnostic: the reductions collapse that axis whether the
            # grids are 2-D (cone) or 3-D (hemisphere).
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
        avg = stats_df.loc["Average", self.ref_substituent]
        std = stats_df.loc["Standard Deviation", self.ref_substituent]
        return df[SUBSTITUENTS].map(lambda grid: (grid - avg) / std)

    def _create_output_directories(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if not self.save_pdb:
            return
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
        # Every file carries the geometry name, so runs of different geometries
        # can share one output directory without overwriting each other.
        stem = f"{base_filename}_{self.geometry.name}"

        _save_pickle(results_df, self.out_dir / f"{stem}_data.pickle")
        _save_pickle(stats_df, self.out_dir / f"{stem}_stats.pickle")
        _save_pickle(zscores_df, self.out_dir / f"{stem}_zscores.pickle")

    def _save_scan_pdb(
        self, chlorophylls: list[dict], zscores_df: pd.DataFrame
    ) -> None:
        geometry = self.geometry
        for chl in chlorophylls:
            for atom, substituent in zip(ATOMS, SUBSTITUENTS, strict=True):
                amp = chl[f"scan_amp_{substituent}"]
                pos = chl[f"map_position_{substituent}"]
                zsc = zscores_df.loc[chl["chl_id"], substituent]

                pdb_lines = []
                pdb_zsc_lines = []
                # Walk every grid point regardless of rank (2-D cone or 3-D
                # hemisphere); ``pos`` carries one extra trailing axis of size 3.
                for serial, idx in enumerate(np.ndindex(amp.shape)):
                    angle = int(geometry.angles[idx[-1]])
                    x, y, z = pos[idx]
                    pdb_lines.append(
                        _mock_pdb(
                            serial % 100000, angle, x, y, z, float(amp[idx]) * 100
                        )
                    )
                    pdb_zsc_lines.append(
                        _mock_pdb(serial % 100000, angle, x, y, z, float(zsc[idx]))
                    )

                pdb_filepaths = _get_pdb_filepaths(
                    atom, chl, self.out_dir, substituent, geometry
                )
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
    angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, atom2, atom3 = three_atoms
    av2, av3 = gemmi.Vec3(*atom2.pos), gemmi.Vec3(*atom3.pos)
    avector = _normalize(av3 - av2)
    scan_amps = []
    map_positions = []
    for theta in angles:
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
    # Fixed-column PDB ATOM record: https://cupnet.net/pdb-format/
    return (
        f"ATOM  {n:5d}  CA  UNK "  # record, serial, name, altLoc, resName
        f"A{resi:4d}    "  # chain, resSeq, iCode
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{temp:6.2f}\n"  # x, y, z, occupancy, temp
    )


def _get_pdb_filepaths(
    atom: str,
    chl: dict,
    out_dir: Path,
    substituent: str,
    geometry: "str | Geometry",
) -> list[Path]:
    geometry = get_geometry(geometry)
    filename = (
        f"{geometry.name}_{chl['chl_id']}_"
        f"{chl['chl_structure'].name}_{atom}_{substituent}"
    )
    return [
        out_dir / "pdb_intensity" / (filename + ".pdb"),
        out_dir / "pdb_zscores" / (filename + ".pdb"),
    ]
