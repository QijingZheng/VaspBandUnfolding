#!/usr/bin/env python3
"""
Utilities for parsing VASP BSEFATBAND output, plotting excitons in k-space,
and reconstructing fixed-particle exciton densities in real space from WAVECAR.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))

import matplotlib
import matplotlib.patheffects as mpe
import numpy as np
from ase.build import make_supercell
from ase.io import read as ase_read
from ase.io import write as ase_write
from matplotlib import patches as mpatches
from matplotlib.colors import LogNorm
from matplotlib.path import Path as MplPath

try:
    from scipy.spatial import Voronoi
except Exception:
    Voronoi = None

try:
    from scipy.interpolate import RectBivariateSpline, griddata
except Exception:
    RectBivariateSpline = None
    griddata = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.fftpack import ifftn

from vaspwfc import vaspwfc

HEADER_RE = re.compile(
    r"^\s*(\d+)\s*BSE eigenvalue\s+([+\-]?\d+(?:\.\d*)?(?:[EeDd][+\-]?\d+)?)"
    r"\s+IP-eigenvalue:\s+([+\-]?\d+(?:\.\d*)?(?:[EeDd][+\-]?\d+)?)\s*$"
)

BZ_PATH_EPS = 1e-8


def parse_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "E"))


def canonical_frac(frac: np.ndarray | Sequence[float], decimals: int = 8) -> np.ndarray:
    vals = np.mod(np.asarray(frac, dtype=float), 1.0)
    # Wrap again after rounding so values like 0.999999999 -> 1.0 are restored to 0.0.
    return np.mod(np.round(vals, decimals), 1.0)


def frac_key(frac: np.ndarray | Sequence[float], decimals: int = 8) -> Tuple[float, ...]:
    vals = canonical_frac(frac, decimals=decimals).reshape(-1)
    return tuple(float(x) for x in vals)


@dataclass
class ExcitonFatband:
    filepath: Path | None = None
    xdim: int | None = None
    nexciton: int | None = None
    index: int | None = None
    bse_eigenvalue: float | None = None
    ip_eigenvalue: float | None = None
    k_frac: np.ndarray | None = None
    evb: np.ndarray | None = None
    ecb: np.ndarray | None = None
    column_weight: np.ndarray | None = None
    ivb: np.ndarray | None = None
    icb: np.ndarray | None = None
    amplitude: np.ndarray | None = None

    @classmethod
    def from_file(cls, filepath: Path) -> "ExcitonFatband":
        file_path = Path(filepath)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        instance = cls(filepath=file_path)
        instance.xdim, instance.nexciton = instance._read_dimensions()
        return instance

    def _read_dimensions(self) -> Tuple[int, int]:
        if self.filepath is None:
            raise ValueError("No BSEFATBAND filepath set.")
        with self.filepath.open("r", encoding="utf-8") as handle:
            line = handle.readline()
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"Invalid first line in {self.filepath}: {line.rstrip()}")
        return int(fields[0]), int(fields[1])

    def _require_parser_context(self) -> Tuple[Path, int, int]:
        if self.filepath is None or self.xdim is None or self.nexciton is None:
            raise RuntimeError("Parser context is missing. Initialize with ExcitonFatband.from_file(...).")
        return self.filepath, self.xdim, self.nexciton

    @classmethod
    def _parse_exciton_header(cls, line: str) -> "ExcitonFatband":
        match = HEADER_RE.match(line.rstrip("\n"))
        if not match:
            raise ValueError(f"Invalid exciton header line: {line.rstrip()}")
        return cls(
            index=int(match.group(1)),
            bse_eigenvalue=parse_float(match.group(2)),
            ip_eigenvalue=parse_float(match.group(3)),
        )

    @staticmethod
    def _parse_transition_line(line: str) -> Tuple[np.ndarray, float, float, float, int, int, complex]:
        fields = line.split()
        if len(fields) < 11:
            raise ValueError(f"Invalid transition line: {line.rstrip()}")
        k_frac = np.array(
            [parse_float(fields[0]), parse_float(fields[1]), parse_float(fields[2])],
            dtype=float,
        )
        evb = parse_float(fields[3])
        ecb = parse_float(fields[4])
        column_weight = parse_float(fields[5])
        ivb = int(fields[6])
        icb = int(fields[7])
        amplitude = complex(parse_float(fields[8]), parse_float(fields[10]))
        return k_frac, evb, ecb, column_weight, ivb, icb, amplitude

    def iter_metadata(self) -> Iterable["ExcitonFatband"]:
        filepath, xdim, nexciton = self._require_parser_context()
        with filepath.open("r", encoding="utf-8") as handle:
            _ = handle.readline()
            for _ in range(nexciton):
                header = handle.readline()
                if not header:
                    raise EOFError("Unexpected EOF while reading exciton header.")
                meta = self._parse_exciton_header(header)
                yield meta
                for _ in range(xdim):
                    if not handle.readline():
                        raise EOFError("Unexpected EOF while skipping transition data.")

    def read_excitons(self, indices: Sequence[int]) -> List["ExcitonFatband"]:
        _, xdim, nexciton = self._require_parser_context()
        wanted = sorted(set(indices))
        if not wanted:
            return []
        for idx in wanted:
            if idx < 1 or idx > nexciton:
                raise ValueError(f"Exciton index out of range: {idx}, valid range is 1..{nexciton}")

        wanted_set = set(wanted)
        found: Dict[int, ExcitonFatband] = {}
        filepath = self.filepath
        if filepath is None:
            raise RuntimeError("Parser filepath is missing.")
        with filepath.open("r", encoding="utf-8") as handle:
            _ = handle.readline()
            for _ in range(nexciton):
                header = handle.readline()
                if not header:
                    raise EOFError("Unexpected EOF while reading exciton header.")
                meta = self._parse_exciton_header(header)
                if meta.index is None:
                    raise ValueError("Parsed exciton header without index.")
                parse_block = meta.index in wanted_set

                if parse_block:
                    k_frac = np.empty((xdim, 3), dtype=float)
                    evb = np.empty(xdim, dtype=float)
                    ecb = np.empty(xdim, dtype=float)
                    column_weight = np.empty(xdim, dtype=float)
                    ivb = np.empty(xdim, dtype=int)
                    icb = np.empty(xdim, dtype=int)
                    amplitude = np.empty(xdim, dtype=np.complex128)

                    for i in range(xdim):
                        line = handle.readline()
                        if not line:
                            raise EOFError("Unexpected EOF while reading transition data.")
                        k, ev, ec, cw, vb, cb, amp = self._parse_transition_line(line)
                        k_frac[i] = k
                        evb[i] = ev
                        ecb[i] = ec
                        column_weight[i] = cw
                        ivb[i] = vb
                        icb[i] = cb
                        amplitude[i] = amp

                    found[meta.index] = ExcitonFatband(
                        index=meta.index,
                        bse_eigenvalue=meta.bse_eigenvalue,
                        ip_eigenvalue=meta.ip_eigenvalue,
                        k_frac=k_frac,
                        evb=evb,
                        ecb=ecb,
                        column_weight=column_weight,
                        ivb=ivb,
                        icb=icb,
                        amplitude=amplitude,
                    )
                else:
                    for _ in range(xdim):
                        if not handle.readline():
                            raise EOFError("Unexpected EOF while skipping transition data.")

        missing = [idx for idx in wanted if idx not in found]
        if missing:
            raise RuntimeError(f"Failed to read requested excitons: {missing}")
        return [found[idx] for idx in wanted]


# -----------------------------
# Generic utilities
# -----------------------------


def parse_float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "E"))


def write_scalar_vasp(path: Path, poscar_header_lines: Sequence[str], grid: np.ndarray, ncol: int = 10) -> None:
    data = np.asarray(grid, dtype=float)
    if data.ndim != 3:
        raise ValueError("Grid must be 3D.")
    nx, ny, nz = data.shape
    flat = data.flatten(order="F")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for line in poscar_header_lines:
            fh.write(line.rstrip("\n") + "\n")
        fh.write("\n")
        fh.write(f"{nx:5d}{ny:5d}{nz:5d}\n")
        for i in range(0, flat.size, ncol):
            chunk = flat[i:i+ncol]
            fh.write("".join(f"{x:16.8E}" for x in chunk) + "\n")


def read_poscar(path: Path) -> Tuple[List[str], np.ndarray, np.ndarray, List[str], List[int], str, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 8:
        raise ValueError(f"POSCAR too short: {path}")
    comment = lines[0]
    scale = parse_float(lines[1].split()[0])
    lattice = np.array([[parse_float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], dtype=float)
    lattice *= scale

    species_line = 5
    try:
        counts = [int(x) for x in lines[6].split()]
        species = lines[5].split()
    except ValueError:
        species = [f"X{i+1}" for i in range(len(lines[5].split()))]
        counts = [int(x) for x in lines[5].split()]
        species_line = 4
        raise ValueError("POSCAR without species names is not supported by this helper.")

    coord_line = 7
    selective = False
    if lines[7].strip().lower().startswith("s"):
        selective = True
        coord_line = 8
    coord_mode = lines[coord_line].strip()
    nat = sum(counts)
    start = coord_line + 1
    coords = []
    for i in range(nat):
        coords.append([parse_float(x) for x in lines[start + i].split()[:3]])
    coords = np.array(coords, dtype=float)

    if coord_mode.lower().startswith("c"):
        coords = coords @ np.linalg.inv(lattice)
    elif not coord_mode.lower().startswith("d"):
        raise ValueError(f"Unsupported coordinate mode in POSCAR: {coord_mode}")

    header_lines = lines[:start + nat]
    return header_lines, lattice, coords, species, counts, coord_mode, coords


def make_diagonal_supercell_poscar(
    poscar_path: Path,
    supercell: Tuple[int, int, int],
    output_path: Path,
    shift_frac: np.ndarray | None = None,
) -> List[str]:
    sx, sy, sz = supercell
    lines = poscar_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 8:
        raise ValueError(f"POSCAR too short: {poscar_path}")

    atoms = ase_read(poscar_path)
    transform = np.diag([sx, sy, sz])
    sc_atoms = make_supercell(atoms, transform)
    if "momenta" in sc_atoms.arrays:
        del sc_atoms.arrays["momenta"]

    if shift_frac is not None:
        scaled = sc_atoms.get_scaled_positions(wrap=False)
        scaled = wrap_frac(scaled + np.asarray(shift_frac, dtype=float)[None, :])
        sc_atoms.set_scaled_positions(scaled)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(output_path, sc_atoms, vasp5=True, direct=True, sort=True)

    header = output_path.read_text(encoding="utf-8").splitlines()
    header[0] = lines[0] + f" | supercell {sx}x{sy}x{sz}"
    output_path.write_text("\n".join(header) + "\n", encoding="utf-8")
    return header


def parse_triplet(text: str) -> Tuple[int, int, int]:
    vals = [int(x) for x in text.replace(",", " ").split()]
    if len(vals) != 3:
        raise ValueError("Expected three integers for supercell, e.g. '6 6 1'.")
    if any(v <= 0 for v in vals):
        raise ValueError("Supercell entries must be positive.")
    return vals[0], vals[1], vals[2]


def parse_frac_triplet(text: str) -> np.ndarray:
    vals = [float(x) for x in text.replace(",", " ").split()]
    if len(vals) != 3:
        raise ValueError("Expected three floats for hole position, e.g. '0.333333 0.666667 0.5'.")
    return np.array(vals, dtype=float)


def wrap_frac(frac: np.ndarray) -> np.ndarray:
    return frac - np.floor(frac)


def wrap_frac_signed(frac: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(frac, dtype=float) + 0.5, 1.0) - 0.5


def nearest_grid_index(frac: np.ndarray, shape: Sequence[int]) -> Tuple[int, int, int]:
    # Match VASP EXCITON_WF in bse.F:
    #   NG = FLOOR(R*NGPTAR) + 1    (Fortran 1-based)
    # which corresponds to 0-based Python indices idx = FLOOR(R*N).
    idx = np.floor(wrap_frac(frac) * np.array(shape, dtype=float)).astype(int)
    idx %= np.array(shape, dtype=int)
    return int(idx[0]), int(idx[1]), int(idx[2])


def infer_regular_mesh(k_frac: np.ndarray, tol: float = 1e-8) -> Tuple[int, int, int] | None:
    dims = []
    for ax in range(3):
        vals = canonical_frac(k_frac[:, ax], decimals=8)
        uniq = np.unique(vals)
        dims.append(len(uniq))
    prod = dims[0] * dims[1] * dims[2]
    canonical = canonical_frac(k_frac, decimals=8)
    if prod == len(np.unique(canonical, axis=0)):
        return int(dims[0]), int(dims[1]), int(dims[2])
    return None


def threshold_indices(weights: np.ndarray, cumulative: float) -> np.ndarray:
    if cumulative >= 1.0:
        return np.arange(weights.size)
    order = np.argsort(weights)[::-1]
    sorted_w = weights[order]
    csum = np.cumsum(sorted_w)
    total = float(csum[-1]) if len(csum) else 0.0
    if total <= 0:
        return np.array([], dtype=int)
    keep_n = int(np.searchsorted(csum / total, cumulative, side="left")) + 1
    return np.sort(order[:keep_n])


def circular_shift_grid(grid: np.ndarray, shift: Sequence[int]) -> np.ndarray:
    shift_i = tuple(int(x) for x in np.asarray(shift, dtype=int))
    if not any(shift_i):
        return np.array(grid, copy=True)
    return np.roll(grid, shift_i, axis=(0, 1, 2))


def center_shift_from_hole_index(
    hole_idx: Sequence[int],
    primitive_grid: Sequence[int],
    supercell: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    hole_idx = np.asarray(hole_idx, dtype=int)
    super_grid = np.asarray(primitive_grid, dtype=int) * np.asarray(supercell, dtype=int)
    shift = (super_grid // 2 - hole_idx) % super_grid
    return shift, shift / super_grid.astype(float)


def center_shift_from_density_expectation(
    rho: np.ndarray,
    tol: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 3:
        raise ValueError("Density grid must be 3D.")
    total = float(np.sum(rho))
    grid = np.asarray(rho.shape, dtype=int)
    if total <= 0:
        zero = np.zeros(3, dtype=int)
        return zero, np.zeros(3, dtype=float), np.zeros(3, dtype=float)

    mean_idx = np.zeros(3, dtype=float)
    for axis, n in enumerate(grid):
        other_axes = tuple(i for i in range(3) if i != axis)
        marginal = np.sum(rho, axis=other_axes)
        phases = np.exp(2j * np.pi * np.arange(n, dtype=float) / float(n))
        moment = complex(np.dot(marginal, phases))
        if abs(moment) <= tol * float(np.sum(marginal)):
            mean_idx[axis] = 0.0
            continue
        angle = float(np.angle(moment))
        if angle < 0:
            angle += 2.0 * np.pi
        mean_idx[axis] = angle * float(n) / (2.0 * np.pi)

    shift = np.mod(np.rint(grid.astype(float) / 2.0 - mean_idx).astype(int), grid)
    return shift, shift / grid.astype(float), mean_idx / grid.astype(float)


def sample_supercell_hole(
    hole_super_frac: np.ndarray,
    primitive_grid: Sequence[int],
    supercell: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    primitive_grid = np.asarray(primitive_grid, dtype=int)
    super_grid = primitive_grid * np.asarray(supercell, dtype=int)
    hole_super_idx = np.asarray(nearest_grid_index(hole_super_frac, super_grid), dtype=int)
    hole_super_sampled = hole_super_idx / super_grid.astype(float)
    hole_primitive_idx = hole_super_idx % primitive_grid
    hole_primitive_sampled = hole_primitive_idx / primitive_grid.astype(float)
    placement_shift = (hole_super_idx - hole_primitive_idx) % super_grid
    return hole_super_idx, hole_super_sampled, hole_primitive_idx, hole_primitive_sampled, placement_shift


def sample_supercell_point(
    point_super_frac: np.ndarray,
    primitive_grid: Sequence[int],
    supercell: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return sample_supercell_hole(point_super_frac, primitive_grid, supercell)


def origin_supercell_image_from_primitive_frac(
    point_primitive_frac: np.ndarray,
    supercell: Tuple[int, int, int],
) -> np.ndarray:
    return wrap_frac(np.asarray(point_primitive_frac, dtype=float)) / np.asarray(supercell, dtype=float)


def reconstruct_fixed_hole(
    recon: "ExcitonConstructor",
    block: "ExcitonFatband",
    hole: np.ndarray,
    ngrid: np.ndarray,
    supercell: Tuple[int, int, int],
    keep: np.ndarray,
) -> ReconstructionResult:
    weights = np.abs(block.amplitude) ** 2
    used_weight_fraction = float(np.sum(weights[keep]) / np.sum(weights)) if np.sum(weights) > 0 else 0.0

    beta: Dict[Tuple[int, int, int, bool, int, Tuple[float, float, float]], complex] = {}
    for i in keep:
        kmatch = recon.wav_kpoint_match(block.k_frac[i])
        vb = int(block.ivb[i])
        cb = int(block.icb[i])
        spin_match = recon.infer_transition_spin(
            ikpt=kmatch.ikpt,
            ivb=vb,
            evb=float(block.evb[i]),
            icb=cb,
            ecb=float(block.ecb[i]),
        )
        A = complex(block.amplitude[i])
        psi_v_at_h = recon.sample_band_at_point(
            ispin=spin_match.valence_spin,
            ikpt=kmatch.ikpt,
            iband=vb,
            frac=hole,
            ngrid=ngrid,
            time_reversed=kmatch.time_reversed,
            symm_op=kmatch.symm_op,
        )
        ktuple = tuple(float(x) for x in np.asarray(block.k_frac[i], dtype=float))
        key = (kmatch.ikpt, cb, spin_match.conduction_spin, kmatch.time_reversed, kmatch.symm_op, ktuple)
        beta[key] = beta.get(key, 0.0 + 0.0j) + A * np.conjugate(psi_v_at_h)

    super_grid = np.asarray(ngrid, dtype=int) * np.asarray(supercell, dtype=int)
    if recon.can_use_reciprocal_reconstruction():
        psi_q = np.zeros(tuple(int(x) for x in super_grid), dtype=np.complex128)
        supercell_arr = np.asarray(supercell, dtype=float)
        for (ikpt, cb, cspin, time_reversed, symm_op, _), beta_coeff in beta.items():
            q_frac, band_coeff = recon.matched_full_bloch_coefficients(
                ispin=cspin,
                ikpt=ikpt,
                iband=cb,
                time_reversed=time_reversed,
                symm_op=symm_op,
            )
            q_super = q_frac * supercell_arr[None, :]
            q_index = np.rint(q_super).astype(int)
            mismatch = np.max(np.abs(q_super - q_index))
            if mismatch > 1e-6:
                raise ValueError(
                    f"Transformed reciprocal coefficients do not map onto the inferred supercell grid {supercell}; "
                    f"largest mismatch = {mismatch:.3e}."
                )
            q_index %= super_grid[None, :]
            np.add.at(
                psi_q,
                (q_index[:, 0], q_index[:, 1], q_index[:, 2]),
                beta_coeff * band_coeff,
            )
        psi_e = ifftn(psi_q) * math.sqrt(np.prod(super_grid))
    else:
        psi_e = np.zeros(tuple(int(x) for x in super_grid), dtype=np.complex128)
        for (ikpt, cb, cspin, time_reversed, symm_op, ktuple), coeff in beta.items():
            kfrac = np.array(ktuple, dtype=float)
            psi_c_pc = recon.primitive_wavefunction_at_k(
                ispin=cspin,
                ikpt=ikpt,
                iband=cb,
                ngrid=ngrid,
                time_reversed=time_reversed,
                symm_op=symm_op,
            )
            psi_c_sc = recon.tile_to_supercell(psi_c_pc, kfrac, supercell)
            psi_e += coeff * psi_c_sc

    rho = np.abs(psi_e) ** 2
    norm = float(np.sum(rho))
    if norm > 0:
        rho /= norm
        psi_e /= math.sqrt(norm)

    return ReconstructionResult(
        psi_electron=psi_e,
        density=rho,
        raw_density_sum=norm,
        used_transitions=len(keep),
        total_transitions=len(weights),
        used_weight_fraction=used_weight_fraction,
        supercell=supercell,
        primitive_grid=tuple(int(x) for x in np.asarray(ngrid, dtype=int)),
        supercell_grid=tuple(int(x) for x in psi_e.shape),
    )


def reconstruct_fixed_electron(
    recon: "ExcitonConstructor",
    block: "ExcitonFatband",
    electron: np.ndarray,
    ngrid: np.ndarray,
    supercell: Tuple[int, int, int],
    keep: np.ndarray,
) -> ReconstructionResult:
    weights = np.abs(block.amplitude) ** 2
    used_weight_fraction = float(np.sum(weights[keep]) / np.sum(weights)) if np.sum(weights) > 0 else 0.0

    gamma: Dict[Tuple[int, int, int, bool, int, Tuple[float, float, float]], complex] = {}
    for i in keep:
        kmatch = recon.wav_kpoint_match(block.k_frac[i])
        vb = int(block.ivb[i])
        cb = int(block.icb[i])
        spin_match = recon.infer_transition_spin(
            ikpt=kmatch.ikpt,
            ivb=vb,
            evb=float(block.evb[i]),
            icb=cb,
            ecb=float(block.ecb[i]),
        )
        A = complex(block.amplitude[i])
        psi_c_at_e = recon.sample_band_at_point(
            ispin=spin_match.conduction_spin,
            ikpt=kmatch.ikpt,
            iband=cb,
            frac=electron,
            ngrid=ngrid,
            time_reversed=kmatch.time_reversed,
            symm_op=kmatch.symm_op,
        )
        ktuple = tuple(float(x) for x in np.asarray(block.k_frac[i], dtype=float))
        key = (kmatch.ikpt, vb, spin_match.valence_spin, kmatch.time_reversed, kmatch.symm_op, ktuple)
        kfrac = np.asarray(block.k_frac[i], dtype=float)
        ephi = np.exp(2j * np.pi * np.dot(kfrac, np.asarray(electron, dtype=float)))
        gamma[key] = gamma.get(key, 0.0 + 0.0j) + A * np.conjugate(psi_c_at_e) * (ephi * ephi)

    super_grid = np.asarray(ngrid, dtype=int) * np.asarray(supercell, dtype=int)
    if recon.can_use_reciprocal_reconstruction():
        psi_q = np.zeros(tuple(int(x) for x in super_grid), dtype=np.complex128)
        supercell_arr = np.asarray(supercell, dtype=float)
        for (ikpt, vb, vspin, time_reversed, symm_op, _), gamma_coeff in gamma.items():
            q_frac, band_coeff = recon.matched_full_bloch_coefficients(
                ispin=vspin,
                ikpt=ikpt,
                iband=vb,
                time_reversed=time_reversed,
                symm_op=symm_op,
            )
            q_super = q_frac * supercell_arr[None, :]
            q_index = np.rint(q_super).astype(int)
            mismatch = np.max(np.abs(q_super - q_index))
            if mismatch > 1e-6:
                raise ValueError(
                    f"Transformed reciprocal coefficients do not map onto the inferred supercell grid {supercell}; "
                    f"largest mismatch = {mismatch:.3e}."
                )
            q_index %= super_grid[None, :]
            np.add.at(
                psi_q,
                (q_index[:, 0], q_index[:, 1], q_index[:, 2]),
                gamma_coeff * band_coeff,
            )
        psi_h = ifftn(psi_q) * math.sqrt(np.prod(super_grid))
    else:
        psi_h = np.zeros(tuple(int(x) for x in super_grid), dtype=np.complex128)
        for (ikpt, vb, vspin, time_reversed, symm_op, ktuple), coeff in gamma.items():
            kfrac = np.array(ktuple, dtype=float)
            psi_v_pc = recon.primitive_wavefunction_at_k(
                ispin=vspin,
                ikpt=ikpt,
                iband=vb,
                ngrid=ngrid,
                time_reversed=time_reversed,
                symm_op=symm_op,
            )
            psi_v_sc = recon.tile_to_supercell(psi_v_pc, kfrac, supercell)
            psi_h += coeff * psi_v_sc

    rho = np.abs(psi_h) ** 2
    norm = float(np.sum(rho))
    if norm > 0:
        rho /= norm
        psi_h /= math.sqrt(norm)

    return ReconstructionResult(
        psi_electron=psi_h,
        density=rho,
        raw_density_sum=norm,
        used_transitions=len(keep),
        total_transitions=len(weights),
        used_weight_fraction=used_weight_fraction,
        supercell=supercell,
        primitive_grid=tuple(int(x) for x in np.asarray(ngrid, dtype=int)),
        supercell_grid=tuple(int(x) for x in psi_h.shape),
    )


def dominant_fixed_particle_from_block(
    recon: "ExcitonConstructor",
    block: "ExcitonFatband",
    ngrid: np.ndarray,
    supercell: Tuple[int, int, int],
    fixed_particle: str,
) -> DominantFixedParticle:
    weights = np.abs(block.amplitude) ** 2
    if weights.size == 0:
        raise ValueError("Exciton block has no transitions.")

    idx = int(np.argmax(weights))
    kmatch = recon.wav_kpoint_match(block.k_frac[idx])
    vb = int(block.ivb[idx])
    cb = int(block.icb[idx])
    spin_match = recon.infer_transition_spin(
        ikpt=kmatch.ikpt,
        ivb=vb,
        evb=float(block.evb[idx]),
        icb=cb,
        ecb=float(block.ecb[idx]),
    )
    if fixed_particle == "hole":
        source_state = "valence"
        selected_band = vb
        selected_spin = spin_match.valence_spin
    elif fixed_particle == "electron":
        source_state = "conduction"
        selected_band = cb
        selected_spin = spin_match.conduction_spin
    else:
        raise ValueError(f"Unsupported fixed particle '{fixed_particle}'.")

    psi_fixed = recon.primitive_wavefunction_at_k(
        ispin=selected_spin,
        ikpt=kmatch.ikpt,
        iband=selected_band,
        ngrid=ngrid,
        time_reversed=kmatch.time_reversed,
        symm_op=kmatch.symm_op,
    )
    rho_fixed = np.abs(psi_fixed) ** 2
    primitive_index = np.array(np.unravel_index(int(np.argmax(rho_fixed)), rho_fixed.shape), dtype=int)
    primitive_frac = primitive_index / np.asarray(ngrid, dtype=float)
    supercell_frac = primitive_frac / np.asarray(supercell, dtype=float)

    return DominantFixedParticle(
        transition_index=idx,
        k_frac=np.asarray(block.k_frac[idx], dtype=float),
        ikpt=kmatch.ikpt,
        time_reversed=kmatch.time_reversed,
        symm_op=kmatch.symm_op,
        valence_band=vb,
        conduction_band=cb,
        valence_spin=spin_match.valence_spin,
        conduction_spin=spin_match.conduction_spin,
        amplitude=complex(block.amplitude[idx]),
        weight=float(weights[idx]),
        fixed_particle=fixed_particle,
        source_state=source_state,
        selected_band=selected_band,
        selected_spin=selected_spin,
        primitive_index=tuple(int(x) for x in primitive_index),
        primitive_frac=primitive_frac,
        supercell_frac=supercell_frac,
    )


# -----------------------------
# Exciton reconstruction
# -----------------------------

@dataclass
class ReconstructionResult:
    psi_electron: np.ndarray
    density: np.ndarray
    raw_density_sum: float
    used_transitions: int
    total_transitions: int
    used_weight_fraction: float
    supercell: Tuple[int, int, int]
    primitive_grid: Tuple[int, int, int]
    supercell_grid: Tuple[int, int, int]


@dataclass(frozen=True)
class KpointMatch:
    ikpt: int
    time_reversed: bool = False
    symm_op: int = 0


@dataclass(frozen=True)
class FullBZKpoint:
    k_frac: np.ndarray
    ikpt: int
    time_reversed: bool = False


@dataclass(frozen=True)
class TransitionSpinMatch:
    valence_spin: int
    conduction_spin: int


@dataclass(frozen=True)
class DominantFixedParticle:
    transition_index: int
    k_frac: np.ndarray
    ikpt: int
    time_reversed: bool
    symm_op: int
    valence_band: int
    conduction_band: int
    valence_spin: int
    conduction_spin: int
    amplitude: complex
    weight: float
    fixed_particle: str
    source_state: str
    selected_band: int
    selected_spin: int
    primitive_index: Tuple[int, int, int]
    primitive_frac: np.ndarray
    supercell_frac: np.ndarray


def kpoint_match_priority(match: KpointMatch) -> Tuple[int, int, int, int]:
    return (
        int(match.time_reversed),
        int(match.symm_op != 0),
        int(match.symm_op),
        int(match.ikpt),
    )


@dataclass(frozen=True)
class SymmetryOp:
    irot: int
    real_matrix: np.ndarray
    reciprocal_matrix: np.ndarray
    tau_frac: np.ndarray


def rodrigues_rotation(axis: Sequence[float], angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3, dtype=float)
    axis /= norm
    theta = math.radians(angle_deg)
    x, y, z = axis
    K = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )
    ident = np.eye(3, dtype=float)
    return (
        math.cos(theta) * ident
        + (1.0 - math.cos(theta)) * np.outer(axis, axis)
        + math.sin(theta) * K
    )


def parse_outcar_symmetry_ops(outcar: Path, lattice: np.ndarray, tol: float = 1e-5) -> List[SymmetryOp]:
    lines = outcar.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "Space group operators:":
            start = i + 2
            break
    if start is None:
        return []

    basis = np.asarray(lattice, dtype=float).T
    ops: List[SymmetryOp] = []
    for line in lines[start:]:
        text = line.strip()
        if not text:
            if ops:
                break
            continue
        if text.startswith("Subroutine"):
            break
        parts = text.split()
        if len(parts) != 9:
            continue
        try:
            irot = int(parts[0])
            det_a = parse_float(parts[1])
            alpha = parse_float(parts[2])
            axis = [parse_float(x) for x in parts[3:6]]
            tau = np.array([parse_float(x) for x in parts[6:9]], dtype=float)
        except ValueError:
            continue

        rot_cart = det_a * rodrigues_rotation(axis, alpha)
        rot_frac = np.linalg.inv(basis) @ rot_cart @ basis
        rot_frac_i = np.rint(rot_frac).astype(int)
        if np.max(np.abs(rot_frac - rot_frac_i)) > tol:
            raise ValueError(
                f"Failed to convert symmetry operator {irot} from {outcar} into an integer lattice-basis matrix."
            )
        recip_frac = np.linalg.inv(rot_frac_i).T
        recip_frac_i = np.rint(recip_frac).astype(int)
        if np.max(np.abs(recip_frac - recip_frac_i)) > tol:
            raise ValueError(
                f"Failed to convert reciprocal symmetry operator {irot} from {outcar} into an integer matrix."
            )
        ops.append(
            SymmetryOp(
                irot=irot,
                real_matrix=rot_frac_i,
                reciprocal_matrix=recip_frac_i,
                tau_frac=tau,
            )
        )
    return ops


def load_symmetry_ops(search_dirs: Sequence[Path], lattice: np.ndarray) -> List[SymmetryOp]:
    seen: set[Path] = set()
    for dpath in search_dirs:
        for name in ("OUTCAR.symm", "OUTCAR"):
            candidate = (dpath / name)
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            ops = parse_outcar_symmetry_ops(candidate, lattice)
            if ops:
                return ops
    ident = np.eye(3, dtype=int)
    return [SymmetryOp(irot=1, real_matrix=ident, reciprocal_matrix=ident, tau_frac=np.zeros(3, dtype=float))]


def parse_outcar_full_bz_kpoints(outcar: Path) -> List[FullBZKpoint]:
    lines = outcar.read_text(encoding="utf-8", errors="ignore").splitlines()
    for i, line in enumerate(lines):
        if "Subroutine IBZKPT_HF returns following result" not in line:
            continue
        start = None
        for j in range(i + 1, len(lines)):
            if "Following reciprocal coordinates:" in lines[j]:
                start = j + 1
                break
        if start is None:
            continue

        entries: List[FullBZKpoint] = []
        for line in lines[start:]:
            text = line.strip()
            if not text:
                if entries:
                    return entries
                continue
            fields = text.split()
            if len(fields) < 7:
                if entries:
                    return entries
                continue
            try:
                k_frac = np.array([parse_float(fields[0]), parse_float(fields[1]), parse_float(fields[2])], dtype=float)
                ikpt = int(fields[4])
            except ValueError:
                if entries:
                    return entries
                continue
            time_reversed = fields[6].upper().startswith("T")
            entries.append(
                FullBZKpoint(
                    k_frac=wrap_frac(k_frac),
                    ikpt=ikpt,
                    time_reversed=time_reversed,
                )
            )
        if entries:
            return entries
    return []


def load_full_bz_kpoints(search_dirs: Sequence[Path]) -> List[FullBZKpoint]:
    seen: set[Path] = set()
    for dpath in search_dirs:
        for name in ("OUTCAR.symm", "OUTCAR"):
            candidate = dpath / name
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            entries = parse_outcar_full_bz_kpoints(candidate)
            if entries:
                return entries
    return []


def signed_permutation_transform(
    psi: np.ndarray,
    real_matrix: np.ndarray,
    tau_frac: np.ndarray,
    k_frac: np.ndarray | None = None,
    tol: float = 1e-6,
) -> np.ndarray:
    real_matrix = np.asarray(real_matrix, dtype=int)
    if real_matrix.shape != (3, 3):
        raise ValueError("Symmetry matrix must be 3x3.")
    if not np.all(np.sum(np.abs(real_matrix), axis=0) == 1) or not np.all(np.sum(np.abs(real_matrix), axis=1) == 1):
        raise ValueError("Only signed-permutation symmetry matrices are supported for wavefunction restoration.")

    s_inv = np.linalg.inv(real_matrix)
    s_inv_i = np.rint(s_inv).astype(int)
    if np.max(np.abs(s_inv - s_inv_i)) > tol:
        raise ValueError("Failed to invert symmetry matrix as an integer signed permutation.")
    shift_frac = -s_inv_i @ np.asarray(tau_frac, dtype=float)

    perm: List[int] = []
    signs: List[int] = []
    shifts: List[int] = []
    for out_axis in range(3):
        src_axis = int(np.argmax(np.abs(s_inv_i[:, out_axis])))
        sign = int(s_inv_i[src_axis, out_axis])
        if sign == 0:
            raise ValueError("Invalid signed-permutation symmetry matrix.")
        perm.append(src_axis)
        signs.append(sign)
        shift_grid = shift_frac[src_axis] * psi.shape[src_axis]
        shift_int = int(round(shift_grid))
        if abs(shift_grid - shift_int) > tol:
            raise ValueError(
                f"Symmetry translation {tau_frac} does not map cleanly onto FFT grid {psi.shape}."
            )
        shifts.append(shift_int)

    out = np.transpose(psi, axes=perm)
    k_vec = None if k_frac is None else np.asarray(k_frac, dtype=float)
    for axis, (src_axis, sign, shift_int) in enumerate(zip(perm, signs, shifts)):
        n = out.shape[axis]
        unwrapped = sign * np.arange(n, dtype=int) + shift_int
        idx = unwrapped % n
        out = np.take(out, idx, axis=axis)
        if k_vec is not None:
            wraps = np.floor_divide(unwrapped, n)
            phase = np.exp(2j * np.pi * k_vec[src_axis] * wraps)
            shape = [1] * out.ndim
            shape[axis] = n
            out = out * phase.reshape(shape)
    return out


def bloch_wavefunction_from_full_coefficients(
    q_frac: np.ndarray,
    coeff: np.ndarray,
    ngrid: np.ndarray,
    tol: float = 1e-6,
) -> np.ndarray:
    q_frac = np.asarray(q_frac, dtype=float)
    coeff = np.asarray(coeff, dtype=np.complex128)
    ngrid = np.asarray(ngrid, dtype=int)
    if q_frac.ndim != 2 or q_frac.shape[1] != 3:
        raise ValueError("Full Bloch coefficients must have shape (nplane, 3).")
    if coeff.ndim != 1 or coeff.shape[0] != q_frac.shape[0]:
        raise ValueError("Plane-wave coefficient array is incompatible with q vectors.")

    k_frac = wrap_frac(q_frac[0])
    gvec = q_frac - k_frac[None, :]
    gvec_i = np.rint(gvec).astype(int)
    mismatch = float(np.max(np.abs(gvec - gvec_i))) if gvec.size else 0.0
    if mismatch > tol:
        raise ValueError(
            "Transformed full-Bloch coefficients do not separate into one k vector plus integer G vectors; "
            f"largest mismatch = {mismatch:.3e}."
        )

    phi_q = np.zeros(tuple(int(x) for x in ngrid), dtype=np.complex128)
    gmod = gvec_i % ngrid[None, :]
    np.add.at(phi_q, (gmod[:, 0], gmod[:, 1], gmod[:, 2]), coeff)
    psi = ifftn(phi_q) * math.sqrt(np.prod(ngrid))

    grid_frac = (
        np.mgrid[0:ngrid[0], 0:ngrid[1], 0:ngrid[2]]
        .reshape((3, int(np.prod(ngrid))))
        .T
        / ngrid.astype(float)
    )
    phase = np.exp(2j * np.pi * (grid_frac @ k_frac)).reshape(tuple(int(x) for x in ngrid))
    return psi * phase


class ExcitonConstructor:
    def __init__(
        self,
        wavecar: Path,
        bsefatband: Path,
        poscar: Path,
        lsorbit: bool = False,
        lgamma: bool = False,
        gamma_half: str = "x",
        spin_channel: int | None = None,
    ) -> None:
        self.poscar = poscar
        _, lattice, _, _, _, _, _ = read_poscar(poscar)
        search_dirs = [
            Path(str(wavecar)).resolve().parent,
            Path(str(bsefatband)).resolve().parent,
            Path(str(poscar)).resolve().parent,
        ]
        self.symm_ops = load_symmetry_ops(search_dirs, lattice)
        self.wfc = vaspwfc(
            str(wavecar),
            lsorbit=lsorbit,
            lgamma=lgamma,
            gamma_half=gamma_half,
        )
        if spin_channel is not None and not (1 <= spin_channel <= self.wfc._nspin):
            raise ValueError(f"--spin-channel must be in 1..{self.wfc._nspin}")
        self.spin_channel = spin_channel
        self.fatband = ExcitonFatband.from_file(bsefatband)
        self._psi_cache: Dict[Tuple[int, int, int, Tuple[int, int, int]], np.ndarray] = {}
        self._psi_match_cache: Dict[Tuple[int, int, int, Tuple[int, int, int], bool, int], np.ndarray] = {}
        self._coeff_cache: Dict[Tuple[int, int, int, bool, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._sample_cache: Dict[Tuple[int, int, int, Tuple[int, int, int], bool, int], complex] = {}
        self._resolved_kpoint_cache: Dict[Tuple[float, float, float], KpointMatch] = {}
        self._kpoint_lookup: Dict[Tuple[float, float, float], int] = {}
        self._expanded_kpoint_lookup: Dict[Tuple[float, float, float], KpointMatch] = {}
        self._full_bz_lookup: Dict[Tuple[float, float, float], KpointMatch] = {}
        self._transition_spin_cache: Dict[Tuple[int, int, float, int, float], TransitionSpinMatch] = {}
        self._expanded_kpoint_coords: List[np.ndarray] = []
        self._expanded_kpoint_matches: List[KpointMatch] = []
        for ik, k in enumerate(np.asarray(self.wfc._kvecs, dtype=float), start=1):
            key = frac_key(k, decimals=8)
            if key not in self._kpoint_lookup:
                self._kpoint_lookup[key] = ik
            for isym, op in enumerate(self.symm_ops):
                k_symm = wrap_frac(op.reciprocal_matrix @ np.asarray(k, dtype=float))
                match = KpointMatch(ikpt=ik, time_reversed=False, symm_op=isym)
                self._store_expanded_match(k_symm, match)
                self._expanded_kpoint_coords.append(k_symm)
                self._expanded_kpoint_matches.append(match)
                k_symm_tr = wrap_frac(-k_symm)
                match_tr = KpointMatch(ikpt=ik, time_reversed=True, symm_op=isym)
                self._store_expanded_match(k_symm_tr, match_tr)
                self._expanded_kpoint_coords.append(k_symm_tr)
                self._expanded_kpoint_matches.append(match_tr)
        self._expanded_kpoint_coords_arr = np.asarray(self._expanded_kpoint_coords, dtype=float)
        self._full_bz_kpoint_coords_arr = np.empty((0, 3), dtype=float)
        self._full_bz_kpoint_matches: List[KpointMatch] = []
        full_bz_entries = load_full_bz_kpoints(search_dirs)
        full_bz_coords: List[np.ndarray] = []
        full_bz_matches: List[KpointMatch] = []
        for entry in full_bz_entries:
            if not (1 <= entry.ikpt <= self.wfc._nkpts):
                continue
            match = self._resolve_symmetry_match(entry.k_frac, entry.ikpt, entry.time_reversed)
            if match is None:
                continue
            self._store_full_bz_match(entry.k_frac, match)
            full_bz_coords.append(wrap_frac(entry.k_frac))
            full_bz_matches.append(match)
        if full_bz_coords:
            self._full_bz_kpoint_coords_arr = np.asarray(full_bz_coords, dtype=float)
            self._full_bz_kpoint_matches = full_bz_matches

    def _store_expanded_match(self, k_frac: np.ndarray, match: KpointMatch) -> None:
        key = frac_key(k_frac, decimals=8)
        current = self._expanded_kpoint_lookup.get(key)
        if current is None or kpoint_match_priority(match) < kpoint_match_priority(current):
            self._expanded_kpoint_lookup[key] = match

    def _store_full_bz_match(self, k_frac: np.ndarray, match: KpointMatch) -> None:
        key = frac_key(k_frac, decimals=5)
        current = self._full_bz_lookup.get(key)
        if current is None or kpoint_match_priority(match) < kpoint_match_priority(current):
            self._full_bz_lookup[key] = match

    def _resolve_symmetry_match(
        self,
        target_k: np.ndarray,
        ikpt: int,
        time_reversed: bool = False,
        tol: float = 5e-5,
    ) -> KpointMatch | None:
        base_k = np.asarray(self.wfc._kvecs[ikpt - 1], dtype=float)
        if time_reversed:
            base_k = -base_k
        target_k = wrap_frac(np.asarray(target_k, dtype=float))
        candidates: List[Tuple[float, KpointMatch]] = []
        for isym, op in enumerate(self.symm_ops):
            trial = wrap_frac(op.reciprocal_matrix @ base_k)
            diff = wrap_frac_signed(trial - target_k)
            max_abs = float(np.max(np.abs(diff)))
            if max_abs <= tol:
                candidates.append((max_abs, KpointMatch(ikpt=ikpt, time_reversed=time_reversed, symm_op=isym)))
        if not candidates:
            return None
        best = min(item[0] for item in candidates)
        tied = [item for item in candidates if item[0] <= best + 1e-8]
        tied.sort(key=lambda item: kpoint_match_priority(item[1]))
        return tied[0][1]

    def _best_kpoint_match(
        self,
        k_frac: np.ndarray,
        coords: np.ndarray,
        matches: Sequence[KpointMatch],
        tol: float = 5e-5,
    ) -> KpointMatch | None:
        if coords.size == 0:
            return None
        diff = wrap_frac_signed(coords - k_frac[None, :])
        max_abs = np.max(np.abs(diff), axis=1)
        best = float(np.min(max_abs))
        if best > tol:
            return None
        candidates = np.where(max_abs <= best + 1e-8)[0]
        idx = min(candidates, key=lambda i: kpoint_match_priority(matches[int(i)]))
        return matches[int(idx)]

    def _band_spin_candidates(
        self,
        ikpt: int,
        iband: int,
        energy: float,
        tol: float = 5e-4,
    ) -> List[Tuple[float, int]]:
        if not (1 <= iband <= self.wfc._nbands):
            return []
        candidates: List[Tuple[float, int]] = []
        for ispin in range(1, self.wfc._nspin + 1):
            ref = float(self.wfc._bands[ispin - 1, ikpt - 1, iband - 1])
            err = abs(ref - energy)
            if err <= tol:
                candidates.append((err, ispin))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates

    def infer_transition_spin(
        self,
        ikpt: int,
        ivb: int,
        evb: float,
        icb: int,
        ecb: float,
        tol: float = 5e-4,
    ) -> TransitionSpinMatch:
        if self.wfc._nspin == 1:
            return TransitionSpinMatch(valence_spin=1, conduction_spin=1)

        key = (int(ikpt), int(ivb), round(float(evb), 6), int(icb), round(float(ecb), 6))
        if key in self._transition_spin_cache:
            return self._transition_spin_cache[key]

        vb_candidates = self._band_spin_candidates(ikpt, ivb, evb, tol=tol)
        cb_candidates = self._band_spin_candidates(ikpt, icb, ecb, tol=tol)
        vb_spins = {spin for _, spin in vb_candidates}
        cb_spins = {spin for _, spin in cb_candidates}
        common = sorted(vb_spins & cb_spins)

        if self.spin_channel is not None:
            if self.spin_channel not in common:
                raise ValueError(
                    f"Transition (k={ikpt}, v={ivb}, Ev={evb:.6f}, c={icb}, Ec={ecb:.6f}) "
                    f"does not match requested spin channel {self.spin_channel} within tolerance {tol:.1e}."
                )
            match = TransitionSpinMatch(valence_spin=self.spin_channel, conduction_spin=self.spin_channel)
            self._transition_spin_cache[key] = match
            return match

        if len(common) == 1:
            spin = common[0]
            match = TransitionSpinMatch(valence_spin=spin, conduction_spin=spin)
            self._transition_spin_cache[key] = match
            return match

        if len(common) > 1:
            raise ValueError(
                f"Ambiguous spin assignment for transition (k={ikpt}, v={ivb}, Ev={evb:.6f}, c={icb}, Ec={ecb:.6f}). "
                "Both collinear spin channels match; rerun with --spin-channel."
            )

        if vb_candidates and cb_candidates:
            raise ValueError(
                f"Transition (k={ikpt}, v={ivb}, Ev={evb:.6f}, c={icb}, Ec={ecb:.6f}) "
                "matches different spin channels for valence and conduction. Spin-flip transitions are not supported."
            )

        raise ValueError(
            f"Could not infer a spin channel for transition (k={ikpt}, v={ivb}, Ev={evb:.6f}, c={icb}, Ec={ecb:.6f}) "
            f"from WAVECAR band energies within tolerance {tol:.1e}. "
            "If this is a spin-polarized calculation, inspect the band indexing or pass --spin-channel if the channels are exactly degenerate."
        )

    def wav_kpoint_match(self, k_frac: np.ndarray) -> KpointMatch:
        k_frac = np.asarray(k_frac, dtype=float)
        cache_key = frac_key(k_frac, decimals=5)
        if cache_key in self._resolved_kpoint_cache:
            return self._resolved_kpoint_cache[cache_key]
        k_frac = wrap_frac(k_frac)
        if cache_key in self._full_bz_lookup:
            match = self._full_bz_lookup[cache_key]
            self._resolved_kpoint_cache[cache_key] = match
            return match
        full_bz_match = self._best_kpoint_match(
            k_frac,
            self._full_bz_kpoint_coords_arr,
            self._full_bz_kpoint_matches,
        )
        if full_bz_match is not None:
            self._resolved_kpoint_cache[cache_key] = full_bz_match
            return full_bz_match
        lookup_key = frac_key(k_frac, decimals=8)
        if lookup_key in self._expanded_kpoint_lookup:
            match = self._expanded_kpoint_lookup[lookup_key]
            self._resolved_kpoint_cache[cache_key] = match
            return match
        expanded_match = self._best_kpoint_match(
            k_frac,
            self._expanded_kpoint_coords_arr,
            self._expanded_kpoint_matches,
        )
        if expanded_match is not None:
            self._resolved_kpoint_cache[cache_key] = expanded_match
            return expanded_match
        diff = wrap_frac_signed(self._expanded_kpoint_coords_arr - k_frac[None, :])
        best = float(np.min(np.linalg.norm(diff, axis=1)))
        raise ValueError(
            f"Failed to match BSE k point {k_frac} to WAVECAR k mesh; "
            f"nearest mismatch = {best:.3e}. "
            "Exact, time-reversed, and symmetry-equivalent matching were all attempted."
        )

    def primitive_wavefunction(self, ispin: int, ikpt: int, iband: int, ngrid: np.ndarray) -> np.ndarray:
        key = (ispin, ikpt, iband, tuple(int(x) for x in np.asarray(ngrid, dtype=int)))
        if key not in self._psi_cache:
            psi = self.wfc.wfc_r(ispin=ispin, ikpt=ikpt, iband=iband, ngrid=ngrid, kr_phase=True)
            if isinstance(psi, list):
                raise NotImplementedError(
                    "SOC/noncollinear spinor WAVECAR returned a two-component spinor. "
                    "This helper currently supports scalar wavefunctions only."
                )
            self._psi_cache[key] = np.asarray(psi, dtype=np.complex128)
        return self._psi_cache[key]

    def can_use_reciprocal_reconstruction(self) -> bool:
        return (not self.wfc._lsoc) and (not self.wfc._lgam)

    def matched_full_bloch_coefficients(
        self,
        ispin: int,
        ikpt: int,
        iband: int,
        time_reversed: bool = False,
        symm_op: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = (ispin, ikpt, iband, time_reversed, symm_op)
        if key not in self._coeff_cache:
            gvec = np.asarray(self.wfc.gvectors(ikpt=ikpt), dtype=float)
            coeff = np.asarray(
                self.wfc.readBandCoeff(ispin=ispin, ikpt=ikpt, iband=iband, norm=False),
                dtype=np.complex128,
            )
            if coeff.ndim != 1:
                raise NotImplementedError(
                    "Reciprocal-space reconstruction currently supports scalar WAVECAR coefficients only."
                )

            q_frac = gvec + np.asarray(self.wfc._kvecs[ikpt - 1], dtype=float)[None, :]
            if time_reversed:
                q_frac = -q_frac
                coeff = np.conjugate(coeff)
            if symm_op:
                op = self.symm_ops[symm_op]
                q_frac = q_frac @ op.reciprocal_matrix.T
                phase = np.exp(-2j * np.pi * np.sum(q_frac * op.tau_frac[None, :], axis=1))
                coeff = coeff * phase
            self._coeff_cache[key] = (q_frac, coeff)
        return self._coeff_cache[key]

    def primitive_wavefunction_at_k(
        self,
        ispin: int,
        ikpt: int,
        iband: int,
        ngrid: np.ndarray,
        time_reversed: bool = False,
        symm_op: int = 0,
    ) -> np.ndarray:
        key = (
            ispin,
            ikpt,
            iband,
            tuple(int(x) for x in np.asarray(ngrid, dtype=int)),
            time_reversed,
            symm_op,
        )
        if key not in self._psi_match_cache:
            if self.can_use_reciprocal_reconstruction():
                q_frac, coeff = self.matched_full_bloch_coefficients(
                    ispin=ispin,
                    ikpt=ikpt,
                    iband=iband,
                    time_reversed=time_reversed,
                    symm_op=symm_op,
                )
                psi = bloch_wavefunction_from_full_coefficients(q_frac, coeff, ngrid)
            else:
                psi = self.primitive_wavefunction(ispin, ikpt, iband, ngrid)
                k_frac = np.asarray(self.wfc._kvecs[ikpt - 1], dtype=float)
                if time_reversed:
                    psi = np.conjugate(psi)
                    k_frac = -k_frac
                if symm_op:
                    op = self.symm_ops[symm_op]
                    psi = signed_permutation_transform(psi, op.real_matrix, op.tau_frac, k_frac=k_frac)
            self._psi_match_cache[key] = psi
        return self._psi_match_cache[key]

    def sample_band_at_point(
        self,
        ispin: int,
        ikpt: int,
        iband: int,
        frac: np.ndarray,
        ngrid: np.ndarray,
        time_reversed: bool = False,
        symm_op: int = 0,
    ) -> complex:
        idx = nearest_grid_index(frac, ngrid)
        key = (ispin, ikpt, iband, idx, time_reversed, symm_op)
        if key not in self._sample_cache:
            if self.can_use_reciprocal_reconstruction():
                q_frac, coeff = self.matched_full_bloch_coefficients(
                    ispin,
                    ikpt,
                    iband,
                    time_reversed=time_reversed,
                    symm_op=symm_op,
                )
                sample_frac = idx / np.asarray(ngrid, dtype=float)
                phase = np.exp(2j * np.pi * (q_frac @ sample_frac))
                norm_fac = math.sqrt(np.prod(np.asarray(ngrid, dtype=int)))
                self._sample_cache[key] = complex(np.dot(coeff, phase) / norm_fac)
            else:
                psi = self.primitive_wavefunction_at_k(
                    ispin,
                    ikpt,
                    iband,
                    ngrid,
                    time_reversed=time_reversed,
                    symm_op=symm_op,
                )
                self._sample_cache[key] = complex(psi[idx])
        return self._sample_cache[key]

    def tile_to_supercell(self, psi_pc: np.ndarray, k_frac: np.ndarray, supercell: Tuple[int, int, int]) -> np.ndarray:
        sx, sy, sz = supercell
        nx, ny, nz = psi_pc.shape
        out = np.zeros((sx * nx, sy * ny, sz * nz), dtype=np.complex128)
        for tx, ty, tz in product(range(sx), range(sy), range(sz)):
            phase = np.exp(2j * np.pi * (k_frac[0] * tx + k_frac[1] * ty + k_frac[2] * tz))
            xs = slice(tx * nx, (tx + 1) * nx)
            ys = slice(ty * ny, (ty + 1) * ny)
            zs = slice(tz * nz, (tz + 1) * nz)
            out[xs, ys, zs] = phase * psi_pc
        return out






def read_poscar_lattice(poscar_path: Path) -> np.ndarray:
    lines = poscar_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 5:
        raise ValueError(f"POSCAR is too short: {poscar_path}")

    scale = parse_float(lines[1].split()[0])
    lattice = np.array(
        [
            [parse_float(x) for x in lines[2].split()[:3]],
            [parse_float(x) for x in lines[3].split()[:3]],
            [parse_float(x) for x in lines[4].split()[:3]],
        ],
        dtype=float,
    )

    if scale < 0:
        target_volume = abs(scale)
        current_volume = abs(np.linalg.det(lattice))
        if current_volume <= 0:
            raise ValueError("Invalid lattice in POSCAR: zero volume.")
        factor = (target_volume / current_volume) ** (1.0 / 3.0)
        lattice *= factor
    else:
        lattice *= scale
    return lattice


def reciprocal_lattice(lattice: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * np.linalg.inv(lattice).T


def build_plane_basis(b1: np.ndarray, b2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    norm_b1 = np.linalg.norm(b1)
    if norm_b1 < 1e-14:
        raise ValueError("Reciprocal vector b1 is zero.")
    e1 = b1 / norm_b1

    b2_ortho = b2 - np.dot(b2, e1) * e1
    norm_b2_ortho = np.linalg.norm(b2_ortho)
    if norm_b2_ortho < 1e-14:
        raise ValueError("Reciprocal vectors b1 and b2 are nearly collinear.")
    e2 = b2_ortho / norm_b2_ortho
    return e1, e2


def project_to_2d(vectors: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    return np.column_stack((vectors @ e1, vectors @ e2))


def clip_polygon_halfplane(
    polygon: np.ndarray,
    normal: np.ndarray,
    bound: float,
    tol: float = 1e-12,
) -> np.ndarray:
    if len(polygon) == 0:
        return polygon

    def inside(point: np.ndarray) -> bool:
        return np.dot(normal, point) <= bound + tol

    clipped: List[np.ndarray] = []
    prev = polygon[-1]
    prev_inside = inside(prev)
    prev_val = np.dot(normal, prev) - bound

    for curr in polygon:
        curr_inside = inside(curr)
        curr_val = np.dot(normal, curr) - bound
        if curr_inside != prev_inside:
            denom = curr_val - prev_val
            if abs(denom) > tol:
                t = -prev_val / denom
                inter = prev + t * (curr - prev)
                clipped.append(inter)
        if curr_inside:
            clipped.append(curr)
        prev, prev_inside, prev_val = curr, curr_inside, curr_val

    if not clipped:
        return np.empty((0, 2), dtype=float)
    return np.array(clipped, dtype=float)


def dedup_polygon_vertices(polygon: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    if len(polygon) <= 1:
        return polygon
    cleaned = [polygon[0]]
    for point in polygon[1:]:
        if np.linalg.norm(point - cleaned[-1]) > tol:
            cleaned.append(point)
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= tol:
        cleaned.pop()
    return np.array(cleaned, dtype=float)


def lattice_points_2d(cell: np.ndarray, n: int = 2) -> Tuple[np.ndarray, int]:
    grid = np.array(
        [(i, j) for i in range(-n, n + 1) for j in range(-n, n + 1)],
        dtype=float,
    )
    points = grid @ cell
    origin_idx = n * (2 * n + 1) + n
    return points, origin_idx


def bz_polygon_from_voronoi(cell: np.ndarray, n: int = 3) -> np.ndarray:
    if Voronoi is None:
        raise RuntimeError("scipy.spatial.Voronoi is unavailable.")
    points, origin_idx = lattice_points_2d(cell, n=n)
    vor = Voronoi(points)
    region_idx = vor.point_region[origin_idx]
    region = vor.regions[region_idx]
    if -1 in region or len(region) == 0:
        raise RuntimeError("Voronoi region is unbounded; increase n.")
    poly = vor.vertices[np.array(region, dtype=int)]
    center = poly.mean(axis=0)
    angles = np.arctan2(poly[:, 1] - center[1], poly[:, 0] - center[0])
    return poly[np.argsort(angles)]


def first_bz_polygon_halfplane(g1: np.ndarray, g2: np.ndarray, shell: int = 4) -> np.ndarray:
    scale = 4.0 * max(np.linalg.norm(g1), np.linalg.norm(g2))
    polygon = np.array(
        [
            [-scale, -scale],
            [scale, -scale],
            [scale, scale],
            [-scale, scale],
        ],
        dtype=float,
    )

    for i in range(-shell, shell + 1):
        for j in range(-shell, shell + 1):
            if i == 0 and j == 0:
                continue
            g = i * g1 + j * g2
            g2norm = np.dot(g, g)
            if g2norm < 1e-16:
                continue
            polygon = clip_polygon_halfplane(polygon, g, 0.5 * g2norm)
            if len(polygon) == 0:
                raise RuntimeError("Failed to construct first BZ polygon.")

    polygon = dedup_polygon_vertices(polygon)
    center = np.mean(polygon, axis=0)
    angles = np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0])
    return polygon[np.argsort(angles)]


def first_bz_polygon_2d(g1: np.ndarray, g2: np.ndarray, shell: int = 4) -> np.ndarray:
    # Prefer Voronoi Wigner-Seitz construction (as in user reference); fallback to half-plane clipping.
    cell = np.vstack([g1, g2])
    if Voronoi is not None:
        try:
            poly = bz_polygon_from_voronoi(cell, n=max(3, shell))
            return dedup_polygon_vertices(poly)
        except Exception:
            pass
    return first_bz_polygon_halfplane(g1, g2, shell=shell)


def reciprocal_shift_vectors(
    g1: np.ndarray,
    g2: np.ndarray,
    search_radius: int = 1,
) -> np.ndarray:
    search_radius = max(0, int(search_radius))
    shifts = np.array(
        [
            [i, j]
            for i in range(-search_radius, search_radius + 1)
            for j in range(-search_radius, search_radius + 1)
        ],
        dtype=float,
    )
    return shifts[:, 0:1] * g1[None, :] + shifts[:, 1:2] * g2[None, :]


def select_display_shifts(
    bz_polygon: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    view_mins: np.ndarray,
    view_maxs: np.ndarray,
    search_radius: int = 1,
) -> np.ndarray:
    selected: List[np.ndarray] = []
    for shift in reciprocal_shift_vectors(g1, g2, search_radius=search_radius):
        shifted = bz_polygon + shift
        poly_mins = np.min(shifted, axis=0)
        poly_maxs = np.max(shifted, axis=0)
        intersects = (
            poly_maxs[0] >= view_mins[0]
            and poly_mins[0] <= view_maxs[0]
            and poly_maxs[1] >= view_mins[1]
            and poly_mins[1] <= view_maxs[1]
        )
        if intersects:
            selected.append(np.array(shift, dtype=float))

    if not selected:
        return np.zeros((1, 2), dtype=float)
    return np.array(selected, dtype=float)


def tile_density_for_display(
    k_xy: np.ndarray,
    density: np.ndarray,
    shifts: np.ndarray,
    decimals: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    tiled_k = np.concatenate([k_xy + shift for shift in shifts], axis=0)
    tiled_density = np.tile(density, len(shifts))
    return deduplicate_density_max(tiled_k, tiled_density, decimals=decimals)


def points_inside_polygons(
    points_xy: np.ndarray,
    polygons: Sequence[np.ndarray],
    radius: float = BZ_PATH_EPS,
) -> np.ndarray:
    inside = np.zeros(len(points_xy), dtype=bool)
    for polygon in polygons:
        inside |= MplPath(polygon).contains_points(points_xy, radius=radius)
    return inside


def compound_path_from_polygons(polygons: Sequence[np.ndarray]) -> MplPath | None:
    if not polygons:
        return None
    try:
        return MplPath.make_compound_path_from_polys(np.array(polygons, dtype=float))
    except Exception:
        return None


def fold_points_to_first_bz(
    points_xy: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    search_radius: int = 3,
) -> np.ndarray:
    shifts = np.array(
        [
            [i, j]
            for i in range(-search_radius, search_radius + 1)
            for j in range(-search_radius, search_radius + 1)
        ],
        dtype=float,
    )
    g_shifts = shifts[:, 0:1] * g1[None, :] + shifts[:, 1:2] * g2[None, :]

    folded = np.empty_like(points_xy)
    for idx, point in enumerate(points_xy):
        candidates = point[None, :] - g_shifts
        dist2 = np.einsum("ij,ij->i", candidates, candidates)
        folded[idx] = candidates[np.argmin(dist2)]
    return folded


def aggregate_density(
    k_xy: np.ndarray,
    weights: np.ndarray,
    decimals: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    rounded = np.round(k_xy, decimals=decimals)
    unique_k, inverse = np.unique(rounded, axis=0, return_inverse=True)
    density = np.zeros(unique_k.shape[0], dtype=float)
    np.add.at(density, inverse, weights)
    return unique_k, density


def deduplicate_density_max(
    k_xy: np.ndarray,
    density: np.ndarray,
    decimals: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    rounded = np.round(k_xy, decimals=decimals)
    unique_k, inverse = np.unique(rounded, axis=0, return_inverse=True)
    density_max = np.full(unique_k.shape[0], -np.inf, dtype=float)
    np.maximum.at(density_max, inverse, density)
    return unique_k, density_max


def complete_periodic_edges_first_bz(
    k_xy: np.ndarray,
    density: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    bz_polygon: np.ndarray,
    decimals: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    shifts = np.array(
        [
            [0, 0],
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
        ],
        dtype=float,
    )
    shift_vecs = shifts[:, 0:1] * g1[None, :] + shifts[:, 1:2] * g2[None, :]
    k_all = np.concatenate([k_xy + shift for shift in shift_vecs], axis=0)
    d_all = np.tile(density, len(shift_vecs))

    path_bz = MplPath(bz_polygon)
    inside = path_bz.contains_points(k_all, radius=BZ_PATH_EPS)
    return deduplicate_density_max(k_all[inside], d_all[inside], decimals=decimals)


def axis_edges_from_centers(values: np.ndarray, clamp_ends_to_centers: bool = False) -> np.ndarray:
    values = np.array(values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Need at least two center values to build cell edges.")
    diffs = np.diff(values)
    edges = np.empty(len(values) + 1, dtype=float)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    if clamp_ends_to_centers:
        # Useful when extreme centers are already BZ boundaries.
        edges[0] = values[0]
        edges[-1] = values[-1]
    else:
        edges[0] = values[0] - 0.5 * diffs[0]
        edges[-1] = values[-1] + 0.5 * diffs[-1]
    return edges


def build_rect_grid_from_points(
    k_xy: np.ndarray,
    density: np.ndarray,
    decimals: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    rounded = np.round(k_xy, decimals=decimals)
    x_vals = np.unique(rounded[:, 0])
    y_vals = np.unique(rounded[:, 1])
    nx, ny = len(x_vals), len(y_vals)
    if nx * ny != len(rounded):
        return None

    x_index = {val: i for i, val in enumerate(x_vals)}
    y_index = {val: i for i, val in enumerate(y_vals)}
    z = np.full((ny, nx), np.nan, dtype=float)
    for (xv, yv), dv in zip(rounded, density):
        z[y_index[yv], x_index[xv]] = dv
    if np.isnan(z).any():
        return None
    return x_vals, y_vals, z


def interpolate_rect_grid_cubic(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    z_vals: np.ndarray,
    factor: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if RectBivariateSpline is None or factor <= 1 or len(x_vals) < 4 or len(y_vals) < 4:
        return x_vals, y_vals, z_vals

    nx = (len(x_vals) - 1) * factor + 1
    ny = (len(y_vals) - 1) * factor + 1
    x_dense = np.linspace(x_vals[0], x_vals[-1], nx)
    y_dense = np.linspace(y_vals[0], y_vals[-1], ny)

    spline = RectBivariateSpline(x_vals, y_vals, z_vals.T, kx=3, ky=3, s=0.0)
    z_dense = spline(x_dense, y_dense).T

    zmin = float(np.nanmin(z_vals))
    zmax = float(np.nanmax(z_vals))
    z_dense = np.clip(z_dense, zmin, zmax)
    return x_dense, y_dense, z_dense


def interpolate_scattered_cubic_to_grid(
    points_xy: np.ndarray,
    values: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
) -> np.ndarray | None:
    if griddata is None or len(points_xy) < 8:
        return None
    xx, yy = np.meshgrid(x_vals, y_vals)
    zz = griddata(points_xy, values, (xx, yy), method="cubic")
    if zz is None:
        return None

    if np.isnan(zz).any():
        zz_lin = griddata(points_xy, values, (xx, yy), method="linear")
        if zz_lin is not None:
            zz = np.where(np.isnan(zz), zz_lin, zz)
        if np.isnan(zz).any():
            zz_near = griddata(points_xy, values, (xx, yy), method="nearest")
            if zz_near is not None:
                zz = np.where(np.isnan(zz), zz_near, zz)

    if np.isnan(zz).all():
        return None
    zmin = float(np.nanmin(values))
    zmax = float(np.nanmax(values))
    zz = np.clip(zz, zmin, zmax)
    return zz


def normalize_density(density: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return density
    if mode == "sum":
        total = float(np.sum(density))
        return density / total if total > 0 else density
    if mode == "max":
        vmax = float(np.max(density))
        return density / vmax if vmax > 0 else density
    raise ValueError(f"Unsupported normalization mode: {mode}")


def auto_figsize_for_window(
    mins: np.ndarray,
    maxs: np.ndarray,
    axis_long_side: float = 5.0,
    min_axis_short_side: float = 3.6,
    colorbar_extra_width: float = 1.0,
    title_extra_height: float = 0.8,
) -> Tuple[float, float]:
    span = np.maximum(np.asarray(maxs, dtype=float) - np.asarray(mins, dtype=float), 1e-12)
    aspect = float(span[0] / span[1])

    if aspect >= 1.0:
        axis_width = axis_long_side
        axis_height = max(min_axis_short_side, axis_long_side / aspect)
    else:
        axis_height = axis_long_side
        axis_width = max(min_axis_short_side, axis_long_side * aspect)

    return axis_width + colorbar_extra_width, axis_height + title_extra_height


def plot_density_first_bz(
    k_xy: np.ndarray,
    density: np.ndarray,
    bz_polygon: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    output_png: Path,
    title: str,
    cmap: str = "magma",
    log_scale: bool = False,
    interp: str = "cubic",
    interp_factor: int = 6,
    figsize: Tuple[float, float] | None = None,
    bz_pad_ratio: float = 0.015,
    view_scale: float = 1.12,
    tile_radius: int = 1,
    show_neighbor_bz: bool = False,
    dpi: int = 300,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    base_mins = np.min(bz_polygon, axis=0)
    base_maxs = np.max(bz_polygon, axis=0)
    center = 0.5 * (base_mins + base_maxs)
    base_half_span = 0.5 * (base_maxs - base_mins)
    view_scale = max(1.0, float(view_scale))
    view_half_span = np.maximum(base_half_span * view_scale, 1e-9)
    view_mins = center - view_half_span
    view_maxs = center + view_half_span

    display_shifts = select_display_shifts(
        bz_polygon,
        g1,
        g2,
        view_mins,
        view_maxs,
        search_radius=tile_radius,
    )
    display_polygons = [bz_polygon + shift for shift in display_shifts]

    display_k, display_density = tile_density_for_display(k_xy, density, display_shifts, decimals=8)
    inside = points_inside_polygons(display_k, display_polygons, radius=BZ_PATH_EPS)
    plot_k = display_k[inside]
    plot_density = display_density[inside]
    if len(plot_k) == 0:
        raise RuntimeError("No k-points remain inside the displayed BZ area for plotting.")

    pad = max(0.0, float(bz_pad_ratio)) * (view_maxs - view_mins + 1e-9)
    plot_mins = view_mins - pad
    plot_maxs = view_maxs + pad
    if figsize is None:
        figsize = auto_figsize_for_window(plot_mins, plot_maxs)

    fig, ax = plt.subplots(figsize=figsize)
    norm = None
    value_for_plot = plot_density
    clip_patch = None
    compound_path = compound_path_from_polygons(display_polygons)
    if compound_path is not None:
        clip_patch = mpatches.PathPatch(compound_path, facecolor="none", edgecolor="none")
        ax.add_patch(clip_patch)

    if log_scale:
        positive = plot_density[plot_density > 0]
        if len(positive) > 0:
            vmin = max(float(np.min(positive)), float(np.max(plot_density)) * 1e-8)
            norm = LogNorm(vmin=vmin, vmax=float(np.max(plot_density)))
            value_for_plot = np.clip(plot_density, vmin, None)

    mappable = None
    grid = build_rect_grid_from_points(plot_k, value_for_plot, decimals=8)
    if grid is not None:
        x_vals, y_vals, z_vals = grid
        if interp == "cubic":
            x_plot, y_plot, z_plot = interpolate_rect_grid_cubic(
                x_vals,
                y_vals,
                z_vals,
                factor=max(1, interp_factor),
            )
        else:
            x_plot, y_plot, z_plot = x_vals, y_vals, z_vals

        xx, yy = np.meshgrid(x_plot, y_plot)
        inside_grid = points_inside_polygons(
            np.column_stack((xx.ravel(), yy.ravel())),
            display_polygons,
            radius=BZ_PATH_EPS,
        ).reshape(xx.shape)
        z_plot = np.where(inside_grid, z_plot, np.nan)

        x_edges = axis_edges_from_centers(x_plot, clamp_ends_to_centers=True)
        y_edges = axis_edges_from_centers(y_plot, clamp_ends_to_centers=True)
        mappable = ax.pcolormesh(
            x_edges,
            y_edges,
            z_plot,
            cmap=cmap,
            norm=norm,
            shading="auto",
        )
        if clip_patch is not None:
            mappable.set_clip_path(clip_patch)
    elif len(plot_k) >= 3:
        try:
            if interp == "cubic":
                span = np.maximum(view_maxs - view_mins, 1e-12)
                nx = max(120, 24 * max(1, interp_factor))
                ny = max(120, int(round(nx * span[1] / span[0])))
                x_dense = np.linspace(view_mins[0], view_maxs[0], nx)
                y_dense = np.linspace(view_mins[1], view_maxs[1], ny)
                z_dense = interpolate_scattered_cubic_to_grid(
                    plot_k,
                    value_for_plot,
                    x_dense,
                    y_dense,
                )
                if z_dense is not None:
                    xx, yy = np.meshgrid(x_dense, y_dense)
                    inside_grid = points_inside_polygons(
                        np.column_stack((xx.ravel(), yy.ravel())),
                        display_polygons,
                        radius=BZ_PATH_EPS,
                    ).reshape(xx.shape)
                    z_dense = np.where(inside_grid, z_dense, np.nan)
                    x_edges = axis_edges_from_centers(x_dense, clamp_ends_to_centers=True)
                    y_edges = axis_edges_from_centers(y_dense, clamp_ends_to_centers=True)
                    mappable = ax.pcolormesh(
                        x_edges,
                        y_edges,
                        z_dense,
                        cmap=cmap,
                        norm=norm,
                        shading="auto",
                    )
                    if clip_patch is not None:
                        mappable.set_clip_path(clip_patch)

            if mappable is None:
                tri = mtri.Triangulation(plot_k[:, 0], plot_k[:, 1])
                mappable = ax.tricontourf(
                    tri,
                    value_for_plot,
                    levels=120,
                    cmap=cmap,
                    norm=norm,
                )
                if clip_patch is not None:
                    for coll in mappable.collections:
                        coll.set_clip_path(clip_patch)
        except Exception:
            mappable = None

    if mappable is None:
        mappable = ax.scatter(
            plot_k[:, 0],
            plot_k[:, 1],
            c=value_for_plot,
            cmap=cmap,
            norm=norm,
            s=36,
            edgecolors="none",
        )

    if show_neighbor_bz:
        for polygon in display_polygons:
            if np.allclose(polygon, bz_polygon):
                continue
            closed = np.vstack([polygon, polygon[0]])
            ax.plot(closed[:, 0], closed[:, 1], color="white", linewidth=0.8, alpha=0.18)

    closed = np.vstack([bz_polygon, bz_polygon[0]])
    first_bz_line, = ax.plot(
        closed[:, 0],
        closed[:, 1],
        color="white",
        linewidth=1.3,
        linestyle=(0, (4, 3)),
        zorder=4,
    )
    first_bz_line.set_path_effects(
        [mpe.Stroke(linewidth=2.3, foreground="black", alpha=0.35), mpe.Normal()]
    )
    ax.scatter([0.0], [0.0], marker="+", s=80, c="white", linewidths=1.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, loc="center")
    ax.set_xlabel(r"$k_x$ (1/$\AA$)")
    ax.set_ylabel(r"$k_y$ (1/$\AA$)")

    ax.set_xlim(plot_mins[0], plot_maxs[0])
    ax.set_ylim(plot_mins[1], plot_maxs[1])

    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Exciton density")
    fig.tight_layout()
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)


def parse_exciton_selection(selection: str, nexciton: int) -> List[int]:
    result: List[int] = []
    for part in selection.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if end < start:
                raise ValueError(f"Invalid range '{part}': end < start")
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    dedup = sorted(set(result))
    for idx in dedup:
        if idx < 1 or idx > nexciton:
            raise ValueError(f"Exciton index {idx} out of range 1..{nexciton}")
    return dedup


def choose_default_input() -> Path:
    for name in ("BSEFATBAND_TDA", "BSEFATBAND_FULL", "BSEFATBAND"):
        path = Path(name)
        if path.is_file():
            return path
    raise FileNotFoundError("No BSEFATBAND file found. Set --input explicitly.")


ExcitonReconstructor = ExcitonConstructor


@dataclass
class BZDensityMap:
    k_xy: np.ndarray
    density: np.ndarray
    bz_polygon: np.ndarray
    g1: np.ndarray
    g2: np.ndarray


def exciton_bz_density(
    block: ExcitonFatband,
    poscar: Path,
    weight_source: str = "amplitude",
    normalize: str = "max",
) -> BZDensityMap:
    if block.k_frac is None:
        raise RuntimeError(f"Exciton {block.index} has no k-point data.")

    lattice = read_poscar_lattice(Path(poscar))
    recip = reciprocal_lattice(lattice)
    b1, b2 = recip[0], recip[1]
    e1, e2 = build_plane_basis(b1, b2)
    g1 = np.array([np.dot(b1, e1), np.dot(b1, e2)], dtype=float)
    g2 = np.array([np.dot(b2, e1), np.dot(b2, e2)], dtype=float)
    bz_poly = first_bz_polygon_2d(g1, g2, shell=4)

    if weight_source == "amplitude":
        if block.amplitude is None:
            raise RuntimeError(f"Exciton {block.index} has no complex amplitude data.")
        weights = np.abs(block.amplitude) ** 2
    elif weight_source == "column":
        if block.column_weight is None:
            raise RuntimeError(f"Exciton {block.index} has no column-weight data.")
        weights = np.asarray(block.column_weight, dtype=float).copy()
    else:
        raise ValueError(f"Unsupported weight source: {weight_source}")

    k_cart = block.k_frac[:, 0:1] * recip[0] + block.k_frac[:, 1:2] * recip[1]
    k_xy = project_to_2d(k_cart, e1, e2)
    k_xy_fold = fold_points_to_first_bz(k_xy, g1, g2, search_radius=3)
    k_unique, density = aggregate_density(k_xy_fold, weights, decimals=8)
    k_unique, density = complete_periodic_edges_first_bz(
        k_unique,
        density,
        g1,
        g2,
        bz_poly,
        decimals=8,
    )
    density = normalize_density(density, normalize)
    return BZDensityMap(k_xy=k_unique, density=density, bz_polygon=bz_poly, g1=g1, g2=g2)


def write_kdensity_table(path: Path, k_xy: np.ndarray, density: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = np.column_stack((k_xy[:, 0], k_xy[:, 1], density))
    header = "kx_proj(1/Ang) ky_proj(1/Ang) exciton_density"
    np.savetxt(path, table, header=header, fmt="%.10e")


def add_realspace_arguments(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Reconstruct a fixed-hole or fixed-electron exciton density in real space "
        "from BSEFATBAND + WAVECAR."
    )
    parser.epilog = (
        "This workflow uses symmetry operators and full-BZ/IBZ mapping parsed from "
        "OUTCAR (or OUTCAR.symm). Keep OUTCAR alongside WAVECAR, BSEFATBAND, or POSCAR "
        "so the script can restore the correct full-BZ Bloch phases when WAVECAR stores "
        "only irreducible k points."
    )
    parser.add_argument("--wavecar", type=Path, default=Path("WAVECAR"))
    parser.add_argument("--bsefatband", type=Path, default=Path("BSEFATBAND"))
    parser.add_argument("--poscar", type=Path, default=Path("POSCAR"))
    parser.add_argument("--exciton", type=int, required=True, help="1-based exciton index")
    parser.add_argument("--hole", type=str, default=None, help="Fix the hole at this primitive-cell fractional position and reconstruct the electron density.")
    parser.add_argument("--hole-from-max-akcv", action="store_true", help="Locate the transition with the largest |A_kcv|^2, find the maximum of the corresponding valence-state density on the primitive FFT grid, and use that point as the hole.")
    parser.add_argument("--electron", type=str, default=None, help="Fix the electron at this primitive-cell fractional position and reconstruct the hole density.")
    parser.add_argument("--electron-from-max-akcv", action="store_true", help="Locate the transition with the largest |A_kcv|^2, find the maximum of the corresponding conduction-state density on the primitive FFT grid, and use that point as the electron.")
    parser.add_argument("--supercell", type=str, default=None, help="Diagonal supercell 'Nx Ny Nz'. If omitted, infer from exciton k mesh.")
    parser.add_argument("--cumulative-weight", type=float, default=1.0, help="Keep the strongest transitions until this cumulative |A|^2 fraction is reached. Use 1.0 to match VASP bse.F exactly.")
    parser.add_argument("--fft-grid", type=str, default=None, help="Primitive-cell FFT grid 'Nx Ny Nz'. Default: 2 * minimal WAVECAR grid.")
    parser.add_argument("--prefix", type=str, default="exciton")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--write-amplitude", action="store_true", help="Also write real/imag parts of the reconstructed amplitude.")
    parser.add_argument("--spin-channel", type=int, default=None, help="For NSPIN=2 WAVECARs, force a collinear spin channel (1-based). If omitted, infer it from BSEFATBAND band energies and raise on ambiguity.")
    parser.add_argument("--lsorbit", action="store_true", help="Pass lsorbit=True to vaspwfc when reading WAVECAR.")
    parser.add_argument("--lgamma", action="store_true", help="Pass lgamma=True to vaspwfc when reading WAVECAR.")
    parser.add_argument("--gamma-half", type=str, default="x", choices=["x", "z"])
    parser.set_defaults(center_fixed_particle=False)
    parser.add_argument("--center-fixed-particle", dest="center_fixed_particle", action="store_true", help="Circularly shift the output so the fixed particle lies near the supercell center.")
    parser.add_argument("--center-expectation-value", action="store_true", help="Circularly shift the output so the moving-particle density expectation value lies near the supercell center.")
    parser.add_argument("--center-r", dest="center_expectation_value", action="store_true", help="Alias for --center-expectation-value.")


def add_bz_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", "--bsefatband", dest="input", type=Path, default=None, help="BSEFATBAND file path (default: auto-detect TDA/FULL).")
    parser.add_argument("--poscar", type=Path, default=Path("POSCAR"), help="POSCAR path for reciprocal lattice.")
    parser.add_argument("--exciton", default="1", help="Exciton index/range list (e.g. '1', '1,2,5', '1-10').")
    parser.add_argument("--all", action="store_true", help="Process all excitons in the file.")
    parser.add_argument("--list", action="store_true", help="List exciton energies only (no plotting).")
    parser.add_argument("--weight-source", choices=("amplitude", "column"), default="amplitude", help="Use |A|^2 ('amplitude') or column-6 weight ('column') for density.")
    parser.add_argument("--normalize", choices=("max", "sum", "none"), default="max", help="Normalize density before plotting.")
    parser.add_argument("--log-scale", action="store_true", help="Use logarithmic color normalization.")
    parser.add_argument("--interp", choices=("cubic", "none"), default="cubic", help="Interpolation mode for smoother density map.")
    parser.add_argument("--interp-factor", type=int, default=6, help="Refinement factor for cubic interpolation (>=1).")
    parser.add_argument("--cmap", default="magma", help="Matplotlib colormap.")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory for figures/data.")
    parser.add_argument("--prefix", default=None, help="Output filename prefix (default: input stem).")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument("--figsize", nargs=2, type=float, metavar=("WIDTH", "HEIGHT"), default=None, help="Figure size in inches (width height). Default: auto from displayed BZ shape.")
    parser.add_argument("--bz-pad", type=float, default=0.015, help="Relative padding around first-BZ limits.")
    parser.add_argument("--view-scale", type=float, default=1.12, help="Scale factor for the plotting window relative to first-BZ bounds.")
    parser.add_argument("--tile-radius", type=int, default=1, help="Neighbor-shell radius used to tile first-BZ density for display.")
    parser.add_argument("--show-neighbor-bz", action="store_true", help="Draw faint neighboring BZ outlines in addition to the dashed first-BZ marker.")
    parser.add_argument("--save-kdensity", action="store_true", help="Save folded k-density table (*.dat) for each exciton.")


def run_realspace(args: argparse.Namespace) -> int:
    manual_fixed_primitive: np.ndarray | None = None
    auto_fixed: DominantFixedParticle | None = None
    if args.hole_from_max_akcv and args.electron_from_max_akcv:
        raise SystemExit("Pass only one of --hole-from-max-akcv or --electron-from-max-akcv.")
    if args.hole_from_max_akcv or args.electron_from_max_akcv:
        if args.hole is not None or args.electron is not None:
            raise SystemExit("Pass either an auto-selected fixed particle (--hole-from-max-akcv/--electron-from-max-akcv) or an explicit --hole/--electron coordinate, not both.")
    if args.hole_from_max_akcv:
        fixed_particle = "hole"
        moving_particle = "electron"
        auto_mode = "hole"
    elif args.electron_from_max_akcv:
        fixed_particle = "electron"
        moving_particle = "hole"
        auto_mode = "electron"
    else:
        if (args.hole is None) == (args.electron is None):
            raise SystemExit("Pass exactly one of --hole or --electron.")
        if args.hole is not None:
            fixed_particle = "hole"
            moving_particle = "electron"
            manual_fixed_primitive = parse_frac_triplet(args.hole)
        else:
            fixed_particle = "electron"
            moving_particle = "hole"
            manual_fixed_primitive = parse_frac_triplet(args.electron)
        auto_mode = None

    ngrid = np.array(parse_triplet(args.fft_grid), dtype=int) if args.fft_grid else None
    supercell = parse_triplet(args.supercell) if args.supercell else None

    recon = ExcitonConstructor(
        wavecar=args.wavecar,
        bsefatband=args.bsefatband,
        poscar=args.poscar,
        lsorbit=args.lsorbit,
        lgamma=args.lgamma,
        gamma_half=args.gamma_half,
        spin_channel=args.spin_channel,
    )
    block = recon.fatband.read_excitons([args.exciton])[0]

    if ngrid is None:
        ngrid = recon.wfc._ngrid.copy() * 2
    if supercell is None:
        inferred = infer_regular_mesh(block.k_frac)
        if inferred is None:
            raise SystemExit("Could not infer a regular supercell from the BSE k mesh. Please pass --supercell.")
        supercell = inferred
    if auto_mode is not None:
        auto_fixed = dominant_fixed_particle_from_block(recon, block, ngrid, supercell, fixed_particle=auto_mode)
        fixed_primitive = auto_fixed.primitive_frac
    else:
        if manual_fixed_primitive is None:
            raise SystemExit("Internal error: fixed particle position was not initialized.")
        fixed_primitive = wrap_frac(manual_fixed_primitive)
    fixed_super = origin_supercell_image_from_primitive_frac(fixed_primitive, supercell)

    weights = np.abs(block.amplitude) ** 2
    keep = threshold_indices(weights, args.cumulative_weight)
    used_weight_fraction = float(np.sum(weights[keep]) / np.sum(weights)) if np.sum(weights) > 0 else 0.0

    super_grid = np.asarray(ngrid, dtype=int) * np.asarray(supercell, dtype=int)
    (
        base_fixed_super_idx,
        base_fixed_super_sampled,
        _base_fixed_idx,
        base_fixed_sampled,
        base_fixed_placement_shift,
    ) = sample_supercell_point(fixed_super, ngrid, supercell)
    center_by_fixed_particle = args.center_fixed_particle
    fixed_center_shift = np.zeros(3, dtype=int)
    total_shift_frac = np.zeros(3, dtype=float)
    if center_by_fixed_particle:
        fixed_center_shift, fixed_shift_frac = center_shift_from_hole_index(base_fixed_super_idx, ngrid, supercell)
        total_shift_frac = (total_shift_frac + fixed_shift_frac) % 1.0

    if moving_particle == "electron":
        result = reconstruct_fixed_hole(recon, block, base_fixed_sampled, ngrid, supercell, keep)
    else:
        result = reconstruct_fixed_electron(recon, block, base_fixed_sampled, ngrid, supercell, keep)
    psi_moving = result.psi_electron
    rho = result.density
    raw_density_sum = result.raw_density_sum
    total_shift = base_fixed_placement_shift.copy()
    if center_by_fixed_particle:
        total_shift = (total_shift + fixed_center_shift) % super_grid
    if np.any(total_shift):
        rho = circular_shift_grid(rho, total_shift)
        if psi_moving is not None:
            psi_moving = circular_shift_grid(psi_moving, total_shift)

    expectation_shift = np.zeros(3, dtype=int)
    expectation_mean_frac = np.zeros(3, dtype=float)
    if args.center_expectation_value:
        expectation_shift, expectation_shift_frac, expectation_mean_frac = center_shift_from_density_expectation(rho)
        total_shift_frac = (total_shift_frac + expectation_shift_frac) % 1.0
        if np.any(expectation_shift):
            rho = circular_shift_grid(rho, expectation_shift)
            if psi_moving is not None:
                psi_moving = circular_shift_grid(psi_moving, expectation_shift)

    output_rho = np.array(rho, copy=True)
    output_psi = None if psi_moving is None else np.array(psi_moving, copy=True)
    output_normalization = "vasp"
    if raw_density_sum is None:
        raise SystemExit("Internal error: missing raw density norm for VASP-style output scaling.")
    density_scale = float(raw_density_sum * np.prod(np.asarray(ngrid, dtype=int)) * np.prod(super_grid))
    output_rho *= density_scale
    if output_psi is not None:
        output_psi *= math.sqrt(density_scale)

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    sc_poscar = outdir / f"{args.prefix}_SC_POSCAR"
    poscar_header = make_diagonal_supercell_poscar(
        args.poscar,
        supercell,
        sc_poscar,
        shift_frac=total_shift_frac if (center_by_fixed_particle or args.center_expectation_value) else None,
    )

    rho_path = outdir / f"{args.prefix}_{args.exciton:03d}_{moving_particle}_rho.vasp"
    write_scalar_vasp(rho_path, poscar_header, output_rho)

    if args.write_amplitude and output_psi is not None:
        re_path = outdir / f"{args.prefix}_{args.exciton:03d}_{moving_particle}_real.vasp"
        im_path = outdir / f"{args.prefix}_{args.exciton:03d}_{moving_particle}_imag.vasp"
        write_scalar_vasp(re_path, poscar_header, output_psi.real)
        write_scalar_vasp(im_path, poscar_header, output_psi.imag)

    print(f"Exciton index           : {args.exciton}")
    print(f"Fixed particle          : {fixed_particle}")
    print(f"Moving particle         : {moving_particle}")
    if auto_fixed is not None:
        print(f"Fixed position source   : dominant {auto_fixed.source_state} maximum of largest |A_kcv|^2")
        print(f"Dominant transition row : {auto_fixed.transition_index + 1}")
        print(f"Dominant transition k   : {auto_fixed.k_frac.tolist()}")
        print(f"Dominant k match        : ikpt={auto_fixed.ikpt}, time_reversed={auto_fixed.time_reversed}, symm_op={auto_fixed.symm_op}")
        print(f"Dominant v/c/spin       : v={auto_fixed.valence_band}, c={auto_fixed.conduction_band}, sv={auto_fixed.valence_spin}, sc={auto_fixed.conduction_spin}")
        print(f"Dominant amplitude      : {auto_fixed.amplitude}  |A|^2={auto_fixed.weight:.6f}")
        print(f"Auto {fixed_particle} (PC frac) : {auto_fixed.primitive_frac.tolist()}")
    print(f"Fixed position (PC frac): {fixed_primitive.tolist()}")
    print(f"Fixed position (SC frac): {fixed_super.tolist()}")
    print(f"Sampled fixed (SC frac) : {base_fixed_super_sampled.tolist()}")
    print(f"Sampled fixed (PC frac) : {base_fixed_sampled.tolist()}")
    print(f"Primitive FFT grid      : {tuple(int(x) for x in ngrid)}")
    print(f"Supercell               : {supercell}")
    print(f"Supercell grid          : {rho.shape}")
    print(f"Used transitions        : {len(keep)} / {len(weights)}")
    print(f"Retained |A|^2 fraction : {used_weight_fraction:.6f}")
    print(f"Output normalization    : {output_normalization}")
    print(f"Output density scale    : {density_scale:.12e}")
    if np.any(base_fixed_placement_shift):
        print(f"Placement grid shift    : {tuple(int(x) for x in base_fixed_placement_shift)}")
    if center_by_fixed_particle:
        print(f"Fixed-center shift      : {tuple(int(x) for x in fixed_center_shift)}")
    if args.center_expectation_value:
        print(f"Expectation <r> (SC frac): {expectation_mean_frac.tolist()}")
        print(f"Expectation shift       : {tuple(int(x) for x in expectation_shift)}")
    print(f"Supercell POSCAR        : {sc_poscar}")
    print(f"Density file            : {rho_path}")
    if args.write_amplitude and output_psi is not None:
        print(f"Amplitude (real)        : {re_path}")
        print(f"Amplitude (imag)        : {im_path}")
    print()
    print("Notes:")
    print("  1) This reconstructs pseudo-wavefunctions from WAVECAR and ignores PAW augmentation.")
    print("  2) Exactly one of --hole, --electron, --hole-from-max-akcv, or --electron-from-max-akcv must define the fixed particle. Explicit --hole/--electron coordinates are interpreted in primitive-cell fractional coordinates, matching VASP BSEHOLE/BSEELECTRON.")
    print("  3) Standard scalar outputs use VASP-style raw grid scaling and are not renormalized to unit integral after fixing the hole/electron.")
    print("  4) The script assumes BSEFATBAND rows correspond to one regular k mesh. If not, pass --supercell explicitly and inspect the result carefully.")
    print("  5) For large k meshes, start with --cumulative-weight 0.90~0.98 to reduce cost.")
    if recon.wfc._nspin > 1 and args.spin_channel is None:
        print("  6) For NSPIN=2, the script infers a collinear spin channel from the BSEFATBAND band energies and raises if the assignment is ambiguous.")
    elif recon.wfc._nspin > 1:
        print(f"  6) Forced spin channel     : {args.spin_channel}")
    if center_by_fixed_particle and args.center_expectation_value:
        print("  7) The output density and supercell POSCAR were circularly shifted using both the fixed-particle position and the moving-particle density expectation value.")
    elif center_by_fixed_particle:
        print("  7) The output density and supercell POSCAR were circularly shifted so the sampled fixed particle lies near the box center.")
    elif args.center_expectation_value:
        print("  7) The output density and supercell POSCAR were circularly shifted so the moving-particle density expectation value lies near the box center.")
    return 0


def run_bz(args: argparse.Namespace) -> int:
    input_path = args.input if args.input is not None else choose_default_input()
    fatband = ExcitonFatband.from_file(input_path)
    if fatband.nexciton is None or fatband.xdim is None:
        raise RuntimeError("Failed to initialize BSEFATBAND parser context.")

    if args.list:
        print(f"# file: {input_path}")
        print(f"# xdim={fatband.xdim} nexciton={fatband.nexciton}")
        print("# idx  BSE_eigenvalue(eV)  IP_eigenvalue(eV)")
        for meta in fatband.iter_metadata():
            print(f"{meta.index:4d}  {meta.bse_eigenvalue:18.8f}  {meta.ip_eigenvalue:17.8f}")
        return 0

    if args.all:
        exciton_indices = list(range(1, fatband.nexciton + 1))
    else:
        exciton_indices = parse_exciton_selection(args.exciton, fatband.nexciton)

    blocks = fatband.read_excitons(exciton_indices)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix if args.prefix else input_path.stem

    for block in blocks:
        bz_map = exciton_bz_density(
            block,
            poscar=args.poscar,
            weight_source=args.weight_source,
            normalize=args.normalize,
        )
        out_png = output_dir / f"{prefix}_exciton_{block.index:03d}_bz.png"
        title = (
            f"{input_path.name} exciton {block.index}\n"
            f"E_BSE={block.bse_eigenvalue:.6f} eV  "
            f"E_IP={block.ip_eigenvalue:.6f} eV"
        )
        plot_density_first_bz(
            bz_map.k_xy,
            bz_map.density,
            bz_map.bz_polygon,
            bz_map.g1,
            bz_map.g2,
            out_png,
            title=title,
            cmap=args.cmap,
            log_scale=args.log_scale,
            interp=args.interp,
            interp_factor=max(1, args.interp_factor),
            figsize=None if args.figsize is None else (float(args.figsize[0]), float(args.figsize[1])),
            bz_pad_ratio=float(args.bz_pad),
            view_scale=max(1.0, float(args.view_scale)),
            tile_radius=max(0, int(args.tile_radius)),
            show_neighbor_bz=bool(args.show_neighbor_bz),
            dpi=args.dpi,
        )
        print(f"[saved] {out_png}")

        if args.save_kdensity:
            out_data = output_dir / f"{prefix}_exciton_{block.index:03d}_k_density.dat"
            write_kdensity_table(out_data, bz_map.k_xy, bz_map.density)
            print(f"[saved] {out_data}")

    return 0


def infer_default_mode(argv: Sequence[str]) -> str:
    realspace_flags = {
        "--wavecar",
        "--hole",
        "--hole-from-max-akcv",
        "--electron",
        "--electron-from-max-akcv",
        "--supercell",
        "--cumulative-weight",
        "--fft-grid",
        "--write-amplitude",
        "--spin-channel",
        "--lsorbit",
        "--lgamma",
        "--gamma-half",
        "--center-fixed-particle",
        "--center-expectation-value",
        "--center-r",
    }
    for token in argv:
        if token in realspace_flags:
            return "realspace"
    return "kspace"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot VASP BSE excitons in reciprocal space or reconstruct them in real space."
    )
    subparsers = parser.add_subparsers(dest="mode")
    subparsers.required = True

    p_real = subparsers.add_parser(
        "realspace",
        aliases=["real"],
        help="Reconstruct a fixed-hole or fixed-electron exciton density in real space.",
    )
    add_realspace_arguments(p_real)
    p_real.set_defaults(_handler=run_realspace)

    p_bz = subparsers.add_parser(
        "kspace",
        aliases=["bz"],
        help="Plot exciton density in the first Brillouin zone.",
    )
    add_bz_arguments(p_bz)
    p_bz.set_defaults(_handler=run_bz)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    known_modes = {"realspace", "real", "kspace", "bz", "-h", "--help"}
    if argv and argv[0] not in known_modes:
        argv = [infer_default_mode(argv)] + argv
    parser = build_parser()
    args = parser.parse_args(argv)
    return args._handler(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
