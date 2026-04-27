#!/usr/bin/env python3

"""
bsematrix.py — BSE matrix builder aligned to the VASP 5.4.4 storage conventions.

This script reconstructs the Bethe-Salpeter Equation (BSE) Hamiltonian directly
from VASP wavefunctions and metadata, with emphasis on reproducing the matrix
layout and k-point bookkeeping used by VASP's own BSE implementation.

Physical model
--------------
In the excitonic pair basis

    I = (v, c, k1, spin)  with  k3 = k1 + q_ext

the assembled matrix is

    A_IJ = (eps_c(I) - eps_v(I)) delta_IJ + K_H(I, J) + K_D(I, J)

with

    K_H(I, J) = + w_k * spin_factor * sum_G
                rho_cv(1, 3; G)^* v(q_H + G) rho_vc(2, 4; G)

    K_D(I, J) = - w_k * sum_G
                rho_cc(3, 4; G)^* W(q_D + G) rho_vv(1, 2; G)

and

    q_H = k1 - k3 = -q_ext
    q_D = k1 - k2 = k3 - k4

The pair densities are built from the periodic parts of the Bloch states on the
FFT grid:

    rho_ab(G) = FFT[ u_a(r)^* u_b(r) ](G)

where `u_{n,k}(r)` is reconstructed from the WAVECAR plane-wave coefficients
without the Bloch phase factor.

Implementation outline
----------------------
The main steps in this file are:

1. read WAVECAR, OUTCAR, KPOINTS, and optionally POSCAR/POTCAR;
2. recover irreducible-k and full-BZ mappings from OUTCAR, including symmetry
   operators and time-reversal tags;
3. build the electron-hole pair basis with the same full-BZ ordering needed for
   VASP's half-stored dense AMAT layout;
4. reconstruct cell-periodic wavefunctions on the VASP charge grid, applying
   symmetry rotation, time reversal, and SETPHASE-compatible phase shifts;
5. optionally orthogonalize pseudo-wavefunctions with PAW projector-overlap
   corrections in `paw_orth_only` mode;
6. contract Hartree and/or direct kernels either on the full FFT grid or, when
   available, on the selected response basis loaded from `WFULL/W*`;
7. diagonalize the Hermitian matrix and optionally write a VASP-style
   `BSEFATBAND`.

Alignment goals and conventions
-------------------------------
- Preserve VASP full-BZ pair ordering and storage masking rules.
- Treat Hartree momentum as fixed at `-q_ext` and direct momentum as pair
  dependent.
- Keep diagonal excitation energies at matrix-assembly time, not at pair
  construction time.
- Use VASP FFT-grid conventions from OUTCAR (`NGXF/NGYF/NGZF`) whenever
  available.
- In `paw_orth_only`, include projector-overlap orthogonalization but do not
  modify the source pair density.
- In `paw_full`, rebuild the PAW source object from the VASP FAST_AUG dumps.

Current scope and limitations
-----------------------------
- Supported matrix modes: `pw_only`, `paw_orth_only`, and `paw_full`.
- Supported interactions: `hartree`, `direct`, and `both`.
- The script can consume the selected response basis for screened direct-term
  contractions, but its core object reconstruction path remains full-grid based.
- `paw_full` requires `BSE_TRANS_MATRIX_FOCK.bin` and `BSE_FASTAUG_FOCK.bin`
  from the matching VASP run.

CLI usage
---------
Typical usage from the repository root:

    python bsematrix.py \
        --wavecar WAVECAR \
        --outcar OUTCAR \
        --kpoints KPOINTS \
        --poscar POSCAR \
        --potcar POTCAR \
        --mode paw_orth_only \
        --interaction both \
        --vb-num 2 --cb-num 2 --ewin 0 6 \
        --use-response-basis \
        --output-prefix AMAT_bse \
        --bsefatband-output BSEFATBAND

Credits:
  - Ionizing
  - OpenAI Codex
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.fft import fftn, ifftn
from scipy.special import spherical_jn as _spherical_jn
from sph_harm import sph_r as _sph_r
from vasp_constant import EDEPS, HSQDTM, TPI
from wfull import ScreenedPotentialData, find_screened_potential_file, read_screened_potential, read_screened_potential_diag

try:
    from ase.io import read as ase_read
except ImportError:
    ase_read = None


class _DefaultsFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action: argparse.Action) -> str:
        """Append default values to help text unless argparse already does so."""
        help_text = action.help or ""
        if "%(default)" in help_text:
            return help_text
        if action.default in (None, argparse.SUPPRESS):
            return help_text
        if action.required:
            return help_text
        return f"{help_text} (default: %(default)s)"


def _resolve_existing_path(path_str: str | None, *, label: str) -> Path | None:
    """Resolve an optional path and fail early if the file does not exist."""
    if path_str is None:
        return None
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _wrap_frac(v: np.ndarray | Sequence[float]) -> np.ndarray:
    """Wrap fractional coordinates into the half-open interval [0, 1)."""
    return np.mod(np.asarray(v, dtype=float), 1.0)


def _wrap_frac_signed(v: np.ndarray | Sequence[float]) -> np.ndarray:
    """Wrap fractional coordinates into the interval [-0.5, 0.5)."""
    return np.mod(np.asarray(v, dtype=float) + 0.5, 1.0) - 0.5


def _parse_fortran_float(text: str) -> float:
    """Parse a float written with Fortran D/E exponent notation."""
    return float(text.replace("D", "E").replace("d", "E"))


def _frac_key(frac: np.ndarray | Sequence[float], decimals: int = 8) -> Tuple[float, ...]:
    """Build a rounded hashable key for wrapped fractional coordinates."""
    vals = np.mod(np.round(_wrap_frac(np.asarray(frac, dtype=float)), decimals), 1.0)
    return tuple(float(x) for x in vals.reshape(-1))


def _wrap_frac_signed_key(frac: np.ndarray | Sequence[float], decimals: int = 8) -> Tuple[float, ...]:
    """Build a rounded hashable key for signed wrapped fractional coordinates."""
    vals = np.round(_wrap_frac_signed(np.asarray(frac, dtype=float)), decimals)
    return tuple(float(x) for x in vals.reshape(-1))


@dataclass(frozen=True)
class KpointMatch:
    """Reference from a full-BZ point back to an irreducible-k representative."""
    ikpt: int
    time_reversed: bool = False
    symm_op: int = 0


@dataclass(frozen=True)
class FullBZKpoint:
    """Full-BZ k-point together with the matched irreducible representative."""
    full_index: int
    k_frac: Tuple[float, float, float]
    match: KpointMatch


@dataclass(frozen=True)
class SymmetryOp:
    """Symmetry operator parsed from OUTCAR in fractional coordinates."""
    irot: int
    real_matrix: np.ndarray
    reciprocal_matrix: np.ndarray
    tau_frac: np.ndarray


@dataclass(frozen=True)
class PairState:
    """Excitonic transition state stored in the same conventions as VASP BSE."""
    iv: int
    ic: int
    ik: int
    ik3: int
    ik_ir: int
    ik3_ir: int
    ispin: int
    eps_v: float
    eps_c: float
    k1_frac: Tuple[float, float, float]
    k3_frac: Tuple[float, float, float]
    k1_time_reversed: bool = False
    k3_time_reversed: bool = False
    k1_symm_op: int = 0
    k3_symm_op: int = 0

    @property
    def excitation_energy(self) -> float:
        """Return the independent-particle transition energy Ec - Ev."""
        return float(self.eps_c - self.eps_v)


@dataclass(frozen=True)
class FatbandExciton:
    """Container for one BSEFATBAND exciton block."""
    index: int
    bse_eigenvalue: float
    ip_eigenvalue: float
    column_weight: np.ndarray
    amplitude: np.ndarray


@dataclass(frozen=True)
class TransMatrixDump:
    """Serialized TRANS_MATRIX_FOCK metadata dumped from VASP."""
    version: int
    dims: Tuple[int, int, int, int]
    ntyp: int
    npro: int
    nprod: int
    nrspinors: int
    lmmx_wdes: np.ndarray
    lmmx_aug: np.ndarray
    lmax_fast_aug: np.ndarray
    lmax_fockae: int
    nmax_fockae: int
    trans_matrix: np.ndarray


@dataclass(frozen=True)
class FastAugSourceRecord:
    """One ion block from BSE_FASTAUG_FOCK.bin."""
    iatom: int
    ntype: int
    lmmax: int
    indmax: int
    nli: np.ndarray
    xfrac: Optional[np.ndarray]
    crrexp: Optional[np.ndarray]
    rproj: np.ndarray


@dataclass(frozen=True)
class FastAugSourceDump:
    """Serialized FAST_AUG_FOCK support and basis data dumped from VASP."""
    version: int
    grid_shape: Tuple[int, int, int]
    nions: int
    ntypes: int
    records: Tuple[FastAugSourceRecord, ...]


def _read_i4_from_buf(buf: bytes, offset: int) -> Tuple[int, int]:
    """Read one int32 from a byte buffer."""
    arr = np.frombuffer(buf, dtype=np.int32, count=1, offset=offset)
    if arr.size != 1:
        raise ValueError("Unexpected EOF while reading int32")
    return int(arr[0]), offset + 4


def _read_i4_vec_from_buf(buf: bytes, offset: int, n: int) -> Tuple[np.ndarray, int]:
    """Read a contiguous int32 vector from a byte buffer."""
    arr = np.frombuffer(buf, dtype=np.int32, count=n, offset=offset)
    if arr.size != n:
        raise ValueError("Unexpected EOF while reading int32 vector")
    return arr.astype(np.int32, copy=True), offset + 4 * n


def _read_f8_vec_from_buf(buf: bytes, offset: int, n: int) -> Tuple[np.ndarray, int]:
    """Read a contiguous float64 vector from a byte buffer."""
    arr = np.frombuffer(buf, dtype=np.float64, count=n, offset=offset)
    if arr.size != n:
        raise ValueError("Unexpected EOF while reading float64 vector")
    return arr.astype(np.float64, copy=True), offset + 8 * n


def _read_c16_from_buf(buf: bytes, offset: int, n: int) -> Tuple[np.ndarray, int]:
    """Read a contiguous complex128 vector from a byte buffer."""
    arr = np.frombuffer(buf, dtype=np.complex128, count=n, offset=offset)
    if arr.size != n:
        raise ValueError("Unexpected EOF while reading complex128 vector")
    return arr.astype(np.complex128, copy=True), offset + 16 * n


def read_bse_trans_matrix_dump(path: str | Path) -> TransMatrixDump:
    """Read BSE_TRANS_MATRIX_FOCK.bin."""
    dump_path = Path(path).expanduser().resolve()
    data = dump_path.read_bytes()
    offset = 0
    magic = data[offset:offset + 8]
    offset += 8
    if magic != b"BSETMTR1":
        raise ValueError(f"Unexpected magic {magic!r} in {dump_path}")

    version, offset = _read_i4_from_buf(data, offset)
    d1, offset = _read_i4_from_buf(data, offset)
    d2, offset = _read_i4_from_buf(data, offset)
    d3, offset = _read_i4_from_buf(data, offset)
    d4, offset = _read_i4_from_buf(data, offset)
    ntyp, offset = _read_i4_from_buf(data, offset)
    npro, offset = _read_i4_from_buf(data, offset)
    nprod, offset = _read_i4_from_buf(data, offset)
    nrspinors, offset = _read_i4_from_buf(data, offset)
    lmmx_wdes, offset = _read_i4_vec_from_buf(data, offset, ntyp)
    lmmx_aug, offset = _read_i4_vec_from_buf(data, offset, ntyp)
    lmax_fast_aug, offset = _read_i4_vec_from_buf(data, offset, ntyp)
    lmax_fockae, offset = _read_i4_from_buf(data, offset)
    nmax_fockae, offset = _read_i4_from_buf(data, offset)

    n_tm = d1 * d2 * d3 * d4
    tm_flat, offset = _read_f8_vec_from_buf(data, offset, n_tm)
    trans_matrix = tm_flat.reshape((d1, d2, d3, d4), order="F")
    if offset != len(data):
        raise ValueError(f"Trailing bytes in dump: {len(data) - offset}")

    return TransMatrixDump(
        version=version,
        dims=(d1, d2, d3, d4),
        ntyp=ntyp,
        npro=npro,
        nprod=nprod,
        nrspinors=nrspinors,
        lmmx_wdes=lmmx_wdes,
        lmmx_aug=lmmx_aug,
        lmax_fast_aug=lmax_fast_aug,
        lmax_fockae=lmax_fockae,
        nmax_fockae=nmax_fockae,
        trans_matrix=trans_matrix,
    )


def read_bse_fastaug_source_dump(path: str | Path) -> FastAugSourceDump:
    """Read BSE_FASTAUG_FOCK.bin."""
    dump_path = Path(path).expanduser().resolve()
    data = dump_path.read_bytes()
    offset = 0
    magic = data[offset:offset + 8]
    offset += 8
    if magic != b"BSEFAS01":
        raise ValueError(f"Unexpected magic {magic!r} in {dump_path}")

    version, offset = _read_i4_from_buf(data, offset)
    ngx, offset = _read_i4_from_buf(data, offset)
    ngy, offset = _read_i4_from_buf(data, offset)
    ngz, offset = _read_i4_from_buf(data, offset)
    nions, offset = _read_i4_from_buf(data, offset)
    ntypes, offset = _read_i4_from_buf(data, offset)

    records: List[FastAugSourceRecord] = []
    for _ in range(nions):
        iatom, offset = _read_i4_from_buf(data, offset)
        ntype, offset = _read_i4_from_buf(data, offset)
        lmmax, offset = _read_i4_from_buf(data, offset)
        indmax, offset = _read_i4_from_buf(data, offset)
        nli, offset = _read_i4_vec_from_buf(data, offset, indmax)
        xfrac: Optional[np.ndarray] = None
        crrexp: Optional[np.ndarray] = None
        if version >= 2:
            xfrac_flat, offset = _read_f8_vec_from_buf(data, offset, 3 * indmax)
            xfrac = xfrac_flat.reshape((3, indmax), order="F").T
        else:
            crrexp, offset = _read_c16_from_buf(data, offset, indmax)
        rproj_flat, offset = _read_f8_vec_from_buf(data, offset, indmax * lmmax)
        rproj = rproj_flat.reshape((indmax, lmmax), order="F")
        records.append(
            FastAugSourceRecord(
                iatom=iatom,
                ntype=ntype,
                lmmax=lmmax,
                indmax=indmax,
                nli=np.asarray(nli, dtype=np.int32),
                xfrac=None if xfrac is None else np.asarray(xfrac, dtype=np.float64),
                crrexp=None if crrexp is None else np.asarray(crrexp, dtype=np.complex128),
                rproj=np.asarray(rproj, dtype=np.float64),
            )
        )
    if offset != len(data):
        raise ValueError(f"Trailing bytes in FAST_AUG dump: {len(data) - offset}")
    return FastAugSourceDump(
        version=version,
        grid_shape=(ngx, ngy, ngz),
        nions=nions,
        ntypes=ntypes,
        records=tuple(records),
    )


def _kpoint_match_priority(match: KpointMatch) -> Tuple[int, int, int, int]:
    """Rank candidate k-point matches the same way the builder prefers them."""
    return (
        int(match.time_reversed),
        int(match.symm_op != 0),
        int(match.symm_op),
        int(match.ikpt),
    )


def _rodrigues_rotation(axis: Sequence[float], angle_deg: float) -> np.ndarray:
    """Construct a Cartesian rotation matrix from axis-angle data."""
    axis_arr = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis_arr))
    if norm < 1e-12:
        return np.eye(3, dtype=float)
    axis_arr /= norm
    theta = math.radians(angle_deg)
    x, y, z = axis_arr
    k_mat = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    ident = np.eye(3, dtype=float)
    return math.cos(theta) * ident + (1.0 - math.cos(theta)) * np.outer(axis_arr, axis_arr) + math.sin(theta) * k_mat


def _parse_outcar_symmetry_ops(outcar_path: Path, lattice: np.ndarray, tol: float = 1e-5) -> List[SymmetryOp]:
    """Parse fractional-space symmetry operators from an OUTCAR file."""
    lines = outcar_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = None
    for line_idx, line in enumerate(lines):
        if line.strip() == "Space group operators:":
            start = line_idx + 2
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
            det_a = _parse_fortran_float(parts[1])
            alpha = _parse_fortran_float(parts[2])
            axis = [_parse_fortran_float(value) for value in parts[3:6]]
            tau = np.array([_parse_fortran_float(value) for value in parts[6:9]], dtype=float)
        except ValueError:
            continue
        rot_cart = det_a * _rodrigues_rotation(axis, alpha)
        rot_frac = np.linalg.inv(basis) @ rot_cart @ basis
        rot_frac_i = np.rint(rot_frac).astype(int)
        if np.max(np.abs(rot_frac - rot_frac_i)) > tol:
            raise ValueError(f"Failed to convert symmetry operator {irot} into an integer lattice matrix")
        recip_frac = np.linalg.inv(rot_frac_i).T
        recip_frac_i = np.rint(recip_frac).astype(int)
        if np.max(np.abs(recip_frac - recip_frac_i)) > tol:
            raise ValueError(f"Failed to convert reciprocal symmetry operator {irot} into an integer matrix")
        ops.append(SymmetryOp(irot=irot, real_matrix=rot_frac_i, reciprocal_matrix=recip_frac_i, tau_frac=tau))
    return ops


def _parse_outcar_full_bz_kpoints(outcar_path: Path) -> List[Tuple[np.ndarray, int, bool]]:
    """Parse the full-BZ k-point list emitted by IBZKPT_HF in OUTCAR."""
    lines = outcar_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line_idx, line in enumerate(lines):
        if "Subroutine IBZKPT_HF returns following result" not in line:
            continue
        start = None
        for inner_idx in range(line_idx + 1, len(lines)):
            if "Following reciprocal coordinates:" in lines[inner_idx]:
                start = inner_idx + 1
                break
        if start is None:
            continue
        entries: List[Tuple[np.ndarray, int, bool]] = []
        for row in lines[start:]:
            text = row.strip()
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
                k_frac = np.array([_parse_fortran_float(fields[0]), _parse_fortran_float(fields[1]), _parse_fortran_float(fields[2])], dtype=float)
                ikpt = int(fields[4])
            except ValueError:
                if entries:
                    return entries
                continue
            time_reversed = fields[6].upper().startswith("T")
            entries.append((_wrap_frac(k_frac), ikpt, time_reversed))
        if entries:
            return entries
    return []


def _parse_outcar_ibzkpt_sections(outcar_path: Path) -> List[List[np.ndarray]]:
    """Parse all IBZKPT reciprocal-coordinate tables found in OUTCAR."""
    lines = outcar_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    sections: List[List[np.ndarray]] = []
    line_idx = 0
    while line_idx < len(lines):
        if "Subroutine IBZKPT returns following result:" not in lines[line_idx]:
            line_idx += 1
            continue
        start = None
        for inner_idx in range(line_idx + 1, len(lines)):
            if "Following reciprocal coordinates:" in lines[inner_idx]:
                start = inner_idx + 2
                break
        if start is None:
            line_idx += 1
            continue
        coords: List[np.ndarray] = []
        for row in lines[start:]:
            text = row.strip()
            if not text:
                break
            fields = text.split()
            if len(fields) < 4:
                break
            try:
                coords.append(_wrap_frac(np.array([_parse_fortran_float(fields[0]), _parse_fortran_float(fields[1]), _parse_fortran_float(fields[2])], dtype=float)))
            except ValueError:
                break
        if coords:
            sections.append(coords)
        line_idx = start if start is not None else line_idx + 1
    return sections


def _parse_outcar_charge_grid(outcar_path: Path) -> Optional[Tuple[int, int, int]]:
    """Read the VASP charge-density FFT grid from OUTCAR."""
    pattern = re.compile(
        r"dimension x,y,z NGXF=\s*(\d+) NGYF=\s*(\d+) NGZF=\s*(\d+)",
        re.IGNORECASE,
    )
    for line in outcar_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match is not None:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None


def _parse_outcar_encutgw(outcar_path: Path) -> Optional[float]:
    """Read ENCUTGW from OUTCAR if the calculation wrote it."""
    pattern = re.compile(r"\bENCUTGW\s*=\s*([-+0-9.DEded]+)", re.IGNORECASE)
    for line in outcar_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match is not None:
            return _parse_fortran_float(match.group(1))
    return None


def _parse_outcar_ir_kpoints(outcar_path: Path, nkpts: int) -> Optional[np.ndarray]:
    """Recover the irreducible-k coordinates from the OUTCAR band blocks."""
    pattern = re.compile(r"^\s*k-point\s+(\d+)\s*:\s*([-+0-9.DEded]+)\s+([-+0-9.DEded]+)\s+([-+0-9.DEded]+)\s+plane waves:")
    lines = outcar_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    block: List[np.ndarray] = []
    expected = 1
    for line in lines:
        match = pattern.match(line)
        if match is None:
            if len(block) == nkpts:
                return np.asarray(block, dtype=float)
            continue
        idx = int(match.group(1))
        kvec = np.array([_parse_fortran_float(match.group(2)), _parse_fortran_float(match.group(3)), _parse_fortran_float(match.group(4))], dtype=float)
        if idx == 1:
            block = [kvec]
            expected = 2
            continue
        if block and idx == expected:
            block.append(kvec)
            expected += 1
            if len(block) == nkpts:
                return np.asarray(block, dtype=float)
            continue
        block = []
        expected = 1
    if len(block) == nkpts:
        return np.asarray(block, dtype=float)
    return None


def _load_ir_kvecs_from_search_dirs(search_dirs: Sequence[Path], fallback_kvecs: np.ndarray) -> np.ndarray:
    """Prefer OUTCAR-derived irreducible k-vectors over the WAVECAR fallback."""
    fallback = np.asarray(fallback_kvecs, dtype=float)
    nkpts = int(fallback.shape[0])
    for dpath in search_dirs:
        candidate = dpath / "OUTCAR"
        if not candidate.is_file():
            continue
        parsed = _parse_outcar_ir_kpoints(candidate, nkpts)
        if parsed is None:
            continue
        resolved = np.asarray(fallback, dtype=float).copy()
        parsed_signed = _wrap_frac_signed(parsed)
        fallback_signed = _wrap_frac_signed(fallback)
        neg_signed = _wrap_frac_signed(-fallback)
        direct_err = np.max(np.abs(_wrap_frac_signed(fallback_signed - parsed_signed)), axis=1)
        neg_err = np.max(np.abs(_wrap_frac_signed(neg_signed - parsed_signed)), axis=1)
        use_neg = neg_err + 1e-8 < direct_err
        resolved[use_neg] = -fallback[use_neg]
        return resolved
    return fallback


def _generate_gvectors_for_kvec(wfc: vaspwfc, kvec: Sequence[float]) -> np.ndarray:
    """Generate plane-wave G vectors for an explicit k-point and ENCUT."""
    fx, fy, fz = [np.arange(n, dtype=int) for n in wfc._ngrid]
    fx[wfc._ngrid[0] // 2 + 1:] -= wfc._ngrid[0]
    fy[wfc._ngrid[1] // 2 + 1:] -= wfc._ngrid[1]
    fz[wfc._ngrid[2] // 2 + 1:] -= wfc._ngrid[2]
    gz, gy, gx = np.array(np.meshgrid(fz, fy, fx, indexing="ij")).reshape((3, -1))
    kgrid = np.array([gx, gy, gz], dtype=float).T
    kvec_arr = np.asarray(kvec, dtype=float)
    kenergy = HSQDTM * np.linalg.norm(np.dot(kgrid + kvec_arr[np.newaxis, :], TPI * wfc._Bcell), axis=1) ** 2
    return np.asarray(kgrid[np.where(kenergy < wfc._encut)[0]], dtype=int)


def _build_gaunt_table(l_max: int) -> dict:
    """Numerically compute Gaunt coefficients for real spherical harmonics."""
    L_max = 2 * l_max
    ntheta, nphi = 300, 600
    theta = np.linspace(0.0, np.pi, ntheta)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    st = np.sin(T).ravel()
    xyz = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], axis=-1).reshape(-1, 3)
    dtheta = np.pi / (ntheta - 1)
    dphi = 2.0 * np.pi / (nphi - 1)
    dA = st * dtheta * dphi
    ylm_all: dict = {}
    for l in range(L_max + 1):
        ylm = _sph_r(xyz, l)
        for m_idx, m in enumerate(range(-l, l + 1)):
            ylm_all[(l, m)] = ylm[:, m_idx]
    gaunt: dict = {}
    for l1 in range(l_max + 1):
        for m1 in range(-l1, l1 + 1):
            Y1 = ylm_all[(l1, m1)]
            for l2 in range(l_max + 1):
                for m2 in range(-l2, l2 + 1):
                    Y12 = Y1 * ylm_all[(l2, m2)]
                    for L in range(abs(l1 - l2), l1 + l2 + 1):
                        if (l1 + l2 + L) % 2 != 0:
                            continue
                        for M in range(-L, L + 1):
                            G = float(np.dot(Y12 * ylm_all[(L, M)], dA))
                            if abs(G) > 1e-13:
                                gaunt[(l1, m1, l2, m2, L, M)] = G
    return gaunt


_GAUNT_CACHE: dict = {}


def _get_gaunt_table(l_max: int) -> dict:
    """Return the cached Gaunt table for a given projector l_max."""
    if l_max not in _GAUNT_CACHE:
        _GAUNT_CACHE[l_max] = _build_gaunt_table(l_max)
    return _GAUNT_CACHE[l_max]


def _compute_Qij_lm(pp: Any) -> np.ndarray:
    """Build the full-multipole PAW augmentation matrix Q_{ij}^{LM}."""
    lmmax = pp.lmmax
    l_max = max(l for (_n, l, _m) in pp.ilm)
    L_max = 2 * l_max
    n_LM = (L_max + 1) ** 2
    gaunt = _get_gaunt_table(l_max)
    Q_lm = np.zeros((n_LM, lmmax, lmmax))
    for ii in range(lmmax):
        n1, l1, m1 = pp.ilm[ii]
        ae1 = pp.paw_ae_wfc[n1]
        ps1 = pp.paw_ps_wfc[n1]
        for jj in range(lmmax):
            n2, l2, m2 = pp.ilm[jj]
            ae2 = pp.paw_ae_wfc[n2]
            ps2 = pp.paw_ps_wfc[n2]
            for L in range(abs(l1 - l2), l1 + l2 + 1):
                if (l1 + l2 + L) % 2 != 0 or L > L_max:
                    continue
                R_L = pp.radial_simp_int(pp.rgrid**L * (ae1 * ae2 - ps1 * ps2), inside_rcomp=True)
                if abs(R_L) < 1e-30:
                    continue
                for M in range(-L, L + 1):
                    G = gaunt.get((l1, m1, l2, m2, L, M), 0.0)
                    if abs(G) > 1e-15:
                        Q_lm[L * L + L + M, ii, jj] += R_L * G
    return Q_lm


def _depsum_two_bands_rholm_trace_py(
    beta_left: np.ndarray,
    beta_right: np.ndarray,
    q_lm: np.ndarray,
) -> np.ndarray:
    """Contract two projector vectors with Q_{ij}^{LM} to obtain rho_LM."""
    q_lm_arr = np.asarray(q_lm, dtype=np.complex128)
    b_left = np.asarray(beta_left, dtype=np.complex128).reshape(-1)
    b_right = np.asarray(beta_right, dtype=np.complex128).reshape(-1)
    return np.einsum("i,kij,j->k", np.conj(b_left), q_lm_arr, b_right)


def _depsum_two_bands_rholm_trace_fortran_py(
    cproj1: np.ndarray,
    cproj2: np.ndarray,
    trans_matrix_lm_first: np.ndarray,
) -> np.ndarray:
    """Mirror VASP RHOLM_KERNEL ordering for TRANS_MATRIX_FOCK contractions."""
    trans_arr = np.asarray(trans_matrix_lm_first, dtype=np.complex128)
    left = np.asarray(cproj1, dtype=np.complex128).reshape(-1)
    right = np.asarray(cproj2, dtype=np.complex128).reshape(-1)
    if trans_arr.ndim != 3:
        raise ValueError(f"trans_matrix_lm_first must be rank-3, got shape {trans_arr.shape}")
    lmmax = trans_arr.shape[1]
    if trans_arr.shape[2] != lmmax:
        raise ValueError(f"trans_matrix_lm_first last two dimensions must match, got {trans_arr.shape}")
    if left.size < lmmax or right.size < lmmax:
        raise ValueError(
            "Projector vectors are smaller than lmmax: "
            f"left={left.size}, right={right.size}, lmmax={lmmax}"
        )
    return np.einsum("i,kji,j->k", left[:lmmax], trans_arr, np.conj(right[:lmmax]))


class _PawOrthHelper:
    """PAW helper for projector overlaps and full-BZ projector reconstruction."""
    def __init__(self, *, poscar_path: Path, potcar_path: Path, wfc: vaspwfc, ir_kvecs: np.ndarray, lgamma: bool, gamma_half: str, search_dirs: Sequence[Path]) -> None:
        """Load PAW metadata and prepare projector caches."""
        from paw import pawpotcar, _build_elem_groups

        if ase_read is None:
            raise ImportError("ASE is required for PAW-enabled modes")
        self.atoms = ase_read(str(poscar_path))
        self.wfc = wfc
        self.ir_kvecs = np.asarray(ir_kvecs, dtype=float)
        self.lgamma = bool(lgamma)
        self.gamma_half = str(gamma_half)
        self.encut = float(wfc._encut)
        self.pawpp = [pawpotcar(potstr) for potstr in potcar_path.read_text().split("End of Dataset")[:-1]]
        _, _, self.element_idx = _build_elem_groups(self.atoms, self.pawpp, str(potcar_path))
        self.natoms = len(self.atoms)
        self._Qij_per_type = [np.asarray(pp.get_Qij(), dtype=np.complex128) for pp in self.pawpp]
        self._Qij_lm_per_type = [_compute_Qij_lm(pp) for pp in self.pawpp]
        self._nonlq_cache: Dict[Tuple[float, ...], nonlq] = {}
        self._nonlq_kvec_cache: Dict[Tuple[float, ...], nonlq] = {}
        self._proj_cache: Dict[Tuple[int, int, int], List[np.ndarray]] = {}
        self._search_dirs = [Path(d).resolve() for d in search_dirs]
        self._source_trans_dump_path: Optional[Path] = None
        self._source_trans_dump: Optional[TransMatrixDump] = None
        self._source_fastaug_dump_path: Optional[Path] = None
        self._source_fastaug_dump: Optional[FastAugSourceDump] = None
        self._source_aug_cache: Dict[str, np.ndarray] = {}
        self._Bcell = np.asarray(wfc._Bcell, dtype=float)
        self._Omega = float(wfc._Omega)
        self._pos_frac = np.asarray(self.atoms.get_scaled_positions(), dtype=float)
        self._multipole_kernel_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    def _split_projector_vector(self, beta_flat: np.ndarray) -> List[np.ndarray]:
        """Split a flattened projector vector into per-atom blocks."""
        flat = np.asarray(beta_flat, dtype=np.complex128).reshape(-1)
        blocks: List[np.ndarray] = []
        offset = 0
        for iatom in range(self.natoms):
            lmmax = int(self.pawpp[self.element_idx[iatom]].lmmax)
            blocks.append(np.asarray(flat[offset:offset + lmmax], dtype=np.complex128))
            offset += lmmax
        return blocks

    def _projector_for_k(self, kvec: Sequence[float]) -> nonlq:
        """Return the projector object for an irreducible-k representative."""
        from paw import nonlq

        key = _wrap_frac_signed_key(kvec)
        cached = self._nonlq_cache.get(key)
        if cached is not None:
            return cached
        projector = nonlq(self.atoms, self.encut, potcar=self.pawpp, k=np.asarray(kvec, dtype=float), lgam=self.lgamma, gamma_half=self.gamma_half)
        self._nonlq_cache[key] = projector
        return projector

    def get_proj(self, ispin: int, ik: int, iband: int, coeff: np.ndarray) -> List[np.ndarray]:
        """Project one irreducible-k band onto PAW projectors atom by atom."""
        key = (ispin, ik, iband)
        cached = self._proj_cache.get(key)
        if cached is not None:
            return cached
        projector = self._projector_for_k(self.ir_kvecs[ik - 1])
        beta = projector.proj(np.asarray(coeff, dtype=np.complex128))
        blocks = self._split_projector_vector(beta)
        self._proj_cache[key] = blocks
        return blocks

    def _projector_for_full_k(self, kvec: Sequence[float]) -> nonlq:
        """Return the projector object for an arbitrary full-BZ k-point."""
        from paw import nonlq

        key = _frac_key(kvec)
        cached = self._nonlq_kvec_cache.get(key)
        if cached is not None:
            return cached
        projector = nonlq(
            self.atoms,
            self.encut,
            potcar=self.pawpp,
            k=np.asarray(kvec, dtype=float),
            lgam=self.lgamma,
            gamma_half=self.gamma_half,
        )
        self._nonlq_kvec_cache[key] = projector
        return projector

    def project_coefficients(
        self,
        coeff: np.ndarray,
        gvecs: np.ndarray,
        k_frac: Sequence[float],
    ) -> List[np.ndarray]:
        """Project a full-BZ coefficient vector after remapping onto projector G vectors."""
        projector = self._projector_for_full_k(k_frac)
        coeff_arr = np.asarray(coeff, dtype=np.complex128).reshape(-1)
        gvecs_arr = np.asarray(gvecs, dtype=int)
        if gvecs_arr.shape == projector.Gvec.shape and np.all(gvecs_arr == projector.Gvec):
            matched = coeff_arr
        else:
            gdict = {tuple(int(x) for x in g): idx for idx, g in enumerate(gvecs_arr)}
            matched = np.zeros(projector.Gvec.shape[0], dtype=np.complex128)
            for idx, gvec in enumerate(projector.Gvec):
                src_idx = gdict.get(tuple(int(x) for x in gvec))
                if src_idx is not None:
                    matched[idx] = coeff_arr[src_idx]
        return [
            np.asarray(projector.proj(matched, whichatom=iatom), dtype=np.complex128)
            for iatom in range(self.natoms)
        ]

    def _get_multipole_kernel_data(
        self,
        ngrid: Tuple[int, int, int],
        q_frac: np.ndarray,
        L_max_global: int,
    ) -> Dict[str, Any]:
        """Return cached geometry-dependent data for PAW compensation charges."""
        q_frac_arr = np.asarray(q_frac, dtype=float)
        cache_key = (
            tuple(int(n) for n in ngrid),
            tuple(float(round(x, 12)) for x in q_frac_arr),
            int(L_max_global),
        )
        cached = self._multipole_kernel_cache.get(cache_key)
        if cached is not None:
            return cached
        nx, ny, nz = ngrid
        q_cart = q_frac_arr @ (TPI * self._Bcell)
        gx = (np.fft.fftfreq(nx) * nx).astype(float)
        gy = (np.fft.fftfreq(ny) * ny).astype(float)
        gz = (np.fft.fftfreq(nz) * nz).astype(float)
        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
        G_frac = np.stack([GX, GY, GZ], axis=-1)
        G_frac_flat = G_frac.reshape(-1, 3)
        Gcart = G_frac @ (TPI * self._Bcell)
        qpG_cart = (Gcart + q_cart).reshape(-1, 3)
        qpGnorm = np.linalg.norm(qpG_cart, axis=-1)
        mask_nz = qpGnorm > 1e-10
        qpGnorm_safe = np.where(mask_nz, qpGnorm, 1.0)
        qpGhat = qpG_cart / qpGnorm_safe[:, np.newaxis]
        q_dot_R = TPI * (self._pos_frac @ q_frac_arr)
        G_dot_R = TPI * (G_frac_flat @ self._pos_frac.T)
        phase = np.exp(1j * (G_dot_R + q_dot_R[np.newaxis, :]))
        ylm_grid: Dict[Tuple[int, int], np.ndarray] = {}
        for L in range(L_max_global + 1):
            YLM_all = _sph_r(qpGhat, L)
            for M_idx, M in enumerate(range(-L, L + 1)):
                ylm_grid[(L, M)] = YLM_all[:, M_idx]
        cached = {"qpGnorm": qpGnorm, "mask_nz": mask_nz, "phase": phase, "ylm_grid": ylm_grid}
        self._multipole_kernel_cache[cache_key] = cached
        return cached

    def _compensation_charge_G(
        self,
        ngrid: Tuple[int, int, int],
        q_frac: np.ndarray,
        beta_src_left: Optional[List[np.ndarray]],
        beta_src_right: Optional[List[np.ndarray]],
    ) -> np.ndarray:
        """Build the PAW compensation charge on the FFT grid for one source pair."""
        if beta_src_left is None or beta_src_right is None:
            return np.zeros(tuple(int(n) for n in ngrid), dtype=np.complex128)
        grid_shape = tuple(int(n) for n in ngrid)
        N = int(np.prod(grid_shape))
        q_frac_arr = np.asarray(q_frac, dtype=float)
        L_max_global = max(max(l for (_n, l, _m) in pp.ilm) * 2 for pp in self.pawpp)
        kernel_data = self._get_multipole_kernel_data(grid_shape, q_frac_arr, L_max_global)
        qpGnorm = kernel_data["qpGnorm"]
        mask_nz = kernel_data["mask_nz"]
        phase = kernel_data["phase"]
        ylm_grid = kernel_data["ylm_grid"]
        delta_rho = np.zeros(N, dtype=np.complex128)
        scale = N / self._Omega
        for iatom in range(self.natoms):
            ntype = self.element_idx[iatom]
            pp = self.pawpp[ntype]
            rcomp = pp.rcomp
            Q_lm = self._Qij_lm_per_type[ntype]
            l_max = max(l for (_n, l, _m) in pp.ilm)
            L_max = 2 * l_max
            n_LM = (L_max + 1) ** 2
            q_vec = _depsum_two_bands_rholm_trace_py(
                beta_src_left[iatom],
                beta_src_right[iatom],
                Q_lm[:n_LM],
            )
            phase_atom = phase[:, iatom]
            for L in range(L_max + 1):
                iL = (1j) ** L
                coeff_L = 4.0 * np.pi * iL / (2.0 * L + 1.0)
                jL_vals = _spherical_jn(L, qpGnorm * rcomp)
                jL_vals = np.where(mask_nz, jL_vals, (1.0 if L == 0 else 0.0))
                for M in range(-L, L + 1):
                    lm_idx = L * L + L + M
                    if abs(q_vec[lm_idx]) < 1e-20:
                        continue
                    delta_rho += scale * q_vec[lm_idx] * coeff_L * jL_vals * ylm_grid[(L, M)] * phase_atom
        return delta_rho.reshape(grid_shape)

    def _resolve_source_trans_matrix_dump_path(self) -> Optional[Path]:
        """Locate BSE_TRANS_MATRIX_FOCK.bin near the example inputs."""
        if self._source_trans_dump_path is not None:
            return self._source_trans_dump_path
        for directory in self._search_dirs:
            candidate = directory / "BSE_TRANS_MATRIX_FOCK.bin"
            if candidate.exists():
                self._source_trans_dump_path = candidate.resolve()
                return self._source_trans_dump_path
        return None

    def _get_source_trans_matrix_dump(self) -> TransMatrixDump:
        """Load TRANS_MATRIX_FOCK from the nearest VASP dump."""
        if self._source_trans_dump is not None:
            return self._source_trans_dump
        dump_path = self._resolve_source_trans_matrix_dump_path()
        if dump_path is None or not dump_path.exists():
            raise FileNotFoundError("paw_full requires BSE_TRANS_MATRIX_FOCK.bin next to the example inputs")
        self._source_trans_dump = read_bse_trans_matrix_dump(dump_path)
        return self._source_trans_dump

    def _resolve_source_fastaug_dump_path(self) -> Optional[Path]:
        """Locate BSE_FASTAUG_FOCK.bin near the example inputs."""
        if self._source_fastaug_dump_path is not None:
            return self._source_fastaug_dump_path
        for directory in self._search_dirs:
            for name in ("BSE_FAST_AUG_FOCK.bin", "BSE_FASTAUG_FOCK.bin"):
                candidate = directory / name
                if candidate.exists():
                    self._source_fastaug_dump_path = candidate.resolve()
                    return self._source_fastaug_dump_path
        return None

    def _get_source_fastaug_dump(self) -> FastAugSourceDump:
        """Load the dumped FAST_AUG support and basis data."""
        if self._source_fastaug_dump is not None:
            return self._source_fastaug_dump
        dump_path = self._resolve_source_fastaug_dump_path()
        if dump_path is None or not dump_path.exists():
            raise FileNotFoundError(
                "paw_full requires BSE_FASTAUG_FOCK.bin "
                "(or legacy BSE_FAST_AUG_FOCK.bin) next to the example inputs"
            )
        self._source_fastaug_dump = read_bse_fastaug_source_dump(dump_path)
        return self._source_fastaug_dump

    @staticmethod
    def _hash_beta_list(hasher: "hashlib._Hash", beta_list: Optional[List[np.ndarray]]) -> None:
        """Hash one projector-block list into a stable source-augmentation cache key."""
        if beta_list is None:
            hasher.update(b"none")
            return
        hasher.update(np.asarray([len(beta_list)], dtype=np.int64).tobytes())
        for arr in beta_list:
            arr_c = np.ascontiguousarray(np.asarray(arr, dtype=np.complex128))
            hasher.update(np.asarray(arr_c.shape, dtype=np.int64).tobytes())
            hasher.update(arr_c.view(np.uint8).tobytes())

    def _source_aug_cache_key(
        self,
        ngrid: Tuple[int, int, int],
        q_frac: np.ndarray,
        beta_src_left: Optional[List[np.ndarray]],
        beta_src_right: Optional[List[np.ndarray]],
    ) -> str:
        """Build a content hash for exact FAST_AUG source augmentation."""
        hasher = hashlib.sha256()
        hasher.update(np.asarray(tuple(int(n) for n in ngrid), dtype=np.int64).tobytes())
        hasher.update(np.asarray(q_frac, dtype=np.float64).tobytes())
        self._hash_beta_list(hasher, beta_src_left)
        self._hash_beta_list(hasher, beta_src_right)
        return hasher.hexdigest()

    def _build_full_crholm_blocks(
        self,
        trans_dump: TransMatrixDump,
        beta_left: List[np.ndarray],
        beta_right: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Build per-atom FAST_AUG source coefficients using TRANS_MATRIX_FOCK."""
        blocks: List[np.ndarray] = []
        for iatom in range(self.natoms):
            ntype = self.element_idx[iatom]
            n_aug = int(trans_dump.lmmx_aug[ntype])
            n_lm = (int(trans_dump.lmax_fast_aug[ntype]) + 1) ** 2
            tm = np.asarray(trans_dump.trans_matrix[:, :, :n_lm, ntype], dtype=np.float64)
            tm_lm_first = np.transpose(tm, (2, 0, 1))
            coeff = np.zeros(n_aug, dtype=np.complex128)
            coeff[:n_lm] = _depsum_two_bands_rholm_trace_fortran_py(
                beta_left[iatom],
                beta_right[iatom],
                tm_lm_first,
            )
            blocks.append(coeff)
        return blocks

    def _exact_fastaug_source_density_G(
        self,
        rho_G: np.ndarray,
        q_frac: np.ndarray,
        beta_src_left: Optional[List[np.ndarray]],
        beta_src_right: Optional[List[np.ndarray]],
    ) -> np.ndarray:
        """Rebuild the VASP FAST_AUG source object exactly on the dumped GRIDHF support."""
        rho_pw = np.asarray(rho_G, dtype=np.complex128)
        if beta_src_left is None or beta_src_right is None:
            return np.array(rho_pw, copy=True)

        grid_shape = tuple(int(n) for n in rho_pw.shape)
        cache_key = self._source_aug_cache_key(
            grid_shape,
            np.asarray(q_frac, dtype=float),
            beta_src_left,
            beta_src_right,
        )
        cached = self._source_aug_cache.get(cache_key)
        if cached is None:
            trans_dump = self._get_source_trans_matrix_dump()
            fastaug_dump = self._get_source_fastaug_dump()
            if tuple(int(n) for n in fastaug_dump.grid_shape) != grid_shape:
                raise ValueError(
                    f"FAST_AUG source grid {fastaug_dump.grid_shape} does not match requested grid {grid_shape}"
                )
            coeff_blocks = self._build_full_crholm_blocks(trans_dump, beta_src_left, beta_src_right)
            aug_r = np.zeros(grid_shape, dtype=np.complex128)
            q_arr = np.asarray(q_frac, dtype=float)
            for record in fastaug_dump.records:
                atom_index = int(record.iatom) - 1
                coeff = np.asarray(coeff_blocks[atom_index], dtype=np.complex128)
                lm_count = min(coeff.size, int(record.lmmax))
                if lm_count == 0 or int(record.indmax) == 0:
                    continue
                work = np.asarray(record.rproj[:, :lm_count] @ coeff[:lm_count], dtype=np.complex128)
                if record.xfrac is not None:
                    scatter_phase = np.exp(1j * 2.0 * np.pi * (np.asarray(record.xfrac, dtype=np.float64) @ q_arr))
                elif record.crrexp is not None:
                    scatter_phase = np.conj(np.asarray(record.crrexp, dtype=np.complex128))
                else:
                    raise ValueError("FAST_AUG source record must provide either xfrac or crrexp")
                coords = np.asarray(
                    np.unravel_index(np.asarray(record.nli, dtype=np.int64) - 1, grid_shape, order="F"),
                    dtype=np.int64,
                )
                aug_r[coords[0], coords[1], coords[2]] += work * scatter_phase
            cached = np.asarray(fftn(aug_r), dtype=np.complex128)
            self._source_aug_cache[cache_key] = cached
        return rho_pw + np.asarray(cached, dtype=np.complex128)

    def effective_source_density_G(
        self,
        rho_G: np.ndarray,
        q_frac: np.ndarray,
        beta_src_left: Optional[List[np.ndarray]] = None,
        beta_src_right: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Upgrade a PW source density with the exact VASP FAST_AUG source object."""
        return self._exact_fastaug_source_density_G(
            rho_G,
            q_frac,
            beta_src_left=beta_src_left,
            beta_src_right=beta_src_right,
        )

class HartreeFullGridBuilder:
    """Assemble the BSE matrix directly on the real-space FFT grid."""
    def __init__(self, *, wavecar_path: Path, outcar_path: Path, kpoints_path: Path, q_ext: Sequence[float], vb_num: Optional[int], cb_num: Optional[int], ewin: Optional[Tuple[float, float]], mode: str, poscar_path: Optional[Path], potcar_path: Optional[Path], charge_grid: Optional[Sequence[int]], wfc_ifft_scale: str, interaction: str, epsilon: Optional[float], use_response_basis: bool, verbose: bool = False) -> None:
        """Load wavefunction metadata, symmetry data, caches, and PAW helpers."""
        self.wavecar_path = Path(wavecar_path).resolve()
        self.outcar_path = Path(outcar_path).resolve()
        self.kpoints_path = Path(kpoints_path).resolve()
        self.mode = mode
        self.verbose = bool(verbose)
        self.q_ext = np.asarray(q_ext, dtype=float)
        self.vb_num = vb_num
        self.cb_num = cb_num
        self.ewin = ewin
        self.interaction = str(interaction)
        if self.interaction not in {"hartree", "direct", "both"}:
            raise ValueError("interaction must be one of {'hartree', 'direct', 'both'}")
        self.epsilon = None if epsilon is None else float(epsilon)
        self.use_response_basis = bool(use_response_basis)
        if self.epsilon is not None and self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.occ_threshold = 0.5
        self.wfc_ifft_scale = str(wfc_ifft_scale).strip()
        if self.wfc_ifft_scale not in {"sqrtN", "N", "none"}:
            raise ValueError("wfc_ifft_scale must be one of {'sqrtN', 'N', 'none'}")

        from vaspwfc import vaspwfc

        self.wfc = vaspwfc(str(self.wavecar_path))
        self.Acell = np.asarray(self.wfc._Acell, dtype=float)
        self.Bcell = np.asarray(self.wfc._Bcell, dtype=float)
        self.Omega = float(self.wfc._Omega)
        self._nspin = int(self.wfc._nspin)
        self._nkpts = int(self.wfc._nkpts)
        self._nbands = int(self.wfc._nbands)
        self.spin_factor = 2.0 if self._nspin == 1 else 1.0

        if charge_grid is None:
            self.ngrid = tuple(int(2 * n) for n in self.wfc._ngrid)
        else:
            if len(charge_grid) != 3:
                raise ValueError("charge_grid must contain exactly three integers")
            self.ngrid = tuple(int(n) for n in charge_grid)
        self.Nfft = int(np.prod(self.ngrid))
        self._Gcart_grid = self._build_Gcart_grid()
        self.encutgw = _parse_outcar_encutgw(self.outcar_path)

        search_dirs = [self.wavecar_path.parent, self.outcar_path.parent, self.kpoints_path.parent]
        if poscar_path is not None:
            search_dirs.append(Path(poscar_path).resolve().parent)
        if potcar_path is not None:
            search_dirs.append(Path(potcar_path).resolve().parent)
        seen_dirs: set[Path] = set()
        self._search_dirs: List[Path] = []
        for dpath in search_dirs:
            resolved = dpath.resolve()
            if resolved in seen_dirs:
                continue
            seen_dirs.add(resolved)
            self._search_dirs.append(resolved)

        self._ir_kvecs = _load_ir_kvecs_from_search_dirs(self._search_dirs, self.wfc._kvecs)
        self._kmap: Dict[Tuple[float, ...], int] = {}
        for ik0, kvec in enumerate(self._ir_kvecs):
            self._kmap[_frac_key(kvec)] = ik0

        self.symm_ops = self._load_symmetry_ops()
        self._expanded_kpoint_lookup: Dict[Tuple[float, ...], KpointMatch] = {}
        self._expanded_kpoint_coords: List[np.ndarray] = []
        self._expanded_kpoint_matches: List[KpointMatch] = []
        self._full_bz_lookup: Dict[Tuple[float, ...], KpointMatch] = {}
        self._full_bz_index_lookup: Dict[Tuple[float, ...], int] = {}
        self._resolved_kpoint_cache: Dict[Tuple[float, ...], KpointMatch] = {}
        self._full_bz_kpoint_coords: List[np.ndarray] = []
        self._full_bz_kpoint_matches: List[KpointMatch] = []
        self._setphase_grid_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._build_expanded_kpoint_lookup()
        self.full_kpoints = self._load_full_bz_kpoints()
        self.kpoint_weight = self._infer_kpoint_weight()

        self._wfc_cache: Dict[Tuple[Any, ...], np.ndarray] = {}
        self._matched_wfc_cache: Dict[Tuple[Any, ...], np.ndarray] = {}
        self._grid_wfc_cache: Dict[Tuple[Any, ...], np.ndarray] = {}
        self._grid_matched_wfc_cache: Dict[Tuple[Any, ...], np.ndarray] = {}
        self._matched_coeff_grid_cache: Dict[Tuple[Any, ...], np.ndarray] = {}
        self._matched_proj_cache: Dict[Tuple[Any, ...], List[np.ndarray]] = {}
        self._orth_coeff_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._orth_gvec_cache: Dict[int, np.ndarray] = {}
        self._screened_kernel_cache: Dict[Tuple[float, ...], np.ndarray] = {}
        self._screened_operator_cache: Dict[Tuple[float, ...], Tuple[np.ndarray, np.ndarray]] = {}
        self._screened_diag_cache: Dict[int, np.ndarray] = {}
        self._screened_matrix_cache: Dict[int, ScreenedPotentialData] = {}
        self._response_gvec_cache: Dict[int, np.ndarray] = {}
        self._orth_ready = False
        self.pairs: List[PairState] = []

        self._paw: Optional[_PawOrthHelper] = None
        if self.mode in {"paw_orth_only", "paw_full"}:
            if poscar_path is None or potcar_path is None:
                raise FileNotFoundError(f"{self.mode} requires POSCAR and POTCAR")
            self._paw = _PawOrthHelper(
                poscar_path=Path(poscar_path).resolve(),
                potcar_path=Path(potcar_path).resolve(),
                wfc=self.wfc,
                ir_kvecs=self._ir_kvecs,
                lgamma=getattr(self.wfc, "_lgam", False),
                gamma_half=getattr(self.wfc, "_gam_half", "x"),
                search_dirs=self._search_dirs,
            )
            if self.mode == "paw_full":
                fastaug_dump = self._paw._get_source_fastaug_dump()
                self.ngrid = tuple(int(n) for n in fastaug_dump.grid_shape)
                self.Nfft = int(np.prod(self.ngrid))
                self._Gcart_grid = self._build_Gcart_grid()

    def _log(self, msg: str) -> None:
        """Emit a progress message when verbose logging is enabled."""
        if self.verbose:
            print(msg, flush=True)

    def _infer_kpoint_weight(self) -> float:
        """Infer the per-k-point weight from KPOINTS or fall back to 1/nk."""
        try:
            lines = [line.strip() for line in self.kpoints_path.read_text().splitlines()]
        except OSError:
            return 1.0 / max(self._nkpts, 1)
        if len(lines) >= 4 and lines[1] == "0":
            try:
                mesh = [int(x) for x in lines[3].split()[:3]]
            except ValueError:
                mesh = []
            if len(mesh) == 3 and all(n > 0 for n in mesh):
                return 1.0 / float(np.prod(mesh))
        return 1.0 / max(self._nkpts, 1)

    def orthogonalize_pair_subspaces(self) -> None:
        """Orthogonalize each irreducible-k band manifold in the active overlap metric."""
        self._wfc_cache.clear()
        self._grid_wfc_cache.clear()
        self._matched_wfc_cache.clear()
        self._grid_matched_wfc_cache.clear()
        self._matched_coeff_grid_cache.clear()
        self._matched_proj_cache.clear()
        self._orth_coeff_cache.clear()
        self._orth_gvec_cache.clear()
        self._orth_ready = False
        band_list = list(range(1, self._nbands + 1))
        if len(band_list) <= 1:
            self._orth_ready = True
            return
        self._log("Orthogonalizing full NBANDS manifolds for all spin/k blocks ...")
        for ispin in range(1, self._nspin + 1):
            for ik in range(1, self._nkpts + 1):
                target_kvec = np.asarray(self._ir_kvecs[ik - 1], dtype=float)
                target_gvecs = np.asarray(
                    self.wfc.gvectors(
                        ikpt=ik,
                        check_consistency=False,
                        kvec=target_kvec,
                    ),
                    dtype=int,
                )
                raw_gvecs = np.asarray(
                    self.wfc.gvectors(ikpt=ik, check_consistency=False),
                    dtype=int,
                )
                coeff_cols = []
                for iband in band_list:
                    raw_coeff = self._raw_band_coeff(ispin, ik, iband)
                    source_map = {
                        tuple(int(x) for x in gvec): raw_coeff[idx]
                        for idx, gvec in enumerate(raw_gvecs)
                    }
                    coeff_cols.append(
                        np.asarray(
                            [
                                source_map.get(tuple(int(x) for x in gvec), 0.0j)
                                for gvec in target_gvecs
                            ],
                            dtype=np.complex128,
                        )
                    )
                coeff_mat = np.column_stack(coeff_cols)
                overlap_mat = coeff_mat.conj().T @ coeff_mat
                if self.mode == "paw_orth_only" and self._paw is not None:
                    proj_cols = [self._paw.get_proj(ispin, ik, iband, coeff_cols[idx]) for idx, iband in enumerate(band_list)]
                    overlap_mat += self._projector_overlap_block(proj_cols)
                overlap_mat = 0.5 * (overlap_mat + overlap_mat.conj().T)
                transform = self._inverse_cholesky_upper(overlap_mat)
                coeff_orth = coeff_mat @ transform
                self._orth_gvec_cache[ik] = target_gvecs
                for col_idx, iband in enumerate(band_list):
                    self._orth_coeff_cache[(ispin, ik, iband)] = np.asarray(coeff_orth[:, col_idx], dtype=np.complex128)
        self._orth_ready = True

    def _projector_overlap_block(self, proj_cols: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
        """Assemble the PAW overlap correction matrix for one band block."""
        if self._paw is None:
            raise RuntimeError("Projector overlap requested without PAW helper")
        nbands = len(proj_cols)
        overlap = np.zeros((nbands, nbands), dtype=np.complex128)
        for iatom in range(self._paw.natoms):
            ntype = self._paw.element_idx[iatom]
            qij = self._paw._Qij_per_type[ntype]
            beta_mat = np.column_stack([proj_cols[idx][iatom] for idx in range(nbands)])
            overlap += beta_mat.conj().T @ qij @ beta_mat
        return overlap

    @staticmethod
    def _inverse_cholesky_upper(overlap_mat: np.ndarray) -> np.ndarray:
        """Return the inverse upper-triangular Cholesky factor of an overlap matrix."""
        try:
            lower = np.linalg.cholesky(overlap_mat)
        except np.linalg.LinAlgError:
            jitter = 1e-10 * max(1.0, float(np.max(np.abs(np.diag(overlap_mat)))))
            lower = np.linalg.cholesky(overlap_mat + jitter * np.eye(overlap_mat.shape[0], dtype=overlap_mat.dtype))
        return np.linalg.inv(lower.conj().T)

    def _ensure_orthogonalized_wavefunctions(self) -> None:
        """Run the orthogonalization pass once before matrix assembly."""
        if not self._orth_ready:
            self.orthogonalize_pair_subspaces()

    def _load_symmetry_ops(self) -> List[SymmetryOp]:
        """Load symmetry operators from the nearest available OUTCAR file."""
        for dpath in self._search_dirs:
            for name in ("OUTCAR.symm", "OUTCAR"):
                candidate = dpath / name
                if not candidate.is_file():
                    continue
                ops = _parse_outcar_symmetry_ops(candidate, self.Acell)
                if ops:
                    return ops
        ident = np.eye(3, dtype=int)
        return [SymmetryOp(irot=1, real_matrix=ident, reciprocal_matrix=ident, tau_frac=np.zeros(3, dtype=float))]

    def _store_expanded_match(self, k_frac: np.ndarray, match: KpointMatch) -> None:
        """Store the best irreducible-k match for an expanded k-point."""
        key = _frac_key(k_frac)
        current = self._expanded_kpoint_lookup.get(key)
        if current is None or _kpoint_match_priority(match) < _kpoint_match_priority(current):
            self._expanded_kpoint_lookup[key] = match

    def _store_full_bz_match(self, k_frac: np.ndarray, match: KpointMatch) -> None:
        """Store the preferred full-BZ match for a wrapped fractional point."""
        key = _frac_key(k_frac)
        current = self._full_bz_lookup.get(key)
        if current is None or _kpoint_match_priority(match) < _kpoint_match_priority(current):
            self._full_bz_lookup[key] = match

    def _store_full_bz_index(self, k_frac: np.ndarray, full_index: int) -> None:
        """Remember the VASP full-BZ loop index for one k-point."""
        self._full_bz_index_lookup[_frac_key(k_frac)] = int(full_index)

    def _full_bz_index(self, k_frac: Sequence[float]) -> Optional[int]:
        """Look up the VASP full-BZ loop index of a wrapped k-point.

        Finite-q examples can mix truncated decimal coordinates from OUTCAR with
        exact fractional arithmetic from `q_ext`, so the exact hash lookup needs
        a tolerance-based fallback.
        """
        key = _frac_key(k_frac)
        direct = self._full_bz_index_lookup.get(key)
        if direct is not None:
            return direct
        target = _wrap_frac(np.asarray(k_frac, dtype=float))
        for full_k in self.full_kpoints:
            diff = _wrap_frac_signed(np.asarray(full_k.k_frac, dtype=float) - target)
            if float(np.max(np.abs(diff))) <= 5e-5:
                return int(full_k.full_index)
        return None

    def _build_expanded_kpoint_lookup(self) -> None:
        """Expand irreducible k-points through symmetry and time reversal."""
        for ik, kvec in enumerate(np.asarray(self._ir_kvecs, dtype=float), start=1):
            for isym, op in enumerate(self.symm_ops):
                k_symm = _wrap_frac(op.reciprocal_matrix @ kvec)
                direct_match = KpointMatch(ikpt=ik, time_reversed=False, symm_op=isym)
                self._store_expanded_match(k_symm, direct_match)
                self._expanded_kpoint_coords.append(k_symm)
                self._expanded_kpoint_matches.append(direct_match)
                k_symm_tr = _wrap_frac(-k_symm)
                tr_match = KpointMatch(ikpt=ik, time_reversed=True, symm_op=isym)
                self._store_expanded_match(k_symm_tr, tr_match)
                self._expanded_kpoint_coords.append(k_symm_tr)
                self._expanded_kpoint_matches.append(tr_match)
        self._expanded_kpoint_coords_arr = np.asarray(self._expanded_kpoint_coords, dtype=float)

    def _resolve_symmetry_match(self, target_k: np.ndarray, ikpt: int, time_reversed: bool = False, tol: float = 5e-5) -> Optional[KpointMatch]:
        """Resolve which symmetry operator maps one irreducible point to target_k."""
        base_k = np.asarray(self._ir_kvecs[ikpt - 1], dtype=float)
        if time_reversed:
            base_k = -base_k
        target_arr = _wrap_frac(np.asarray(target_k, dtype=float))
        candidates: List[Tuple[float, KpointMatch]] = []
        for isym, op in enumerate(self.symm_ops):
            trial = _wrap_frac(op.reciprocal_matrix @ base_k)
            diff = _wrap_frac_signed(trial - target_arr)
            max_abs = float(np.max(np.abs(diff)))
            if max_abs <= tol:
                candidates.append((max_abs, KpointMatch(ikpt=ikpt, time_reversed=time_reversed, symm_op=isym)))
        if not candidates:
            return None
        best = min(item[0] for item in candidates)
        tied = [item for item in candidates if item[0] <= best + 1e-8]
        tied.sort(key=lambda item: _kpoint_match_priority(item[1]))
        return tied[0][1]

    def _best_kpoint_match(self, k_frac: np.ndarray, coords: np.ndarray, matches: Sequence[KpointMatch], tol: float = 5e-5) -> Optional[KpointMatch]:
        """Pick the nearest stored k-point match subject to the VASP-style priority order."""
        if coords.size == 0:
            return None
        diff = _wrap_frac_signed(coords - k_frac[np.newaxis, :])
        max_abs = np.max(np.abs(diff), axis=1)
        best = float(np.min(max_abs))
        if best > tol:
            return None
        candidates = np.where(max_abs <= best + 1e-8)[0]
        idx = min(candidates, key=lambda item: _kpoint_match_priority(matches[int(item)]))
        return matches[int(idx)]

    def _load_full_bz_kpoints(self) -> List[FullBZKpoint]:
        """Build the full-BZ k-point loop used by VASP BSE storage conventions."""
        entries: List[FullBZKpoint] = []
        seen: set[Tuple[float, ...]] = set()
        for dpath in self._search_dirs:
            for name in ("OUTCAR.symm", "OUTCAR"):
                candidate = dpath / name
                if not candidate.is_file():
                    continue
                parsed = _parse_outcar_full_bz_kpoints(candidate)
                if not parsed:
                    continue
                resolved_rows: List[Tuple[np.ndarray, KpointMatch]] = []
                seen_keys: set[Tuple[float, ...]] = set()
                for k_frac, ikpt, time_reversed in parsed:
                    match = self._resolve_symmetry_match(k_frac, ikpt, time_reversed)
                    if match is None:
                        match = self._best_kpoint_match(_wrap_frac(np.asarray(k_frac, dtype=float)), self._expanded_kpoint_coords_arr, self._expanded_kpoint_matches)
                    if match is None:
                        continue
                    key = _frac_key(k_frac)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    resolved_rows.append((_wrap_frac(k_frac), match))
                labeled_rows = list(parsed)
                matching_sections = [section for section in _parse_outcar_ibzkpt_sections(candidate) if len(section) == len(parsed)]
                if matching_sections:
                    target_order = matching_sections[-1]
                    direct_rows = [row for row in parsed if not row[2]]
                    time_reversed_by_key = {_frac_key(k_frac): row for row in parsed for k_frac in [row[0]] if row[2]}
                    labeled_rows = list(direct_rows)
                    for k_frac in target_order:
                        entry = time_reversed_by_key.pop(_frac_key(k_frac), None)
                        if entry is not None:
                            labeled_rows.append(entry)
                    if time_reversed_by_key:
                        labeled_rows.extend(row for row in parsed if row[2] and _frac_key(row[0]) in time_reversed_by_key)
                label_index_by_key = {_frac_key(k_frac): full_index for full_index, (k_frac, _, _) in enumerate(labeled_rows, start=1)}
                match_by_key = {_frac_key(k_frac): match for k_frac, match in resolved_rows}
                ordered_rows = [(label_index_by_key[_frac_key(k_frac)], _wrap_frac(k_frac), match_by_key[_frac_key(k_frac)]) for k_frac, _, _ in labeled_rows if _frac_key(k_frac) in match_by_key]
                positions_by_ir: Dict[int, Dict[bool, int]] = {}
                for pos, (_, _, match) in enumerate(ordered_rows):
                    ir_positions = positions_by_ir.setdefault(match.ikpt, {})
                    ir_positions[match.time_reversed] = pos
                for ir_positions in positions_by_ir.values():
                    pos_direct = ir_positions.get(False)
                    pos_tr = ir_positions.get(True)
                    if pos_direct is not None and pos_tr is not None and pos_direct > pos_tr:
                        ordered_rows[pos_direct], ordered_rows[pos_tr] = ordered_rows[pos_tr], ordered_rows[pos_direct]
                for full_index, k_frac, match in ordered_rows:
                    key = _frac_key(k_frac)
                    if key in seen:
                        continue
                    seen.add(key)
                    self._store_full_bz_match(k_frac, match)
                    self._store_full_bz_index(k_frac, full_index)
                    self._full_bz_kpoint_coords.append(_wrap_frac(k_frac))
                    self._full_bz_kpoint_matches.append(match)
                    entries.append(FullBZKpoint(full_index=full_index, k_frac=tuple(float(x) for x in _wrap_frac(k_frac).tolist()), match=match))
        if not entries:
            for full_index, (k_key, match) in enumerate(sorted(self._expanded_kpoint_lookup.items()), start=1):
                k_frac = np.asarray(k_key, dtype=float)
                entries.append(FullBZKpoint(full_index=full_index, k_frac=k_key, match=match))
                self._store_full_bz_match(k_frac, match)
                self._store_full_bz_index(k_frac, full_index)
                self._full_bz_kpoint_coords.append(k_frac)
                self._full_bz_kpoint_matches.append(match)
        self._full_bz_kpoint_coords_arr = np.asarray(self._full_bz_kpoint_coords, dtype=float)
        return entries

    def _match_full_kpoint(self, k_frac: np.ndarray) -> Optional[KpointMatch]:
        """Match a wrapped full-BZ point to the stored irreducible-k representation."""
        k_arr = _wrap_frac(np.asarray(k_frac, dtype=float))
        cache_key = _frac_key(k_arr)
        cached = self._resolved_kpoint_cache.get(cache_key)
        if cached is not None:
            return cached
        direct = self._full_bz_lookup.get(cache_key)
        if direct is not None:
            self._resolved_kpoint_cache[cache_key] = direct
            return direct
        full_match = self._best_kpoint_match(k_arr, self._full_bz_kpoint_coords_arr, self._full_bz_kpoint_matches)
        if full_match is not None:
            self._resolved_kpoint_cache[cache_key] = full_match
            return full_match
        expanded = self._expanded_kpoint_lookup.get(cache_key)
        if expanded is not None:
            self._resolved_kpoint_cache[cache_key] = expanded
            return expanded
        expanded_match = self._best_kpoint_match(k_arr, self._expanded_kpoint_coords_arr, self._expanded_kpoint_matches)
        if expanded_match is not None:
            self._resolved_kpoint_cache[cache_key] = expanded_match
            return expanded_match
        return None

    def _raw_band_coeff(self, ispin: int, ik: int, iband: int) -> np.ndarray:
        """Read one raw WAVECAR coefficient vector and normalize its shape."""
        coeff = self.wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=iband, norm=False)
        if np.iscomplexobj(coeff) and np.ndim(coeff) == 2:
            coeff = coeff[:, 0]
        coeff_arr = np.asarray(coeff, dtype=np.complex128)
        gvecs = self.wfc.gvectors(ikpt=ik, check_consistency=False)
        if coeff_arr.shape[0] != gvecs.shape[0]:
            raise ValueError(f"Coefficient/G-vector size mismatch for {(ispin, ik, iband)}")
        return coeff_arr

    def _matched_gvecs_and_coeffs(self, ispin: int, ik: int, iband: int, *, time_reversed: bool, symm_op: int, k_frac: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Remap one state onto the G-vector basis of a requested full-BZ k-point."""
        coeff = self._orth_coeff_cache.get((ispin, ik, iband))
        if coeff is None:
            coeff = self._raw_band_coeff(ispin, ik, iband)
            source_gvecs_base = np.asarray(self.wfc.gvectors(ikpt=ik, check_consistency=False), dtype=int)
            source_rep_k = np.asarray(self.wfc._kvecs[ik - 1], dtype=float)
        else:
            source_gvecs_base = self._orth_gvec_cache.get(ik)
            if source_gvecs_base is None:
                raise RuntimeError(f"Missing orthogonalized G-vector basis for ik={ik}")
            source_rep_k = np.asarray(self._ir_kvecs[ik - 1], dtype=float)
        coeff_arr = np.asarray(coeff, dtype=np.complex128).copy()
        if symm_op == 0:
            target_k = _wrap_frac_signed(np.asarray(k_frac, dtype=float))
            target_gvecs = np.asarray(
                self.wfc.gvectors(
                    ikpt=ik,
                    check_consistency=False,
                    kvec=target_k,
                ),
                dtype=int,
            )
            source_gvecs = -source_gvecs_base if time_reversed else source_gvecs_base
            source_coeffs = np.conjugate(coeff_arr) if time_reversed else coeff_arr
            source_map = {tuple(int(x) for x in gvec): source_coeffs[idx] for idx, gvec in enumerate(source_gvecs)}
            remapped_coeffs = np.asarray([source_map.get(tuple(int(x) for x in gvec), 0.0j) for gvec in target_gvecs], dtype=np.complex128)
            return target_gvecs, remapped_coeffs
        corrected_rep_k = np.asarray(self._ir_kvecs[ik - 1], dtype=float)
        gvecs = source_gvecs_base.astype(float)
        q_frac = gvecs + source_rep_k[np.newaxis, :]
        if time_reversed:
            q_frac = -q_frac
            coeff_arr = np.conjugate(coeff_arr)
        if symm_op:
            op = self.symm_ops[symm_op]
            q_frac = q_frac @ op.reciprocal_matrix.T
            phase = np.exp(-2j * np.pi * np.sum(q_frac * op.tau_frac[np.newaxis, :], axis=1))
            coeff_arr *= phase
        target_k = _wrap_frac_signed(np.asarray(k_frac, dtype=float))
        gvec_target = q_frac - target_k[np.newaxis, :]
        gvec_target_i = np.rint(gvec_target).astype(int)
        mismatch = float(np.max(np.abs(gvec_target - gvec_target_i))) if gvec_target.size else 0.0
        if mismatch > 5e-5:
            raise ValueError(f"Failed to recover integer G vectors for {(ispin, ik, iband)} at full-BZ k={target_k.tolist()}")
        return gvec_target_i, coeff_arr

    def _coeff_on_grid(self, ispin: int, ik: int, iband: int, *, time_reversed: bool = False, symm_op: int = 0, k_frac: Optional[Sequence[float]] = None, grid_shape: Optional[Sequence[int]] = None) -> np.ndarray:
        """Scatter plane-wave coefficients onto an FFT grid."""
        target_grid = self.ngrid if grid_shape is None else tuple(int(n) for n in grid_shape)
        if k_frac is None:
            gvecs = self.wfc.gvectors(ikpt=ik, check_consistency=False)
            coeff = self._orth_coeff_cache.get((ispin, ik, iband))
            if coeff is None:
                coeff = self._raw_band_coeff(ispin, ik, iband)
            cache_key = None
        else:
            frac_key = _frac_key(k_frac)
            cache_key = (ispin, ik, iband, time_reversed, symm_op, frac_key)
            if target_grid == self.ngrid:
                cached = self._matched_coeff_grid_cache.get(cache_key)
                if cached is not None:
                    return cached
            gvecs, coeff = self._matched_gvecs_and_coeffs(ispin, ik, iband, time_reversed=time_reversed, symm_op=symm_op, k_frac=k_frac)
        nx, ny, nz = target_grid
        ix = gvecs[:, 0] % nx
        iy = gvecs[:, 1] % ny
        iz = gvecs[:, 2] % nz
        c_grid = np.zeros(target_grid, dtype=np.complex128)
        np.add.at(c_grid, (ix, iy, iz), coeff)
        if cache_key is not None and target_grid == self.ngrid:
            self._matched_coeff_grid_cache[cache_key] = c_grid
        return c_grid

    def _get_periodic_wfc(self, ispin: int, ik: int, iband: int, *, time_reversed: bool = False, symm_op: int = 0, k_frac: Optional[Sequence[float]] = None, grid_shape: Optional[Sequence[int]] = None) -> np.ndarray:
        """Reconstruct the cell-periodic wavefunction on the requested FFT grid."""
        target_grid = self.ngrid if grid_shape is None else tuple(int(n) for n in grid_shape)
        if k_frac is None:
            if target_grid == self.ngrid:
                key = (ispin, ik, iband)
                cached = self._wfc_cache.get(key)
                if cached is not None:
                    return cached
            else:
                key = (target_grid, ispin, ik, iband)
                cached = self._grid_wfc_cache.get(key)
                if cached is not None:
                    return cached
            c_on_grid = self._coeff_on_grid(ispin, ik, iband, grid_shape=target_grid)
        else:
            frac_key = _frac_key(k_frac)
            if target_grid == self.ngrid:
                key = (ispin, ik, iband, time_reversed, symm_op, frac_key)
                cached = self._matched_wfc_cache.get(key)
            else:
                key = (target_grid, ispin, ik, iband, time_reversed, symm_op, frac_key)
                cached = self._grid_matched_wfc_cache.get(key)
            if cached is not None:
                return cached
            c_on_grid = self._coeff_on_grid(ispin, ik, iband, time_reversed=time_reversed, symm_op=symm_op, k_frac=k_frac, grid_shape=target_grid)
        ngrid_pts = int(np.prod(target_grid))
        if self.wfc_ifft_scale == "sqrtN":
            scale = np.sqrt(ngrid_pts)
        elif self.wfc_ifft_scale == "N":
            scale = float(ngrid_pts)
        else:
            scale = 1.0
        u = ifftn(c_on_grid) * scale
        if k_frac is None and target_grid == self.ngrid:
            self._wfc_cache[key] = u
        elif k_frac is None:
            self._grid_wfc_cache[key] = u
        elif target_grid == self.ngrid:
            self._matched_wfc_cache[key] = u
        else:
            self._grid_matched_wfc_cache[key] = u
        return u

    def _get_state_projectors(
        self,
        ispin: int,
        ik: int,
        iband: int,
        *,
        time_reversed: bool = False,
        symm_op: int = 0,
        k_frac: Optional[Sequence[float]] = None,
    ) -> List[np.ndarray]:
        """Project one state onto PAW projectors in either IR or full-BZ form."""
        if self._paw is None:
            raise RuntimeError("Projector overlaps requested without PAW helper")
        if k_frac is None:
            coeff = self._orth_coeff_cache.get((ispin, ik, iband))
            if coeff is None:
                coeff = self._raw_band_coeff(ispin, ik, iband)
            return self._paw.get_proj(ispin, ik, iband, coeff)
        frac_key = _frac_key(k_frac)
        cache_key = (ispin, ik, iband, time_reversed, symm_op, frac_key)
        cached = self._matched_proj_cache.get(cache_key)
        if cached is not None:
            return cached
        if time_reversed and symm_op == 0:
            raw_blocks = self._get_state_projectors(ispin, ik, iband, k_frac=None)
            projected = [np.conj(np.asarray(block, dtype=np.complex128)) for block in raw_blocks]
            self._matched_proj_cache[cache_key] = projected
            return projected
        gvecs, coeff = self._matched_gvecs_and_coeffs(
            ispin,
            ik,
            iband,
            time_reversed=time_reversed,
            symm_op=symm_op,
            k_frac=k_frac,
        )
        projected = self._paw.project_coefficients(coeff, gvecs, k_frac)
        self._matched_proj_cache[cache_key] = projected
        return projected

    def _effective_source_density_G(
        self,
        rho_G: np.ndarray,
        q_frac: np.ndarray,
        beta_src_left: Optional[List[np.ndarray]] = None,
        beta_src_right: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Return the source density used by the Coulomb contraction."""
        if self.mode == "paw_full" and self._paw is not None:
            return self._paw.effective_source_density_G(
                rho_G,
                q_frac,
                beta_src_left=beta_src_left,
                beta_src_right=beta_src_right,
            )
        return np.asarray(rho_G, dtype=np.complex128)

    def _pair_density_G(self, u_left: np.ndarray, u_right: np.ndarray) -> np.ndarray:
        """Build the pair density rho(G) = FFT[u_left^* u_right]."""
        return fftn(np.conj(u_left) * u_right)

    def _build_Gcart_grid(self) -> np.ndarray:
        """Construct the Cartesian G-grid corresponding to the FFT mesh."""
        nx, ny, nz = self.ngrid
        gx = np.fft.fftfreq(nx) * nx
        gy = np.fft.fftfreq(ny) * ny
        gz = np.fft.fftfreq(nz) * nz
        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
        return np.stack([GX, GY, GZ], axis=-1) @ (TPI * self.Bcell)

    def _coulomb_kernel(self, q_frac: np.ndarray) -> np.ndarray:
        """Return the bare Coulomb kernel 4pi/|q+G|^2 on the FFT grid."""
        q_arr = np.asarray(q_frac, dtype=float)
        q_cart = q_arr @ (TPI * self.Bcell)
        qpG_cart = self._Gcart_grid + q_cart[np.newaxis, np.newaxis, np.newaxis, :]
        qpGsq = np.sum(qpG_cart**2, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(qpGsq > 0.0, EDEPS / qpGsq, 0.0)

    def _screened_kernel(self, q_frac: np.ndarray) -> np.ndarray:
        """Return the diagonal screened kernel, falling back to bare Coulomb."""
        cache_key = _wrap_frac_signed_key(q_frac)
        cached = self._screened_kernel_cache.get(cache_key)
        if cached is not None:
            return cached
        kernel = self._screened_kernel_from_dump(q_frac)
        if kernel is not None:
            self._screened_kernel_cache[cache_key] = kernel
            return kernel
        kernel = self._coulomb_kernel(q_frac)
        if self.epsilon is not None:
            kernel = kernel / self.epsilon
        self._screened_kernel_cache[cache_key] = kernel
        return kernel

    def _response_gvectors_for_kfrac(self, k_frac: Sequence[float]) -> np.ndarray:
        """Generate response-basis G vectors at a given k-point using ENCUTGW."""
        original_encut = float(self.wfc._encut)
        try:
            self.wfc._encut = float(self.encutgw)
            return np.asarray(self.wfc.gvectors(kvec=np.asarray(k_frac, dtype=float), check_consistency=False), dtype=int)
        finally:
            self.wfc._encut = original_encut

    def _response_basis_values(self, values_G: np.ndarray, gvecs: np.ndarray) -> np.ndarray:
        """Sample a full-grid G object on the selected response-basis vectors."""
        nx, ny, nz = self.ngrid
        values_arr = np.asarray(values_G, dtype=np.complex128)
        return np.asarray(values_arr[gvecs[:, 0] % nx, gvecs[:, 1] % ny, gvecs[:, 2] % nz], dtype=np.complex128)

    def _response_basis_gvectors_for_q(self, q_frac: np.ndarray) -> Optional[np.ndarray]:
        """Return response-basis G vectors for one momentum transfer."""
        if self.encutgw is None:
            return None
        screened_operator = self._screened_operator_on_response_basis(q_frac)
        if screened_operator is not None:
            return np.asarray(screened_operator[0], dtype=int)
        q_target = _wrap_frac_signed(np.asarray(q_frac, dtype=float))
        return np.asarray(self._response_gvectors_for_kfrac(q_target), dtype=int)

    def _screened_operator_on_response_basis(self, q_frac: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load the full screened interaction matrix on the response basis for q."""
        if self.encutgw is None:
            return None
        cache_key = _wrap_frac_signed_key(q_frac)
        cached = self._screened_operator_cache.get(cache_key)
        if cached is not None:
            return cached
        q_target = _wrap_frac_signed(np.asarray(q_frac, dtype=float))
        match = self._match_full_kpoint(_wrap_frac(q_target))
        if match is None:
            return None
        screen_path = find_screened_potential_file(self._search_dirs, match.ikpt)
        if screen_path is None:
            return None
        screen_data = self._screened_matrix_cache.get(match.ikpt)
        if screen_data is None:
            screen_data = read_screened_potential(screen_path)
            self._screened_matrix_cache[match.ikpt] = screen_data
            self._screened_diag_cache[match.ikpt] = np.diag(screen_data.matrix).copy()
        source_gvecs = self._response_gvec_cache.get(match.ikpt)
        if source_gvecs is None:
            source_gvecs = self._response_gvectors(match.ikpt)
            self._response_gvec_cache[match.ikpt] = source_gvecs
        target_gvecs = self._response_gvectors_for_kfrac(q_target)
        if target_gvecs.shape[0] != source_gvecs.shape[0] or screen_data.ngvector != source_gvecs.shape[0]:
            return None
        transformed = np.asarray(source_gvecs, dtype=int)
        if match.time_reversed:
            transformed = -transformed
        if match.symm_op:
            op = self.symm_ops[match.symm_op]
            transformed = np.rint(transformed.astype(float) @ op.reciprocal_matrix.T).astype(int)
        index_map = {tuple(int(x) for x in gvec): idx for idx, gvec in enumerate(transformed)}
        try:
            perm = np.asarray([index_map[tuple(int(x) for x in gvec)] for gvec in target_gvecs], dtype=np.int32)
        except KeyError:
            return None
        matrix = np.asarray(screen_data.matrix[np.ix_(perm, perm)], dtype=np.complex128)
        if match.time_reversed:
            matrix = np.conjugate(matrix)
        matrix = matrix / float(self.Nfft)
        result = (target_gvecs, matrix)
        self._screened_operator_cache[cache_key] = result
        return result

    def _screened_kernel_from_dump(self, q_frac: np.ndarray) -> Optional[np.ndarray]:
        """Load the diagonal screened kernel from WFULL/W on the FFT mesh."""
        if self.encutgw is None:
            return None
        match = self._match_full_kpoint(_wrap_frac(q_frac))
        if match is None:
            return None
        screen_path = find_screened_potential_file(self._search_dirs, match.ikpt)
        if screen_path is None:
            return None
        diag = self._screened_diag_cache.get(match.ikpt)
        if diag is None:
            diag = read_screened_potential_diag(screen_path)
            self._screened_diag_cache[match.ikpt] = diag
        gvecs = self._response_gvec_cache.get(match.ikpt)
        if gvecs is None:
            gvecs = self._response_gvectors(match.ikpt)
            self._response_gvec_cache[match.ikpt] = gvecs
        if diag.shape[0] != gvecs.shape[0]:
            raise ValueError(f"Screened-kernel size mismatch for q index {match.ikpt}: {diag.shape[0]} != {gvecs.shape[0]}")
        values = np.asarray(diag, dtype=np.complex128) / float(self.Nfft)
        g_target = np.asarray(gvecs, dtype=float)
        if match.time_reversed:
            values = np.conjugate(values)
            g_target = -g_target
        if match.symm_op:
            op = self.symm_ops[match.symm_op]
            g_target = g_target @ op.reciprocal_matrix.T
        g_target_i = np.rint(g_target).astype(int)
        mismatch = float(np.max(np.abs(g_target - g_target_i))) if g_target.size else 0.0
        if mismatch > 5e-5:
            raise ValueError(f"Failed to rotate screened G vectors for q index {match.ikpt}")
        nx, ny, nz = self.ngrid
        kernel = np.zeros(self.ngrid, dtype=np.complex128)
        np.add.at(kernel, (g_target_i[:, 0] % nx, g_target_i[:, 1] % ny, g_target_i[:, 2] % nz), values)
        return kernel

    def _response_gvectors(self, ikpt: int) -> np.ndarray:
        """Return the response-basis G vectors for one irreducible q index."""
        original_encut = float(self.wfc._encut)
        try:
            self.wfc._encut = float(self.encutgw)
            return np.asarray(self.wfc.gvectors(kvec=np.asarray(self._ir_kvecs[ikpt - 1], dtype=float), check_consistency=False), dtype=int)
        finally:
            self.wfc._encut = original_encut

    @staticmethod
    def _setphase_integer_shift(setphase_shift_frac: np.ndarray, tol: float = 1e-6) -> Tuple[int, int, int]:
        """Convert a fractional SETPHASE shift into exact integer grid offsets."""
        shift = np.asarray(setphase_shift_frac, dtype=float)
        rounded = np.rint(shift)
        if np.max(np.abs(shift - rounded)) > tol:
            raise ValueError(f"SETPHASE shift is not integer within tolerance: {shift.tolist()}")
        return int(rounded[0]), int(rounded[1]), int(rounded[2])

    def _get_setphase_grid(self, shift_int: Tuple[int, int, int]) -> np.ndarray:
        """Build and cache the real-space phase factor for a SETPHASE shift."""
        cached = self._setphase_grid_cache.get(shift_int)
        if cached is not None:
            return cached
        sx, sy, sz = shift_int
        nx, ny, nz = self.ngrid
        x = np.arange(nx, dtype=np.float64)[:, None, None]
        y = np.arange(ny, dtype=np.float64)[None, :, None]
        z = np.arange(nz, dtype=np.float64)[None, None, :]
        phase = np.exp(1j * 2.0 * np.pi * ((sx * x / float(nx)) + (sy * y / float(ny)) + (sz * z / float(nz))))
        self._setphase_grid_cache[shift_int] = phase
        return phase

    def _apply_setphase_to_G_object(self, values_G: np.ndarray, setphase_shift_frac: Optional[np.ndarray]) -> np.ndarray:
        """Apply VASP's SETPHASE convention to a reciprocal-space object."""
        values_arr = np.asarray(values_G, dtype=np.complex128)
        if setphase_shift_frac is None:
            return values_arr
        shift_int = self._setphase_integer_shift(setphase_shift_frac)
        if shift_int == (0, 0, 0):
            return values_arr
        phase = self._get_setphase_grid(shift_int)
        return np.asarray(fftn(ifftn(values_arr) * phase), dtype=np.complex128)

    def build_pair_indices(self) -> List[PairState]:
        """Enumerate the electron-hole pair basis used for the BSE matrix."""
        self.pairs = []
        for ispin in range(1, self._nspin + 1):
            occs = self.wfc._occs[ispin - 1]
            bands = self.wfc._bands[ispin - 1]
            for full_k in self.full_kpoints:
                k1_frac_wrapped = np.asarray(full_k.k_frac, dtype=float)
                k1_frac = _wrap_frac_signed(k1_frac_wrapped)
                ik1_full = full_k.full_index
                k1_match = full_k.match
                ik1_0 = k1_match.ikpt - 1
                ik1_ir = k1_match.ikpt
                k3_frac_wrapped = _wrap_frac(k1_frac_wrapped + self.q_ext)
                k3_frac = _wrap_frac_signed(k3_frac_wrapped)
                k3_match = self._match_full_kpoint(k3_frac_wrapped)
                if k3_match is None:
                    continue
                ik3_full = self._full_bz_index(k3_frac_wrapped)
                if ik3_full is None:
                    continue
                ik3_0 = k3_match.ikpt - 1
                ik3_ir = k3_match.ikpt
                occ1 = occs[ik1_0]
                occ3 = occs[ik3_0]
                eps1 = bands[ik1_0]
                eps3 = bands[ik3_0]
                val_indices = np.where(occ1 > self.occ_threshold)[0]
                cond_indices = np.where(occ3 < 1.0 - self.occ_threshold)[0]
                if len(val_indices) == 0 or len(cond_indices) == 0:
                    continue
                if self.vb_num is not None:
                    val_indices = val_indices[-self.vb_num:]
                if self.cb_num is not None:
                    cond_indices = cond_indices[:self.cb_num]
                for iv0 in val_indices:
                    for ic0 in cond_indices:
                        eps_v = float(eps1[iv0])
                        eps_c = float(eps3[ic0])
                        de = eps_c - eps_v
                        if de <= 0.0:
                            continue
                        if self.ewin is not None and (de < self.ewin[0] or de > self.ewin[1]):
                            continue
                        self.pairs.append(PairState(iv=iv0 + 1, ic=ic0 + 1, ik=ik1_full, ik3=ik3_full, ik_ir=ik1_ir, ik3_ir=ik3_ir, ispin=ispin, eps_v=eps_v, eps_c=eps_c, k1_frac=tuple(float(x) for x in k1_frac.tolist()), k3_frac=tuple(float(x) for x in k3_frac.tolist()), k1_time_reversed=k1_match.time_reversed, k3_time_reversed=k3_match.time_reversed, k1_symm_op=k1_match.symm_op, k3_symm_op=k3_match.symm_op))
        if not self.pairs:
            raise RuntimeError("No valid electron-hole pairs found")
        # Match VASP pair-basis ordering consistently for both q=0 and finite-q:
        # spin -> full-BZ source k -> full-BZ target k -> conduction band -> valence band.
        self.pairs.sort(key=lambda pair: (pair.ispin, pair.ik, pair.ik3, pair.ic, pair.iv))
        self._log(f"Number of BSE pair states: {len(self.pairs)}")
        return self.pairs

    def _full_k_loop_position(self, k_frac: Sequence[float]) -> int:
        """Return the position of a k-point in the VASP full-BZ loop ordering."""
        key = _frac_key(_wrap_frac(k_frac))
        for idx, full_k in enumerate(self.full_kpoints):
            if _frac_key(_wrap_frac(full_k.k_frac)) == key:
                return idx
        raise KeyError(f"Unknown full-BZ k-point: {tuple(k_frac)}")

    def _should_store_vasp_bse_entry(self, pair_i: PairState, pair_j: PairState) -> bool:
        """Apply VASP's half-storage rule for dense BSE matrix output."""
        if pair_j.ispin < pair_i.ispin:
            return False
        if pair_j.ispin > pair_i.ispin:
            return True
        return self._full_k_loop_position(pair_j.k1_frac) >= self._full_k_loop_position(pair_i.k1_frac)

    def build_bse_matrix(self, *, vasp_storage: bool) -> np.ndarray:
        """Assemble the excitonic Hamiltonian in the current pair basis."""
        self._ensure_orthogonalized_wavefunctions()
        npairs = len(self.pairs)
        A = np.zeros((npairs, npairs), dtype=np.complex128)
        for idx, pair in enumerate(self.pairs):
            A[idx, idx] = pair.excitation_energy
        pair_weight = self.kpoint_weight
        use_hartree = self.interaction in {"hartree", "both"}
        use_direct = self.interaction in {"direct", "both"}
        q_h = -self.q_ext
        v_h = self._coulomb_kernel(q_h) if use_hartree else None
        hartree_response_gvecs = None
        if use_hartree and self.use_response_basis:
            hartree_response_gvecs = self._response_basis_gvectors_for_q(q_h)
        self._log(f"Building full-grid BSE matrix (interaction={self.interaction}) ...")
        for i, pair_i in enumerate(self.pairs):
            u_v1 = self._get_periodic_wfc(pair_i.ispin, pair_i.ik_ir, pair_i.iv, time_reversed=pair_i.k1_time_reversed, symm_op=pair_i.k1_symm_op, k_frac=pair_i.k1_frac)
            u_c3 = self._get_periodic_wfc(pair_i.ispin, pair_i.ik3_ir, pair_i.ic, time_reversed=pair_i.k3_time_reversed, symm_op=pair_i.k3_symm_op, k_frac=pair_i.k3_frac)
            beta_v1 = None
            beta_c3 = None
            if self.mode == "paw_full" and self._paw is not None:
                beta_v1 = self._get_state_projectors(pair_i.ispin, pair_i.ik_ir, pair_i.iv, time_reversed=pair_i.k1_time_reversed, symm_op=pair_i.k1_symm_op, k_frac=pair_i.k1_frac)
                beta_c3 = self._get_state_projectors(pair_i.ispin, pair_i.ik3_ir, pair_i.ic, time_reversed=pair_i.k3_time_reversed, symm_op=pair_i.k3_symm_op, k_frac=pair_i.k3_frac)
            hartree_left = None
            hartree_left_resp = None
            if use_hartree:
                q13_raw = np.asarray(pair_i.k1_frac, dtype=float) - np.asarray(pair_i.k3_frac, dtype=float)
                shift13 = q13_raw - np.asarray(q_h, dtype=float)
                rho13 = self._effective_source_density_G(
                    self._pair_density_G(u_c3, u_v1),
                    q13_raw,
                    beta_v1,
                    beta_c3,
                )
                pot13 = self._apply_setphase_to_G_object(rho13 * (v_h / self.Omega), shift13)
                hartree_left = np.conj(pot13) / float(self.Nfft)
                hartree_left_resp = (
                    self._response_basis_values(hartree_left, hartree_response_gvecs)
                    if hartree_response_gvecs is not None
                    else None
                )
            for j, pair_j in enumerate(self.pairs):
                if vasp_storage and not self._should_store_vasp_bse_entry(pair_i, pair_j):
                    continue
                if pair_i.ispin != pair_j.ispin:
                    continue
                u_v2 = self._get_periodic_wfc(pair_j.ispin, pair_j.ik_ir, pair_j.iv, time_reversed=pair_j.k1_time_reversed, symm_op=pair_j.k1_symm_op, k_frac=pair_j.k1_frac)
                u_c4 = self._get_periodic_wfc(pair_j.ispin, pair_j.ik3_ir, pair_j.ic, time_reversed=pair_j.k3_time_reversed, symm_op=pair_j.k3_symm_op, k_frac=pair_j.k3_frac)
                beta_v2 = None
                beta_c4 = None
                if self.mode == "paw_full" and self._paw is not None:
                    beta_v2 = self._get_state_projectors(pair_j.ispin, pair_j.ik_ir, pair_j.iv, time_reversed=pair_j.k1_time_reversed, symm_op=pair_j.k1_symm_op, k_frac=pair_j.k1_frac)
                    beta_c4 = self._get_state_projectors(pair_j.ispin, pair_j.ik3_ir, pair_j.ic, time_reversed=pair_j.k3_time_reversed, symm_op=pair_j.k3_symm_op, k_frac=pair_j.k3_frac)
                if use_hartree:
                    q24_raw = np.asarray(pair_j.k1_frac, dtype=float) - np.asarray(pair_j.k3_frac, dtype=float)
                    shift24 = q24_raw - np.asarray(q_h, dtype=float)
                    rho24 = self._effective_source_density_G(
                        self._pair_density_G(u_c4, u_v2),
                        q24_raw,
                        beta_v2,
                        beta_c4,
                    )
                    hartree_right = np.conj(self._apply_setphase_to_G_object(rho24, shift24)) / float(self.Nfft)
                    if hartree_left_resp is not None:
                        hartree_right_resp = self._response_basis_values(hartree_right, hartree_response_gvecs)
                        kh = pair_weight * self.spin_factor * np.sum(np.conj(hartree_left_resp) * hartree_right_resp)
                    else:
                        kh = pair_weight * self.spin_factor * np.sum(np.conj(hartree_left) * hartree_right)
                    A[i, j] += kh
                if use_direct:
                    q12_raw = np.asarray(pair_i.k1_frac, dtype=float) - np.asarray(pair_j.k1_frac, dtype=float)
                    q34_raw = np.asarray(pair_i.k3_frac, dtype=float) - np.asarray(pair_j.k3_frac, dtype=float)
                    q_d = _wrap_frac_signed(q12_raw)
                    shift12 = q12_raw - q_d
                    shift34 = q34_raw - q_d
                    rho_vv_pw = self._pair_density_G(u_v2, u_v1)
                    rho_vv_12 = self._apply_setphase_to_G_object(
                        self._effective_source_density_G(
                            rho_vv_pw,
                            q12_raw,
                            beta_v1,
                            beta_v2,
                        ),
                        shift12,
                    )
                    rho_cc_pw = self._pair_density_G(u_c4, u_c3)
                    rho_cc_34 = np.conjugate(
                        self._apply_setphase_to_G_object(
                            self._effective_source_density_G(
                                rho_cc_pw,
                                q34_raw,
                                beta_c3,
                                beta_c4,
                            ),
                            shift34,
                        )
                    ) / float(self.Nfft)
                    screened_operator = self._screened_operator_on_response_basis(q_d) if self.use_response_basis else None
                    if screened_operator is not None:
                        response_gvecs, w_d_mat = screened_operator
                        rho_vv_resp = self._response_basis_values(rho_vv_12, response_gvecs)
                        rho_cc_resp = self._response_basis_values(rho_cc_34, response_gvecs)
                        # local_field.F applies the screened operator to the
                        # selected 12-side object with ZGEMM(...,'N',...) on the
                        # response-function matrix as stored in WFULL/W.tmp.
                        # After read_screened_potential() this means a direct
                        # matrix-vector product, not an extra transpose.
                        kd = -pair_weight * np.sum(rho_cc_resp * (w_d_mat @ rho_vv_resp))
                    else:
                        w_d = self._screened_kernel(q_d)
                        kd = -pair_weight * np.sum(rho_cc_34 * w_d * rho_vv_12)
                    A[i, j] += kd
        return A


def _parse_q_ext(values: list[float]) -> np.ndarray:
    """Parse the external exciton momentum from CLI input."""
    if len(values) != 3:
        raise ValueError("q_ext must have exactly three fractional components")
    return np.asarray(values, dtype=float)


def _pair_metadata_array(pairs: list[PairState]) -> np.ndarray:
    """Pack pair metadata into a structured array for NPZ output."""
    dtype = np.dtype(
        [
            ("iv", np.int32),
            ("ic", np.int32),
            ("ik", np.int32),
            ("ik3", np.int32),
            ("ik_ir", np.int32),
            ("ik3_ir", np.int32),
            ("ispin", np.int32),
            ("eps_v", np.float64),
            ("eps_c", np.float64),
            ("k1_frac", np.float64, (3,)),
            ("k3_frac", np.float64, (3,)),
            ("k1_time_reversed", np.bool_),
            ("k3_time_reversed", np.bool_),
            ("k1_symm_op", np.int32),
            ("k3_symm_op", np.int32),
        ]
    )
    table = np.empty(len(pairs), dtype=dtype)
    for idx, pair in enumerate(pairs):
        table[idx] = (
            pair.iv,
            pair.ic,
            pair.ik,
            pair.ik3,
            pair.ik_ir,
            pair.ik3_ir,
            pair.ispin,
            pair.eps_v,
            pair.eps_c,
            np.asarray(pair.k1_frac, dtype=float),
            np.asarray(pair.k3_frac, dtype=float),
            pair.k1_time_reversed,
            pair.k3_time_reversed,
            pair.k1_symm_op,
            pair.k3_symm_op,
        )
    return table


def _write_text_matrix(path: Path, matrix: np.ndarray, *, mode: str, q_ext: np.ndarray, vasp_storage: bool) -> None:
    """Write the assembled BSE matrix in a simple text format."""
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# AMAT exchange matrix\n")
        handle.write(f"# mode {mode}\n")
        handle.write("# implementation full_grid_only\n")
        handle.write("# response_basis TODO\n")
        handle.write(f"# shape {matrix.shape[0]} {matrix.shape[1]}\n")
        handle.write(f"# q_ext {q_ext[0]:.12f} {q_ext[1]:.12f} {q_ext[2]:.12f}\n")
        handle.write(f"# vasp_storage {int(vasp_storage)}\n")
        handle.write("# i j real imag\n")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if vasp_storage and value == 0.0j:
                    continue
                handle.write(f"{i + 1:6d} {j + 1:6d} {value.real: .16e} {value.imag: .16e}\n")


def _masked_vasp_storage_matrix(builder: HartreeFullGridBuilder, matrix: np.ndarray) -> np.ndarray:
    """Zero entries that VASP would omit in its half-stored dense layout."""
    masked = np.asarray(matrix, dtype=np.complex128).copy()
    for i, pair_i in enumerate(builder.pairs):
        for j, pair_j in enumerate(builder.pairs):
            if not builder._should_store_vasp_bse_entry(pair_i, pair_j):
                masked[i, j] = 0.0j
    return masked


def _diagonalize_bse_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize the Hermitian part of the assembled BSE matrix."""
    hermitian = 0.5 * (np.asarray(matrix, dtype=np.complex128) + np.asarray(matrix, dtype=np.complex128).conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    return np.asarray(eigenvalues, dtype=float), np.asarray(eigenvectors, dtype=np.complex128)


def _write_bsefatband(
    path: Path,
    *,
    pairs: Sequence[PairState],
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    kpoint_weight: float,
    nexciton: int,
) -> None:
    """Write a VASP-style BSEFATBAND file from eigenpairs."""
    ntrans = len(pairs)
    nwrite = min(int(nexciton), int(eigenvalues.shape[0]))
    ip_energies = np.sort(np.asarray([pair.excitation_energy for pair in pairs], dtype=float))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{ntrans:18d}{nwrite:18d}\n")
        for iexc in range(nwrite):
            handle.write(
                f"{iexc + 1:6d}BSE eigenvalue{float(eigenvalues[iexc]):14.8f}"
                f"      IP-eigenvalue:{float(ip_energies[iexc]):14.8f}\n"
            )
            vec = np.asarray(eigenvectors[:, iexc], dtype=np.complex128)
            for row_idx, pair in enumerate(pairs):
                amp = complex(vec[row_idx])
                column_weight = abs(amp) / float(kpoint_weight)
                kx, ky, kz = [float(x) for x in pair.k1_frac]
                handle.write(
                    f"{kx:9.5f}{ky:9.5f}{kz:9.5f}"
                    f"{pair.eps_v:14.7f}{pair.eps_c:14.7f}{column_weight:14.7f}"
                    f"{pair.iv:6d}{pair.ic:6d}{amp.real:14.6f}+i* {amp.imag:12.6f}\n"
                )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Build the command-line interface for the standalone script."""
    parser = argparse.ArgumentParser(
        description="Build Hartree/direct BSE matrix elements using the local full-grid path.",
        formatter_class=_DefaultsFormatter,
    )
    parser.add_argument("--wavecar", default="WAVECAR", help="Path to WAVECAR")
    parser.add_argument("--outcar", default="OUTCAR", help="Path to OUTCAR")
    parser.add_argument("--kpoints", default="KPOINTS", help="Path to KPOINTS")
    parser.add_argument("--poscar", default="POSCAR", help="Path to POSCAR")
    parser.add_argument("--potcar", default="POTCAR", help="Path to POTCAR")
    parser.add_argument("--mode", choices=["pw_only", "paw_orth_only", "paw_full"], default="paw_orth_only", help="Matrix construction mode")
    parser.add_argument("--interaction", choices=["hartree", "direct", "both"], default="hartree", help="Which interaction term to assemble")
    parser.add_argument("--q-ext", nargs=3, type=float, metavar=("QX", "QY", "QZ"), default=[0.0, 0.0, 0.0], help="External exciton momentum in fractional reciprocal coordinates")
    parser.add_argument("--vb-num", type=int, required=True, help="Number of valence bands")
    parser.add_argument("--cb-num", type=int, required=True, help="Number of conduction bands")
    parser.add_argument("--ewin", nargs=2, type=float, metavar=("EMIN", "EMAX"), default=(0.0, 6.0), help="Excitation-energy window in eV")
    parser.add_argument("--epsilon", type=float, default=None, help="Static dielectric constant used to screen the direct term")
    parser.add_argument("--output-prefix", default="AMAT_exchange", help="Output prefix for .txt and .npz files")
    parser.add_argument("--full-hermitian", action="store_true", help="Write the full Hermitian Python matrix instead of VASP storage layout")
    parser.add_argument("--nexciton", type=int, default=None, help="Number of lowest excitons to write into BSEFATBAND output")
    parser.add_argument("--bsefatband-output", default=None, help="Optional path for VASP-format BSEFATBAND output")
    parser.add_argument("--use-response-basis", action="store_true", help="Use selected response-basis screened operator for the direct term instead of full-grid contraction")
    parser.add_argument("--verbose", action="store_true", help="Enable progress logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI entry point and emit AMAT/BSEFATBAND outputs."""
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)
    if not argv:
        parse_args(["--help"])
    args = parse_args(argv)
    wavecar = _resolve_existing_path(args.wavecar, label="WAVECAR")
    outcar = _resolve_existing_path(args.outcar, label="OUTCAR")
    kpoints = _resolve_existing_path(args.kpoints, label="KPOINTS")
    poscar = _resolve_existing_path(args.poscar, label="POSCAR") if args.mode in {"paw_orth_only", "paw_full"} else None
    potcar = _resolve_existing_path(args.potcar, label="POTCAR") if args.mode in {"paw_orth_only", "paw_full"} else None
    q_ext = _parse_q_ext(list(args.q_ext))
    charge_grid = _parse_outcar_charge_grid(outcar)
    if charge_grid is None:
        raise ValueError(f"Failed to read NGXF/NGYF/NGZF charge grid from OUTCAR: {outcar}")
    builder = HartreeFullGridBuilder(
        wavecar_path=wavecar,
        outcar_path=outcar,
        kpoints_path=kpoints,
        q_ext=q_ext.tolist(),
        vb_num=args.vb_num,
        cb_num=args.cb_num,
        ewin=tuple(float(x) for x in args.ewin),
        mode=args.mode,
        poscar_path=poscar,
        potcar_path=potcar,
        charge_grid=charge_grid,
        wfc_ifft_scale="N",
        interaction=args.interaction,
        epsilon=args.epsilon,
        use_response_basis=bool(args.use_response_basis),
        verbose=bool(args.verbose),
    )
    builder.build_pair_indices()
    matrix_full = np.asarray(builder.build_bse_matrix(vasp_storage=False), dtype=np.complex128)
    matrix_for_output = matrix_full.copy() if args.full_hermitian else _masked_vasp_storage_matrix(builder, matrix_full)
    np.fill_diagonal(matrix_for_output, 0.0j)
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    txt_path = output_prefix.with_suffix(".txt")
    npz_path = output_prefix.with_suffix(".npz")
    pair_table = _pair_metadata_array(builder.pairs)
    _write_text_matrix(txt_path, matrix_for_output, mode=args.mode, q_ext=q_ext, vasp_storage=not args.full_hermitian)
    implementation_tag = "full_grid_fastaug_source" if args.mode == "paw_full" else "full_grid_only"
    np.savez(npz_path, amat=matrix_for_output, pairs=pair_table, q_ext=q_ext, mode=np.asarray(args.mode), interaction=np.asarray(args.interaction), epsilon=np.asarray(args.epsilon if args.epsilon is not None else np.nan), ifft_scale=np.asarray("N"), implementation=np.asarray(implementation_tag), response_basis=np.asarray(bool(args.use_response_basis)), vasp_storage=np.asarray(not args.full_hermitian), wavecar=np.asarray(str(wavecar)), outcar=np.asarray(str(outcar)), kpoints=np.asarray(str(kpoints)), charge_grid=np.asarray(builder.ngrid, dtype=np.int32))
    eigenvalues, eigenvectors = _diagonalize_bse_matrix(matrix_full)
    bsefatband_path = Path(args.bsefatband_output).expanduser().resolve() if args.bsefatband_output else None
    if bsefatband_path is not None:
        _write_bsefatband(
            bsefatband_path,
            pairs=builder.pairs,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            kpoint_weight=builder.kpoint_weight,
            nexciton=len(builder.pairs) if args.nexciton is None else int(args.nexciton),
        )
    print(f"mode            : {args.mode}")
    print(f"interaction     : {args.interaction}")
    if args.epsilon is not None:
        print(f"epsilon         : {args.epsilon}")
    elif args.interaction in {"direct", "both"}:
        print("epsilon         : auto (WFULL/W if found, else bare Coulomb)")
    print(f"implementation  : {implementation_tag}")
    print(f"response_basis  : {bool(args.use_response_basis)}")
    print(f"pairs           : {len(builder.pairs)}")
    print(f"matrix_shape    : {matrix_for_output.shape}")
    print(f"q_ext           : {q_ext.tolist()}")
    print(f"charge_grid     : {builder.ngrid}")
    print(f"vasp_storage    : {not args.full_hermitian}")
    print(f"read_outcar     : {outcar}")
    print(f"read_kpoints    : {kpoints}")
    if args.mode in {"paw_orth_only", "paw_full"}:
        print(f"read_poscar     : {poscar}")
        print(f"read_potcar     : {potcar}")
    print(f"txt_output      : {txt_path}")
    print(f"npz_output      : {npz_path}")
    print(f"lowest_eigenvalue: {eigenvalues[0]:.8f}")
    if bsefatband_path is not None:
        print(f"bsefatband_out  : {bsefatband_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
