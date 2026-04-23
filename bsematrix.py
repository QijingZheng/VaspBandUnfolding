#!/usr/bin/env python3

"""
Build the Hartree/exchange part of the BSE matrix from VASP wavefunctions using
the local full-grid path.

The workflow in this script is:
1. Read the irreducible-k wavefunctions from WAVECAR together with symmetry,
    full-BZ mapping, and charge-grid information from OUTCAR.
2. Expand each excitonic pair state (v, k) -> (c, k + q) onto the full BZ while
    preserving VASP's k-point ordering, time-reversal handling, and symmetry
    phase conventions.
3. Reconstruct the cell-periodic wavefunctions on the VASP charge grid, form the
    pair densities in G space, apply the Coulomb kernel, and assemble the
    Hartree/exchange contribution to the BSE matrix element by element.
4. Optionally orthogonalize the PAW pseudo-wavefunctions with projector-overlap
    corrections in `paw_orth_only` mode before building the pair densities.
5. Diagonalize the resulting matrix and optionally write a VASP-style
    `BSEFATBAND` file from the eigenvectors.

This implementation currently targets the full-grid Hartree/exchange path only;
response-basis acceleration remains a TODO.

Credits:
  - Ionizing
  - OpenAI Codex
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.fft import fftn, ifftn

from vasp_constant import EDEPS, HSQDTM, TPI

try:
    from ase.io import read as ase_read
except ImportError:
    ase_read = None


class _DefaultsFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""
        if "%(default)" in help_text:
            return help_text
        if action.default in (None, argparse.SUPPRESS):
            return help_text
        if action.required:
            return help_text
        return f"{help_text} (default: %(default)s)"


def _resolve_existing_path(path_str: str | None, *, label: str) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _wrap_frac(v: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.mod(np.asarray(v, dtype=float), 1.0)


def _wrap_frac_signed(v: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.mod(np.asarray(v, dtype=float) + 0.5, 1.0) - 0.5


def _parse_fortran_float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "E"))


def _frac_key(frac: np.ndarray | Sequence[float], decimals: int = 8) -> Tuple[float, ...]:
    vals = np.mod(np.round(_wrap_frac(np.asarray(frac, dtype=float)), decimals), 1.0)
    return tuple(float(x) for x in vals.reshape(-1))


def _wrap_frac_signed_key(frac: np.ndarray | Sequence[float], decimals: int = 8) -> Tuple[float, ...]:
    vals = np.round(_wrap_frac_signed(np.asarray(frac, dtype=float)), decimals)
    return tuple(float(x) for x in vals.reshape(-1))


@dataclass(frozen=True)
class KpointMatch:
    ikpt: int
    time_reversed: bool = False
    symm_op: int = 0


@dataclass(frozen=True)
class FullBZKpoint:
    full_index: int
    k_frac: Tuple[float, float, float]
    match: KpointMatch


@dataclass(frozen=True)
class SymmetryOp:
    irot: int
    real_matrix: np.ndarray
    reciprocal_matrix: np.ndarray
    tau_frac: np.ndarray


@dataclass(frozen=True)
class PairState:
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
        return float(self.eps_c - self.eps_v)


@dataclass(frozen=True)
class FatbandExciton:
    index: int
    bse_eigenvalue: float
    ip_eigenvalue: float
    column_weight: np.ndarray
    amplitude: np.ndarray


def _kpoint_match_priority(match: KpointMatch) -> Tuple[int, int, int, int]:
    return (
        int(match.time_reversed),
        int(match.symm_op != 0),
        int(match.symm_op),
        int(match.ikpt),
    )


def _rodrigues_rotation(axis: Sequence[float], angle_deg: float) -> np.ndarray:
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
    pattern = re.compile(
        r"dimension x,y,z NGXF=\s*(\d+) NGYF=\s*(\d+) NGZF=\s*(\d+)",
        re.IGNORECASE,
    )
    for line in outcar_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match is not None:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None


def _parse_outcar_ir_kpoints(outcar_path: Path, nkpts: int) -> Optional[np.ndarray]:
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
    fx, fy, fz = [np.arange(n, dtype=int) for n in wfc._ngrid]
    fx[wfc._ngrid[0] // 2 + 1:] -= wfc._ngrid[0]
    fy[wfc._ngrid[1] // 2 + 1:] -= wfc._ngrid[1]
    fz[wfc._ngrid[2] // 2 + 1:] -= wfc._ngrid[2]
    gz, gy, gx = np.array(np.meshgrid(fz, fy, fx, indexing="ij")).reshape((3, -1))
    kgrid = np.array([gx, gy, gz], dtype=float).T
    kvec_arr = np.asarray(kvec, dtype=float)
    kenergy = HSQDTM * np.linalg.norm(np.dot(kgrid + kvec_arr[np.newaxis, :], TPI * wfc._Bcell), axis=1) ** 2
    return np.asarray(kgrid[np.where(kenergy < wfc._encut)[0]], dtype=int)


class _PawOrthHelper:
    def __init__(self, *, poscar_path: Path, potcar_path: Path, wfc: vaspwfc, ir_kvecs: np.ndarray, lgamma: bool, gamma_half: str) -> None:
        from paw import pawpotcar, _build_elem_groups

        if ase_read is None:
            raise ImportError("ASE is required for paw_orth_only mode")
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
        self._nonlq_cache: Dict[Tuple[float, ...], nonlq] = {}
        self._proj_cache: Dict[Tuple[int, int, int], List[np.ndarray]] = {}

    def _split_projector_vector(self, beta_flat: np.ndarray) -> List[np.ndarray]:
        flat = np.asarray(beta_flat, dtype=np.complex128).reshape(-1)
        blocks: List[np.ndarray] = []
        offset = 0
        for iatom in range(self.natoms):
            lmmax = int(self.pawpp[self.element_idx[iatom]].lmmax)
            blocks.append(np.asarray(flat[offset:offset + lmmax], dtype=np.complex128))
            offset += lmmax
        return blocks

    def _projector_for_k(self, kvec: Sequence[float]) -> nonlq:
        from paw import nonlq

        key = _wrap_frac_signed_key(kvec)
        cached = self._nonlq_cache.get(key)
        if cached is not None:
            return cached
        projector = nonlq(self.atoms, self.encut, potcar=self.pawpp, k=np.asarray(kvec, dtype=float), lgam=self.lgamma, gamma_half=self.gamma_half)
        self._nonlq_cache[key] = projector
        return projector

    def get_proj(self, ispin: int, ik: int, iband: int, coeff: np.ndarray) -> List[np.ndarray]:
        key = (ispin, ik, iband)
        cached = self._proj_cache.get(key)
        if cached is not None:
            return cached
        projector = self._projector_for_k(self.ir_kvecs[ik - 1])
        beta = projector.proj(np.asarray(coeff, dtype=np.complex128))
        blocks = self._split_projector_vector(beta)
        self._proj_cache[key] = blocks
        return blocks


class HartreeFullGridBuilder:
    def __init__(self, *, wavecar_path: Path, outcar_path: Path, kpoints_path: Path, q_ext: Sequence[float], vb_num: Optional[int], cb_num: Optional[int], ewin: Optional[Tuple[float, float]], mode: str, poscar_path: Optional[Path], potcar_path: Optional[Path], charge_grid: Optional[Sequence[int]], wfc_ifft_scale: str, verbose: bool) -> None:
        self.wavecar_path = Path(wavecar_path).resolve()
        self.outcar_path = Path(outcar_path).resolve()
        self.kpoints_path = Path(kpoints_path).resolve()
        self.mode = mode
        self.verbose = bool(verbose)
        self.q_ext = np.asarray(q_ext, dtype=float)
        self.vb_num = vb_num
        self.cb_num = cb_num
        self.ewin = ewin
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
        self._orth_coeff_cache: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._orth_ready = False
        self.pairs: List[PairState] = []

        self._paw: Optional[_PawOrthHelper] = None
        if self.mode == "paw_orth_only":
            if poscar_path is None or potcar_path is None:
                raise FileNotFoundError("paw_orth_only requires POSCAR and POTCAR")
            self._paw = _PawOrthHelper(poscar_path=Path(poscar_path).resolve(), potcar_path=Path(potcar_path).resolve(), wfc=self.wfc, ir_kvecs=self._ir_kvecs, lgamma=getattr(self.wfc, "_lgam", False), gamma_half=getattr(self.wfc, "_gam_half", "x"))

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _infer_kpoint_weight(self) -> float:
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
        self._wfc_cache.clear()
        self._grid_wfc_cache.clear()
        self._matched_wfc_cache.clear()
        self._grid_matched_wfc_cache.clear()
        self._matched_coeff_grid_cache.clear()
        self._orth_coeff_cache.clear()
        self._orth_ready = False
        if self._paw is None:
            self._orth_ready = True
            return
        band_list = list(range(1, self._nbands + 1))
        if len(band_list) <= 1:
            self._orth_ready = True
            return
        self._log("Orthogonalizing full NBANDS manifolds for all spin/k blocks ...")
        for ispin in range(1, self._nspin + 1):
            for ik in range(1, self._nkpts + 1):
                coeff_cols = [self._raw_band_coeff(ispin, ik, iband) for iband in band_list]
                coeff_mat = np.column_stack(coeff_cols)
                overlap_mat = coeff_mat.conj().T @ coeff_mat
                proj_cols = [self._paw.get_proj(ispin, ik, iband, coeff_cols[idx]) for idx, iband in enumerate(band_list)]
                overlap_mat += self._projector_overlap_block(proj_cols)
                overlap_mat = 0.5 * (overlap_mat + overlap_mat.conj().T)
                transform = self._inverse_cholesky_upper(overlap_mat)
                coeff_orth = coeff_mat @ transform
                for col_idx, iband in enumerate(band_list):
                    self._orth_coeff_cache[(ispin, ik, iband)] = np.asarray(coeff_orth[:, col_idx], dtype=np.complex128)
        self._orth_ready = True

    def _projector_overlap_block(self, proj_cols: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
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
        try:
            lower = np.linalg.cholesky(overlap_mat)
        except np.linalg.LinAlgError:
            jitter = 1e-10 * max(1.0, float(np.max(np.abs(np.diag(overlap_mat)))))
            lower = np.linalg.cholesky(overlap_mat + jitter * np.eye(overlap_mat.shape[0], dtype=overlap_mat.dtype))
        return np.linalg.inv(lower.conj().T)

    def _ensure_orthogonalized_wavefunctions(self) -> None:
        if not self._orth_ready:
            self.orthogonalize_pair_subspaces()

    def _load_symmetry_ops(self) -> List[SymmetryOp]:
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
        key = _frac_key(k_frac)
        current = self._expanded_kpoint_lookup.get(key)
        if current is None or _kpoint_match_priority(match) < _kpoint_match_priority(current):
            self._expanded_kpoint_lookup[key] = match

    def _store_full_bz_match(self, k_frac: np.ndarray, match: KpointMatch) -> None:
        key = _frac_key(k_frac)
        current = self._full_bz_lookup.get(key)
        if current is None or _kpoint_match_priority(match) < _kpoint_match_priority(current):
            self._full_bz_lookup[key] = match

    def _store_full_bz_index(self, k_frac: np.ndarray, full_index: int) -> None:
        self._full_bz_index_lookup[_frac_key(k_frac)] = int(full_index)

    def _full_bz_index(self, k_frac: Sequence[float]) -> Optional[int]:
        return self._full_bz_index_lookup.get(_frac_key(k_frac))

    def _build_expanded_kpoint_lookup(self) -> None:
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
        coeff = self.wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=iband, norm=False)
        if np.iscomplexobj(coeff) and np.ndim(coeff) == 2:
            coeff = coeff[:, 0]
        coeff_arr = np.asarray(coeff, dtype=np.complex128)
        gvecs = self.wfc.gvectors(ikpt=ik, check_consistency=False)
        if coeff_arr.shape[0] != gvecs.shape[0]:
            raise ValueError(f"Coefficient/G-vector size mismatch for {(ispin, ik, iband)}")
        return coeff_arr

    def _matched_gvecs_and_coeffs(self, ispin: int, ik: int, iband: int, *, time_reversed: bool, symm_op: int, k_frac: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        coeff = self._orth_coeff_cache.get((ispin, ik, iband))
        if coeff is None:
            coeff = self._raw_band_coeff(ispin, ik, iband)
        coeff_arr = np.asarray(coeff, dtype=np.complex128).copy()
        raw_gvecs = np.asarray(self.wfc.gvectors(ikpt=ik, check_consistency=False), dtype=int)
        raw_rep_k = np.asarray(self.wfc._kvecs[ik - 1], dtype=float)
        corrected_rep_k = np.asarray(self._ir_kvecs[ik - 1], dtype=float)
        rep_flipped = bool(np.max(np.abs(_wrap_frac_signed(corrected_rep_k - raw_rep_k))) > 5e-5)
        if rep_flipped and symm_op == 0:
            target_rep_k = -corrected_rep_k if time_reversed else corrected_rep_k
            target_gvecs = _generate_gvectors_for_kvec(self.wfc, target_rep_k)
            source_gvecs = -raw_gvecs if time_reversed else raw_gvecs
            source_coeffs = np.conjugate(coeff_arr) if time_reversed else coeff_arr
            source_map = {tuple(int(x) for x in gvec): source_coeffs[idx] for idx, gvec in enumerate(source_gvecs)}
            remapped_coeffs = np.asarray([source_map.get(tuple(int(x) for x in gvec), 0.0j) for gvec in target_gvecs], dtype=np.complex128)
            return target_gvecs, remapped_coeffs
        gvecs = raw_gvecs.astype(float)
        q_frac = gvecs + corrected_rep_k[np.newaxis, :]
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

    def _pair_density_G(self, u_left: np.ndarray, u_right: np.ndarray) -> np.ndarray:
        return fftn(np.conj(u_left) * u_right)

    def _build_Gcart_grid(self) -> np.ndarray:
        nx, ny, nz = self.ngrid
        gx = np.fft.fftfreq(nx) * nx
        gy = np.fft.fftfreq(ny) * ny
        gz = np.fft.fftfreq(nz) * nz
        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
        return np.stack([GX, GY, GZ], axis=-1) @ (TPI * self.Bcell)

    def _coulomb_kernel(self, q_frac: np.ndarray) -> np.ndarray:
        q_arr = np.asarray(q_frac, dtype=float)
        q_cart = q_arr @ (TPI * self.Bcell)
        qpG_cart = self._Gcart_grid + q_cart[np.newaxis, np.newaxis, np.newaxis, :]
        qpGsq = np.sum(qpG_cart**2, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(qpGsq > 0.0, EDEPS / qpGsq, 0.0)

    @staticmethod
    def _setphase_integer_shift(setphase_shift_frac: np.ndarray, tol: float = 1e-6) -> Tuple[int, int, int]:
        shift = np.asarray(setphase_shift_frac, dtype=float)
        rounded = np.rint(shift)
        if np.max(np.abs(shift - rounded)) > tol:
            raise ValueError(f"SETPHASE shift is not integer within tolerance: {shift.tolist()}")
        return int(rounded[0]), int(rounded[1]), int(rounded[2])

    def _get_setphase_grid(self, shift_int: Tuple[int, int, int]) -> np.ndarray:
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
        values_arr = np.asarray(values_G, dtype=np.complex128)
        if setphase_shift_frac is None:
            return values_arr
        shift_int = self._setphase_integer_shift(setphase_shift_frac)
        if shift_int == (0, 0, 0):
            return values_arr
        phase = self._get_setphase_grid(shift_int)
        return np.asarray(fftn(ifftn(values_arr) * phase), dtype=np.complex128)

    def build_pair_indices(self) -> List[PairState]:
        self.pairs = []
        for ispin in range(1, self._nspin + 1):
            occs = self.wfc._occs[ispin - 1]
            bands = self.wfc._bands[ispin - 1]
            for full_k in self.full_kpoints:
                k1_frac = np.asarray(full_k.k_frac, dtype=float)
                ik1_full = full_k.full_index
                k1_match = full_k.match
                ik1_0 = k1_match.ikpt - 1
                ik1_ir = k1_match.ikpt
                k3_frac = _wrap_frac(k1_frac + self.q_ext)
                k3_match = self._match_full_kpoint(k3_frac)
                if k3_match is None:
                    continue
                ik3_full = self._full_bz_index(k3_frac)
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
        self._log(f"Number of BSE pair states: {len(self.pairs)}")
        return self.pairs

    def _full_k_loop_position(self, k_frac: Sequence[float]) -> int:
        key = _frac_key(k_frac)
        for idx, full_k in enumerate(self.full_kpoints):
            if _frac_key(full_k.k_frac) == key:
                return idx
        raise KeyError(f"Unknown full-BZ k-point: {tuple(k_frac)}")

    def _should_store_vasp_bse_entry(self, pair_i: PairState, pair_j: PairState) -> bool:
        if pair_j.ispin < pair_i.ispin:
            return False
        if pair_j.ispin > pair_i.ispin:
            return True
        return self._full_k_loop_position(pair_j.k1_frac) >= self._full_k_loop_position(pair_i.k1_frac)

    def build_bse_matrix(self, *, vasp_storage: bool) -> np.ndarray:
        self._ensure_orthogonalized_wavefunctions()
        npairs = len(self.pairs)
        A = np.zeros((npairs, npairs), dtype=np.complex128)
        for idx, pair in enumerate(self.pairs):
            A[idx, idx] = pair.excitation_energy
        pair_weight = self.kpoint_weight
        if self.wfc_ifft_scale == "N":
            pair_density_scale = float(self.Nfft)
        elif self.wfc_ifft_scale == "none":
            pair_density_scale = 1.0 / float(self.Nfft)
        else:
            pair_density_scale = 1.0
        pair_density_norm = 1.0 / (pair_density_scale**2)
        hartree_pair_norm = (self.Omega / float(self.Nfft)) * pair_density_norm
        q_h = -self.q_ext
        v_h = self._coulomb_kernel(q_h)
        self._log("Building full-grid Hartree matrix ...")
        for i, pair_i in enumerate(self.pairs):
            u_v1 = self._get_periodic_wfc(pair_i.ispin, pair_i.ik_ir, pair_i.iv, time_reversed=pair_i.k1_time_reversed, symm_op=pair_i.k1_symm_op, k_frac=pair_i.k1_frac)
            u_c3 = self._get_periodic_wfc(pair_i.ispin, pair_i.ik3_ir, pair_i.ic, time_reversed=pair_i.k3_time_reversed, symm_op=pair_i.k3_symm_op, k_frac=pair_i.k3_frac)
            rho13 = self._pair_density_G(u_c3, u_v1)
            shift13 = np.asarray(pair_i.k1_frac, dtype=float) - np.asarray(pair_i.k3_frac, dtype=float) - np.asarray(q_h, dtype=float)
            rho13_eff = self._apply_setphase_to_G_object(rho13, shift13)
            for j, pair_j in enumerate(self.pairs):
                if vasp_storage and not self._should_store_vasp_bse_entry(pair_i, pair_j):
                    continue
                if pair_i.ispin != pair_j.ispin:
                    continue
                u_v2 = self._get_periodic_wfc(pair_j.ispin, pair_j.ik_ir, pair_j.iv, time_reversed=pair_j.k1_time_reversed, symm_op=pair_j.k1_symm_op, k_frac=pair_j.k1_frac)
                u_c4 = self._get_periodic_wfc(pair_j.ispin, pair_j.ik3_ir, pair_j.ic, time_reversed=pair_j.k3_time_reversed, symm_op=pair_j.k3_symm_op, k_frac=pair_j.k3_frac)
                rho24 = self._pair_density_G(u_c4, u_v2)
                shift24 = np.asarray(pair_j.k1_frac, dtype=float) - np.asarray(pair_j.k3_frac, dtype=float) - np.asarray(q_h, dtype=float)
                rho24_eff = self._apply_setphase_to_G_object(rho24, shift24)
                kh = pair_weight * self.spin_factor * hartree_pair_norm * np.sum(rho13_eff * (v_h / self.Omega) * np.conj(rho24_eff))
                A[i, j] += kh
        return A


def _parse_q_ext(values: list[float]) -> np.ndarray:
    if len(values) != 3:
        raise ValueError("q_ext must have exactly three fractional components")
    return np.asarray(values, dtype=float)


def _pair_metadata_array(pairs: list[PairState]) -> np.ndarray:
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
    masked = np.asarray(matrix, dtype=np.complex128).copy()
    for i, pair_i in enumerate(builder.pairs):
        for j, pair_j in enumerate(builder.pairs):
            if not builder._should_store_vasp_bse_entry(pair_i, pair_j):
                masked[i, j] = 0.0j
    return masked


def _diagonalize_bse_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    parser = argparse.ArgumentParser(
        description="Build a Hartree-only BSE exchange matrix using the local full-grid path.",
        formatter_class=_DefaultsFormatter,
    )
    parser.add_argument("--wavecar", default="WAVECAR", help="Path to WAVECAR")
    parser.add_argument("--outcar", default="OUTCAR", help="Path to OUTCAR")
    parser.add_argument("--kpoints", default="KPOINTS", help="Path to KPOINTS")
    parser.add_argument("--poscar", default="POSCAR", help="Path to POSCAR")
    parser.add_argument("--potcar", default="POTCAR", help="Path to POTCAR")
    parser.add_argument("--mode", choices=["pw_only", "paw_orth_only"], default="paw_orth_only", help="Matrix construction mode")
    parser.add_argument("--q-ext", nargs=3, type=float, metavar=("QX", "QY", "QZ"), default=[0.0, 0.0, 0.0], help="External exciton momentum in fractional reciprocal coordinates")
    parser.add_argument("--vb-num", type=int, required=True, help="Number of valence bands")
    parser.add_argument("--cb-num", type=int, required=True, help="Number of conduction bands")
    parser.add_argument("--ewin", nargs=2, type=float, metavar=("EMIN", "EMAX"), default=(0.0, 6.0), help="Excitation-energy window in eV")
    parser.add_argument("--output-prefix", default="AMAT_exchange", help="Output prefix for .txt and .npz files")
    parser.add_argument("--full-hermitian", action="store_true", help="Write the full Hermitian Python matrix instead of VASP storage layout")
    parser.add_argument("--nexciton", type=int, default=None, help="Number of lowest excitons to write into BSEFATBAND output")
    parser.add_argument("--bsefatband-output", default=None, help="Optional path for VASP-format BSEFATBAND output")
    parser.add_argument("--verbose", action="store_true", help="Enable progress logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
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
    poscar = _resolve_existing_path(args.poscar, label="POSCAR") if args.mode == "paw_orth_only" else None
    potcar = _resolve_existing_path(args.potcar, label="POTCAR") if args.mode == "paw_orth_only" else None
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
    np.savez(npz_path, amat=matrix_for_output, pairs=pair_table, q_ext=q_ext, mode=np.asarray(args.mode), ifft_scale=np.asarray("N"), implementation=np.asarray("full_grid_only"), response_basis=np.asarray("TODO"), vasp_storage=np.asarray(not args.full_hermitian), wavecar=np.asarray(str(wavecar)), outcar=np.asarray(str(outcar)), kpoints=np.asarray(str(kpoints)), charge_grid=np.asarray(charge_grid, dtype=np.int32))
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
    print("implementation  : full_grid_only")
    print("response_basis  : TODO")
    print(f"pairs           : {len(builder.pairs)}")
    print(f"matrix_shape    : {matrix_for_output.shape}")
    print(f"q_ext           : {q_ext.tolist()}")
    print(f"charge_grid     : {charge_grid}")
    print(f"vasp_storage    : {not args.full_hermitian}")
    print(f"read_outcar     : {outcar}")
    print(f"read_kpoints    : {kpoints}")
    if args.mode == "paw_orth_only":
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
