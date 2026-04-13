#!/usr/bin/env python3

"""
Build SOC spinor WAVECAR from VASP scalar/ISPIN=2 outputs.
Provides a command-line spinor maker with optional k-point correction
and post-mixing purification controls.

Contributors:
- @Ionizing
- Xiang Jiang (@realxiangjiang)
- Qijing Zheng
- OpenAI Codex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from vaspwfc import vaspwfc
from spinorb import paw_core_soc_mat, read_cproj_NormalCar

DEFAULT_WAVECAR = "WAVECAR"
DEFAULT_WAVECAR_OUT = "WAVECAR_spinor"
DEFAULT_MIXWAVE_IKPT = 1
DEFAULT_SPIN_BALANCE_TOL = 1.0e-5


def _parse_int_tokens(tokens: list[str]) -> tuple[int, ...]:
    """
    Parse a list of integer tokens that may be space- or comma-separated.
    Example:
        ["1", "2,4", "7"] -> (1, 2, 4, 7)
    """
    vals: list[int] = []
    for tok in tokens:
        for part in tok.split(","):
            s = part.strip()
            if s:
                vals.append(int(s))
    return tuple(vals)


def _to_zero_based_indices(indices_1based: tuple[int, ...], *, name: str) -> tuple[int, ...]:
    if indices_1based and min(indices_1based) < 1:
        raise ValueError(f"{name} expects 1-based positive indices.")
    return tuple(idx - 1 for idx in indices_1based)


def ReCombineAB(a: NDArray[np.complex128],
                b: NDArray[np.complex128],
                N: int) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Python port of paraHFNAMD/src/math.cpp::ReCombineAB.

    `a` and `b` are 2N vectors in block order [up(N), down(N)].
    """
    assert a.ndim == 1 and b.ndim == 1 and a.shape == b.shape
    assert a.shape[0] == 2 * N

    res1 = np.zeros_like(a)
    res2 = np.zeros_like(a)

    for ispin in range(2):
        a_s = a[ispin * N:(ispin + 1) * N]
        b_s = b[ispin * N:(ispin + 1) * N]

        norm_a2 = np.linalg.norm(a_s) ** 2
        norm_b2 = np.linalg.norm(b_s) ** 2
        cdtmp = np.vdot(a_s, b_s)  # conj(a_s) dot b_s

        abmat = np.zeros((4, 4), dtype=float)
        abmat[0, 0] = norm_a2
        abmat[1, 1] = norm_a2
        abmat[2, 2] = norm_b2
        abmat[3, 3] = norm_b2

        abmat[0, 2] = cdtmp.real
        abmat[1, 3] = cdtmp.real
        abmat[2, 0] = cdtmp.real
        abmat[3, 1] = cdtmp.real

        abmat[0, 3] = -cdtmp.imag
        abmat[1, 2] = cdtmp.imag
        abmat[2, 1] = cdtmp.imag
        abmat[3, 0] = -cdtmp.imag

        # Dsyev(..., 'V', 'L', ...): eigenvectors in columns
        _eigvals, eigvecs = np.linalg.eigh(abmat)

        minf = np.inf
        min_ind = 0
        for ii in range(4):
            z1 = eigvecs[0, ii] + 1j * eigvecs[1, ii]
            z2 = eigvecs[2, ii] + 1j * eigvecs[3, ii]
            amp = np.linalg.norm(z1 * a_s + z2 * b_s)
            if amp < minf:
                minf = amp
                min_ind = ii

        z1 = eigvecs[0, min_ind] + 1j * eigvecs[1, min_ind]
        z2 = eigvecs[2, min_ind] + 1j * eigvecs[3, min_ind]
        mixed = z1 * a + z2 * b

        # Keep C++ assignment order exactly.
        if ispin:
            res1[:] = mixed
        else:
            res2[:] = mixed

    return res1, res2


class socclass:
    """
    Single-process SOC builder that follows paraHFNAMD soc.cpp flow.
    """

    def __init__(self, wavecar: str = DEFAULT_WAVECAR, *,
                 lsorbit: bool = False,
                 correct_kpts: tuple[int, ...] | list[int] | None = None):
        self.wvc = vaspwfc(wavecar, lsorbit=lsorbit)
        assert not self.wvc._lsoc, "This WAVECAR already contains spinors."
        assert self.wvc._nplws[0] == self.wvc.gvectors(ikpt=1).shape[0], (
            "Gamma-half / existing ncl WAVECAR is not supported."
        )

        self.nspns = int(self.wvc._nspin)
        self.nkpts = int(self.wvc._nkpts)
        self.nbnds = int(self.wvc._nbands)
        self.twonbnds = 2 * self.nbnds

        self.maxnpw = int(max(np.max(self.wvc._nplws), 2 + 3 * self.nbnds))
        self.recl = int(2 * self.maxnpw * np.dtype(np.complex64).itemsize)

        # Corresponds to socmat / eigenvals in C++
        self.socmat = np.zeros((self.twonbnds, self.twonbnds), dtype=np.complex128)
        self.eigenvals = np.zeros(self.twonbnds, dtype=np.float64)

        # K-points where CorrectSocmat/ReCombineAB is applied.
        # Internal indexing is 0-based, matching soc.cpp call sites.
        self.read_in_kpts = self._normalize_correct_kpts(correct_kpts)

        # Prepare SOC building blocks (single-process replacement of Setup_hsoc + projector contractions).
        self._setup_soc_blocks()

        self.isSettotweight = False
        self.totweight = 0.0

    def _normalize_correct_kpts(
        self, correct_kpts: tuple[int, ...] | list[int] | None
    ) -> list[int]:
        if not correct_kpts:
            return []
        sel = sorted(set(int(k) for k in correct_kpts))
        for k in sel:
            if not (0 <= k < self.nkpts):
                raise ValueError(f"Invalid corrected k-point index: {k + 1}")
        return sel

    def _setup_soc_blocks(self) -> None:
        # Equivalent high-level data needed to assemble socmat for each k-point.
        soc = paw_core_soc_mat()  # [4, nproj, nproj]
        cproj = read_cproj_NormalCar()  # [nspns*nkpts*nbnds, nproj]

        assert soc.shape[-1] == cproj.shape[-1], (
            "No. of projectors from SocCar and NormalCAR not consistent."
        )
        assert cproj.shape[0] == self.nspns * self.nkpts * self.nbnds, (
            "No. of bands in NormalCAR and WAVECAR not consistent."
        )

        self._soc = soc
        self._cproj = cproj.reshape((self.nspns, self.nkpts, self.nbnds, soc.shape[-1]))

    def CheckSpinorWavecar(self, wavecar_out: str = DEFAULT_WAVECAR_OUT) -> int:
        """
        Port of soc.cpp::CheckSpinorWavecar.
        """
        p = Path(wavecar_out)
        if not p.exists():
            return 0
        with p.open("rb") as f:
            rdum = np.fromfile(f, dtype=np.float64, count=1)
            if rdum.size != 1:
                return 0
            if int(rdum[0]) != self.recl:
                return 0
            f.seek(0, 2)
            fsize = f.tell()

        expected = (2 + self.nkpts * (1 + self.twonbnds)) * self.recl
        return 1 if fsize == expected else 0

    def SetSocmatDiag(self, in_kpt: int) -> None:
        """
        Port of soc.cpp::SetSocmatDiag.
        in_kpt is 0-based.
        """
        self.socmat.fill(0.0)
        for ispin in range(2):
            iss = min(ispin, self.nspns - 1)
            e = self.wvc._bands[iss, in_kpt, :]
            idx = ispin * self.nbnds + np.arange(self.nbnds)
            self.socmat[idx, idx] = e

    def AddSocOnsite(self, in_kpt: int) -> None:
        """
        Single-process replacement for soc.cpp onsite projector contractions in MakeSpinor.
        in_kpt is 0-based.
        """
        c0 = self._cproj[0, in_kpt, :, :]  # [nb, nproj]
        if self.nspns == 1:
            c1 = c0
        else:
            c1 = self._cproj[1, in_kpt, :, :]

        self.socmat[:self.nbnds, :self.nbnds] += c0.conj() @ self._soc[0] @ c0.T
        self.socmat[:self.nbnds, self.nbnds:] += c0.conj() @ self._soc[1] @ c1.T
        self.socmat[self.nbnds:, :self.nbnds] += c1.conj() @ self._soc[2] @ c0.T
        self.socmat[self.nbnds:, self.nbnds:] += c1.conj() @ self._soc[3] @ c1.T

    def CorrectSocmat(self, ikpt: int, eigvecs: NDArray[np.complex128]) -> None:
        """
        Port of soc.cpp::CorrectSocmat.
        """
        if ikpt not in self.read_in_kpts:
            return

        for ib in range(self.nbnds):
            i0 = 2 * ib
            i1 = i0 + 1
            a = eigvecs[:, i0].copy()
            b = eigvecs[:, i1].copy()
            ra, rb = ReCombineAB(a, b, self.nbnds)
            eigvecs[:, i0] = ra
            eigvecs[:, i1] = rb

            newen = 0.5 * (self.eigenvals[i0] + self.eigenvals[i1])
            self.eigenvals[i0] = newen
            self.eigenvals[i1] = newen

    def GetSpinorCoeff(self, in_kpt: int, eigvecs: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Port of soc.cpp::GetSpinorCoeff (PW part only).
        Returns shape [twonbnds, 2*npw] with [up_pw, down_pw] concatenation per band.
        """
        npw = int(self.wvc._nplws[in_kpt])

        # C_{G,ib}
        cup = np.zeros((npw, self.nbnds), dtype=np.complex128)
        cdn = np.zeros((npw, self.nbnds), dtype=np.complex128)
        ispin_dn = min(2, self.nspns)
        for ib in range(self.nbnds):
            cup[:, ib] = self.wvc.readBandCoeff(ispin=1, ikpt=in_kpt + 1, iband=ib + 1, norm=False)
            cdn[:, ib] = self.wvc.readBandCoeff(ispin=ispin_dn, ikpt=in_kpt + 1, iband=ib + 1, norm=False)

        # S_{ib, n}: columns are SOC eigenvectors (twonbnds x twonbnds)
        spin_up = cup @ eigvecs[:self.nbnds, :]
        spin_dn = cdn @ eigvecs[self.nbnds:, :]

        spinorpw = np.zeros((self.twonbnds, 2 * npw), dtype=np.complex128)
        spinorpw[:, :npw] = spin_up.T
        spinorpw[:, npw:] = spin_dn.T
        return spinorpw

    def _get_spinor_irec(self, *, ikpoint: int, iband: int) -> int:
        assert 1 <= ikpoint <= self.nkpts
        assert 1 <= iband <= self.twonbnds
        return 2 + (ikpoint - 1) * (self.twonbnds + 1) + iband

    def _get_total_weight(self) -> float:
        if self.isSettotweight:
            return self.totweight

        # Mirror soc.cpp logic: sum first-kpoint weights, then multiply by (3 - nspns).
        wt = 0.0
        for ispin in range(self.nspns):
            wt += float(np.sum(self.wvc._occs[ispin, 0, :]))
        wt *= (3 - self.nspns)
        self.totweight = wt
        self.isSettotweight = True
        return wt

    def WriteSpinorHead(self, wavecar_out: str = DEFAULT_WAVECAR_OUT) -> None:
        """
        Port of soc.cpp::WriteSpinorHead.
        """
        with open(wavecar_out, "wb") as f:
            rdum = float(self.recl)
            totnspns = 1.0
            rtag = 45200.0
            np.array([rdum, totnspns, rtag], dtype=np.float64).tofile(f)

            # second record
            f.seek(self.recl)
            rec2 = np.array(
                [
                    float(self.nkpts),
                    float(self.twonbnds),
                    float(self.wvc._encut),
                    *self.wvc._Acell.flatten(),
                    float(self.wvc._efermi),
                ],
                dtype=np.float64,
            )
            rec2.tofile(f)

    def WriteSpinor(self, in_kpt: int, spinorpw: NDArray[np.complex128], *,
                    wavecar_out: str = DEFAULT_WAVECAR_OUT) -> None:
        """
        Port of soc.cpp::WriteSpinor.
        in_kpt is 0-based.
        """
        npw = int(self.wvc._nplws[in_kpt])
        totweight = self._get_total_weight()

        with open(wavecar_out, "rb+") as f:
            # k-point header record
            irec = self._get_spinor_irec(ikpoint=in_kpt + 1, iband=1)
            f.seek((irec - 1) * self.recl)

            buf = np.zeros((4 + 3 * self.twonbnds,), dtype=np.float64)
            buf[0] = float(2 * npw)
            buf[1:4] = self.wvc._kvecs[in_kpt, :]
            for ib in range(self.twonbnds):
                buf[4 + 3 * ib + 0] = self.eigenvals[ib]
                buf[4 + 3 * ib + 1] = 0.0
                buf[4 + 3 * ib + 2] = 1.0 if (ib + 1) < (totweight + 0.1) else 0.0
            buf.tofile(f)

            # band records
            for ib in range(self.twonbnds):
                irec = self._get_spinor_irec(ikpoint=in_kpt + 1, iband=ib + 1)
                f.seek(irec * self.recl)
                spinorpw[ib, :].astype(np.complex64).tofile(f)

    @staticmethod
    def _sigma_elements(phi1: np.ndarray, phi2: np.ndarray) -> tuple[float, float, complex]:
        a11 = float(np.sum(np.abs(phi1[0]) ** 2) - np.sum(np.abs(phi1[1]) ** 2))
        a22 = float(np.sum(np.abs(phi2[0]) ** 2) - np.sum(np.abs(phi2[1]) ** 2))
        a21 = (
            np.einsum("ijk,ijk->", np.conj(phi2[0]), phi1[0])
            - np.einsum("ijk,ijk->", np.conj(phi2[1]), phi1[1])
        )
        return a11, a22, a21

    @staticmethod
    def _read_pair_coeffs(wav: vaspwfc, *, ikpt: int, ib: int, ncoeff: int) -> tuple[np.ndarray, list[float]]:
        psi_nat: list[NDArray[np.complexfloating]] = []
        norms: list[float] = []
        for iw in (0, 1):
            rec = wav.whereRec(ispin=1, ikpt=ikpt, iband=ib + iw)
            wav._wfc.seek(rec * wav._recl)
            dump = np.fromfile(wav._wfc, dtype=wav._WFPrec, count=ncoeff)
            nrm = float(np.linalg.norm(dump))
            if nrm > 0.0:
                dump = dump / nrm
            else:
                nrm = 1.0
            psi_nat.append(dump)
            norms.append(nrm)
        return np.array(psi_nat), norms

    @staticmethod
    def _mix_pair(psi_nat: np.ndarray, *, a11: float, a22: float,
                  a21: complex, spin_balance_tol: float) -> np.ndarray:
        # Preferred branch: exact mixWave.py formula (assumes trace~0 pair).
        # Fallback branch: diagonalize 2x2 sigma_z matrix for robust purification.
        if (abs(a11 + a22) <= spin_balance_tol) and (not np.isclose(abs(a21), 0.0)):
            reverse = False
            if a11 > 0.0:
                a11, a22 = a22, a11
                reverse = True

            phase = a21 / np.abs(a21)
            theta = (np.arctan(np.abs(a21) / a11) + np.pi) / 2.0
            a = np.cos(theta)
            b = phase * np.sin(theta)
            U = np.array([[a, b], [-np.conj(b), np.conj(a)]], dtype=np.complex128)
            psi_in = psi_nat[[1, 0], :] if reverse else psi_nat
            return np.dot(U, psi_in)

        sigma_z = np.array(
            [
                [a11, np.conj(a21)],
                [a21, a22],
            ],
            dtype=np.complex128,
        )
        _evals, evecs = np.linalg.eigh(sigma_z)
        order = np.argsort(_evals.real)  # negative first, positive second
        coeffs = evecs[:, order]         # columns are new states in old basis
        U = coeffs.T
        return np.dot(U, psi_nat)

    @staticmethod
    def _enforce_spin_order(psi_mix: np.ndarray, *, nplw: int) -> np.ndarray:
        sz0 = float(np.linalg.norm(psi_mix[0, :nplw]) ** 2 - np.linalg.norm(psi_mix[0, nplw:]) ** 2)
        sz1 = float(np.linalg.norm(psi_mix[1, :nplw]) ** 2 - np.linalg.norm(psi_mix[1, nplw:]) ** 2)
        return psi_mix[[1, 0], :] if sz0 < sz1 else psi_mix

    @staticmethod
    def _restore_norms(psi_mix: np.ndarray, norms: list[float]) -> np.ndarray:
        out = psi_mix.copy()
        for iw in range(2):
            mix_nrm = float(np.linalg.norm(out[iw]))
            if mix_nrm > 0.0:
                out[iw] = out[iw] / mix_nrm * norms[iw]
        return out

    @staticmethod
    def _write_pair_coeffs(wf, wav: vaspwfc, *, ikpt: int, ib: int, psi_mix: np.ndarray) -> None:
        for iw in range(2):
            rec = wav.whereRec(ispin=1, ikpt=ikpt, iband=ib + iw)
            wf.seek(rec * wav._recl)
            psi_mix[iw].astype(wav._WFPrec).tofile(wf)

    def MixWavePurify(self, *, wavecar_out: str = DEFAULT_WAVECAR_OUT,
                      ibs: tuple[int, ...] = (211, 213, 215, 217, 219),
                      ikpt: int = DEFAULT_MIXWAVE_IKPT,
                      spin_balance_tol: float = DEFAULT_SPIN_BALANCE_TOL) -> None:
        """
        Port/merge of VaspBandUnfolding/mixWave.py for a single WAVECAR.

        The same 2x2 unitary rotation is applied to each selected SOC pair
        (ib, ib+1) at a chosen k-point to make spinors purer.
        """
        wav = vaspwfc(wavecar_out, lsorbit=True)
        if int(wav._nspin) != 1:
            raise ValueError("MixWavePurify expects LSORBIT/NCL WAVECAR (nspin=1).")
        if not (1 <= ikpt <= int(wav._nkpts)):
            raise ValueError("Invalid ikpt in MixWavePurify.")

        # In LSORBIT WAVECAR, _nplws already equals full spinor coeff length:
        # [up(npw), down(npw)] -> length = 2*npw.
        ncoeff = int(wav._nplws[ikpt - 1])
        if ncoeff % 2 != 0:
            raise ValueError("Unexpected odd spinor coeff length.")
        nplw = ncoeff // 2

        with open(wavecar_out, "rb+") as wf:
            for ib in ibs:
                if not (1 <= ib < int(wav._nbands)):
                    raise ValueError(f"Invalid band pair start index: {ib}")

                phi1 = np.array(wav.wfc_r(iband=ib, ikpt=ikpt))
                phi2 = np.array(wav.wfc_r(iband=ib + 1, ikpt=ikpt))
                a11, a22, a21 = self._sigma_elements(phi1, phi2)

                psi_nat, norms = self._read_pair_coeffs(wav, ikpt=ikpt, ib=ib, ncoeff=ncoeff)
                psi_mix = self._mix_pair(
                    psi_nat,
                    a11=a11,
                    a22=a22,
                    a21=a21,
                    spin_balance_tol=spin_balance_tol,
                )
                psi_mix = self._enforce_spin_order(psi_mix, nplw=nplw)
                psi_mix = self._restore_norms(psi_mix, norms)
                self._write_pair_coeffs(wf, wav, ikpt=ikpt, ib=ib, psi_mix=psi_mix)

    def MakeSpinor(self, *, wavecar_out: str = DEFAULT_WAVECAR_OUT,
                   overwrite: bool = True,
                   mixwave_purify: bool = True,
                   mixwave_ibs: tuple[int, ...] = (211, 213, 215, 217, 219),
                   mixwave_ikpt: int = DEFAULT_MIXWAVE_IKPT,
                   mixwave_spin_balance_tol: float = DEFAULT_SPIN_BALANCE_TOL) -> bool:
        """
        Port of soc.cpp::MakeSpinor (single-process).
        """
        if (not overwrite) and self.CheckSpinorWavecar(wavecar_out):
            return False

        self.WriteSpinorHead(wavecar_out)

        for ik in range(self.nkpts):
            self.SetSocmatDiag(ik)
            self.AddSocOnsite(ik)

            # Zheev(..., 'V', 'U', ...): Hermitian eigenproblem
            eigvals, eigvecs = np.linalg.eigh(self.socmat)
            self.eigenvals[:] = eigvals.real

            self.CorrectSocmat(ik, eigvecs)

            # C_{G, i} x S_{i, n}
            spinorpw = self.GetSpinorCoeff(ik, eigvecs)
            self.WriteSpinor(ik, spinorpw, wavecar_out=wavecar_out)

        if mixwave_purify:
            self.MixWavePurify(
                wavecar_out=wavecar_out,
                ibs=mixwave_ibs,
                ikpt=mixwave_ikpt,
                spin_balance_tol=mixwave_spin_balance_tol,
            )

        return True


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build SOC spinor WAVECAR from spinless/ISPIN=2 WAVECAR."
    )
    p.add_argument("--wavecar", default=DEFAULT_WAVECAR,
                   help=f"Input scalar WAVECAR path (default: {DEFAULT_WAVECAR}).")
    p.add_argument("--wavecar-out", default=DEFAULT_WAVECAR_OUT,
                   help=f"Output spinor WAVECAR path (default: {DEFAULT_WAVECAR_OUT}).")
    p.add_argument("--lsorbit", action="store_true",
                   help="Interpret input WAVECAR as LSORBIT when reading (normally off).")
    p.add_argument("--no-overwrite", dest="overwrite", action="store_false", default=True,
                   help="Do not overwrite existing output; skip build when output already exists and passes header-size checks.")
    p.add_argument("--correct-kpts", nargs="*", default=[],
                   metavar="KPT",
                   help=("Apply CorrectSocmat/ReCombineAB on selected 1-based k-point indices. "
                         "Accepts space/comma-separated values, e.g. --correct-kpts 1 3 or 1,3."))
    p.add_argument("--no-mixwave", dest="mixwave_purify", action="store_false",
                   help="Disable mixWave-based post purification.")
    p.add_argument("--mixwave-ikpt", type=int, default=DEFAULT_MIXWAVE_IKPT,
                   help=f"1-based k-point index for mixWave purification (default: {DEFAULT_MIXWAVE_IKPT}).")
    p.add_argument("--mixwave-ibs", nargs="*", default=None,
                   metavar="IB",
                   help=("Band-pair starts (ib, ib+1) for mixWave purification. "
                         "Accepts space/comma-separated values."))
    p.add_argument("--mixwave-spin-balance-tol", type=float, default=DEFAULT_SPIN_BALANCE_TOL,
                   help=f"Tolerance on |a11+a22| to use direct mixWave branch (default: {DEFAULT_SPIN_BALANCE_TOL:g}).")
    p.add_argument("--full-kpts", action="store_true",
                   help=("Apply CorrectSocmat and mixWave purification on all k-points. "
                         "Band pairs still follow --mixwave-ibs."))
    p.add_argument("--full-bands", action="store_true",
                   help=("Apply mixWave purification to all odd/even band pairs "
                         "(1,3,5,...,2*NBANDS-1)."))
    p.add_argument("--full-kpts-full-bands", action="store_true",
                   help=("Equivalent to --full-kpts --full-bands."))
    return p


def _resolve_mixwave_targets(
    maker: socclass,
    *,
    full_kpts: bool,
    full_bands: bool,
    mixwave_ikpt: int,
    mixwave_ibs: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if full_kpts:
        maker.read_in_kpts = list(range(maker.nkpts))
        mixwave_ikpts = tuple(range(1, maker.nkpts + 1))
    else:
        mixwave_ikpts = (mixwave_ikpt,)

    if full_bands:
        mixwave_ibs_eff = tuple(range(1, maker.twonbnds, 2))
    else:
        mixwave_ibs_eff = mixwave_ibs
    return mixwave_ikpts, mixwave_ibs_eff


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    full_kpts = bool(args.full_kpts or args.full_kpts_full_bands)
    full_bands = bool(args.full_bands or args.full_kpts_full_bands)

    if (args.mixwave_ibs is None) and (not full_bands):
        parser.print_help()
        return

    kpts_1based = _parse_int_tokens(args.correct_kpts)
    correct_kpts_0based = _to_zero_based_indices(kpts_1based, name="--correct-kpts")

    mixwave_ibs = _parse_int_tokens(args.mixwave_ibs or [])
    if (not mixwave_ibs) and (not full_bands):
        raise ValueError("Empty --mixwave-ibs is not allowed.")

    maker = socclass(args.wavecar, lsorbit=args.lsorbit, correct_kpts=correct_kpts_0based)
    mixwave_ikpts, mixwave_ibs_eff = _resolve_mixwave_targets(
        maker,
        full_kpts=full_kpts,
        full_bands=full_bands,
        mixwave_ikpt=args.mixwave_ikpt,
        mixwave_ibs=mixwave_ibs,
    )

    maker.MakeSpinor(
        wavecar_out=args.wavecar_out,
        overwrite=args.overwrite,
        mixwave_purify=False,
        mixwave_ibs=mixwave_ibs_eff,
        mixwave_ikpt=args.mixwave_ikpt,
        mixwave_spin_balance_tol=args.mixwave_spin_balance_tol,
    )

    if args.mixwave_purify:
        for ikpt_now in mixwave_ikpts:
            maker.MixWavePurify(
                wavecar_out=args.wavecar_out,
                ibs=mixwave_ibs_eff,
                ikpt=ikpt_now,
                spin_balance_tol=args.mixwave_spin_balance_tol,
            )


if __name__ == "__main__":
    main()
