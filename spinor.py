#!/usr/bin/env python3

from typing import BinaryIO
import numpy as np
from numpy.typing import NDArray
from vaspwfc import vaspwfc
from spinorb import paw_core_soc_mat, \
                    read_cproj_NormalCar


class SpinorMaker:
    def __init__(self, wavecar: str="WAVECAR", *,
                 lsorbit: bool=False):
        wfn = vaspwfc(wavecar, lsorbit=lsorbit)
        assert wfn._lsoc == False, "This WAVECAR contains spinors already"
        assert wfn._nplws[0] == wfn.gvectors(ikpt=1).shape[0], "This script do not support WAVECAR with _gam or _ncl yet."

        self.nspin     = wfn._nspin
        self.nkpoint   = wfn._nkpts
        self.nbands    = wfn._nbands
        self.band_eigs = wfn._bands
        self.band_occs = wfn._occs
        self.kvecs     = wfn._kvecs
        self.nplws     = wfn._nplws
        self.wfn  = wfn

        self.generate_wavecar_header()
        self.calc_Hmm(nspin=self.nspin, nkpoint=self.nkpoint, nbands=self.nbands)
        return


    def calc_Hmm(self, *, nspin: int=1, nkpoint: int=1, nbands: int):
        """
        Calculate spin-orbit coupling matrix:

        returns: [nkpoint, nbands, nbands], where each [nbands, nbands] be like
                     [[up-up, up-dn]                ]
                 H = [[dn-up, dn-dn],     ...       ]
                     [                              ]
                     [    ....            ...       ]
        """
        socmat = paw_core_soc_mat()     # [4, nproj, nproj]
        cprojs = read_cproj_NormalCar() # [nspin*nkpoint*nbands, nproj]

        assert socmat.shape[-1] == cprojs.shape[-1], "No. of projectors from SocCar and NormalCAR not consistent."
        assert cprojs.shape[0] == nspin * nkpoint * nbands, "No. of bands in SocCar and input not consistent."

        nproj = socmat.shape[-1]
        cprojs = cprojs.reshape((nspin, nkpoint, nbands, nproj))

        # for ISPIN=1 system, expand cprojs that cprojs[1,...] == cprojs[0,...]
        s = 0 if 1 == nspin else nspin

        ibs = np.mgrid[0:2*nbands]
        Hmm = np.zeros((nkpoint, 2*nbands, 2*nbands), dtype=np.complex128)
        nb = nbands
        for ikpoint in range(nkpoint):
            Hmm[ikpoint, :nb , :nb ] = cprojs[0, ikpoint, :, :].conj() @ socmat[0, :, :] @ cprojs[0, ikpoint, :, :].T
            Hmm[ikpoint, :nb ,  nb:] = cprojs[0, ikpoint, :, :].conj() @ socmat[1, :, :] @ cprojs[s, ikpoint, :, :].T
            Hmm[ikpoint,  nb:, :nb ] = cprojs[s, ikpoint, :, :].conj() @ socmat[2, :, :] @ cprojs[0, ikpoint, :, :].T
            Hmm[ikpoint,  nb:,  nb:] = cprojs[s, ikpoint, :, :].conj() @ socmat[3, :, :] @ cprojs[s, ikpoint, :, :].T

            # Add the band eigenvalue to the diagonal of Hmm
            eigs = self.band_eigs[:, ikpoint, :].flatten()
            if nspin == 1:
                eigs = np.hstack((eigs, eigs))
            Hmm[ikpoint, ibs, ibs] += eigs

            assert np.allclose(
                    np.abs(Hmm[ikpoint, :, :] - Hmm[ikpoint, :, :].T.conj()),
                    np.zeros((2*nbands, 2*nbands)),
                   ), "Hmm not Hermitian."

        self.Hmm = Hmm
        return


    @staticmethod
    def recombine_spinor(A: NDArray, B: NDArray) -> tuple[NDArray, NDArray]:
        """
        Construct the spin-polarized states using SOC matrix using Lagrange multiplier.

        Input: A = [2N] = [2,N]
               B = [2N] = [2,N]

        Solve the matrix:

        [   <a|a>,        0,  Re<a|b>, -Im<a|b>] [x1]   [x1]
        [       0,    <a|a>,  Im<a|b>,  Re<a|b>] [y1] = [y1]
        [ Re<a|b>,  Im<a|b>,    <b|b>,        0] [x2]   [x2]
        [-Im<a|b>,  Re<a|b>,        0,    <b|b>] [y2]   [y2]

        to make   |z1*A[0,:] + z2*B[0,:]| -> maximum
            and   |z3*A[1,:] + z4*B[1,:]| -> maximum

        A and B are column vectors with length of `2N`
        """
        assert len(A.shape) == 1 and A.shape == B.shape
        N = A.shape[0] // 2

        A = A.reshape(2, N)
        B = B.reshape(2, N)

        ret = [np.zeros(1), np.zeros(1)]    # make pyright happy
        idiag = np.mgrid[0:4]

        for ispin in range(2):
            # Lagrange multiplier, matrix is
            nrmsqr_a = np.linalg.norm(A[ispin,:])**2    # <a|a>
            nrmsqr_b = np.linalg.norm(B[ispin,:])**2
            olap_ab  = A[ispin,:].conj() @ B[ispin,:]   # <a|b>

            lmat = np.zeros((4, 4), dtype=float)
            lmat[idiag, idiag] = np.array([nrmsqr_a, nrmsqr_a, nrmsqr_b, nrmsqr_b])

            lmat[0,2] = olap_ab.real
            lmat[1,3] = olap_ab.real
            lmat[2,0] = olap_ab.real
            lmat[3,1] = olap_ab.real

            lmat[0,3] = olap_ab.conj().imag
            lmat[1,2] = olap_ab.imag
            lmat[2,1] = olap_ab.imag
            lmat[3,0] = olap_ab.conj().imag

            _eigvals, eigvecs = np.linalg.eigh(lmat)
            minf = np.inf
            mind = 0
            for ii in range(4):
                z1 = eigvecs[0,ii] + 1j*eigvecs[1,ii]
                z2 = eigvecs[2,ii] + 1j*eigvecs[3,ii]
                imin = np.linalg.norm(z1 * A[ispin,:] + z2 * B[ispin,:])
                if imin < minf:
                    minf = imin
                    mind = ii
                pass

            z1 = eigvecs[0,mind] + 1j*eigvecs[1,mind]
            z2 = eigvecs[2,mind] + 1j*eigvecs[3,mind]

            ret[ispin] = z1 * A.flatten() + z2 * B.flatten()

        rA, rB = ret
        return (rA, rB)


    def make_spinor(self, ikpoint: int) -> tuple[NDArray, NDArray, NDArray]:
        """
        Make the spinor from ISPIN=1/2 system by diagonalizing the spin-orbit coupling (SOC) matrix.

        `eigvals, eigvecs = np.linalg.eigh(Hmm)`

        where column vectors `eigvecs[:,n]` is the eigen vectors.

        Then spinors = MATMUL(eigvecs.T, band_coeffs)

        Return: (eigvals, occs, spinor_coeffs)
        """

        assert 1 <= ikpoint and ikpoint <= self.nkpoint

        nspin  = self.nspin
        nbands = self.nbands
        wfn    = self.wfn
        ibs = np.mgrid[0:2*nbands]

        # spin-orbit coupling matrix, [2*nbands, 2*nbands]
        Hmm = self.Hmm[ikpoint-1, :, :]

        nplw = wfn._nplws[ikpoint - 1]

        # diagonalize Hmm to get new band eigenvalues
        # finally correct it
        eigvals, eigvecs = np.linalg.eigh(Hmm)

        # sort the eigvals and eigvecs in ascending order
        sorted_idx = np.argsort(eigvals)
        eigvals = eigvals[sorted_idx]
        eigvecs = eigvecs[:, sorted_idx]

        # once the eigvals and eigvecs are sorted, the spin-up and spin-down
        # bands are interleaving
        eigvals[0::2] = (eigvals[0::2] + eigvals[1::2]) / 2.0
        eigvals[1::2] = eigvals[0::2]

        ispin = min(nspin, 2)   # for nspin = 1 system, the ispin=2 channel is same as ispin=1
        occs = np.zeros(2*nbands)
        occs[0::2] = self.band_occs[0,       ikpoint-1, :]
        occs[1::2] = self.band_occs[ispin-1, ikpoint-1, :]
        occs[0::2] = (occs[0::2] + occs[1::2]) / 2.0
        occs[1::2] = occs[0::2]

        spinor_coeffs = np.zeros((2*nbands, 2*nplw), dtype=np.complex128)

        for iband in ibs[0::2]:
            rA, rB = SpinorMaker.recombine_spinor(eigvecs[:,iband], eigvecs[:, iband+1])
            eigvecs[:, iband  ] = rA
            eigvecs[:, iband+1] = rB

            
            # spinor up
            spinor_coeffs[iband,   :nplw ] = wfn.readBandCoeff(ispin=1,     ikpt=ikpoint, iband=iband//2 + 1)
            # spinor down
            spinor_coeffs[iband+1,  nplw:] = wfn.readBandCoeff(ispin=ispin, ikpt=ikpoint, iband=iband//2 + 1)

        # Produce the real spinors using the eigen vectors
        return (eigvals, occs, eigvecs.T @ spinor_coeffs)


    def generate_wavecar_header(self):
        # 1st record
        recl  = 2 * self.nplws.max() * 8
        rdum  = float(recl) # record length, double the old length
        nspin = 1.0
        rtag  = 45200.0     # store the complex<f32> version of planewaves

        # 2nd record
        nkpoint = float(self.nkpoint)
        nbands  = float(self.nbands * 2)
        encut   = float(self.wfn._encut)
        Acell   = self.wfn._Acell
        efermi  = self.wfn._efermi

        # number of plane waves for each k-point
        nplws   = self.wfn._nplws * 2

        self.spinor_recl    = recl
        self.spinor_rdum    = rdum
        self.spinor_nspin   = nspin
        self.spinor_rtag    = rtag
        self.spinor_nkpoint = nkpoint
        self.spinor_nbands  = nbands
        self.spinor_encut   = encut
        self.spinor_Acell   = Acell
        self.spinor_efermi  = efermi
        self.spinor_nplws   = nplws
        pass


    def write_wavecar_header(self, fname: str="WAVECAR_spinor"):
        """
        Prepare to write the spinor to new WAVECAR with lsorbit=True.
        """

        # 1st record
        recl    = self.spinor_recl
        rdum    = self.spinor_rdum
        nspin   = self.spinor_nspin
        rtag    = self.spinor_rtag

        # 2nd record
        nkpoint = self.spinor_nkpoint
        nbands  = self.spinor_nbands
        encut   = self.spinor_encut
        Acell   = self.spinor_Acell
        efermi  = self.spinor_efermi

        with open(fname, "wb") as f:
            f.seek(0)
            np.array(
                [rdum, nspin, rtag], dtype=np.float64
            ).tofile(f)

            f.seek(recl)
            np.array(
                [nkpoint, nbands, encut, *Acell.flatten(), efermi], dtype=np.float64
            ).tofile(f)


    def check_wavecar_header(self, fname: str):
        f = open(fname, "rb")
        f.seek(0)
        rdum, nspin, rtag = np.fromfile(f, dtype=np.float64, count=3)
        recl = int(rdum)
        assert self.spinor_recl  == recl  and \
               self.spinor_nspin == nspin and \
               self.spinor_rtag  == rtag, \
               "Inconsistent WAVECAR with current system."

        f.seek(recl)
        dump = np.fromfile(f, dtype=np.float64, count=13)
        nkpoint = dump[0]
        nbands  = dump[1]
        encut   = dump[2]
        Acell   = dump[3:12].reshape(3, 3)
        efermi  = dump[12]
        assert self.spinor_nkpoint == nkpoint and \
               self.spinor_nbands  == nbands  and \
               self.spinor_encut   == encut   and \
               np.allclose(self.spinor_Acell, Acell) and \
               self.spinor_efermi  == efermi, \
               "Inconsistent WAVECAR with current system."
        pass


    def get_spinor_irec(self, *, ikpoint: int, iband: int):
        assert 1 <= ikpoint and ikpoint <= int(self.spinor_nkpoint)
        assert 1 <= iband   and iband   <= int(self.spinor_nbands)

        nbands  = int(self.spinor_nbands)

        # Only one spin for ncl system, thus no relation to ispin
        rec = 2 + (ikpoint - 1) * (nbands + 1) + iband
        return rec


    def write_wavecar_spinors(self, fname: str="WAVECAR_spinor", *,
                              ikpoint: int,
                              eigs: NDArray,
                              occs: NDArray,
                              spinors: NDArray):
        """
        ikpoint starts from 1.
        """

        # aux function
        nbands = int(self.spinor_nbands)
        nplw   = int(self.spinor_nplws[ikpoint-1])
        assert eigs.shape == (nbands,) and \
               eigs.shape == occs.shape and \
               spinors.shape == (nbands, nplw), \
               "Provided eigs or occs or spinor plane waves not consistent with current system."

        self.check_wavecar_header(fname)
        f = open(fname, "rb+")
        recl = int(self.spinor_recl)

        # write the kpoint-vector, band-eigs and occs before the plane waves
        irec = self.get_spinor_irec(ikpoint=ikpoint, iband=1)
        f.seek((irec - 1) * recl)

        np.array([float(nplw), *self.kvecs[ikpoint-1,:]], dtype=np.float64).tofile(f)
        buf = np.zeros((nbands, 3), dtype=np.float64)
        buf[:,0] = eigs
        buf[:,2] = occs
        buf.tofile(f)

        # write the plane waves
        for iband in range(nbands):
            irec = self.get_spinor_irec(ikpoint=ikpoint, iband=iband+1)
            f.seek(irec * recl)
            spinors[iband,:].astype(np.complex64).tofile(f)


    def write_all_spinors(self, fname: str="WAVECAR_spinor"):
        nkpoints = self.nkpoint

        self.write_wavecar_header(fname)
        for ikpoint in range(1, nkpoints+1):
            eigs, occs, spinor_coeffs = self.make_spinor(ikpoint)
            self.write_wavecar_spinors(fname,
                                       ikpoint=ikpoint,
                                       eigs=eigs,
                                       occs=occs,
                                       spinors=spinor_coeffs)


if '__main__' == __name__:
    sm = SpinorMaker()
    sm.write_all_spinors()
    pass
