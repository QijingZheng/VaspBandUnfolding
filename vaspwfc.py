#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from math import sqrt
from vasp_constant import *
from multiprocessing import cpu_count
from scipy.fftpack import fftfreq, fftn, ifftn

############################################################


def save2vesta(phi=None, poscar='POSCAR', prefix='wfc',
               lgam=False, lreal=False, ncol=10):
    '''
    Save the real space pseudo-wavefunction as vesta format.
    '''
    nx, ny, nz = phi.shape
    try:
        pos = open(poscar, 'r')
        head = ''
        for line in pos:
            if line.strip():
                head += line
            else:
                break
        head += '\n%5d%5d%5d\n' % (nx, ny, nz)
    except:
        raise IOError('Failed to open %s' % poscar)

    # Faster IO
    nrow = phi.size // ncol
    nrem = phi.size % ncol
    fmt = "%16.8E"

    psi = phi.copy()
    psi = psi.flatten(order='F')
    psi_h = psi[:nrow * ncol].reshape((nrow, ncol))
    psi_r = psi[nrow * ncol:]

    with open(prefix + '_r.vasp', 'w') as out:
        out.write(head)
        out.write(
            '\n'.join([''.join([fmt % xx for xx in row])
                       for row in psi_h.real])
        )
        out.write("\n" + ''.join([fmt % xx for xx in psi_r.real]))
    if not (lgam or lreal):
        with open(prefix + '_i.vasp', 'w') as out:
            out.write(head)
            out.write(
                '\n'.join([''.join([fmt % xx for xx in row])
                           for row in psi_h.imag])
            )
            out.write("\n" + ''.join([fmt % xx for xx in psi_r.imag]))

############################################################


class vaspwfc(object):
    '''
    Class for processing VASP Pseudowavefunction stored in WAVECAR.  This
    program is motivated by PIESTA written by Ren Hao <renh@upc.edu.cn>.

    The format of VASP WAVECAR, as shown in
        http://www.andrew.cmu.edu/user/feenstra/wavetrans/
    is:
        Record-length #spin components RTAG(a value specifying the precision)
        #k-points #bands ENCUT(maximum energy for plane waves)
        LatVec-A
        LatVec-B
        LatVec-C
        Loop over spin
           Loop over k-points
              #plane waves, k vector
              Loop over bands
                 band energy, band occupation
              End loop over bands
              Loop over bands
                 Loop over plane waves
                    Plane-wave coefficient
                 End loop over plane waves
              End loop over bands
           End loop over k-points
        End loop over spin
    '''

    def __init__(self, fnm='WAVECAR', lsorbit=False, lgamma=False,
                 gamma_half='x', omp_num_threads=1):
        '''
        Initialization.
        '''

        self._fname = fnm
        # the directory containing the input file
        self._dname = os.path.dirname(fnm)
        if self._dname == '':
            self._dname = '.'

        self._lsoc = lsorbit
        self._lgam = lgamma
        self._gam_half = gamma_half.lower()

        # It seems that some modules in scipy uses OPENMP, it is therefore
        # desirable to set the OMP_NUM_THREADS to tune the parallization.
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

        assert not (lsorbit and lgamma), 'The two settings conflict!'
        assert self._gam_half == 'x' or self._gam_half == 'z', \
            'Gamma_half must be "x" or "z"'

        try:
            self._wfc = open(self._fname, 'rb')
        except:
            raise IOError('Failed to open %s' % self._fname)

        # read the basic information
        self.readWFHeader()
        # read the band information
        self.readWFBand()

        if self._lsoc:
            assert self._nspin == 1, "NSPIN = 1 for noncollinear version WAVECAR!"

    def set_omp_num_threads(self, nproc):
        '''
        Set the OMP_NUM_THREADS envrionment variable
        '''
        assert 1 <= nproc <= cpu_count()

        os.envrion['OMP_NUM_THREADS'] = str(nproc)

    def isSocWfc(self):
        """
        Is the WAVECAR from an SOC calculation?
        """
        return True if self._lsoc else False

    def isGammaWfc(self):
        """
        Is the WAVECAR from an SOC calculation?
        """
        return True if self._lgam else False

    def readWFHeader(self):
        '''
        Read the system information from WAVECAR, which is written in the first
        two record.

        rec1: recl, nspin, rtag
        rec2: nkpts, nbands, encut, ((cell(i,j) i=1, 3), j=1, 3)
        '''

        # goto the start of the file and read the first record
        self._wfc.seek(0)
        self._recl, self._nspin, self._rtag = np.array(
            np.fromfile(self._wfc, dtype=np.float64, count=3),
            dtype=np.int64
        )
        self._WFPrec = self.setWFPrec()
        # the second record
        self._wfc.seek(self._recl)
        dump = np.fromfile(self._wfc, dtype=np.float64, count=12)

        self._nkpts = int(dump[0])                     # No. of k-points
        self._nbands = int(dump[1])                     # No. of bands
        self._encut = dump[2]                          # Energy cutoff
        # real space supercell basis
        self._Acell = dump[3:].reshape((3, 3))
        # real space supercell volume
        self._Omega = np.linalg.det(self._Acell)
        # reciprocal space supercell volume
        self._Bcell = np.linalg.inv(self._Acell).T

        # Minimum FFT grid size
        Anorm = np.linalg.norm(self._Acell, axis=1)
        CUTOF = np.ceil(
            sqrt(self._encut / RYTOEV) / (TPI / (Anorm / AUTOA))
        )
        self._ngrid = np.array(2 * CUTOF + 1, dtype=int)

    def setWFPrec(self):
        '''
        Set wavefunction coefficients precision:
            TAG = 45200: single precision complex, np.complex64, or complex(qs)
            TAG = 45210: double precision complex, np.complex128, or complex(q)
        '''
        if self._rtag == 45200:
            return np.complex64
        elif self._rtag == 45210:
            return np.complex128
        elif self._rtag == 53300:
            raise ValueError("VASP5 WAVECAR format, not implemented yet")
        elif self._rtag == 53310:
            raise ValueError("VASP5 WAVECAR format with double precision "
                             + "coefficients, not implemented yet")
        else:
            raise ValueError("Invalid TAG values: {}".format(self._rtag))

    def readWFBand(self):
        '''
        Extract KS energies and Fermi occupations from WAVECAR.
        '''

        self._nplws = np.zeros(self._nkpts, dtype=int)
        self._kvecs = np.zeros((self._nkpts, 3), dtype=float)
        self._bands = np.zeros(
            (self._nspin, self._nkpts, self._nbands), dtype=float)
        self._occs = np.zeros(
            (self._nspin, self._nkpts, self._nbands), dtype=float)

        for ii in range(self._nspin):
            for jj in range(self._nkpts):
                rec = self.whereRec(ii+1, jj+1, 1) - 1
                self._wfc.seek(rec * self._recl)
                dump = np.fromfile(self._wfc, dtype=np.float64,
                                   count=4+3*self._nbands)
                if ii == 0:
                    self._nplws[jj] = int(dump[0])
                    self._kvecs[jj] = dump[1:4]
                dump = dump[4:].reshape((-1, 3))
                self._bands[ii, jj, :] = dump[:, 0]
                self._occs[ii, jj, :] = dump[:, 2]

        if self._nkpts > 1:
            tmp = np.linalg.norm(
                np.dot(np.diff(self._kvecs, axis=0), self._Bcell), axis=1)
            self._kpath = np.concatenate(([0, ], np.cumsum(tmp)))
        else:
            self._kpath = None
        return self._kpath, self._bands

    def get_kpath(self, nkseg=None):
        '''
        Construct k-point path, find out the k-path boundary if possible.

        nkseg is the number of k-points in each k-path segments.
        '''

        if nkseg is None:
            if os.path.isfile(self._dname + "/KPOINTS"):
                kfile = open(self._dname + "/KPOINTS").readlines()
                if kfile[2][0].upper() == 'L':
                    nkseg = int(kfile[1].split()[0])
                else:
                    raise ValueError(
                        'Error reading number of k-points from KPOINTS')
        assert nkseg > 0

        nsec = self._nkpts // nkseg

        v = self._kvecs.copy()
        for ii in range(nsec):
            ki = ii * nkseg
            kj = (ii + 1) * nkseg
            v[ki:kj, :] -= v[ki]

        self._kpath = np.linalg.norm(np.dot(v, self._Bcell), axis=1)
        for ii in range(1, nsec):
            ki = ii * nkseg
            kj = (ii + 1) * nkseg
            self._kpath[ki:kj] += self._kpath[ki - 1]

            self._kbound = np.concatenate(
                (self._kpath[0::nkseg], [self._kpath[-1], ]))

        return self._kpath, self._kbound

    def gvectors(self, ikpt=1, force_Gamma=False, check_consistency=True):
        '''
        Generate the G-vectors that satisfies the following relation
            (G + k)**2 / 2 < ENCUT
        '''
        assert 1 <= ikpt <= self._nkpts,  'Invalid kpoint index!'

        kvec = self._kvecs[ikpt-1]
        # force_Gamma: consider gamma-only case regardless of the actual setting
        lgam = True if force_Gamma else self._lgam

        # fx, fy, fz = [fftfreq(n) * n for n in self._ngrid]
        # fftfreq in scipy.fftpack is a little different with VASP frequencies
        ############################################################
        # Gamma version -50% memory usage and 1x speed.
        ############################################################
        # fx = [ii if ii < self._ngrid[0] // 2 + 1 else ii - self._ngrid[0]
        #       for ii in range(
        #           self._ngrid[0] // 2 + 1
        #           if (lgam and (self._gam_half == 'x'))
        #           else
        #           self._ngrid[0])]
        # fy = [jj if jj < self._ngrid[1] // 2 + 1 else jj - self._ngrid[1]
        #       for jj in range(self._ngrid[1])]
        # fz = [kk if kk < self._ngrid[2] // 2 + 1 else kk - self._ngrid[2]
        #       for kk in range(
        #           self._ngrid[2] // 2 + 1
        #           if (lgam and (self._gam_half == 'z'))
        #           else
        #           self._ngrid[2])]

        fx, fy, fz = [np.arange(n, dtype=int) for n in self._ngrid]
        fx[self._ngrid[0] // 2 + 1:] -= self._ngrid[0]
        fy[self._ngrid[1] // 2 + 1:] -= self._ngrid[1]
        fz[self._ngrid[2] // 2 + 1:] -= self._ngrid[2]
        if lgam:
            if self._gam_half == 'x':
                fx = fx[:self._ngrid[0] // 2 + 1]
            else:
                fz = fz[:self._ngrid[2] // 2 + 1]

        # if lgam:
        #     # parallel gamma version of VASP WAVECAR exclude some planewave
        #     # components, -DwNGZHalf
        #     if self._gam_half == 'z':
        #         kgrid = np.array([(fx[ii], fy[jj], fz[kk])
        #                           for kk in range(self._ngrid[2])
        #                           for jj in range(self._ngrid[1])
        #                           for ii in range(self._ngrid[0])
        #                           if (
        #                               (fz[kk] > 0) or
        #                               (fz[kk] == 0 and fy[jj] > 0) or
        #                               (fz[kk] == 0 and fy[jj]
        #                                == 0 and fx[ii] >= 0)
        #         )], dtype=float)
        #     else:
        #         kgrid = np.array([(fx[ii], fy[jj], fz[kk])
        #                           for kk in range(self._ngrid[2])
        #                           for jj in range(self._ngrid[1])
        #                           for ii in range(self._ngrid[0])
        #                           if (
        #                               (fx[ii] > 0) or
        #                               (fx[ii] == 0 and fy[jj] > 0) or
        #                               (fx[ii] == 0 and fy[jj]
        #                                == 0 and fz[kk] >= 0)
        #         )], dtype=float)
        # else:
        #     kgrid = np.array([(fx[ii], fy[jj], fz[kk])
        #                       for kk in range(self._ngrid[2])
        #                       for jj in range(self._ngrid[1])
        #                       for ii in range(self._ngrid[0])], dtype=float)

        ############################################################
        # 10x faster
        ############################################################
        # In meshgrid, fx run the fastest, fz the slowest
        gz, gy, gx = np.array(
            np.meshgrid(fz, fy, fx, indexing='ij')
        ).reshape((3, -1))
        kgrid = np.array([gx, gy, gz], dtype=float).T
        if lgam:
            if self._gam_half == 'z':
                kgrid = kgrid[
                    (gz > 0) |
                    ((gz == 0) & (gy > 0)) |
                    ((gz == 0) & (gy == 0) & (gx >= 0))
                ]
            else:
                kgrid = kgrid[
                    (gx > 0) |
                    ((gx == 0) & (gy > 0)) |
                    ((gx == 0) & (gy == 0) & (gz >= 0))
                ]

        # Kinetic_Energy = (G + k)**2 / 2
        # HSQDTM    =  hbar**2/(2*ELECTRON MASS)
        KENERGY = HSQDTM * np.linalg.norm(
            np.dot(kgrid + kvec[np.newaxis, :], TPI*self._Bcell), axis=1
        )**2
        # find Gvectors where (G + k)**2 / 2 < ENCUT
        Gvec = kgrid[np.where(KENERGY < self._encut)[0]]

        # Check if the calculated number of planewaves and the one recorded in the
        # WAVECAR are equal
        if check_consistency:
            # if self._lsoc:
            #     assert Gvec.shape[0] == self._nplws[ikpt - 1] // 2, \
            #         'No. of planewaves not consistent for an SOC WAVECAR! %d %d %d' % \
            #         (Gvec.shape[0], self._nplws[ikpt - 1],
            #          np.prod(self._ngrid))
            # else:
            #     assert Gvec.shape[0] == self._nplws[ikpt - 1], 'No. of planewaves not consistent! %d %d %d' % \
            #         (Gvec.shape[0], self._nplws[ikpt - 1],
            #          np.prod(self._ngrid))

            if Gvec.shape[0] != self._nplws[ikpt - 1]:
                if Gvec.shape[0] * 2 == self._nplws[ikpt - 1]:
                    if not self._lsoc:
                        raise ValueError('''
                        It seems that you are reading a WAVECAR from a NONCOLLINEAR VASP.
                        Please set 'lsorbit = True' when loading the WAVECAR.
                        For example:

                            wfc = vaspwfc('WAVECAR', lsorbit=True)
                        ''')
                elif Gvec.shape[0] == 2 * self._nplws[ikpt - 1] - 1:
                    if not self._lgam:
                        raise ValueError('''
                        It seems that you are reading a WAVECAR from a GAMMA-ONLY VASP.  Please set
                        'lgamma = True' when loading the WAVECAR.  Moreover, you may want to set
                        "gamma_half" if you are using VASP version <= 5.2.x.  For VASP <= 5.2.x, check
                        which FFT VASP uses by the following command:

                            $ grep 'use.* FFT for wave' OUTCAR

                        Then

                            # for parallel FFT, VASP <= 5.2.x
                            wfc = vaspwfc('WAVECAR', lgamma=True, gamma_half='z')

                            # for serial FFT, VASP <= 5.2.x
                            wfc = vaspwfc('WAVECAR', lgamma=True, gamma_half='x')

                        For VASP >= 5.4, WAVECAR is written with x-direction half grid regardless of
                        parallel or serial FFT.

                            # "gamma_half" default to "x" for VASP >= 5.4
                            wfc = vaspwfc('WAVECAR', lgamma=True, gamma_half='x')
                        ''')
                else:
                    raise ValueError('''
                    NO. OF PLANEWAVES NOT CONSISTENT:

                        THIS CODE -> %d
                        FROM VASP -> %d
                           NGRIDS -> %d
                    ''' % (Gvec.shape[0],
                           self._nplws[ikpt - 1] // 2 if self._lsoc else self._nplws[ikpt - 1],
                           np.prod(self._ngrid))
                    )

        return np.asarray(Gvec, dtype=int)

    def save2vesta(self, phi=None, lreal=False, poscar='POSCAR', prefix='wfc',
                   ncol=10):
        '''
        Save the real space pseudo-wavefunction as vesta format.
        '''
        nx, ny, nz = phi.shape
        try:
            pos = open(poscar, 'r')
            head = ''
            for line in pos:
                if line.strip():
                    head += line
                else:
                    break
            head += '\n%5d%5d%5d\n' % (nx, ny, nz)
        except:
            raise IOError('Failed to open %s' % poscar)

        # Faster IO
        nrow = phi.size // ncol
        nrem = phi.size % ncol
        fmt = "%16.8E"

        psi = phi.copy()
        psi = psi.flatten(order='F')
        psi_h = psi[:nrow * ncol].reshape((nrow, ncol))
        psi_r = psi[nrow * ncol:]

        with open(prefix + '_r.vasp', 'w') as out:
            out.write(head)
            out.write(
                '\n'.join([''.join([fmt % xx for xx in row])
                           for row in psi_h.real])
            )
            out.write("\n" + ''.join([fmt % xx for xx in psi_r.real]))
        if not (self._lgam or lreal):
            with open(prefix + '_i.vasp', 'w') as out:
                out.write(head)
                out.write(
                    '\n'.join([''.join([fmt % xx for xx in row])
                               for row in psi_h.imag])
                )
                out.write("\n" + ''.join([fmt % xx for xx in psi_r.imag]))


    def get_ps_wfc(self, *args, **kwargs):
        '''
        Alias for the wfc_r method.
        '''

        return self.wfc_r(*args, **kwargs)


    def wfc_r(self, ispin=1, ikpt=1, iband=1,
              gvec=None, Cg=None, ngrid=None,
              rescale=None,
              norm=False, kr_phase=False, r0=[0.0, 0.0, 0.0]):
        '''
        Obtain the pseudo-wavefunction of the specified KS states in real space
        by performing FT transform on the reciprocal space planewave
        coefficients.  The 3D FT grid size is determined by ngrid, which
        defaults to self._ngrid if not given.  Gvectors of the KS states is used
        to put 1D planewave coefficients back to 3D grid.

        Inputs:
            ispin : spin index of the desired KS states, starting from 1
            ikpt  : k-point index of the desired KS states, starting from 1
            iband : band index of the desired KS states, starting from 1
            gvec  : the G-vectors correspond to the plane-wave coefficients
            Cg    : the plane-wave coefficients. If None, read from WAVECAR
            ngrid : the FFT grid size
            norm  : normalized Cg?
         kr_phase : whether or not to multiply the exp(ikr) phase
               r0 : shift of the kr-phase to get full wfc other than primitive cell

        The return wavefunctions are normalized in a way that

                        \sum_{ijk} | \phi_{ijk} | ^ 2 = 1

        '''
        self.checkIndex(ispin, ikpt, iband)

        if ngrid is None:
            ngrid = self._ngrid.copy() * 2
        else:
            ngrid = np.array(ngrid, dtype=int)
            assert ngrid.shape == (3,)
            assert np.alltrue(ngrid >= self._ngrid), \
                "Minium FT grid size: (%d, %d, %d)" % \
                (self._ngrid[0], self._ngrid[1], self._ngrid[2])

        # By default, the WAVECAR only stores the periodic part of the Bloch
        # wavefunction. In order to get the full Bloch wavefunction, one need to
        # multiply the periodic part with the phase: exp(i k (r + r0). Below, the
        # k-point vector and the real-space grid are both in the direct
        # coordinates.
        if kr_phase:
            phase = np.exp(1j * np.pi * 2 *
                           np.sum(
                               self._kvecs[ikpt-1] *
                               (
                                   # r
                                   np.mgrid[
                                       0:ngrid[0], 0:ngrid[1], 0:ngrid[2]
                                   ].reshape((3, np.prod(ngrid))).T /
                                   ngrid.astype(float) +
                                   # r0
                                   np.array(r0, dtype=float)
                               ),
                               axis=1
                           )).reshape(ngrid)
        else:
            phase = 1.0

        # The default normalization of np.fft.fftn has the direct transforms
        # unscaled and the inverse transforms are scaled by 1/n. It is possible
        # to obtain unitary transforms by setting the keyword argument norm to
        # "ortho" (default is None) so that both direct and inverse transforms
        # will be scaled by 1/\sqrt{n}.

        # default normalization factor so that
        # \sum_{ijk} | \phi_{ijk} | ^ 2 = 1
        normFac = rescale if rescale is not None else np.sqrt(np.prod(ngrid))

        if gvec is None:
            gvec = self.gvectors(ikpt)

        if self._lgam:
            if self._gam_half == 'z':
                phi_k = np.zeros(
                    (ngrid[0], ngrid[1], ngrid[2]//2 + 1), dtype=np.complex128)
            else:
                phi_k = np.zeros(
                    (ngrid[0]//2 + 1, ngrid[1], ngrid[2]), dtype=np.complex128)
        else:
            phi_k = np.zeros(ngrid, dtype=np.complex128)

        gvec %= ngrid[np.newaxis, :]

        if self._lsoc:
            wfc_spinor = []
            if Cg:
                dump = Cg
            else:
                dump = self.readBandCoeff(ispin, ikpt, iband, norm)
            nplw = dump.shape[0] // 2

            # spinor up
            phi_k[gvec[:, 0], gvec[:, 1], gvec[:, 2]] = dump[:nplw]
            wfc_spinor.append(ifftn(phi_k) * normFac * phase)
            # spinor down
            phi_k[:, :, :] = 0.0j
            phi_k[gvec[:, 0], gvec[:, 1], gvec[:, 2]] = dump[nplw:]
            wfc_spinor.append(ifftn(phi_k) * normFac * phase)

            del dump
            return wfc_spinor

        else:
            if Cg is not None:
                phi_k[gvec[:, 0], gvec[:, 1], gvec[:, 2]] = Cg
            else:
                phi_k[gvec[:, 0], gvec[:, 1], gvec[:, 2]
                      ] = self.readBandCoeff(ispin, ikpt, iband, norm)

            if self._lgam:
                # add some components that are excluded and perform c2r FFT
                if self._gam_half == 'z':
                    for ii in range(ngrid[0]):
                        for jj in range(ngrid[1]):
                            fx = ii if ii < ngrid[0] // 2 + \
                                1 else ii - ngrid[0]
                            fy = jj if jj < ngrid[1] // 2 + \
                                1 else jj - ngrid[1]
                            if (fy > 0) or (fy == 0 and fx >= 0):
                                continue
                            phi_k[ii, jj, 0] = phi_k[-ii, -jj, 0].conjugate()

                    # VASP add a factor of SQRT2 for G != 0 in Gamma-only VASP
                    phi_k /= np.sqrt(2.)
                    phi_k[0, 0, 0] *= np.sqrt(2.)
                    return np.fft.irfftn(phi_k, s=ngrid) * normFac
                elif self._gam_half == 'x':
                    for jj in range(ngrid[1]):
                        for kk in range(ngrid[2]):
                            fy = jj if jj < ngrid[1] // 2 + \
                                1 else jj - ngrid[1]
                            fz = kk if kk < ngrid[2] // 2 + \
                                1 else kk - ngrid[2]
                            if (fy > 0) or (fy == 0 and fz >= 0):
                                continue
                            phi_k[0, jj, kk] = phi_k[0, -jj, -kk].conjugate()

                    phi_k /= np.sqrt(2.)
                    phi_k[0, 0, 0] *= np.sqrt(2.)
                    phi_k = np.swapaxes(phi_k, 0, 2)
                    tmp = np.fft.irfftn(
                        phi_k, s=(ngrid[2], ngrid[1], ngrid[0])) * normFac
                    return np.swapaxes(tmp, 0, 2)
            else:
                # perform complex2complex FFT
                return ifftn(phi_k * normFac) * phase

    def poisson(self, rho=None, iband=1, ikpt=1, ispin=1, ngrid=None, norm=False):
        """
        Given a charge density "rho", solve the Poisson equation with periodic
        boundary condition to find out the corresponding electric potential and
        field.

        When "rho" is None, construct the charge density from a chosen Kohn-Sham
        state, i.e. rho(r) = phi_n(r).conj() * phi_n(r).

        In SI units, the real space Poisson equation:

                    \nabla^2 V = - \rho / \varepsilon_0
                             E = - \nabla V

        the reciprocal space Poisson equation:

                    G**2 * V_q = - rho_q / \varepsilon_0
                           E_q = -1j * G * V_q

        Note that the G=(0,0,0) entry is set to 1.0 instead of 0 to avoid
        divergence.
        """

        if rho is not None:
            rho = np.asarray(rho)
            ngrid = np.array(rho.shape, dtype=int)
            assert ngrid.shape == (3,)
        else:
            ngrid = self._ngrid * 2
            # normalization factor so that
            # \sum_{ijk} | \phi_{ijk} | ^ 2 * volume / Ngrid = 1
            normFac = np.prod(ngrid) / self._Omega
            if self._lsoc:
                rho = np.zeros(ngrid, dtype=float)
                phi_spinor = self.wfc_r(iband=iband, ikpt=ikpt, ispin=ispin,
                                        ngrid=ngrid, norm=norm)
                # negative charges, hence the minus sign
                for phi in phi_spinor:
                    rho += -(phi.conj() * phi).real * normFac
            else:
                phi = self.wfc_r(iband=iband, ikpt=ikpt, ispin=ispin,
                                 ngrid=ngrid, norm=norm)
                # negative charges, hence the minus sign
                rho = -(phi.conj() * phi).real * normFac

        fx = [ii if ii < ngrid[0] // 2 + 1 else ii - ngrid[0]
              for ii in range(ngrid[0])]
        fy = [jj if jj < ngrid[1] // 2 + 1 else jj - ngrid[1]
              for jj in range(ngrid[1])]
        fz = [kk if kk < ngrid[2] // 2 + 1 else kk - ngrid[2]
              for kk in range(ngrid[2])]

        # plane-waves: Reciprocal coordinate
        # indexing = 'ij' so that outputs are of shape (ngrid[0], ngrid[1], ngrid[2])
        Dx, Dy, Dz = np.meshgrid(fx, fy, fz, indexing='ij')
        # plane-waves: Cartesian coordinate
        Gx, Gy, Gz = np.tensordot(
            self._Bcell * np.pi * 2, [Dx, Dy, Dz], axes=(0, 0))
        # the norm squared of the G-vectors
        G2 = Gx**2 + Gy**2 + Gz**2
        # Note that the G=(0,0,0) entry is set to 1.0 instead of 0.
        G2[0, 0, 0] = 1.0

        # permittivity of vacuum [F / m]
        _eps0 = 8.85418781762039E-12
        # charge of one electron, in unit of Coulomb [1F * 1V]
        _e = 1.6021766208E-19

        # charge density in reciprocal space, rho in unit of [Coulomb / Angstrom**3]
        rho_q = np.fft.fftn(1E10 * _e * rho / _eps0, norm='ortho')
        # the electric potential in reciprocal space
        # V_q = -rho_q / (-G2)
        V_q = rho_q / G2
        # the electric potential in real space in unit of 'Volt'
        V_r = np.fft.ifftn(V_q, norm='ortho').real
        # the electric field in x/y/z in real space in unit of 'Volt / Angstrom'
        E_x = np.fft.ifftn(-1j * Gx * V_q, norm='ortho').real
        E_y = np.fft.ifftn(-1j * Gy * V_q, norm='ortho').real
        E_z = np.fft.ifftn(-1j * Gz * V_q, norm='ortho').real

        return rho, V_r, E_x, E_y, E_z

    def readBandCoeff(self, ispin=1, ikpt=1, iband=1, norm=False):
        '''
        Read the planewave coefficients of specified KS states.
        '''

        self.checkIndex(ispin, ikpt, iband)

        rec = self.whereRec(ispin, ikpt, iband)
        self._wfc.seek(rec * self._recl)

        nplw = self._nplws[ikpt - 1]
        dump = np.fromfile(self._wfc, dtype=self._WFPrec, count=nplw)

        cg = np.asarray(dump, dtype=np.complex128)
        if norm:
            cg /= np.linalg.norm(cg)
        return cg

    def whereRec(self, ispin=1, ikpt=1, iband=1):
        '''
        Return the rec position for specified KS state.
        '''

        self.checkIndex(ispin, ikpt, iband)

        rec = 2 + (ispin - 1) * self._nkpts * (self._nbands + 1) + \
                  (ikpt - 1) * (self._nbands + 1) + \
            iband
        return rec

    def checkIndex(self, ispin, ikpt, iband):
        '''
        Check if the index is valid!
        '''
        assert 1 <= ispin <= self._nspin,  'Invalid spin index!'
        assert 1 <= ikpt <= self._nkpts,  'Invalid kpoint index!'
        assert 1 <= iband <= self._nbands, 'Invalid band index!'


    def TransitionDipoleMoment(self, ks_i, ks_j):
        '''
        '''
        return self.get_dipole_mat(ks_i, ks_j)

    def get_dipole_mat(self, ks_i, ks_j):
        '''
        Dipole transition within the electric dipole approximation (EDA).
        Please refer to this post for more details.

          https://qijingzheng.github.io/posts/Light-Matter-Interaction-and-Dipole-Transition-Matrix/

        The dipole transition matrix elements in the length gauge is given by:

                <psi_nk | e r | psi_mk>

        where | psi_nk > is the pseudo-wavefunction.  In periodic systems, the
        position operator "r" is not well-defined.  Therefore, we first evaluate
        the momentum operator matrix in the velocity gauge, i.e.

                <psi_nk | p | psi_mk>

        And then use simple "p-r" relation to apprimate the dipole transition
        matrix element

                                          -i⋅h
            <psi_nk | r | psi_mk> =  -------------- ⋅ <psi_nk | p | psi_mk>
                                       m⋅(En - Em)

        Apparently, the above equaiton is not valid for the case Em == En. In
        this case, we just set the dipole matrix element to be 0.

        ################################################################################
        NOTE that, the simple "p-r" relation only applies to molecular or finite
        system, and there might be problem in directly using it for periodic
        system. Please refer to this paper for more details.

          "Relation between the interband dipole and momentum matrix elements in
          semiconductors"
          (https://journals.aps.org/prb/pdf/10.1103/PhysRevB.87.125301)

        ################################################################################
        '''

        # ks_i and ks_j are list containing spin-, kpoint- and band-index of the
        # initial and final states
        assert len(ks_i) == len(ks_j) == 3, 'Must be three indexes!'
        assert ks_i[1] == ks_j[1], 'k-point of the two states differ!'
        self.checkIndex(*ks_i)
        self.checkIndex(*ks_j)

        # energy differences between the two states
        Emk = self._bands[ks_i[0]-1, ks_i[1]-1, ks_i[2]-1]
        Enk = self._bands[ks_j[0]-1, ks_j[1]-1, ks_j[2]-1]
        dE = Enk - Emk

        # if energies of the initial and final states are the same, set the
        # dipole transition moment zero.
        if np.allclose(dE, 0.0):
            return 0.0

        moment_mat = self.get_moment_mat(ks_i, ks_j)
        dipole_mat = -1j / (dE / (2*RYTOEV)) * moment_mat * AUTOA * AUTDEBYE

        return Emk, Enk, dE, dipole_mat

    def get_moment_mat(self, ks_i, ks_j):
        '''
        The momentum operator matrix between the pseudo-wavefunction in the
        velocity gauge

            <psi_nk | p | psi_mk> = hbar <u_nk | k - i nabla | u_mk>

        The nabla operator matrix elements between the pseudo-wavefuncitons

            <u_nk | k - i nabla | u_mk>

           = \sum_G C_nk(G).conj() * C_mk(G) * [k + G]

        where C_nk(G) is the plane-wave coefficients for | u_nk >.
        '''

        # ks_i and ks_j are list containing spin-, kpoint- and band-index of the
        # initial and final states
        assert len(ks_i) == len(ks_j) == 3, 'Must be three indexes!'
        assert ks_i[1] == ks_j[1], 'k-point of the two states differ!'
        self.checkIndex(*ks_i)
        self.checkIndex(*ks_j)

        # k-points in direct coordinate
        k0 = self._kvecs[ks_i[1] - 1]
        # plane-waves in direct coordinates
        G0 = self.gvectors(ikpt=ks_i[1])
        # G + k in Cartesian coordinates
        Gk = np.dot(
            G0 + k0,                            # G in direct coordinates
            self._Bcell * TPI            # reciprocal basis x 2pi
        )

        # plane-wave coefficients for initial (mk) and final (nk) states
        CG_mk = self.readBandCoeff(*ks_i)
        CG_nk = self.readBandCoeff(*ks_j)
        ovlap = CG_nk.conj() * CG_mk

        ################################################################################
        # Momentum operator matrix element between pseudo-wavefunctions
        ################################################################################
        if self._lgam:
            # for gamma-only, only half the plane-wave coefficients are stored.
            # Moreover, the coefficients are multiplied by a factor of sqrt2

            # G > 0 part
            moment_mat_ps = np.sum(ovlap[:,None] * Gk, axis=0)

            # For gamma-only version, add the other half plane-waves, G_ = -G
            # G < 0 part, C(G) = C(-G).conj()
            moment_mat_ps -= np.sum(
                    ovlap[:,None].conj() * Gk,
                    axis=0)

            # remove the sqrt2 factor added by VASP
            moment_mat_ps /= 2.0

        elif self._lsoc:
            moment_mat_ps = np.sum(
                ovlap[:, None] * np.r_[Gk, Gk],
                axis=0)
        else:
            moment_mat_ps = np.sum(
                ovlap[:,None] * Gk, axis=0
            )

        return moment_mat_ps


    def inverse_participation_ratio(self, norm=True):
        '''
        Calculate Inverse Paticipation Ratio (IPR) from the wavefunction. IPR is
        a measure of the localization of Kohn-Sham states. For a particular KS
        state \phi_j, it is defined as

                            \sum_n |\phi_j(n)|^4
            IPR(\phi_j) = -------------------------
                          |\sum_n |\phi_j(n)|^2||^2

        where n iters over the number of grid points.
        '''

        self.ipr = np.zeros((self._nspin, self._nkpts, self._nbands, 3))

        for ispin in range(self._nspin):
            for ikpt in range(self._nkpts):
                for iband in range(self._nbands):
                    phi_j = self.wfc_r(ispin+1, ikpt+1, iband+1,
                                       norm=norm)
                    phi_j_abs = np.abs(phi_j)

                    print('Calculating IPR of #spin %4d, #kpt %4d, #band %4d' %
                          (ispin+1, ikpt+1, iband+1))
                    self.ipr[ispin, ikpt, iband,
                             0] = self._kpath[ikpt] if self._kpath is not None else 0
                    self.ipr[ispin, ikpt, iband,
                             1] = self._bands[ispin, ikpt, iband]
                    self.ipr[ispin, ikpt, iband, 2] = np.sum(
                        phi_j_abs**4) / np.sum(phi_j_abs**2)**2

        np.save('ipr.npy', self.ipr)
        return self.ipr

    def elf(self, kptw, ngrid=None, warn=True):
        '''
        Calculate the electron localization function (ELF) from WAVECAR.

        The following formula was extracted from VASP ELF.F:
                     _
                     h^2    *    2      T.........kinetic energy
          T    =  -2 --- Psi grad Psi   T+TCORR...pos.definite kinetic energy
                   ^ 2 m                TBOS......T of an ideal Bose-gas
                   ^
                   I am not sure if we need to times 2 here, use 1 in this
                   script.

                   _                                (=infimum of T+TCORR)
                 1 h^2      2           DH........T of hom.non-interact.e- - gas
          TCORR= - ---  grad rho                    (acc.to Fermi)
                 2 2 m                  ELF.......electron-localization-function
                   _             2
                 1 h^2 |grad rho|
          TBOS = - --- ----------       D = T + TCORR - TBOS
                 4 2 m    rho
                   _                                \                1
                 3 h^2        2/3  5/3          =====>    ELF = ------------
          DH   = - --- (3 Pi^2)  rho                /                   D   2
                 5 2 m                                           1 + ( ---- )
                                                                        DH

        REF:
            1. Nature, 371, 683-686 (1994)
            2. Becke and Edgecombe, J. Chem. Phys., 92, 5397(1990)
            3. M. Kohout and A. Savin, Int. J. Quantum Chem., 60, 875-882(1996)
            4. http://www2.cpfs.mpg.de/ELF/index.php?content=06interpr.txt
        '''

        if warn:
            warning = """
            ###################################################################
            If you are using VESTA to view the resulting ELF, please rename the
            output file as ELFCAR, otherwise there will be some error in the
            isosurface plot!

            When CHG*/PARCHG/*.vasp are read in to visualize isosurfaces and
            sections, data values are divided by volume in the unit of bohr^3.
            The unit of charge densities input by VESTA is, therefore, bohr^−3.

            For LOCPOT/ELFCAR files, volume data are kept intact.

            You can turn off this warning by setting "warn=False" in the "elf"
            method.
            ###################################################################
            """
            print(warning)

        # the k-point weights
        kptw = np.array(kptw, dtype=float)
        assert kptw.shape == (self._nkpts,), "K-point weights must be provided \
                                              to calculate charge density!"
        # normalization
        kptw /= kptw.sum()

        if ngrid is None:
            ngrid = self._ngrid * 2
        else:
            ngrid = np.array(ngrid, dtype=int)
            assert ngrid.shape == (3,)
            assert np.alltrue(ngrid >= self._ngrid), \
                "Minium FT grid size: (%d, %d, %d)" % \
                (self._ngrid[0], self._ngrid[1], self._ngrid[2])

        fx = [ii if ii < ngrid[0] // 2 + 1 else ii - ngrid[0]
              for ii in range(ngrid[0])]
        fy = [jj if jj < ngrid[1] // 2 + 1 else jj - ngrid[1]
              for jj in range(ngrid[1])]
        fz = [kk if kk < ngrid[2] // 2 + 1 else kk - ngrid[2]
              for kk in range(ngrid[2])]

        # plane-waves: Reciprocal coordinate
        # indexing = 'ij' so that outputs are of shape (ngrid[0], ngrid[1], ngrid[2])
        Dx, Dy, Dz = np.meshgrid(fx, fy, fz, indexing='ij')
        # plane-waves: Cartesian coordinate
        Gx, Gy, Gz = np.tensordot(
            self._Bcell * np.pi * 2, [Dx, Dy, Dz], axes=(0, 0))
        # the norm squared of the G-vectors
        G2 = Gx**2 + Gy**2 + Gz**2
        # k-points vectors in Cartesian coordinate
        vkpts = np.dot(self._kvecs, self._Bcell * 2 * np.pi)

        # normalization factor so that
        # \sum_{ijk} | \phi_{ijk} | ^ 2 * volume / Ngrid = 1
        normFac = np.sqrt(np.prod(ngrid) / self._Omega)

        # electron localization function
        ElectronLocalizationFunction = []
        # Charge density
        rho = np.zeros(ngrid, dtype=complex)
        # Kinetic energy density
        tau = np.zeros(ngrid, dtype=complex)

        for ispin in range(self._nspin):
            # initialization
            rho[...] = 0.0
            tau[...] = 0.0

            for ikpt in range(self._nkpts):

                # plane-wave G-vectors
                igvec = self.gvectors(ikpt+1)
                # for gamma-only version, complete the missing -G vectors
                if self._lgam:
                    tmp = np.array([-k for k in igvec[1:]], dtype=int)
                    igvec = np.vstack([igvec, tmp])
                # plane-wave G-vectors in Cartesian coordinate
                rgvec = np.dot(igvec, self._Bcell * 2 * np.pi)

                k = vkpts[ikpt]                       # k
                gk = rgvec + k[np.newaxis, :]           # G + k
                gk2 = np.linalg.norm(gk, axis=1)**2     # | G + k |^2

                for iband in range(self._nbands):
                    # omit the empty bands
                    if self._occs[ispin, ikpt, iband] == 0.0:
                        continue

                    rspin = 2.0 if self._nspin == 1 else 1.0
                    weight = rspin * kptw[ikpt] * \
                        self._occs[ispin, ikpt, iband]

                    # if self._lgam:
                    #     ########################################
                    #     # slower
                    #     ########################################
                    #     # wavefunction in real space
                    #     # VASP does NOT do normalization in elf.F
                    #     phi_r  = self.wfc_r(ispin=ispin+1, ikpt=ikpt+1,
                    #                         iband=iband+1,
                    #                         ngrid=ngrid,
                    #                         norm=False) * normFac
                    #     # wavefunction in reciprocal space
                    #     phi_q  = np.fft.fftn(phi_r, norm='ortho')
                    #     # grad^2 \phi in reciprocal space
                    #     lap_phi_q = -gk2 * phi_q
                    #     # grad^2 \phi in real space
                    #     lap_phi_r = np.fft.ifftn(lap_phi_q, norm='ortho')
                    # else:

                    ########################################
                    # faster
                    ########################################
                    # wavefunction in reciprocal space
                    # VASP does NOT do normalization in elf.F
                    phi_q = self.readBandCoeff(ispin=ispin+1, ikpt=ikpt+1,
                                               iband=iband+1,
                                               norm=False)
                    # pad the missing planewave coefficients for -G vectors
                    if self._lgam:
                        tmp = [x.conj() for x in phi_q[1:]]
                        phi_q = np.concatenate([phi_q, tmp])
                        # Gamma only, divide a factor of sqrt(2.0) except for
                        # G=0
                        phi_q /= np.sqrt(2.0)
                        phi_q[0] *= np.sqrt(2.0)
                    # wavefunction in real space
                    phi_r = self.wfc_r(ispin=ispin+1, ikpt=ikpt+1,
                                       iband=iband+1,
                                       ngrid=ngrid,
                                       gvec=igvec,
                                       Cg=phi_q) * normFac
                    # grad^2 \phi in reciprocal space
                    lap_phi_q = -gk2 * phi_q
                    # grad^2 \phi in real space
                    lap_phi_r = self.wfc_r(ispin=ispin+1, ikpt=ikpt+1,
                                           iband=iband+1,
                                           ngrid=ngrid,
                                           gvec=igvec,
                                           Cg=lap_phi_q) * normFac

                    # \phi* grad^2 \phi in real space --> kinetic energy density
                    tau += -phi_r * lap_phi_r.conj() * weight

                    # charge density in real space
                    rho += phi_r.conj() * phi_r * weight

            # charge density in reciprocal space
            rho_q = np.fft.fftn(rho, norm='ortho')

            # grad^2 rho: laplacian of charge density
            lap_rho_q = -G2 * rho_q
            lap_rho_r = np.fft.ifftn(lap_rho_q, norm='ortho')

            # charge density gradient: grad rho
            ########################################
            # wrong method for gradient using FFT
            ########################################
            # grad_rho_x = np.fft.ifft(1j * Gx * np.fft.fft(rho, axis=0), axis=0)
            # grad_rho_y = np.fft.ifft(1j * Gy * np.fft.fft(rho, axis=1), axis=1)
            # grad_rho_z = np.fft.ifft(1j * Gz * np.fft.fft(rho, axis=2), axis=2)

            ########################################
            # correct method for gradient using FFT
            ########################################
            grad_rho_x = np.fft.ifftn(1j * Gx * rho_q, norm='ortho')
            grad_rho_y = np.fft.ifftn(1j * Gy * rho_q, norm='ortho')
            grad_rho_z = np.fft.ifftn(1j * Gz * rho_q, norm='ortho')

            grad_rho_sq = np.abs(grad_rho_x)**2 \
                + np.abs(grad_rho_y)**2 \
                + np.abs(grad_rho_z)**2

            rho = rho.real
            tau = tau.real
            lap_rho_r = lap_rho_r.real

            Cf = 3./5 * (3.0 * np.pi**2)**(2./3)
            Dh = np.where(rho > 0.0,
                          Cf * rho**(5./3),
                          0.0)
            eps = 1E-8 / HSQDTM
            Dh[Dh < eps] = eps
            # D0 = T + TCORR - TBOS
            D0 = tau + 0.5 * lap_rho_r - 0.25 * grad_rho_sq / rho

            ElectronLocalizationFunction.append(1. / (1. + (D0 / Dh)**2))

        return ElectronLocalizationFunction

############################################################


if __name__ == '__main__':
    # xx = vaspwfc('wavecar')
    # phi = xx.wfc_r(1, 30, 17, ngrid=(28, 28, 252))
    # xx.save2vesta(phi, poscar='POSCAR')

    # xx = vaspwfc('./gamma/WAVECAR')
    # phi = xx.wfc_r(1, 1, 317, ngrid=(60, 108, 160),
    #                gamma=True)
    # xx.save2vesta(phi, poscar='./gamma/POSCAR',gamma=True)

    # xx = vaspwfc('WAVECAR')
    # dE, ovlap, tdm = xx.TransitionDipoleMoment([1,30,17], [1,30,18], norm=True)
    # print dE, ovlap.real, np.abs(tdm)**2

    # print xx._recl, xx._nspin, xx._rtag
    # print xx._nkpts, xx._nbands, xx._encut
    # print xx._Acell, xx._Bcell
    # # print np.linalg.norm(xx._Acell, axis=1)
    # print xx._ngrid
    # print xx._bands[0,0,:]
    # print xx._kvecs
    # print xx._kpath
    # b = xx.readBandCoeff(1,1,1)
    # xx = np.savetxt('kaka.dat', xx.gvectors(2), fmt='%5d')
    # gvec = xx.gvectors(1)
    # gvec %= xx._ngrid[np.newaxis, :]
    # print gvec

    # ngrid=(28, 28, 252)
    # phi = xx.wfc_r(1, 30, 17, ngrid=(28, 28, 252))
    # header = open('POSCAR').read()
    # with open('wave_real.vasp', 'w') as out:
    #     out.write(header)
    #     out.write('%5d%5d%5d\n' % (ngrid[0], ngrid[1], ngrid[2]))
    #     nwrite=0
    #     for kk in range(ngrid[2]):
    #         for jj in range(ngrid[1]):
    #             for ii in range(ngrid[0]):
    #                 nwrite += 1
    #                 out.write('%22.16f ' % phi.real[ii,jj,kk])
    #                 if nwrite % 10 == 0:
    #                     out.write('\n')
    # with open('wave_imag.vasp', 'w') as out:
    #     out.write(header)
    #     out.write('%5d%5d%5d\n' % (ngrid[0], ngrid[1], ngrid[2]))
    #     nwrite=0
    #     for kk in range(ngrid[2]):
    #         for jj in range(ngrid[1]):
    #             for ii in range(ngrid[0]):
    #                 nwrite += 1
    #                 out.write('%22.16f ' % phi.imag[ii,jj,kk])
    #                 if nwrite % 10 == 0:
    #                     out.write('\n')

    # xx = vaspwfc('wave_tyz')
    # ipr = xx.inverse_participation_ratio()
    # print xx._nbands, xx._nkpts
    #
    # import matplotlib as mpl
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # ax = plt.subplot()
    #
    # ax.scatter(ipr[...,0], ipr[..., 1], s=ipr[..., 2] / ipr[..., 2].max() * 10, c=ipr[..., 2],
    #            cmap='jet_r')
    #
    # plt.show()

    wfc = vaspwfc('WAVECAR', lgamma=True, gamma_half='x')
    # ngrid = [80, 140, 210]
    phi = wfc.wfc_r(iband=190)

    rho = np.abs(phi)**2
    # rho2 = VaspChargeDensity('PARCHG.0158.ALLK').chg[0]
    # rho /= rho.sum()
    # rho2 /= rho2.sum()
    # rho3 = rho - rho2

    wfc.save2vesta(rho, lreal=True)

    pass
