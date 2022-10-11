#!/usr/bin/env python

import numpy as np

from pysbt import pysbt

from vaspwfc import vaspwfc
from vasp_constant import *
from sph_harm import sph_r, sph_c
from paw import nonlq, pawpotcar, gvectors

from ase.io import read
from scipy.fft import ifftn, fftn
from scipy.linalg import block_diag

class vasp_ae_wfc(object):
    '''
    Refer to the following post for details.

    https://qijingzheng.github.io/posts/VASP-All-Electron-WFC
    '''

    def __init__(
        self,
        wavecar,
        poscar: str='POSCAR',
        potcar: str='POTCAR',
        aecut: float=-4,
        ikpt: int=1,
    ):
        '''
        '''

        # wavecar storing the pseudo-wavefunctions
        self._pswfc = wavecar
        # energy cut for the pseudo-wavefunction
        self._pscut = wavecar._encut
        # the k-point vectors in fractional coordinates
        self._ikpt = ikpt
        self._kvec = self._pswfc._kvecs[ikpt - 1]
        
        # the poscar storing the atoms information
        self._atoms  = read(poscar)
        self._natoms = len(self._atoms)

        elements, elem_first_idx, elem_cnts = np.unique(self._atoms.get_chemical_symbols(),
                                                        return_index=True,
                                                        return_counts=True)
        # Sometimes, the order of the elements returned by np.unique may not be
        # consistent with that in POSCAR/POTCAR
        elem_first_idx    = np.argsort(elem_first_idx)
        self._elements    = list(elements[elem_first_idx])
        self._elem_cnts   = elem_cnts[elem_first_idx]
        self._element_idx = [self._elements.index(s) for s in
                             self._atoms.get_chemical_symbols()]

        self._q_proj = nonlq(
            self._atoms,
            self._pscut,
            potcar,
            k=self._kvec,
            lgam=self._pswfc._lgam,
            gamma_half=self._pswfc._gam_half,
        )
        self._pawpp  = self._q_proj.pawpp

        assert len(self._elem_cnts) == len(self._pawpp), \
            "The kind of elements in POTCAR and POSCAR does not match!"
        if not np.alltrue([
            self._pawpp[ii].element.split('_')[0] == elements[ii]
            for ii in range(len(elements))
        ]):
            print(
                "\nWARNING:\nThe name of elements in POTCAR and POSCAR does not match!\n\n" +
                "    POTCAR: {}\n".format(' '.join([pp.element for pp in self._pawpp])) +
                "    POSCAR: {}\n".format(' '.join(self._elements))
            )

        self.set_aecut(aecut)

    def set_aecut(self, aecut=-4):
        '''
        '''
        # negative values means "aecut" is abs(aecut) multiples of 'pscut'
        if aecut < 0:
            self._aecut = -aecut * self._pscut
        else:
            self._aecut = self._pscut if self._pscut > aecut else aecut

        self._aegrid = np.asarray(
            np.sqrt(self._aecut / self._pscut) * self._pswfc._ngrid + 1.0,
            dtype=int
        )


        # once aecut is changed, the following quantities must be re-calculated
        self.get_aeps_gvecs()
        self.set_ae_ylm()
        self.set_ae_phase()
        self.sbt_aeps_core()

    def get_aeps_gvecs(self):
        '''
        '''

        # 0.5*| G + k |^2 < E_c
        self._ps_gv = self._q_proj.Gvec
        # G + k
        self._ps_gk = self._q_proj.Gk
        # |G + k|
        self._ps_gl = self._q_proj.Glen

        # 0.5*| G + k |^2 < E_ae
        self._ae_gv = gvectors(
            self._atoms.cell, self._aecut, self._kvec,
            lgam=self._pswfc._lgam,
            gamma_half=self._pswfc._gam_half,
        )
        # G + k
        self._ae_gk = np.dot(
            self._ae_gv + self._kvec, TPI * self._atoms.cell.reciprocal()
        )
        # G-vectors length
        self._ae_gl = np.linalg.norm(self._ae_gk, axis=1)


    def get_beta_njk(self, Cg):
        '''
        '''
        self._beta_njk = self._q_proj.proj(Cg)

        return self._beta_njk

    def set_ae_ylm(self):
        '''
         Calculate the real spherical harmonics for a set of G-grid points up to
         LMAX.
        '''

        lmax = np.max([p.proj_l.max() for p in self._pawpp])
        self.ylm = []
        for l in range(lmax+1):
            self.ylm.append(
                sph_r(self._ae_gk, l)
            )

    def set_ae_phase(self):
        '''
        Calculates the phasefactor CREXP (exp(-iG.R)) for one k-point
        '''
        self.crexp = np.exp(-1j * TPI *
                            np.dot(
                                self._ae_gv, self._atoms.get_scaled_positions().T
                            ))
        # i^{-L} is stored in CQFAK
        self.cqfak = [
            1j ** np.array([
                -l for l in pp.proj_l
                for ii in range(2 * l + 1)
            ])
            for pp in self._pawpp
        ]

    def sbt_aeps_core(self):
        '''
        perform SBT on the difference of AE and PS partial waves
        '''

        from pysbt import sbt
        from scipy.interpolate import interp1d

        self._q_ae_core = []
        self._q_ps_core = []
        for itype, pp in enumerate(self._pawpp):
            ss    = sbt(pp.rgrid)
            qgrid = np.r_[0, ss.kk]

            t1 = np.zeros((pp.lmmax, self._ae_gl.size))
            t2 = np.zeros((pp.lmmax, self._ae_gl.size))
            iL = 0
            for ii, l in enumerate(pp.proj_l):
                TPL1   = 2*l + 1
                # f1     = pp.paw_ae_wfc[ii] - pp.paw_ps_wfc[ii]
                # g1     = 4*np.pi*ss.run(f1 / pp.rgrid, l=l, include_zero=True)
                # spl_g1 = interp1d(qgrid, g1, kind='cubic')

                f1 = pp.paw_ae_wfc[ii]
                f2 = pp.paw_ps_wfc[ii]

                f1[pp.rgrid >= pp.proj_rmax] = 0.0
                f2[pp.rgrid >= pp.proj_rmax] = 0.0

                g1     = 4*np.pi*ss.run(f1 / pp.rgrid, l=l, include_zero=True)
                g2     = 4*np.pi*ss.run(f2 / pp.rgrid, l=l, include_zero=True)
                spl_g1 = interp1d(qgrid, g1, kind='cubic')
                spl_g2 = interp1d(qgrid, g2, kind='cubic')

                t1[iL:iL+TPL1, :] = spl_g1(self._ae_gl) * self.ylm[l].T
                t2[iL:iL+TPL1, :] = spl_g2(self._ae_gl) * self.ylm[l].T

                iL += TPL1

            t1 /= np.sqrt(self._atoms.get_volume())
            t2 /= np.sqrt(self._atoms.get_volume())

            self._q_ae_core.append(t1)
            self._q_ps_core.append(t2)


        # Q_{ij} = < \phi_i^{AE} | \phi_j^{AE} > - < phi_i^{PS} | phi_j^{PS} >
        self._qij = block_diag(*[
            self._pawpp[self._element_idx[iatom]].get_Qij()
            for iatom in range(self._natoms)
        ])

    def get_ae_norm(self, ispin: int=1, iband: int=1):
        '''
        '''
        
        Cg = self._pswfc.readBandCoeff(ispin, self._ikpt, iband, norm=False)
        beta_njk = self.get_beta_njk(Cg)

        ae_norm = (Cg.conj() * Cg).sum() + \
                  np.dot(
                      np.dot(beta_njk, self._qij), beta_njk
                  )

        return ae_norm.real
    
    def get_ae_wfc(self,
            iband: int=1,
            ispin: int=1,
            lcore: bool=False,
            norm=True,
        ):
        '''
        '''

        Cg = self._pswfc.readBandCoeff(ispin, self._ikpt, iband, norm=False)
        beta_njk = self.get_beta_njk(Cg)

        # The on-site terms
        Sg_ae = np.zeros(self._ae_gl.size, dtype=complex)
        Sg_ps = np.zeros(self._ae_gl.size, dtype=complex)

        nproj = 0
        for ii in range(self._natoms):
            itype   = self._element_idx[ii]
            lmmax   = self._pawpp[itype].lmmax
            iill    = self.cqfak[itype]
            exp_iGR = self.crexp[:,ii]

            Sg_ae += np.sum(
                iill * beta_njk[nproj:nproj+lmmax] * self._q_ae_core[itype].T,
                axis=1
            ) * exp_iGR

            Sg_ps += np.sum(
                iill * beta_njk[nproj:nproj+lmmax] * self._q_ps_core[itype].T,
                axis=1
            ) * exp_iGR

            nproj += lmmax

        phi_ae = np.zeros(self._aegrid, dtype=complex)
        if lcore:
            core_ae_wfc = np.zeros(self._aegrid, dtype=complex)
            core_ps_wfc = np.zeros(self._aegrid, dtype=complex)
        
        g1 = self._ps_gv % self._aegrid[None,:]
        g2 = self._ae_gv % self._aegrid[None,:]

        if self._pswfc._lgam:
            Cg[1:] /= np.sqrt(2.0)

        phi_ae[g1[:,0], g1[:,1], g1[:,2]] += Cg
        phi_ae[g2[:,0], g2[:,1], g2[:,2]] += Sg_ae - Sg_ps

        if lcore:
            core_ae_wfc[g2[:,0], g2[:,1], g2[:,2]] += Sg_ae
            core_ps_wfc[g2[:,0], g2[:,1], g2[:,2]] += Sg_ps

        if self._pswfc._lgam:
            phi_ae[-g1[1:,0], -g1[1:,1], -g1[1:,2]] += Cg[1:].conj()
            phi_ae[-g2[1:,0], -g2[1:,1], -g2[1:,2]] += Sg_ae[1:].conj() - Sg_ps[1:].conj()

            if lcore:
                core_ae_wfc[g2[1:,0], g2[1:,1], g2[1:,2]] += Sg_ae[1:].conj()
                core_ps_wfc[g2[1:,0], g2[1:,1], g2[1:,2]] += Sg_ps[1:].conj()

        fac = np.sqrt(np.prod(self._aegrid)) if norm else 1.0

        if self._pswfc._lgam:
            if lcore:
                return ifftn(phi_ae).real * fac, \
                       ifftn(core_ae_wfc).real * fac, \
                       ifftn(core_ps_wfc).real * fac
            else:
                return ifftn(phi_ae).real * fac
        else:
            if lcore:
                return ifftn(phi_ae) * fac, \
                       ifftn(core_ae_wfc) * fac, \
                       ifftn(core_ps_wfc) * fac
            else:
                return ifftn(phi_ae) * fac

if __name__ == "__main__":
    pass
    # import matplotlib.pyplot as plt
    # from vaspwfc import save2vesta
    #
    # atoms = read('POSCAR')
    # L = atoms.cell[-1, -1]
    #
    # ps_wfc = vaspwfc('WAVECAR', lgamma=True)
    # ae_wfc = vasp_ae_wfc(ps_wfc, aecut=-25)
    #
    # for iband in range(ps_wfc._nbands):
    #     # norm of the PSWFC
    #     cg = ps_wfc.readBandCoeff(iband=iband+1)
    #     ps_norm = np.sum(cg.conj() * cg).real
    #     print(f"#band: {iband+1:3d} -> {1 - ps_norm: 10.4f}")
    #
    # which_band = 8
    #
    # phi_ae = ae_wfc.get_ae_wfc(iband=which_band) * np.sqrt(np.prod(ae_wfc._aegrid))
    # phi_ps = ps_wfc.get_ps_wfc(iband=which_band, norm=False, ngrid=ae_wfc._aegrid)
    #
    # # save2vesta(phi_ae, prefix='ae')
    # # save2vesta(phi_ps, prefix='ps')
    #
    # r_ps = np.arange(phi_ps.shape[-1]) * L / phi_ps.shape[-1]
    # r_ae = np.arange(phi_ae.shape[-1]) * L / phi_ae.shape[-1]
    #
    # fig = plt.figure(
    #   figsize=plt.figaspect(1.0),
    #   dpi=300,
    # )
    # axes = [plt.subplot(2, 2, ii+1) for ii in range(4)]
    #
    # ax = axes[0]
    # ax.plot(r_ps, phi_ps[0, 0].real, color='r', ls='--')
    # # ax.plot(r_ps, phi_ps[10, 0].imag, color='b', ls='--')
    # # ax.plot(r_ps, np.abs(phi_ps.real).sum(axis=(0,1)), color='r', ls='--')
    # # ax.plot(r_ps, phi_ps.real.sum(axis=(0,1)), color='r', ls='--')
    #
    # ax = axes[1]
    # ax.imshow(np.roll(phi_ps[0,:,:].real, phi_ps.shape[1] // 2, axis=0), extent=(0, L, 0, L))
    #
    # ax = axes[0]
    # ax.plot(r_ae, phi_ae[0, 0].real, color='r', ls='-')
    # # ax.plot(r_ae, phi_ae[10, 0].imag, color='b', ls='-')
    # # ax.plot(r_ae, np.abs(phi_ae.real).sum(axis=(0,1)), color='r', ls='-')
    # # ax.plot(r_ae, phi_ae.real.sum(axis=(0,1)), color='r', ls='-')
    #
    # ax = axes[3]
    # ax.imshow(np.roll(phi_ae[0,:,:].real, phi_ae.shape[1] // 2, axis=0), extent=(0, L, 0, L))
    #
    # # for ax in axes:
    # #     ax.set_xlabel(r'$r$ [$\AA$]')
    #
    # plt.tight_layout()
    # plt.show()
    # #
