#!/usr/bin/env python

import numpy as np

try:
    from pysbt import pysbt
except ImportError:
    print("Please install pySBT (https://github.com/QijingZheng/pySBT)!")

from vaspwfc import vaspwfc
from vasp_constant import *
from sph_harm import sph_r, sph_c
from paw import nonlq, pawpotcar, gvectors

from ase.io import read
from scipy.fft import ifftn, fftn
from scipy.sparse import block_diag

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

        # if wavecar._lsoc:
        #     raise NotImplementedError('Non-collinear version currently not supported!')

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

        self._pawpp = [pawpotcar(potstr) for potstr in
                      open(potcar).read().split('End of Dataset')[:-1]]

        self._q_proj = nonlq(
            self._atoms,
            self._pscut,
            self._pawpp,
            k=self._kvec,
            lgam=self._pswfc._lgam,
            gamma_half=self._pswfc._gam_half,
        )

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


    def get_qijs(self):
        '''
        Qij for all the atoms

            Q_{ij} = < \phi_i^{AE} | \phi_j^{AE} > - < phi_i^{PS} | phi_j^{PS} >
        
        and stored in a scipy.sparse block diagonal matrix.
        '''
        if not hasattr(self, "_qijs"):
            self._qijs = block_diag([
                self._pawpp[self._element_idx[iatom]].get_Qij()
                for iatom in range(self._natoms)
            ])

        return self._qijs

    def get_nablaijs(self):
        '''
        Matrix elements of the gradient operator for all the atoms

            nabla_{ij} = < \phi_i^{AE} | nabla_r | \phi_j^{AE} > -
                         < \phi_i^{PS} | nabla_r | \phi_j^{PS} >
        
        and stored in a list of scipy.sparse block diagonal matrices.
        '''
        if not hasattr(self, "_nablaijs"):
            self._nablaijs = [
                block_diag([
                    self._pawpp[self._element_idx[iatom]].get_nablaij()[ii]
                    for iatom in range(self._natoms)
                ])
                for ii in range(3)
            ]

        return self._nablaijs

    def get_ae_norm(self, ispin: int=1, iband: int=1):
        '''
        '''

        Cg = self._pswfc.readBandCoeff(ispin, self._ikpt, iband, norm=False)
        beta_njk = self.get_beta_njk(Cg)

        ae_norm = np.sum(Cg.conj() * Cg) + beta_njk.conj() @ (self.get_qijs() @ beta_njk)

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


    def get_dipole_mat(self, ks_i, ks_j):
        '''
        Dipole transition within the electric dipole approximation (EDA).
        Please refer to this post for more details.

          https://qijingzheng.github.io/posts/Light-Matter-Interaction-and-Dipole-Transition-Matrix/

        The dipole transition matrix elements in the length gauge is given by:

                <psi_nk | e r | psi_mk>

        In periodic systems, the position operator "r" is not well-defined.
        Therefore, we first evaluate the momentum operator matrix in the velocity
        gauge, i.e.

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
        self._pswfc.checkIndex(*ks_i)
        self._pswfc.checkIndex(*ks_j)

        # energy differences between the two states
        Emk = self._pswfc._bands[ks_i[0]-1, ks_i[1]-1, ks_i[2]-1]
        Enk = self._pswfc._bands[ks_j[0]-1, ks_j[1]-1, ks_j[2]-1]
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
        The momentum operator matrix in the velocity gauge

                <psi_nk | p | psi_mk> = hbar <u_nk | k - i nabla | u_mk>

        In PAW, the matrix element can be divided into plane-wave parts and
        one-center parts, i.e.

            <u_nk | k - i nabla | u_mk> = <tilde_u_nk | k - i nabla | tilde_u_mk>
                                         - \sum_ij <tilde_u_nk | p_i><p_j | tilde_u_mk>
                                           \times i [
                                             <phi_i | nabla | phi_j>
                                             -
                                             <tilde_phi_i | nabla | tilde_phi_j>
                                           ]

        where | u_nk > and | tilde_u_nk > are cell-periodic part of the AE/PS
        wavefunctions, | p_j > is the PAW projector function and | phi_j > and
        | tilde_phi_j > are PAW AE/PS partial waves.

        The nabla operator matrix elements between the pseudo-wavefuncitons

            <tilde_u_nk | k - i nabla | tilde_u_mk>

           = \sum_G C_nk(G).conj() * C_mk(G) * [k + G]

        where C_nk(G) is the plane-wave coefficients for | u_nk >.

        '''

        # ks_i and ks_j are list containing spin-, kpoint- and band-index of the
        # initial and final states
        assert len(ks_i) == len(ks_j) == 3, 'Must be three indexes!'
        assert ks_i[1] == ks_j[1], 'k-point of the two states differ!'
        self._pswfc.checkIndex(*ks_i)
        self._pswfc.checkIndex(*ks_j)

        # k-points in direct coordinate
        k0 = self._pswfc._kvecs[ks_i[1] - 1]
        # plane-waves in direct coordinates
        G0 = self._pswfc.gvectors(ikpt=ks_i[1])
        # G + k in Cartesian coordinates
        Gk = np.dot(
            G0 + k0,                            # G in direct coordinates
            self._pswfc._Bcell * TPI            # reciprocal basis x 2pi
        )

        # plane-wave coefficients for initial (mk) and final (nk) states
        CG_mk = self._pswfc.readBandCoeff(*ks_i)
        CG_nk = self._pswfc.readBandCoeff(*ks_j)
        ovlap = CG_nk.conj() * CG_mk

        ################################################################################
        # Momentum operator matrix element between pseudo-wavefunctions
        ################################################################################
        if self._pswfc._lgam:
            # for gamma-only, only half the plane-wave coefficients are stored.
            # Moreover, the coefficients are multiplied by a factor of sqrt2

            # G > 0 part
            moment_mat_ps = np.sum(ovlap[:,None] * Gk, axis=0)

            # For gamma-only version, add the other half plane-waves, G' = -G
            # G < 0 part, C(G) = C(-G).conj()
            moment_mat_ps -= np.sum(
                    ovlap[:,None].conj() * Gk,
                    axis=0)

            # remove the sqrt2 factor added by VASP
            moment_mat_ps /= 2.0
        elif self._pswfc._lsoc:
            moment_mat_ps = np.sum(
                ovlap[:, None] * np.r_[Gk, Gk],
                axis=0)
            # raise NotImplementedError('Non-collinear version currently not supported!')
        else:
            moment_mat_ps = np.sum(
                ovlap[:,None] * Gk, axis=0
            )

        ################################################################################
        # One-center correction
        ################################################################################

        projector = nonlq(
            self._atoms,
            self._pscut,
            self._pawpp,
            k=k0,
            lgam=self._pswfc._lgam,
            gamma_half=self._pswfc._gam_half,
        )

        if self._pswfc._lsoc:
            nplw = Gk.shape[0]
            # spin-up component of the spinor
            beta_mk  = projector.proj(CG_mk[:nplw])
            beta_nk  = projector.proj(CG_nk[:nplw])
            
            # spin-down component of the spinor
            beta_mk2 = projector.proj(CG_mk[nplw:])
            beta_nk2 = projector.proj(CG_nk[nplw:])
        else:
            beta_mk = projector.proj(CG_mk)
            beta_nk = projector.proj(CG_nk)

        # one-center term of momentum operator matrix element
        moment_mat_oc = np.zeros(3, dtype=complex)

        # nproj = 0
        # for ii in range(self._natoms):
        #     itype = self._element_idx[ii]
        #     lmmax = self._pawpp[itype].lmmax
        #     nabla = self._pawpp[itype].get_nablaij(lreal=True)
        #
        #     moment_mat_oc += np.dot(
        #         beta_nk[nproj:nproj+lmmax].conj(),
        #         np.dot(nabla, beta_mk[nproj:nproj+lmmax]).T
        #     )
        #     
        #     if self._pswfc._lsoc:
        #         moment_mat_oc += np.dot(
        #             beta_nk2[nproj:nproj+lmmax].conj(),
        #             np.dot(nabla, beta_mk2[nproj:nproj+lmmax]).T
        #         )
        #
        #     nproj += lmmax

        for ii in range(3):
            moment_mat_oc[ii] = beta_nk.conj() @ (self.get_nablaijs()[ii] @ beta_mk)
            if self._pswfc._lsoc:
                moment_mat_oc[ii] += beta_nk2.conj() @ (self.get_nablaijs()[ii] @ beta_mk2)

        return  moment_mat_ps - 1j*moment_mat_oc



################################################################################
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
