#!/usr/bin/env python3
import numpy as np
from numpy.fft import fftn
from ase.io.vasp import read_vasp

from vasp_constant import (
        TPI,
        HARTREE,
        EDEPS
        )
from vaspwfc import vaspwfc
from paw import (pawpotcar, nonlq)
from coulomb_correction import (
        PAWCoulombCorrection,
        pack
        )


class PWCoulombIntegral(vaspwfc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def density_matrix(self, m: int, n: int):
        '''
              ⌠     *           iGr
        Sₘₙ = ⎮ dr ϕₘ(r) ϕₙ(r) e
              ⌡
        '''
        um = self.wfc_r(ispin=1, ikpt=1, iband=m, ngrid=self._ngrid, norm=False)
        un = self.wfc_r(ispin=1, ikpt=1, iband=n, ngrid=self._ngrid, norm=False)
        Smn = um.conj() * un
        return fftn(Smn)

    @property
    def gvectors_cart(self):
        '''
        G = 2pi * fx * Bcell
        G vectors in cartesian coordinate

        @out:
            - G in 1/Angstrom
        '''
        if not hasattr(self, '_gvectors_cart'):
            fx, fy, fz = [np.arange(n, dtype=int) for n in self._ngrid]
            fx[self._ngrid[0] // 2 + 1:] -= self._ngrid[0]
            fy[self._ngrid[1] // 2 + 1:] -= self._ngrid[1]
            fz[self._ngrid[2] // 2 + 1:] -= self._ngrid[2]
            gx, gy, gz = np.array(
                    np.meshgrid(fx, fy, fz, indexing='ij')
                    ).reshape((3, -1))
            kgrid = np.array([gx, gy, gz], dtype=float).T

            ## Here gvectors_cart is not multiplied with TPI, consistent with CHTOT from POTHAR@pot.F
            self._gvectors_cart = kgrid @ self._Bcell
        return self._gvectors_cart

    def coulomb_integral(self, m: int, n: int, p: int, q: int):
        '''
                  ⌠                 *      1   *
        (mn|pq) = ⎮ dr₁ dr₂ ψₘ(r₁) ψₙ(r₁) ─── ψₚ(r₂) ψ (r₂)
                  ⌡                       r₁₂         q

                    1   ⌠     *   4π
                = ───── ⎮ dq ρ   ──── ρ
                  (2π)³ ⌡     mn |q|²  pq

        where ρₘₙ is also known as Sₘₙ calculated by `density_matrix`

        WARNING: Only states at Gamma point is supported for now

        @in:
            - m,n,p,q: index of states at Gamma point
        @out:
            - Coulomb Integral, in eV
        '''
        rhomn = self.density_matrix(m, n).flatten().conj()
        rhopq = self.density_matrix(p, q).flatten()
        Gsqr  = np.linalg.norm(self.gvectors_cart, axis=-1) ** 2

        # First G is 0, can be filtered out
        # Here EDEPS / self._Omega / TPI**2 is copied from  pot.F subroutine POTHAR
        # which transforms the  unit to  eV
        integral = np.sum(rhomn[1:] * rhopq[1:] / Gsqr[1:]) * (EDEPS / self._Omega / TPI**2)
        return integral


class CoulombIntegral(object):
    '''
    '''
    def __init__(self, poscar="POSCAR", wavecar="WAVECAR", potcar="POTCAR"):
        self.pwci   = PWCoulombIntegral(fnm=wavecar)
        self.atoms  = read_vasp(poscar)
        self.pawcorr = [PAWCoulombCorrection(pawpotcar(potstr=potstr))
                        for potstr in open(potcar).read().split('End of Dataset')[:-1]]
        self.M_pp = [pp.get_coulomb_corrections()[1] for pp in self.pawcorr]
        self.qproj  = nonlq(self.atoms, self.pwci._encut, potcar)

        atom_cnts   = [int(x) for x in open(poscar).readlines()[6].split()]
        self.element_idx = [idx for (i,cnt) in enumerate(atom_cnts)
                                for idx in [i]*cnt]

        # self.coul_corr = block_diag(*[self.M_pp[ie] for ie in self.element_idx])

    def coulomb_integral(self, m:int, n:int, p:int, q:int):
        ci_paw = 0.0
        for (ia,ip) in enumerate(self.element_idx):
            beta_njk = [None, None, None, None]

            # calculate projection <p | psi>
            for (ibeta, iband) in enumerate([m, n, p, q]):
                Cg = self.pwci.readBandCoeff(ispin=1, ikpt=1, iband=iband, norm=False)
                beta_njk[ibeta] = self.qproj.proj(Cg, whichatom=ia)
                pass
            pass

            Pij = pack(np.outer(beta_njk[0],        beta_njk[1].conj()))
            Pkl = pack(np.outer(beta_njk[2].conj(), beta_njk[3]       ))

            ci_paw += Pij @ self.M_pp[ip] @ Pkl

        ci_pw = self.pwci.coulomb_integral(m, n, p, q)
        K_mnpq = ci_pw + 2 * ci_paw * HARTREE
        return (K_mnpq, ci_pw, ci_paw)

    pass


if '__main__' == __name__:
    prefix = 'examples/projectors/lreal_false/'
    ci = CoulombIntegral(poscar=prefix+'POSCAR', wavecar=prefix+'WAVECAR', potcar=prefix+'POTCAR')
    integral_1 = ci.coulomb_integral(9, 10, 11, 12)
    print(integral_1)
    integral_2 = ci.coulomb_integral(9, 9, 9, 9)
    print(integral_2)
    pass
