#!/usr/bin/env python

import re
import numpy as np
from vasp_constant import *


def gvectors(cell, encut, kvec, ngrid=None
             lgam=False, gamma_half='x', force_gamma=False,
             lsoc=False):
    '''
    Generate the G-vectors that satisfies the following relation
        (G + k)**2 / 2 < ENCUT
    '''
    # Minimum FFT grid size
    Bcell = np.linalg.inv(cell).T     # reciprocal space supercell volume
    if ngrid is None:
        Anorm = np.linalg.norm(cell, axis=1)
        CUTOF = np.ceil(
            sqrt(encut / RYTOEV) / (TPI / (Anorm / AUTOA))
        )
        ngrid = np.array(2 * CUTOF + 1, dtype=int)

    kvec = np.asarray(kvec)
    # fx, fy, fz = [fftfreq(n) * n for n in self._ngrid]
    # fftfreq in scipy.fftpack is a little different with VASP frequencies
    fx = [ii if ii < ngrid[0] // 2 + 1 else ii - ngrid[0]
          for ii in range(ngrid[0])]
    fy = [jj if jj < ngrid[1] // 2 + 1 else jj - ngrid[1]
          for jj in range(ngrid[1])]
    fz = [kk if kk < ngrid[2] // 2 + 1 else kk - ngrid[2]
          for kk in range(ngrid[2])]

    # force_Gamma: consider gamma-only case regardless of the real setting
    if force_Gamma:
        lgam = True
    if lgam:
        # parallel gamma version of VASP WAVECAR exclude some planewave
        # components, -DwNGZHalf
        if gamma_half == 'z':
            kgrid = np.array([(fx[ii], fy[jj], fz[kk])
                              for kk in range(ngrid[2])
                              for jj in range(ngrid[1])
                              for ii in range(ngrid[0])
                              if (
                                  (fz[kk] > 0) or
                                  (fz[kk] == 0 and fy[jj] > 0) or
                                  (fz[kk] == 0 and fy[jj] == 0 and fx[ii] >= 0)
            )], dtype=float)
        else:
            kgrid = np.array([(fx[ii], fy[jj], fz[kk])
                              for kk in range(ngrid[2])
                              for jj in range(ngrid[1])
                              for ii in range(ngrid[0])
                              if (
                                  (fx[ii] > 0) or
                                  (fx[ii] == 0 and fy[jj] > 0) or
                                  (fx[ii] == 0 and fy[jj] == 0 and fz[kk] >= 0)
            )], dtype=float)
    else:
        kgrid = np.array([(fx[ii], fy[jj], fz[kk])
                          for kk in range(ngrid[2])
                          for jj in range(ngrid[1])
                          for ii in range(ngrid[0])], dtype=float)

    # Kinetic_Energy = (G + k)**2 / 2
    # HSQDTM    =  hbar**2/(2*ELECTRON MASS)
    KENERGY = HSQDTM * np.linalg.norm(
        np.dot(kgrid + kvec[np.newaxis, :], TPI*Bcell), axis=1
    )**2
    # find Gvectors where (G + k)**2 / 2 < ENCUT
    Gvec = kgrid[np.where(KENERGY < encut)[0]]

    return np.asarray(Gvec, dtype=int)


class pawpot(object):
    '''
    Read projector functions and ae/ps partialwaves from VASP PBE POTCAR.
    '''

    NPSQNL = 100      # no. of data for projectors in reciprocal space
    NPSRNL = 100      # no. of data for projectors in real space

    def __init__(self, potstr):
        '''
        PAW POTCAR provides the PAW projector functions in real and reciprocal
        space. 
        '''

        non_radial_part, radial_part = potstr.split('PAW radial sets', 1)

        # read the projector functions in real/reciprocal space
        self.read_proj(non_radial_part)
        # read the ae/ps partial waves in the core region
        self.read_partial_wfc(radial_part)

    def read_proj(self, datastr):
        '''
        Read the projector functions in reciprocal space.
        '''
        dump = datastr.split('Non local Part')
        head = dump[0].strip().split('\n')
        non_local_part = dump[1:]

        # element of the potcar
        self.element = head[0].split()[1]
        # maximal G for reciprocal non local projectors
        self.proj_gmax = float(head[-1].split()[0])

        qprojs = []
        rprojs = []
        proj_l = []
        for proj in non_local_part:
            dump = proj.split('Reciprocal Space Part')
            ln_rmax_dion_part = dump[0].strip().split()
            l, nlproj = [int(xx) for xx in ln_rmax_dion_part[:2]]

            proj_l += [l] * nlproj
            self.proj_rmax = float(ln_rmax_dion_part[2])

            dion = np.asarray(ln_rmax_dion_part[3:], dtype=float)

            for rr in dump[1:]:
                reci, real = rr.split('Real Space Part')
                qprojs.append(
                    np.fromstring(reci, np.float, sep=' ')
                )
                rprojs.append(
                    np.fromstring(real, np.float, sep=' ')
                )

        # the real space radial grid for the projector functions
        self.proj_rgrid = np.arange(self.NPSRNL) * self.proj_rmax / self.NPSRNL
        # the reciprocal space radial grid for the projector functions
        self.proj_qgrid = np.arange(self.NPSQNL) * self.proj_gmax / self.NPSQNL
        # L quantum number for each projector functions
        self.proj_l = np.asarray(proj_l, dtype=int)
        # projector functions in reciprocal space
        self.qprojs = np.asarray(qprojs, dtype=float)
        # projector functions in real space
        self.rprojs = np.asarray(rprojs, dtype=float)

    def read_partial_wfc(self, datastr):
        '''
        Read the ps/ae partial waves.

        The data structure in POTCAR:

             grid
             aepotential
             core charge-density
             kinetic energy-density
             mkinetic energy-density pseudized
             local pseudopotential core
             pspotential valence only
             core charge-density (pseudized)
             pseudo wavefunction
             ae wavefunction
             ...
             pseudo wavefunction
             ae wavefunction
        '''
        data = datastr.strip().split('\n')
        nmax = int(data[0].split()[0])
        grid_start_idx = data.index(" grid") + 1

        core_data = np.array([
            x for line in data[grid_start_idx:]
            for x in line.strip().split()
            if not re.match(r'\ \w+', line)
        ], dtype=float)
        core_data = core_data.reshape((-1, nmax))
        # number of projectors
        nproj = self.proj_l.size

        # core region logarithmic radial grid
        self.core_rgrid = core_data[0]
        # core region all-electron potential
        self.core_aepot = core_data[1]
        # core region pseudo wavefunction
        self.core_ps_wfc = core_data[-nproj*2::2, :]
        # core region all-electron wavefunctions
        self.core_ae_wfc = core_data[-nproj*2+1::2, :]

    def csplines(self):
        '''
        Cubic spline interpolation of both the real and reciprocal space
        projector functions.
        '''

        from scipy.interpolate import CubicSpline as cs

        # for reciprocal space projector functions, natural boundary condition
        # (Y'' = 0) is applied at both ends.
        self.spl_qproj = [
            cs(self.proj_qgrid, qproj, bc_type='natural') for qproj in
            self.qprojs
        ]
        # for real space projector functions, natural boundary condition
        # (Y'' = 0) is applied at the point N.
        self.spl_rproj = []
        for l, rproj in zip(self.proj_l, self.rprojs):
            # Copy from VASP pseudo.F, I don't know why y1p depend on "l".
            if l == 1:
                yp1 = (rproj[1] - rproj[0]) / (self.proj_rmax / self.NPSRNL)
            else:
                y1p = 0.0
            self.spl_rproj.append(
                cs(self.proj_rgrid, rproj, bc_type=((1, y1p), (2, 0)))
            )

    @property
    def symbol(self):
        '''
        return the symbol of the element
        '''
        return self.element

    @property
    def lmax(self):
        '''
        Return total number of l-channel projector functions.
        '''

        return self.proj_l.size

    @property
    def lmmax(self):
        '''
        Return total number of lm-channel projector functions.
        '''

        return np.sum(2 * self.proj_l + 1)

    def plot(self):
        '''
        '''
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        mpl.rcParams['axes.unicode_minus'] = False
        plt.style.use('ggplot')

        figure = plt.figure(
            figsize=(8.0, 4.0),
            # figsize = plt.figaspect(0.6),
            # dpi=300,
        )

        axes = [
            plt.subplot(121),   # for projector functions
            plt.subplot(122)    # for ps/ae partial waves
        ]

        for ii in range(self.lmax):
            axes[0].plot(
                self.proj_rgrid, self.rprojs[ii], label=f"L = {self.proj_l[ii]}",
            )
            l1, = axes[1].plot(
                self.core_rgrid, self.core_ae_wfc[ii], label=f"L = {self.proj_l[ii]}"
            )
            axes[1].plot(
                self.core_rgrid, self.core_ps_wfc[ii], ls=':',
                color=l1.get_color()
            )

        for ax in axes:
            ax.set_xlabel(r'$r\ [\AA]$', labelpad=5)

            ax.axhline(y=0, ls=':', color='k', alpha=0.6)
            ax.axvline(x=self.proj_rmax, ls=':', color='k', alpha=0.6)

            ax.legend(loc='best')

        axes[0].set_title("Projectors")
        axes[1].set_title("AE/PS Partial Waves")

        plt.tight_layout()
        plt.show()

    def __str__(self):
        '''
        '''
        pstr = f"{self.symbol:>3s}\n"
        pstr += f"\n{'l':>3s}{'rmax':>12s}\n"
        pstr += ''.join(
            [f"{self.proj_l[ii]:>3d}{self.proj_rmax:>12.6f}\n"
                for ii in range(self.lmax)]
        )

        return pstr


class nonlr(object):
    '''
    Nonlocal projection operator from a real-space radial grid to regular 3d grid.
    '''
    pass


class nonlq(object):
    '''
    Nonlocal projection operator from a reciprocal-space radial grid to regular 3d grid.
    '''
    def __init__(
        g, fg, l, cell, encut, gmax,
        R0=[0.0, 0.0, 0.0],
        k=[0.0, 0.0, 0.0], plws=None,
        lgam=False, lsoc=False
    ):
        from sph_harm import sph_r
        from scipy.interpolate import CubicSpline as csp
        # for reciprocal space projector functions, natural boundary condition
        # (Y'' = 0) is applied at both ends.
        spl_fg = csp(g, fg, bc_type='natural')

        Bcell = np.linalg.inv(cell).T     # reciprocal space cell
        if plws is None:
            self.G = gvectors(cell, encut, k, lgam=lgam, lsoc=lsoc)
        else:
            self.G = np.asarray(plws)
        k = np.asarray(k)

        self.Gvec = np.dot(self.G + k[np.newaxis, :], TPI*Bcell)
        self.Glen = np.linalg.norm(self.Gvec, axis=1)
        self.fG = spl_fg(self.Glen)
        self.ylm = sph_r(self.Gvec, l)

        self.qproj = self.fG[:, None] * self.ylm


class radial2grid(object):
    '''
    '''

    def __init__(self,
                 r, fr, cell, encut,
                 R0=[0.0, 0.0, 0.0],
                 # bc_type='natural',
                 rlog=False,
                 reciprocal=False):
        '''
        inputs
            r: the coordinate of the radial grid (r-grid)
            fr: the function values on the r-grid
            cell:  (3,3) ndarray in units of Angstrom, the basis vectors of the regular grid
            encut: the energy cutoff, which determines the grid size of the
                   regurlar grid
            R0: coordinate of the center of the core region
            # bc_type: boundary condition to interpolate r/fr on the radial grid
            rlog: logarithmic radial grid?
            reciprocal: r/fr defined in reciprocal space?
        '''
        from scipy.interpolate import CubicSpline as csp

        if not rlog:
            if reciprocal:
                fr_cs = csp(r, fr, bc_type='natural')
            else:
                fr_cs = csp(r, fr, bc_type='natural')
        else:
            pass


if __name__ == '__main__':
    import time
    xx = open('paw/potcar_ti').read()

    t0 = time.time()
    ps = pawpot(xx)

    # t1 = time.time()
    # ps.csplines()
    # t2 = time.time()
    # print(t1 - t0)
    # print(t2 - t1)

    # print(ps.symbol)
    # print(ps.lmmax, ps.lmax)
    print(ps)
    print(ps.core_ae_wfc[1][-1])

    ps.plot()
