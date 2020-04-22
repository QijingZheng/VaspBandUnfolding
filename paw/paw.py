#!/usr/bin/env python

import re
import numpy as np


class pawpot(object):
    '''
    Read projectors from VASP PBE POTCAR.
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



if __name__ == '__main__':
    import time
    xx = open('potcar_ti').read()
    t0 = time.time()
    ps = pawpot(xx)
    t1 = time.time()
    # print(t1 - t0)
    # print(ps.symbol)

    print(ps.lmmax, ps.lmax)

    # import numpy as np
    # import matplotlib.pyplot as plt
    #
    # figure = plt.figure()
    # ax = plt.subplot()
    #
    # nproj = ps.rprojs.shape[0]
    # # for ii in range(nproj):
    # #     # ax.plot(ps.proj_rgrid, ps.rprojs[ii], label=f"L = {ps.proj_l[ii]}")
    # #     l, = ax.plot(ps.core_rgrid, ps.core_ae_wfc[ii], label=f"L = {ps.proj_l[ii]}")
    # #     ax.plot(ps.core_rgrid, ps.core_ps_wfc[ii], ls = ':',
    # #             color=l.get_color())
    #
    # ax.plot(ps.core_rgrid, ps.core_aepot)
    #
    # ax.axhline(y=0, ls=':', color='k', alpha=0.6)
    # ax.axvline(x=ps.proj_rmax, ls=':', color='k', alpha=0.6)
    #
    # plt.legend(loc='best')
    #
    # plt.tight_layout()
    # plt.show()
