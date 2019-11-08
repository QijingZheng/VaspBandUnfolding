#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from vaspwfc import vaspwfc


############################################################
def save2vesta1(phi=None, poscar='POSCAR', prefix='wfc',
                lgam=False, lreal=False):
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

    with open(prefix + '_r.vasp', 'w') as out:
        out.write(head)
        nwrite = 0
        for kk in range(nz):
            for jj in range(ny):
                for ii in range(nx):
                    nwrite += 1
                    out.write('%16.8E ' % phi.real[ii, jj, kk])
                    if nwrite % 10 == 0:
                        out.write('\n')
    if not (lgam or lreal):
        with open(prefix + '_i.vasp', 'w') as out:
            out.write(head)
            nwrite = 0
            for kk in range(nz):
                for jj in range(ny):
                    for ii in range(nx):
                        nwrite += 1
                        out.write('%16.8E ' % phi.imag[ii, jj, kk])
                        if nwrite % 10 == 0:
                            out.write('\n')


def save2vesta2(phi=None, poscar='POSCAR', prefix='wfc',
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

    # ncol = 10
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

if __name__ == "__main__":
    from time import time
    wfc = vaspwfc('WAVECAR', lsorbit=True)
    phi = wfc.wfc_r(iband=54)[0]

    t0 = time()
    save2vesta1(phi, prefix='s1')
    t1 = time()
    save2vesta2(phi, prefix='s2')
    t2 = time()

    print t1 - t0, t2 - t1
