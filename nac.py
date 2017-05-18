#!/usr/bin/env python
# -*- coding: utf-8 -*-   

############################################################
import os
import numpy as np
import multiprocessing
from vaspwfc import vaspwfc

############################################################

def nac_from_vaspwfc(waveA, waveB, gamma=True,
                     bmin=None, bmax=None,
                     dt=1.0, ikpt=1, ispin=1):
    '''
    Calculate Nonadiabatic Couplings (NAC) from two WAVECARs
    <psi_i(t)| d/dt |(psi_j(t))> ~=~
                                    (<psi_i(t)|psi_j(t+dt)> -
                                     <psi_j(t)|psi_i(t+dt)>) / (2dt)
    inputs:
        waveA:  path of WAVECAR A
        waveB:  path of WAVECAR B
        gamma:  gamma version wavecar
        dt:     ionic time step, in [fs]          
        ikpt:   k-point index, starting from 1 to NKPTS
        ispin:  spin index, 1 or 2

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!! Note, this method is way too slow than fortran code !!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    '''

    phi_i = vaspwfc(waveA)      # wavecar at t
    phi_j = vaspwfc(waveB)      # wavecar at t + dt

    print 'Calculating NACs between <%s> and <%s>' % (waveA, waveB)

    assert phi_i._nbands == phi_j._nbands, '#bands not match!'
    assert phi_i._nplws[ikpt-1] == phi_j._nplws[ikpt-1], '#nplws not match!'

    bmin = 1 if bmin is None else bmin
    bmax = phi_i._nbands if bmax is None else bmax
    nbasis = bmax - bmin + 1

    nacType = np.float if gamma else np.complex
    nacs = np.zeros((nbasis, nbasis), dtype=nacType)

    # from time import time
    for ii in range(nbasis):
        for jj in range(ii):
            # t1 = time()
            ib1 = ii + bmin
            ib2 = jj + bmin

            ci_t   = phi_i.readBandCoeff(ispin, ikpt, ib1, norm=True)
            cj_t   = phi_i.readBandCoeff(ispin, ikpt, ib2, norm=True)

            ci_tdt = phi_j.readBandCoeff(ispin, ikpt, ib1, norm=True)
            cj_tdt = phi_j.readBandCoeff(ispin, ikpt, ib2, norm=True)

            tmp = np.sum(ci_t.conj() * cj_tdt) - np.sum(cj_t.conj() * ci_tdt)

            nacs[ii,jj] = tmp.real if gamma else tmp
            nacs[jj,ii] = -nacs[ii,jj]

            # t2 = time()
            # print 'Elapsed Time: %.4f [s]' % (t2 - t1)

    # EnT = (phi_i._bands[ispin-1,ikpt-1,:] + phi_j._bands[ispin-1,ikpt-1,:]) / 2.
    EnT = phi_i._bands[ispin-1,ikpt-1,:]

    # close the wavecar
    phi_i._wfc.close()
    phi_j._wfc.close()

    return EnT, nacs / (2 * dt)


def parallel_nac_calc(runDirs, nproc=None, gamma=True,
                      bmin=None, bmax=None,
                      ikpt=1, ispin=1, dt=1.0):
    '''
    Parallel calculation of NACs using python multiprocessing package.
    '''
    import multiprocessing
    
    nproc = multiprocessing.cpu_count() if nproc is None else nproc
    pool = multiprocessing.Pool(processes=nproc)

    results = []
    for w1, w2 in zip(runDirs[:-1], runDirs[1:]):
        res = pool.apply_async(nac_from_vaspwfc, (w1, w2, gamma, bmin, bmax, dt, ikpt, ispin,))
        results.append(res)

    for ii in range(len(runDirs)-1):
        et, nc = results[ii].get()

        prefix = os.path.dirname(runDirs[ii])

        np.savetxt(prefix + '/eig.txt', et)
        np.savetxt(prefix + '/nac.txt', nc)


############################################################
# a test
############################################################

if __name__ == '__main__':
    WaveCars = ['./run/%03d/WAVECAR' % (ii + 1) for ii in range(10)]

    parallel_nac_calc(WaveCars, bmin=325, bmax=340)

