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
                                     <psi_i(t+dt)|psi_j(t)>) / (2dt)
    inputs:
        waveA:  path of WAVECAR A
        waveB:  path of WAVECAR B
        gamma:  gamma version wavecar
        dt:     ionic time step, in [fs]          
        ikpt:   k-point index, starting from 1 to NKPTS
        ispin:  spin index, 1 or 2

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!!! Note, this method is much slower than fortran code. !!!!
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

    from time import time
    t1 = time()
    for ii in range(nbasis):
        for jj in range(ii):
            ib1 = ii + bmin
            ib2 = jj + bmin

            # t1 = time()
            ci_t   = phi_i.readBandCoeff(ispin, ikpt, ib1, norm=True)
            cj_t   = phi_i.readBandCoeff(ispin, ikpt, ib2, norm=True)

            ci_tdt = phi_j.readBandCoeff(ispin, ikpt, ib1, norm=True)
            cj_tdt = phi_j.readBandCoeff(ispin, ikpt, ib2, norm=True)
            # t2 = time()
            # print '1. Elapsed Time: %.4f [s]' % (t2 - t1)

            tmp = np.sum(ci_t.conj() * cj_tdt) - np.sum(ci_tdt.conj() * cj_t)
            # t3 = time()
            # print '2. Elapsed Time: %.4f [s]' % (t3 - t2)

            nacs[ii,jj] = tmp.real if gamma else tmp
            if gamma:
                nacs[jj,ii] = -nacs[ii,jj]
            else:
                nacs[jj,ii] = -np.conj(nacs[ii,jj])

    t2 = time()
    print '1. Elapsed Time: %.4f [s]' % (t2 - t1)

    # EnT = (phi_i._bands[ispin-1,ikpt-1,:] + phi_j._bands[ispin-1,ikpt-1,:]) / 2.
    EnT = phi_i._bands[ispin-1,ikpt-1,bmin-1:bmax]

    # close the wavecar
    phi_i._wfc.close()
    phi_j._wfc.close()

    # return EnT, nacs / (2 * dt)
    return EnT, nacs


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

        np.savetxt(prefix + '/eig.txt', et[np.newaxis, :])
        np.savetxt(prefix + '/nac.txt', nc.flatten()[np.newaxis, :])


############################################################
# a test
############################################################

if __name__ == '__main__':
    T_start = 1
    T_end   = 10

    WaveCars = ['./run/%03d/WAVECAR' % (ii + 1) for ii in range(T_start-1, T_end)]

    parallel_nac_calc(WaveCars, bmin=325, bmax=340)

