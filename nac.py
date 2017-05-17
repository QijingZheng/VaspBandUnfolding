#!/usr/bin/env python
# -*- coding: utf-8 -*-   

############################################################
import numpy as np
import multiprocessing
from vaspwfc import vaspwfc

############################################################

def nac_from_vaspwfc(waveA, waveB, dt=1.0
                     ikpt=1, ispin=1):
    '''
    Calculate Nonadiabatic Couplings (NAC) from two WAVECARs
    <psi_i(t)| d/dt |(psi_j(t))> ~=~
                                    (<psi_i(t)|psi_j(t+dt)> -
                                     <psi_j(t)|psi_i(t+dt)>) / (2dt)
    inputs:
        waveA:  path of WAVECAR A
        waveB:  path of WAVECAR B
        dt:     ionic time step, in [fs]          
        ikpt:   which k-point, starting from 1
        ispin:  which spin, 1 or 2
    '''

    phi_i = vaspwfc(waveA)
    phi_j = vaspwfc(waveB)

    print 'Calculating NACs between <%s> and <%s>' % (waveA, waveB)

    assert phi_i._nbands == phi_j._nbands, '#bands not match!'
    assert phi_i._nplws[ikpt-1] == phi_j.nplw[ikpt-1], '#nplws not match!'

    nbands = phi_i._nbands
    nacs = np.zeros((nbands, nbands), dtype=np.complex128)

    for ii in range(nbands):
        for jj in range(ii):
            ib1 = ii + 1
            ib2 = jj + 1

            ci_t   = phi_i.readBandCoeff(ispin, ikpt, ib1, norm=True)
            cj_t   = phi_j.readBandCoeff(ispin, ikpt, ib2, norm=True)

            ci_tdt = phi_i.readBandCoeff(ispin, ikpt, ib1, norm=True)
            cj_tdt = phi_j.readBandCoeff(ispin, ikpt, ib2, norm=True)

            nacs[jj, ii] = np.sum(ci_t.conj() * cj_dtd) - np.sum(cj_t.conj() * ci_tdt)
            nacs[ii, jj] = -nacs[jj, ii]

    return nacs / (2 * dt)
            

