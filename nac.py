#!/usr/bin/env python
# -*- coding: utf-8 -*-   

############################################################
import numpy as np
import multiprocessing
from vaspwfc import vaspwfc

############################################################

def nac_from_vaspwfc(waveA, waveB, dt=1.0):
    '''
    Calculate Nonadiabatic Couplings (NAC) from two WAVECARs
    <psi_i(t)| d/dt |(psi_j(t))> ~=~
                                    (<psi_i(t)|psi_j(t+dt)> -
                                     <psi_j(t)|psi_i(t+dt)>) / (2dt)
    '''

    pass
