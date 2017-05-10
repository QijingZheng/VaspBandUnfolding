#!/usr/bin/env python
# -*- coding: utf-8 -*-   

import numpy as np
from vaspwfc import vaspwfc
############################################################

class unfold():
    '''
    Unfold the band structure from Supercell calculation into a primitive cell and
    obtain the effective band structure (EBS).
    
    REF:
    "Extracting E versus k effective band structure from supercell
     calculations on alloys and impurities"
    Phys. Rev. B 85, 085201 (2012)
    '''

    def __init__(self, M=None, wavecar='WAVECAR'):
        '''
        Initialization.

        M is the transformation matrix between supercell and primitive cell: 

            M = np.dot(A, np.linalg.inv(a))     

        In real space, the basis vectors of Supercell (A) and those of the
        primitive cell (a) satisfy:

            A = np.dot(M, a);      a = np.dot(np.linalg.inv(M), A)

        Whereas in reciprocal space

            b = np.dot(M.T, B);    B = np.dot(np.linalg.inv(M).T, b)    

        wfc is the WAVECAR file that contains the wavefunction information of a
        supercell calculation.
        '''

        self.M = np.array(M, dtype=float)
        assert self.M.shape == (3,3), 'Shape of the tranformation matrix must be (3,3)'

        self.wfc = vaspwfc(wavecar)

    def get_ovlap_G(self, ikpt=1, epsilon=1E-5):
        '''
        Get subset of the reciprocal space vectors of the supercell,
        specifically the ones that match the reciprocal space vectors of the
        primitive cell.
        '''

        # Reciprocal space vectors of the supercell in fractional unit
        Gvecs = self.wfc.gvectors(ikpt=ikpt)
        # Shape of Gvecs: (nplws, 3)
        iGvecs = np.arange(Gvecs.shape[0], dtype=int)

        # Reciprocal space vectors of the primitive cell
        gvecs = np.dot(Gvecs, np.linalg.inv(self.M).T)
        # Deviation from the perfect sites
        gd = gvecs - np.round(gvecs)
        # match = np.linalg.norm(gd, axis=1) < epsilon
        match = np.alltrue(
                    np.abs(gd) < epsilon, axis=1
                )
        return Gvecs[match], iGvecs[match]

    def spectral_weight_k(self, k0):
        '''
        Spectral weight for a given k:

            P_{Km} = \sum_n |<Km | kn>|^2

        which is equivalent to

            P_{Km} = \sum_{G} |C_{Km}(G + k - K)|^2

        where {G} is a subset of the reciprocal space vectors of the supercell.
        '''
        pass
