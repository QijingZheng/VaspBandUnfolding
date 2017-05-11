#!/usr/bin/env python
# -*- coding: utf-8 -*-   

############################################################
import numpy as np
from vaspwfc import vaspwfc
############################################################

def find_K_from_k(k, M):
    '''
    Get the K vector of the supercell onto which the k vector of the primitive
    cell folds. The unfoliding vector G, which satisfy the following equation,
    is also returned.

        k = K + G

    where G is a reciprocal space vector of supercell
    '''

    M = np.array(M)
    Kc = np.dot(k, M.T)
    G = np.array(
            np.round(Kc), dtype=int)
    KG = Kc - np.round(Kc)

    return KG, G

def LorentzSmearing(x, x0, sigma=0.02):
    '''
    Simulate the Delta function by a Lorentzian shape function
        
        \Delta(x) = \lim_{\sigma\to 0}  Lorentzian
    '''

    return 1./ np.pi * sigma**2 / ((x - x0)**2 + sigma**2)

def GaussianSmearing(x, x0, sigma=0.02):
    '''
    Simulate the Delta function by a Lorentzian shape function
        
        \Delta(x) = \lim_{\sigma\to 0} Gaussian
    '''

    return 1. / (np.sqrt(2*np.pi) * sigma) * np.exp(-(x - x0)**2 / (2*sigma**2))

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
        # all the k-point coordinate in reciprocal space.
        self.kvecs = self.wfc._kvecs
        # all the ks energies
        self.bands = self.wfc._bands

        # G-vectors within the cutoff sphere, let's just do it once for all.
        # self.allGvecs = np.array([self.wfc.gvectors(ikpt=kpt+1)
        #                           for kpt in range(self.wfc._nkpts)], dtype=int)

    def get_ovlap_G(self, ikpt=1, epsilon=1E-5):
        '''
        Get subset of the reciprocal space vectors of the supercell,
        specifically the ones that match the reciprocal space vectors of the
        primitive cell.
        '''

        assert 1 <= ikpt <= self.wfc._nkpts, 'Invalid K-point index!'

        # Reciprocal space vectors of the supercell in fractional unit
        Gvecs = self.wfc.gvectors(ikpt=ikpt)
        # Gvecs = self.allGvecs[ikpt - 1]

        # Shape of Gvecs: (nplws, 3)
        # iGvecs = np.arange(Gvecs.shape[0], dtype=int)

        # Reciprocal space vectors of the primitive cell
        gvecs = np.dot(Gvecs, np.linalg.inv(self.M).T)
        # Deviation from the perfect sites
        gd = gvecs - np.round(gvecs)
        # match = np.linalg.norm(gd, axis=1) < epsilon
        match = np.alltrue(
                    np.abs(gd) < epsilon, axis=1
                )

        # return Gvecs[match], iGvecs[match]
        return Gvecs[match], Gvecs

    def find_K_index(self, K0):
        '''
        Find the index of K0.
        '''

        for ii in range(self.wfc._nkpts):
            if np.alltrue(
                    np.abs(self.wfc._kvec[ii] - K0) < 1E-5
               ):
                return ii + 1

    def spectral_weight_k(self, k0):
        '''
        Spectral weight for a given k:

            P_{Km}(k) = \sum_n |<Km | kn>|^2

        which is equivalent to

            P_{Km}(k) = \sum_{G} |C_{Km}(G + k - K)|^2

        where {G} is a subset of the reciprocal space vectors of the supercell.
        '''

        # find the K0 onto which k0 folds
        # k0 = G0 + K0
        K0, G0 = find_K_from_k(k0, self.M)
        # find index of K0
        ikpt = self.find_K_index(K0)

        # get the overlap G-vectors
        Gvalid, Gall = self.get_ovlap_G(ikpt=ikpt)
        # Gnew = Gvalid + k0 - K0
        Goffset = Gvalid + G0[np.newaxis, :]

        # Index of the Gvalid in 3D grid
        GallIndex = Gall % self.wfc._ngrid[np.newaxis, :]
        GoffsetIndex   = Goffset % self.wfc._ngrid[np.newaxis, :]

        # 3d grid for planewave coefficients
        wfc_k_3D = np.zeros(self.wfc._ngrid, dtype=np.complex)

        # the weights and corresponding energies
        P_Km = np.zeros(self.wfc._nbands, dtype=float)
        E_Km = np.zeros(self.wfc._nbands, dtype=float)

        for nb in range(self.wfc._nbands):
            # initialize the array to zero, which is unnecessary since the
            # GallIndex is the same for the same K-point
            # wfc_k_3D[:,:,:] = 0.0

            # pad the coefficients to 3D grid
            wfc_k_3D[GallIndex[:,0], GallIndex[:,1], GallIndex[:,2]] = \
                    self.wfc.readBandCoeff(ispin=1, ikpt=ikpt, iband=nb + 1, norm=True)
            # energy
            E_Km[nb] = self.bands[0,ikpt-1,nb]
            # spectral weight
            P_Km[nb] = np.linalg.norm(wfc_k_3D[GnewIndex[:,0], GnewIndex[:,1], GnewIndex[:,2]])**2

        return E_Km, P_Km
