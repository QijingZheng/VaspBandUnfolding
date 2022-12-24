#!/usr/bin/env python

import numpy as np
from vasp_constant import *
from scipy.special import erf, erfc


class ewaldsum(object):
    '''
    Ewald summation.
    '''

    def __init__(self, atoms, Z: dict={},
            eta: float=None,
            Rcut: float=4.0,
            Gcut: float=4.0):
        '''
        '''

        assert Z, "'Z' can not be empty!\n\
                It is a dictionary containing charges for each element,\
                e.g. {'Na':1.0, 'Cl':-1.0}."

        # the poscar storing the atoms information
        self._atoms  = atoms
        self._na = len(self._atoms)
        # factional coordinates in range [0,1]
        self._scapos = self._atoms.get_scaled_positions()

        elements = np.unique(self._atoms.get_chemical_symbols())
        for elem in elements:
            if elem not in Z:
                raise ValueError(f'Charge for {elem} missing!')

        self._ZZ = np.array([Z[x] for x in self._atoms.get_chemical_symbols()])
        # z_i * z_j
        self._Zij = np.prod(
            np.meshgrid(self._ZZ, self._ZZ, indexing='ij'),
            axis=0)

        # FELECT = (the electronic charge)/(4*pi*the permittivity of free space)
        #          in atomic units this is just e^2
        self._inv_4pi_epsilon0 = FELECT

        self._Acell = np.array(self._atoms.cell)        # real-space cell
        self._Bcell = np.linalg.inv(self._Acell).T      # reciprocal-space cell
        self._omega = np.linalg.det(self._Acell)        # Volume of real-space cell

        # the decaying parameter
        if eta is None:
            self._eta = np.sqrt(np.pi) / (self._omega)**(1./3)
        else:
            self._eta = eta

        self._Rcut = Rcut
        self._Gcut = Gcut

    def get_sum_real(self):
        '''
        Real-space contribution to the Ewald sum.

                 1                              erfc(eta | r_ij + R_N |)
            U = --- \sum_{ij} \sum'_N Z_i Z_j -----------------------------
                 2                                    | r_ij + R_N |

        where the prime in \sum_N means i != j when N = 0.
        '''
        ii, jj = np.mgrid[0:self._na, 0:self._na]

        # r_i - r_j, rij of shape (natoms, natoms, 3)
        rij = self._scapos[ii,:] - self._scapos[jj,:]
        # move rij to the range [-0.5,0.5]
        rij[rij >= 0.5] -= 1.0
        rij[rij < -0.5] += 1.0
        
        ############################################################
        # contribution from N = 0 cell
        ############################################################
        rij0 = np.linalg.norm(
            np.tensordot(self._Acell, rij.T, axes=(0,0)),
            axis=0
        )
        dd = range(self._na)
        # make diagonal term non-zero to avoid divide-by-zero error
        rij0[dd, dd] = 0.1
        Uij = erfc(rij0 * self._eta) / rij0
        # set diagonal term zero
        Uij[dd, dd] = 0

        ############################################################
        # contribution from N != 0 cells
        ############################################################
        rij = rij.reshape((-1, 3)).T

        nx, ny, nz = np.array(
            self._Rcut / self._eta / np.linalg.norm(self._Acell, axis=1),
            dtype=int
        ) + 1
        Rn  = np.mgrid[-nx:nx+1, -ny:ny+1, -nz:nz+1].reshape((3,-1))
        # remove N = 0 term
        cut = np.sum(np.abs(Rn), axis=0) != 0
        Rn  = Rn[:,cut]
        
        # R_N + rij
        Rr = np.linalg.norm(
            np.tensordot(self._Acell, Rn[:,None,:] + rij[:,:,None], axes=(0,0)),
            axis=0
        )
        Uij += np.sum(
            erfc(self._eta * Rr) / Rr, axis=1
        ).reshape((self._na, self._na))

        return 0.5*Uij


    def get_sum_recp(self):
        '''
        Reciprocal-space contribution to the Ewald sum.

                  1            4pi              
            U = ----- \sum'_G ----- exp(-G^2/(4 eta^2)) \sum_{ij} Z_i Z_j exp(-i G r_ij)
                 2 V           G^2

        where the prime in \sum_G means G != 0.
        '''
        nx, ny, nz = np.array(
            self._Gcut * self._eta / np.pi / np.linalg.norm(self._Bcell, axis=1),
            dtype=int
        ) + 1
        Gn  = np.mgrid[-nx:nx+1, -ny:ny+1, -nz:nz+1].reshape((3,-1))
        # remove G = 0 term
        cut = np.sum(np.abs(Gn), axis=0) != 0
        Gn  = Gn[:,cut]

        G2 = np.linalg.norm(
            np.tensordot(self._Bcell*2*np.pi, Gn, axes=(0,0)),
            axis=0
        )**2
        expG2_invG2 = 4*np.pi * np.exp(-G2/4/self._eta**2) / G2

        # r_i - r_j, rij of shape (natoms, natoms, 3)
        # no need to move rij from [0,1] to [-0.5,0.5], which will not affect
        # the phase G*rij
        ii, jj = np.mgrid[0:self._na, 0:self._na]
        rij = self._scapos[ii,:] - self._scapos[jj,:]

        sfac = np.exp(-2j*np.pi * (rij @ Gn))

        Uij  = 0.5 * np.sum(expG2_invG2 * sfac, axis=-1) / self._omega

        return Uij.real


    def get_ewaldsum(self):
        '''
        Total Coulomb energy from Ewald summation.
        '''

        # real-space contribution
        Ur = np.sum(self.get_sum_real() * self._Zij)
        # reciprocal--space contribution
        Ug = np.sum(self.get_sum_recp() * self._Zij)

        # interaction between charges
        Us = -self._eta / np.sqrt(np.pi) * np.sum(self._ZZ**2)
        # interaction with the neutralizing background
        Un = -(2*np.pi / self._eta**2 / self._omega) * self._ZZ.sum()**2 / 4

        # total coulomb energy
        Ut = (Ur + Ug + Us + Un)*self._inv_4pi_epsilon0

        return Ut


    def get_madelung(self):
        '''
        '''
        # index for reference atom
        ii = 0
        # nearest-neighbour of ref atom
        rij = self._scapos - self._scapos[ii]
        rij[rij >= 0.5] -= 1.0
        rij[rij < -0.5] += 1.0
        rij0 = np.linalg.norm(rij @ self._Acell, axis=1)
        dd = np.arange(self._na)
        jj = dd[np.argsort(rij0)[1]]
        r0 = rij0[jj]

        Ur = self.get_sum_real() * self._Zij
        Ug = self.get_sum_recp() * self._Zij
        Ui = (Ur[ii] + Ug[ii]).sum() -\
             self._eta / np.sqrt(np.pi) * self._ZZ[ii]**2
        M  = 2*Ui * r0 / self._ZZ[ii] / self._ZZ[jj]

        return M



if __name__ == '__main__':
    from ase.io import read

    atoms = read('NaCl.vasp')
    ZZ = {'Na':1, 'Cl':-1}

    esum = ewaldsum(atoms, ZZ) 
    print(esum.get_ewaldsum())
    print(esum.get_madelung())

    atoms = read('CsCl.vasp')
    ZZ = {'Cs':1, 'Cl':-1}

    esum = ewaldsum(atoms, ZZ) 
    print(esum.get_ewaldsum())
    print(esum.get_madelung())
