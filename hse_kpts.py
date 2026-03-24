#!/usr/bin/env python

import numpy as np
import spglib


def get_ir_kpts(atoms, kmesh, shift=[0,0,0], symprec=1E-5):
    '''
    Get irreducible k-points in BZ, along with their weights.

    inputs
        atoms: ASE atoms object
        kmesh: k-points mesh, e.g. [10, 10, 10]
      symprec: symmetry tolerance used in spglib. Sometimes, spglib may produce
               more ir-kpts than VASP, which may be tuned by setting symprec.

    return
        kpoints with shape (n, 4), where the first 3 columns are the k-points
        vectors in fraction coordinate and the last column being the k-points
        weight.
    '''

    kmesh = np.asarray(kmesh)
    cell = (
        atoms.cell,
        atoms.get_scaled_positions(),
        atoms.get_atomic_numbers()
    )
    # mapping: a map between the grid points and ir k-points in grid
    mapping, grid = spglib.get_ir_reciprocal_mesh(kmesh, cell,
                                                  is_shift=shift,
                                                  symprec=symprec)

    ir_kpts_idx, ir_kpts_cnt = np.unique(mapping, return_counts=True)
    ir_kpts = grid[ir_kpts_idx] / kmesh.astype(float)

    return np.c_[ir_kpts, ir_kpts_cnt]


def make_kpath():
    '''
    '''
    pass

def save2kpts(kpts, fname='KPOINTS'):
    '''

    '''

    kpts = np.asarray(kpts)

    with open(fname, 'w+') as k:
        k.write('Genrated by hse_kpts.py\n')
        k.write('{:d}\n'.format(kpts.shape[0]))
        k.write('Reciprocal\n')
        np.savetxt(k, kpts, fmt='%20.14f %20.14f %20.14f %10.2f')



if __name__ == '__main__':
    from ase.io import read, write
    xx = read('../VASP_FermiSurface/examples/copper/POSCAR')
    kpts = get_ir_kpts(xx, [21, 21, 21])
    save2kpts(kpts)
