#!/usr/bin/env python

import numpy as np
from ase.io import read
from ewald import ewaldsum

if __name__ == '__main__':

    crystals = [
        'NaCl.vasp',
        'CsCl.vasp',
        'ZnO-Hex.vasp',
        'ZnO-Cub.vasp',
        'TiO2.vasp',
        'CaF2.vasp',
    ]

    ZZ = {
        'Na':  1, 'Ca':  2,
        'Cl': -1,  'F': -1,
        'Cs':  1,
        'Ti':  4,
        'Zn':  2,
         'O': -2,
    }

    print('-' * 30)
    print(f'{"Crystal":>9s} | {"Madelung Constant":>18s}')
    print('-' * 30)

    for crys in crystals:
        atoms = read(crys)
        esum = ewaldsum(atoms, ZZ) 

        # print(esum.get_ewaldsum())
        M = esum.get_madelung()
        C = crys.replace('.vasp', '')
        print(f'{C:>9s} | {M:18.15f}')

    print('-' * 30)
