#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from procar import procar
from unfold import unfold, EBS_scatter
from unfold import make_kpath, removeDuplicateKpoints, find_K_from_k, save2VaspKPOINTS


# The tranformation matrix between supercell and primitive cell.
M = [[3.0, 0.0, 0.0],
     [0.0, 3.0, 0.0],
     [0.0, 0.0, 1.0]]

# high-symmetry point of a Hexagonal BZ in fractional coordinate
kpts = [[0.0, 0.5, 0.0],            # M
        [0.0, 0.0, 0.0],            # G
        [1./3, 1./3, 0.0],          # K
        [0.0, 0.5, 0.0]]            # M

# basis vector of the primitive cell
cell = [[ 3.1903160000000002,    0.0000000000000000,    0.0000000000000000],
        [-1.5951580000000001,    2.7628940000000002,    0.0000000000000000],
        [ 0.0000000000000000,    0.0000000000000000,   30.5692707333920026]]

# create band path from the high-symmetry points, 30 points inbetween each pair
# of high-symmetry points
kpath = make_kpath(kpts, nseg=30)
K_in_sup = []
for kk in kpath:
    kg, g = find_K_from_k(kk, M)
    K_in_sup.append(kg)

# remove the duplicate K-points
reducedK, kmap = removeDuplicateKpoints(K_in_sup, return_map=True)

if not os.path.isfile('KPOINTS'):
    # save to VASP KPOINTS
    save2VaspKPOINTS(reducedK)


if os.path.isfile('WAVECAR'):
    if os.path.isfile('awht.npy'):
        atomic_whts = np.load('awht.npy')
    else:
        p           = procar()
        # The atomic contribution of Ce, Mo, S to each KS states
        atomic_whts = [p.get_pw(0)[:,kmap,:], p.get_pw("1:18")[:,kmap,:], p.get_pw("18:54")[:,kmap,:]]
        np.save('awht.npy', atomic_whts)

    if os.path.isfile('sw.npy'):
        sw = np.load('sw.npy')
    else:
        WaveSuper   = unfold(M=M, wavecar='WAVECAR')
        sw = WaveSuper.spectral_weight(kpath)
        np.save('sw.npy', sw)

    EBS_scatter(kpath, cell, sw,
                atomic_whts,
                atomic_colors=['blue', "red", 'green'],
                nseg=30, eref=-1.0671,
                ylim=(-3, 3), 
                kpath_label = ['M', 'G', "K", "M"],
                factor=20)
