#!/usr/bin/env python

from vaspwfc import vaspwfc
from paw import nonlq, nonlr
from spinorb import read_cproj_NormalCar

import numpy as np
from ase.io import read, write
from scipy.linalg import block_diag
import time

ikpt = 3
iband = 9

poscar = read('POSCAR')
wfc = vaspwfc()

phi = wfc.readBandCoeff(iband=iband, ikpt=ikpt)
p1 = nonlq(poscar, wfc._encut, k=wfc._kvecs[ikpt-1])
beta = p1.proj(phi)

# inner product of the pseudo-wavefunction
ps_inner_prod = np.sum(phi.conj() * phi)

QIJ = block_diag(*[
    p1.pawpp[p1.element_idx[iatom]].get_Qij()
    for iatom in range(len(poscar))
])
# inner product of PAW core parts
core_inner_prod = np.dot(np.dot(beta.conj(), QIJ), beta)

# inner product of all-electron wavefunction
ae_inner_prod = (ps_inner_prod + core_inner_prod).real

# ae inner product should very close to 1.0
print("{:E}".format(ae_inner_prod - 1.0))

ikpt = 3
iband = 10
psi = wfc.readBandCoeff(iband=iband, ikpt=ikpt)
p2 = nonlq(poscar, wfc._encut, k=wfc._kvecs[ikpt-1])
alpha = p2.proj(psi)

ps_inner_prod = np.sum(psi.conj() * phi)
ae_inner_prod = ps_inner_prod + np.dot(np.dot(alpha.conj(), QIJ), beta)
print("{:E}".format(ps_inner_prod))
print("{:E}".format(ae_inner_prod))

# core_inner_prod = 0.0
# ilower = 0
# t0 = time.time()
# for iatom in range(len(poscar)):
#     ntype = p1.element_idx[iatom]
#     lmmax = p1.pawpp[ntype].lmmax
#     core_inner_prod += np.dot(
#         np.dot(
#             beta[ilower:ilower+lmmax].conj(),
#             p1.pawpp[ntype].get_Qij()
#         ),
#         beta[ilower:ilower+lmmax]
#     )
#     ilower += lmmax
# t1 = time.time()
# ae_inner_prod = core_inner_prod + ps_inner_prod
# print(ps_inner_prod)
# print(core_inner_prod)
# print(ae_inner_prod)
#
# t2 = time.time()
# QIJ = block_diag(*[
#     p1.pawpp[p1.element_idx[iatom]].get_Qij()
#     for iatom in range(len(poscar))
# ])
# t3 = time.time()
# print(np.dot(np.dot(beta.conj(), QIJ), beta))
# print(core_inner_prod)
#
# print("old:", t1 - t0)
# print("new:", t3 - t2)
