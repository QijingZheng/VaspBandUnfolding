#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from vaspwfc import vaspwfc

wfc = vaspwfc()

# print "K-points: #{%2d}" "|<u(n,k) | u(n,k+dk)>: {:8.4f}" "|<u(n,k) | u(n+1,k)>: {:8.4f}"
# for ikpt in range(1, wfc._nkpts):
#     u1 = wfc.wfc_r(iband=12, ikpt=ikpt)
#     u2 = wfc.wfc_r(iband=12, ikpt=ikpt + 1)
#     u3 = wfc.wfc_r(iband=13, ikpt=ikpt)
#
#     print ikpt, np.abs(np.sum(u1.conj() * u2)), np.abs(np.sum(u1.conj() * u3))

# for ikpt in range(19, 23) + range(80, 84):
#     u1 = wfc.wfc_r(iband=12, ikpt=ikpt)
#     u2 = wfc.wfc_r(iband=12, ikpt=ikpt - 1)
#     u3 = wfc.wfc_r(iband=13, ikpt=ikpt - 1)
#
#     print ikpt, np.abs(np.sum(u1.conj() * u2)), np.abs(np.sum(u1.conj() * u3))

nbnds = 16
ikpt = 30
M = np.zeros((nbnds, nbnds), dtype=float)
for ii in range(nbnds):
    for jj in range(nbnds):
        M[ii, jj] = np.abs(np.sum(
            wfc.wfc_r(ikpt=ikpt-1, iband=jj+1).conj() *
            wfc.wfc_r(ikpt=ikpt, iband=ii+1)
        ))**2

np.savetxt("M_{:02d}.dat".format(ikpt), M, fmt='%4.2f')
