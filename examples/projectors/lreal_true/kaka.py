#!/usr/bin/env python

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from vaspwfc import vaspwfc
from paw import nonlq, nonlr
from spinorb import read_cproj_NormalCar

import numpy as np
from ase.io import read, write

cprojs = read_cproj_NormalCar()
cprojs.shape = ((3, 16, 34))
iband = 12
ikpt = 3
cc = cprojs[ikpt-1, iband-1]

poscar = read('POSCAR')
wfc = vaspwfc()

ngrid = [20, 20, 120]
phi_r = wfc.wfc_r(iband=iband, ikpt=ikpt, ngrid=ngrid, rescale=1.0, norm=False)
# phi_r /= np.sqrt(np.prod(phi_r.shape))
p1 = nonlr(poscar, wfc._encut, k=wfc._kvecs[ikpt-1], ngrid=ngrid)

ILM = [(i, l, m) for i, it in zip([0, 1, 2], [0, 1, 1])
      for l in p1.pawpp[it].proj_l
      for m in range(-l, l+1)]

import time
t0 = time.time()
# beta = p1.proj(cptwf)
beta = p1.proj(phi_r)
t1 = time.time()
print(t1 - t0)

figure = plt.figure(figsize=(4,4))
ax = plt.subplot()
ax.set_aspect('equal')

ax.plot([0, np.abs(cc).max()], [0, np.abs(cc).max()], ':', lw=1.5, alpha=0.6)
ax.plot(np.abs(beta), np.abs(cc), 'bd', alpha=0.6, ms=5)
# ax.plot(beta.imag, cc.imag, 'bd', alpha=0.6, ms=5)

for ilm, x, y in zip(ILM, np.abs(beta), np.abs(cc)):
    ax.text(x, y, "({},{},{})".format(*ilm),
            ha="center",
            va="center",
            fontsize='small',
            # family='monospace',
            # fontweight='bold'
            transform=ax.transData,
            # bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

ax.set_title("Real Space Projectors")
ax.set_xlabel('My script', labelpad=5)
ax.set_ylabel('VASP NormalCar', labelpad=5)

plt.tight_layout()
plt.savefig('kaka.png', dip=360)
plt.show()
