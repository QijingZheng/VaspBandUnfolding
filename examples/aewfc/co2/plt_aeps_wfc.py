#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from vaspwfc import vaspwfc
from aewfc import vasp_ae_wfc

from ase.io import read

atoms = read('POSCAR')
L = atoms.cell[-1, -1]

# the pseudo-wavefunction
ps_wfc = vaspwfc('WAVECAR', lgamma=True)
# the all-electron wavefunction
ae_wfc = vasp_ae_wfc(ps_wfc, aecut=-25)

# Find out the band that has the most core contributions
for iband in range(ps_wfc._nbands):
    # norm of the PSWFC
    cg = ps_wfc.readBandCoeff(iband=iband+1)
    ps_norm = np.sum(cg.conj() * cg).real
    print(f"#band: {iband+1:3d} -> {1 - ps_norm: 10.4f}")

which_band = 8

phi_ae = ae_wfc.get_ae_wfc(iband=which_band)
phi_ps = ps_wfc.get_ps_wfc(iband=which_band, norm=False, ngrid=ae_wfc._aegrid)

r_ps = np.arange(phi_ps.shape[-1]) * L / phi_ps.shape[-1]
r_ae = np.arange(phi_ae.shape[-1]) * L / phi_ae.shape[-1]

################################################################################
plt.style.use('ggplot')
fig = plt.figure(
  figsize=(4.8, 3.0),
  dpi=300,
)
ax = plt.subplot()

atoms_colors = {
    'C': 'black',
    'O': 'green'
}
atoms_handles = {}
for iatom in range(len(atoms)):
    ss = atoms.get_chemical_symbols()[iatom]
    cc = atoms_colors[ss]
    l, = ax.plot(atoms.positions[iatom, -1], [0],
            ls='none',
            marker='o', mfc='w', ms=5,
            mew=1.0,
            color=cc, label=ss)
    if ss not in atoms_handles:
        atoms_handles[ss] = l

leg1 = ax.legend(
    atoms_handles.values(), atoms_handles.keys(),
    loc='lower left', #ncol=2
)
ax.add_artist(leg1)

ax.plot(
    r_ps, phi_ps[0, 0].real, color='r',
    ls='--', lw=1.0,
    # label=r'$\tilde\psi_j(z)$',
)
ax.plot(
    r_ae, phi_ae[0, 0].real, color='b',
    ls='-', lw=1.0,
    # label=r'$\psi_j(z)$',
)

ax.legend(
    ax.get_lines()[-2:],
    [r'$\tilde\psi_\mathrm{ps}(z)$', r'$\psi_\mathrm{ae}(z)$'],
    loc='lower right')

ax.grid('on', color='gray', alpha=0.5, lw=0.5, ls='--')

ax.set_xlabel(r'$z$ [$\AA$]')
ax.set_ylabel(r'$\psi$ [arb. unit]')

plt.tight_layout()
plt.savefig('co2_homo_aeps_wfc.png')
plt.show()
