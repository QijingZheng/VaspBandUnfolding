#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from vaspwfc import vaspwfc
from aewfc import vasp_ae_wfc

from ase.io import read

atoms = read('POSCAR')
L = atoms.cell[-1, -1]

# the pseudo-wavefunction
ps_wfc = vaspwfc('WAVECAR', lgamma=True)
# the all-electron wavefunction
ae_wfc = vasp_ae_wfc(ps_wfc, aecut=-25)

which_band = 8

phi_ae = ae_wfc.get_ae_wfc(iband=which_band)
phi_ps = ps_wfc.get_ps_wfc(iband=which_band, norm=False, ngrid=ae_wfc._aegrid)

r_ps = np.arange(phi_ps.shape[-1]) * L / phi_ps.shape[-1]
r_ae = np.arange(phi_ae.shape[-1]) * L / phi_ae.shape[-1]

x, y = np.meshgrid(r_ps - L/2, r_ps - L/2)
################################################################################
fig = plt.figure(
  figsize=(9, 2.4),
  dpi=300,
)

axes = [plt.subplot(1, 3, ii+1) for ii in range(3)]
atoms_colors = {
    'C': 'black',
    'O': 'blue'
}

axes[0].pcolor(
    x, y, np.roll(phi_ps[0,:,:], phi_ps.shape[1]//2, axis=0),
    cmap='PiYG', zorder=1,
)
axes[1].pcolor(
    x, y,
    np.roll(phi_ae[0,:,:], phi_ae.shape[1]//2, axis=0) - \
    np.roll(phi_ps[0,:,:], phi_ps.shape[1]//2, axis=0),
    cmap='PiYG', zorder=1,
)
axes[2].pcolor(
    x, y, np.roll(phi_ae[0,:,:], phi_ae.shape[1]//2, axis=0),
    cmap='PiYG', zorder=1,
)

wfc_labels = [
    r'$\tilde\psi_\mathrm{ps}$',
    r'$\psi_\mathrm{ae} - \tilde\psi_\mathrm{ps}$',
    r'$\psi_\mathrm{ae}$'
]
paw_rc = {'C':0.809, 'O':0.822}

for ii in range(3):
    ax = axes[ii]
    ax.set_aspect(1.0)

    atoms_handles = {}
    for iatom in range(len(atoms)):
        ss = atoms.get_chemical_symbols()[iatom]
        cc = atoms_colors[ss]
        l, = ax.plot(atoms.positions[iatom, -1] - L/2, [0],
                ls='none',
                marker='o', mfc='w', ms=3,
                mew=1.0,
                color=cc, label=ss,
                zorder=2,
                )
        if ss not in atoms_handles:
            atoms_handles[ss] = l

        ax.add_artist(
            Circle(
                (atoms.positions[iatom,-1]-L/2, 0), radius=paw_rc[ss],
                fill=False, lw=0.8, color='gray', alpha=0.5, ls='--'
            )
        )

    if ii == 1:
        leg1 = ax.legend(
            atoms_handles.values(), atoms_handles.keys(),
            loc='lower center', ncol=2,
            fontsize='small',
        )
        ax.add_artist(leg1)

    ax.text(0.50, 1.05,
        wfc_labels[ii],
        ha="center",
        va="bottom",
        # fontsize='small',
        transform=ax.transAxes,
        # bbox=dict(boxstyle='round', facecolor='w', alpha=0.5, lw=0.5,)
    )
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)

    ax.set_xlabel(r'$z$ [$\AA$]')
    ax.set_ylabel(r'$y$ [$\AA$]')

plt.tight_layout()
plt.savefig('orb_co2_homo.png')
# plt.show()

from subprocess import call
call('feh -xdF orb_co2_homo.png'.split())

