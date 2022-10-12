#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False

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

phi_ae, phi_core_ae, phi_core_ps = ae_wfc.get_ae_wfc(iband=which_band, lcore=True)
phi_ps = ps_wfc.get_ps_wfc(iband=which_band, norm=False, ngrid=ae_wfc._aegrid)

r_ps = np.arange(phi_ps.shape[-1]) * L / phi_ps.shape[-1]
r_ae = np.arange(phi_ae.shape[-1]) * L / phi_ae.shape[-1]

x, y = np.meshgrid(r_ps - L/2, r_ps - L/2)

orb_min = phi_ae[0].min()
orb_max = phi_ae[0].max()

orb_x0_plane = [
    phi_ae[0],
    phi_ps[0],
    phi_core_ae[0],
    phi_core_ae[0] - phi_core_ps[0],
    phi_core_ps[0],
]
################################################################################
fig = plt.figure(
  figsize=(9.6, 4.5),
  dpi=300,
)

axes = [plt.subplot(2, 3, ii+1) for ii in range(6)]
cax  = axes[1].inset_axes([-0.1, 0.10, 1.20, 0.06])

atoms_colors = {
    'C': 'black',
    'O': 'blue'
}
wfc_labels = [
    r'$\psi_\mathrm{ae}$',
    r'$\tilde\psi_\mathrm{ps}$',
    r'$\psi_\mathrm{ae}^c$',
    r'$\psi_\mathrm{ae}^c - \tilde\psi_\mathrm{ps}^c$',
    r'$\tilde\psi_\mathrm{ps}^c$',
]
paw_rc = {'C':0.809, 'O':0.822}

i_phi = 0
for ii in range(6):
    ax = axes[ii]
    ax.set_aspect(1.0)

    if ii == 1:
        ax.axis('off')
        cbar = plt.colorbar(
            orb_map, ax=axes[1], cax=cax,
            orientation='horizontal',
            extend='both',
            ticks=[-0.01, 0, 0.01]
        )
        cbar.ax.tick_params(labelsize='small')
        cbar.ax.xaxis.set_ticks_position('bottom')

        leg1 = ax.legend(
            atoms_handles.values(), atoms_handles.keys(),
            loc='center', # ncol=2,
            fontsize='small',
        )
        ax.add_artist(leg1)
        continue


    wfc_c = orb_x0_plane[i_phi]
    # orb_map = ax.pcolor(
    orb_map = ax.contourf(
        x, y, np.roll(wfc_c, wfc_c.shape[0] // 2, axis=0),
        levels=30,
        cmap='PiYG', zorder=1,
        vmin=orb_min, 
        vmax=orb_max, 
    )

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

    ax.text(0.95, 0.95,
        wfc_labels[i_phi],
        ha="right",
        va="top",
        # fontsize='small',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='w', alpha=0.5, lw=0.5,)
    )
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)

    if ii >= 3:
        ax.set_xlabel(r'$z$ [$\AA$]')
    if ii % 3 == 0:
        ax.set_ylabel(r'$y$ [$\AA$]')

    i_phi += 1

plt.tight_layout()
plt.savefig('ae-ps-core_co2_homo_wfc.png')

from subprocess import call
call('feh -xdF ae-ps-core_co2_homo_wfc.png'.split())

