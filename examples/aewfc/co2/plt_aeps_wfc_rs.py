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
  figsize=(6.4, 5.4),
  dpi=300,
)

NS           = 4
rs           = np.linspace(0, 0.5, NS, endpoint=True)
axes         = [plt.subplot(2, NS // 2, ii+1) for ii in range(NS)]
atoms_colors = {
    'C': 'black',
    'O': 'green'
}

for ii in range(NS):
    ax = axes[ii]
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

    if ii == 2:
        leg1 = ax.legend(
            atoms_handles.values(), atoms_handles.keys(),
            loc='upper right', # ncol=2,
            fontsize='small',
        )
        ax.add_artist(leg1)

    IRS = int(rs[ii] / L * phi_ps.shape[0])
    ax.plot(
        r_ps, phi_ps[IRS, 0].real, color='r',
        ls='--', lw=1.0,
        # label=r'$\tilde\psi_j(z)$',
    )
    ax.plot(
        r_ae, phi_ae[IRS, 0].real, color='b',
        ls='-', lw=1.0,
        # label=r'$\psi_j(z)$',
    )

    ax.text(0.05, 0.05,
        r"$r = {:.2f}\,\AA$".format(rs[ii]),
        ha="left",
        va="bottom",
        fontsize='small',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='w', alpha=0.5, lw=0.5,)
    )

    if ii == 0:
        ax.legend(
            ax.get_lines()[-2:],
            [r'$\tilde\psi_\mathrm{ps}(r, z)$', r'$\psi_\mathrm{ae}(r, z)$'],
            fontsize='small',
            loc='lower right')

    ax.grid('on', color='gray', alpha=0.5, lw=0.5, ls='--')

    ax.set_xlabel(r'$z$ [$\AA$]')

    ax.set_yticklabels([])
    ax.set_ylabel(r'$\psi$ [arb. unit]')

plt.tight_layout()
plt.savefig('co2_homo_aeps_wfc_rs.png')
# plt.show()

from subprocess import call
call('feh -xdF co2_homo_aeps_wfc_rs.png'.split())

