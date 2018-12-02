#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from vaspwfc import vaspwfc


def reorder_band(wavecar='WAVECAR', max_nbnds=None,
                 olap_cut=0.75, save_olap=False,
                 save_idx=True,
                 nkseg=None):
    '''
    Re-order bands by maximizing the overlap with bands at previous k-point.

    The overlap is defined as the inner product of the periodic part of the
    Bloch wavefunctions.
                            < u(n, k) | u(m, k-1) >

    Note, however, the WAVECAR only contains the pseudo-wavefunction, and thus
    the pseudo u(n,k) are used in this function. Moreover, since the number of
    planewaves for each k-points are different, the inner product is performed
    in real space.

    The overlap maximalization procedure is as follows:
        1. Pick out those bands with large overlap (> olap_cut).
        2. Assign those un-picked bands by maximizing the overlap.
    '''

    wfc = vaspwfc(wavecar)

    if max_nbnds is None:        # Maximum number of bands for band reordering
        nbnds = wfc._nbands
    else:
        nbnds = max_nbnds
    nspin = wfc._nspin           # Number of spin
    nkpts = wfc._nkpts           # Number of k-points
    ebands = wfc._bands.copy()   # Band energies
    kpath, kbound = wfc.get_kpath(nkseg=nkseg)  # The k-points path
    nx, ny, nz = wfc._ngrid      # FFT mesh, minimum grid size is enough

    # the band index arry for each k-points
    band_idx = -1 * np.ones((nspin, nkpts, nbnds), dtype=int)
    # initialize the band index for the first k-point
    for ispin in range(nspin):
        band_idx[ispin, 0, :] = range(nbnds)

    # The real space wavefunction at current k-point
    Unk = np.zeros((nbnds, nx, ny, nz), dtype=complex)
    # The real space wavefunction at previous k-point
    Unk_1 = np.zeros((nbnds, nx, ny, nz), dtype=complex)
    # The overlap matrix between neighbouring k-point
    M = np.zeros((nbnds, nbnds), dtype=float)

    # Loop over spin
    for ispin in range(nspin):
        # Loop over k-points
        for ikpt in range(nkpts):

            G = wfc.gvectors(ikpt=ikpt+1)

            if ikpt == 0:
                # Unk(r) at previous k-points
                for iband in range(nbnds):
                    # No need to perform inner product in real space
                    # Unk_1[iband, ...] = wfc.wfc_r(ikpt=ikpt+1, ispin=ispin+1,
                    #                               iband=iband+1, ngrid=(nx, ny, nz))
                    #
                    Unk_1[iband, G[:, 0], G[:, 1], G[:, 2]] = \
                        wfc.readBandCoeff(
                            ispin=ispin+1, ikpt=ikpt+1, iband=iband+1, norm=True
                        )
                continue

            print("Parsing k-points #{}...".format(ikpt))

            # Unk(r) at current k-points
            for iband in range(nbnds):
                # No need to perform inner product in real space
                # Unk[iband, ...] = wfc.wfc_r(ikpt=ikpt+1, ispin=ispin+1,
                #                             iband=iband+1, ngrid=(nx, ny, nz))
                Unk[iband, G[:, 0], G[:, 1], G[:, 2]] = \
                    wfc.readBandCoeff(
                        ispin=ispin+1, ikpt=ikpt+1, iband=iband+1, norm=True
                    )

            # Calculating the overlap matrix
            for ii in range(nbnds):
                for jj in range(nbnds):
                    M[ii, jj] = np.abs(
                        np.sum(Unk_1[jj, ...].conj() * Unk[ii, ...])
                    )**2

            if save_olap:
                np.savetxt('M_{:d}_{:03d}.dat'.format(ispin+1, ikpt+1), M,
                           fmt='%4.2f')

            ############################################################
            # Use a rotation matrix to re-asign the band index
            ############################################################
            R = np.zeros_like(M, dtype=int)         # the rotation matrix
            R[M >= olap_cut] = 1                    #
            flag = np.zeros(nbnds, dtype=int)       # alread assigned?

            for ii in range(nbnds):
                if not R[ii].sum() == 1:
                    R[ii] = 0
                    for jj in np.argsort(M[ii])[::-1]:
                        if flag[jj] == 0:
                            R[ii, jj] = 1
                            break
                flag += R[ii]
            assert np.sum(flag, dtype=int) == nbnds
            assert int(np.abs(np.linalg.det(R))) == 1
            band_idx[ispin, ikpt] = np.dot(R, band_idx[ispin, ikpt-1])

            ############################################################
            # Method used by QE PP/bands.f90
            ############################################################
            # # assign the band index by maximum overlap > "olap_cut"
            # max_olap = np.max(M, axis=1)
            # max_olap_idx = np.argmax(M, axis=1)
            # olap_filter = max_olap >= olap_cut
            # # find out the maximum overlap
            # band_idx[ispin, ikpt, olap_filter] = band_idx[ispin, ikpt-1,
            #                                               max_olap_idx[olap_filter]]
            #
            # # For those un-assigned bands, maximize the overlap
            # for ii in range(nbnds):
            #     if band_idx[ispin, ikpt, ii] == -1:
            #         # index from maximum to minimum
            #         olap_sort_order = np.argsort(M[ii])[::-1]
            #         for jj in olap_sort_order:
            #             if band_idx[ispin, ikpt-1, jj] in band_idx[ispin,
            #                                                        ikpt]:
            #                 continue
            #             else:
            #                 band_idx[ispin, ikpt, ii] = band_idx[ispin,
            #                                                      ikpt-1, jj]
            #                 break

            # set current wavefunction to previous
            Unk_1[:, :, :, :] = Unk[:, :, :, :]

    wfc._wfc.close()    # close the wavecar

    # re-ordering the band according to the band index
    bands_new = np.zeros((nspin, nkpts, nbnds), dtype=float)
    band_order = np.argsort(band_idx, axis=-1)
    for ispin in range(nspin):
        for ikpt in range(nkpts):
            bands_new[ispin, ikpt, :] = ebands[ispin,
                                               ikpt, band_order[ispin, ikpt, :]]

    if save_idx:
        for ispin in range(nspin):
            np.savetxt('bo_{}.dat'.format(ispin+1), band_idx[ispin] + 1,
                       fmt='%3d')

    return ebands, bands_new, kpath, kbound


############################################################
if __name__ == "__main__":

    if not os.path.isfile('bands_n.npy'):
        bands_o, bands_n, kpath, kbound = reorder_band(max_nbnds=16,
                                                       olap_cut=0.70,
                                                       save_olap=True)

        np.save('bands_o', bands_o)
        np.save('bands_n', bands_n)
        np.savetxt('kpath.dat', kpath)
        np.savetxt('kbound.dat', kbound)

    bands_o = np.load('bands_o.npy')
    bands_n = np.load('bands_n.npy')
    kpath = np.loadtxt('kpath.dat')
    kbound = np.loadtxt('kbound.dat')

    import matplotlib as mpl
    mpl.use('agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=1, ncols=2,
                             sharex=False, sharey=True)
    fig.set_size_inches((5.4, 4.0))

    nspin, nkpts, nbnds = bands_n.shape

    for ispin in range(nspin):
        for iband in range(nbnds):
            l1, = axes[0].plot(kpath, bands_o[ispin, :, iband],
                               ls='None',
                               marker='o', ms=2.5, mfc='none',
                               markeredgewidth=0.6,
                               # alpha=0.5,
                               # markeredgecolor='k',
                               zorder=0)
            l2, = axes[1].plot(kpath, bands_n[ispin, :, iband],
                               ls='-', lw=0.8, zorder=1,
                               color=l1.get_color(),
                               # marker=r'${}$'.format(iband+1),
                               # ms=2,
                               # marker=r'$\mathcircled{{{}}}$'.format(iband+1),
                               )
            # ax.plot(kpath, bands_o[:, iband + 1], marker='o', ms=3, alpha=0.4,
            #         zorder=0)
            # ax.plot(kpath, 1.7681 + bands_n[:, iband + 1], ls='-', lw=1.0, zorder=1)

    for ii in range(2):
        ax = axes[ii]
        for kb in kbound:
            ax.axvline(x=kb, lw=0.5, ls=':')
        # ax.axvline(x=kpath[59], lw=0.5, ls=':')
        if ii == 0:
            ax.set_ylabel('Energy [eV]')

        ax.set_xlim(kpath[0], kpath[-1])
        # ax.set_ylim(-1, 2)

    plt.tight_layout(pad=0.5)
    plt.savefig('kband.png', dpi=400)

    from subprocess import call
    call('feh -xdF kband.png'.split())
