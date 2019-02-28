#!/usr/bin/env python

import os
import numpy as np


def read_cproj_NormalCar(inf='NormalCAR', save_cproj=True):
    '''
    Read NormalCAR of VASP output, which contains the coefficients of the PAW
    projector functions.

    Data stored in NormalCAR:

        WRITE(IU) LMDIM,WDES%NIONS,WDES%NRSPINORS
        WRITE(IU) CQIJ(1:LMDIM,1:LMDIM,1:WDES%NIONS,1:WDES%NRSPINORS) 
        WRITE(IU) WDES%NPROD, WDES%NPRO, WDES%NTYP
        DO NT = 1,  WDES%NTYP
          WRITE(IU) WDES%LMMAX(NT), WDES%NITYP(NT) 
        END DO
        DO ISPIN=1,WDES%ISPIN
          DO NK=1,WDES%NKPTS
            DO N=1,WDES%NB_TOT 
              WRITE(IU) CPROJ(1:WDES1%NPRO_TOT)
            END DO 
          END DO 
        END DO 
    '''
    from scipy.io import FortranFile

    ncr = FortranFile(inf, 'r')

    # rec1 = ncr.read_record(dtype=np.int32)
    rec1 = ncr.read_ints()
    lmdim, nions, nrspinors = rec1

    # rec2 = ncr.read_record(dtype=np.complex)
    cqij = np.array(ncr.read_reals()).reshape((lmdim, lmdim, nions, nrspinors),
                                              order='F')

    rec3 = ncr.read_ints()
    nprod, npro, ntyp = rec3
    # lmmax, natoms for each type of elements
    lmmax_per_typ = np.array([ncr.read_ints() for ii in range(ntyp)])

    cproj = []
    while True:
        try:
            rec = ncr.read_record(dtype=np.complex)
            cproj.append(rec)
        except:
            break
    cproj = np.array(cproj)
    if save_cproj:
        np.save('cproj', cproj)

    ncr.close()

    return cproj


def read_SocCar(inf='SocCar'):
    '''
    SocCar contains the SOC matrix elements on the basis of core AE
    wavefunctions.

    H^{soc}_{ij} = < Y_i; \sigma_1 | S L | Y_j; \sigma_2 > *
                   < f_i(r) | d V_KS(r) / dr * r^{-1} | f_j(r) >
    '''

    soc = np.loadtxt(inf)
    soc = soc[:, 0::2] + 1j * soc[:, 1::2]

    nproj = soc.shape[-1]
    soc.shape = (4, nproj, nproj)

    return soc


def get_bandInfo(inFile='OUTCAR'):
    """
    extract band energies from OUTCAR
    """

    outcar = [line for line in open(inFile) if line.strip()]

    Lvkpts_2 = -1
    for ii, line in enumerate(outcar):
        if 'NKPTS =' in line:
            nkpts = int(line.split()[3])
            nband = int(line.split()[-1])

        if "NELECT =" in line:
            nelect = float(line.split()[2])

        if 'ISPIN  =' in line:
            ispin = int(line.split()[2])

        if "k-points in reciprocal lattice and weights" in line:
            Lvkpts_1 = ii + 1

        if "Following reciprocal coordinates:" in line:
            Lvkpts_2 = ii + 2

        if 'reciprocal lattice vectors' in line:
            ibasis = ii + 1

        if 'E-fermi' in line:
            Efermi = float(line.split()[2])
            LineEfermi = ii + 1
            # break

    # basis vector of reciprocal lattice
    B = np.array([line.split()[3:] for line in outcar[ibasis:ibasis+3]],
                 dtype=float)
    # k-points vectors and weights
    if Lvkpts_2 == -1:
        tmp = np.array([line.split() for line in outcar[Lvkpts_1:Lvkpts_1+nkpts]],
                       dtype=float)
    else:
        tmp = np.array([line.split() for line in outcar[Lvkpts_2:Lvkpts_2+nkpts]],
                       dtype=float)
    vkpts = tmp[:, :3]
    wkpts = tmp[:, -1]
    wkpts /= np.sum(wkpts)

    # for ispin = 2, there are two extra lines "spin component..."
    N = (nband + 2) * nkpts * ispin + (ispin - 1) * 2
    bands = []
    # vkpts = []
    for line in outcar[LineEfermi:LineEfermi + N]:
        if 'spin component' in line or 'band No.' in line:
            continue
        if 'k-point' in line:
            # vkpts += [line.split()[3:]]
            continue
        bands.append(float(line.split()[1]))

    bands = np.array(bands, dtype=float).reshape((ispin, nkpts, nband))

    if os.path.isfile('KPOINTS'):
        kp = open('KPOINTS').readlines()

    kpt_bounds = None
    kpt_path = None
    if os.path.isfile('KPOINTS') and kp[2][0].upper() == 'L':
        Nk_in_seg = int(kp[1].split()[0])
        Nseg = nkpts / Nk_in_seg
        vkpt_diff = np.zeros_like(vkpts, dtype=float)

        for ii in range(Nseg):
            start = ii * Nk_in_seg
            end = (ii + 1) * Nk_in_seg
            vkpt_diff[start:end, :] = vkpts[start:end, :] - vkpts[start, :]

        kpt_path = np.linalg.norm(np.dot(vkpt_diff, B), axis=1)
        # kpt_path = np.sqrt(np.sum(np.dot(vkpt_diff, B)**2, axis=1))
        for ii in range(1, Nseg):
            start = ii * Nk_in_seg
            end = (ii + 1) * Nk_in_seg
            kpt_path[start:end] += kpt_path[start-1]

        # kpt_path /= kpt_path[-1]
        kpt_bounds = np.concatenate((kpt_path[0::Nk_in_seg], [kpt_path[-1], ]))
    # else:
    #     # get band path
    #     vkpt_diff = np.diff(vkpts, axis=0)
    #     kpt_path = np.zeros(nkpts, dtype=float)
    #     kpt_path[1:] = np.cumsum(np.linalg.norm(np.dot(vkpt_diff, B), axis=1))
    #     # kpt_path /= kpt_path[-1]
    #
    #     # get boundaries of band path
    #     xx = np.diff(kpt_path)
    #     kpt_bounds = np.concatenate(
    #         ([0.0, ], kpt_path[np.isclose(xx, 0.0)], [kpt_path[-1], ]))

    return kpt_path, bands, Efermi, kpt_bounds, wkpts, nelect


def setup_ls(L, theta=0.0, phi=0.0, saxis=None):
    '''
    Calculate the spin-orbit matrix in the basis of sphereical
    harmonics:
        < Y_l^m; \sigma_1 | S \cdot L | Y_l'^m'; \sigma_2 >

    Note that
        S \cdot L = (L_{+}S_{-} + L_{-}S_{+} + L_zS_z) / 2

    and
        L_{+}|l, m> = \sqrt{(l-m)(l+m+1)} | l, m+1>
                    = \sqrt{l(l+1) - m(m+1)} | l, m+1>
        L_{-}|l, m> = \sqrt{(l+m)(l-m+1)} | l, m-1>
                    = \sqrt{l(l+1) - m(m-1)} | l, m-1>

    input:
        L:      integer
        theta:  zenith angle in unit of radian (2\pi rad equal 360 degrees)
        phi:    azimuth angle in unit of radian (2\pi rad equal 360 degrees)
        saxis:  spin direction, used to calculate theta and phi
    output:
        spin-orbit matrix of dimension (4, 2L+1, 2L+1)
    '''

    if saxis is None:
        theta = float(theta)
        phi = float(phi)
    else:
        saxis = np.array(saxis, dtype=float)
        saxis /= np.linalg.norm(saxis)
        assert saxis.shape == (3,)
        theta = np.arccos(saxis[2])           # theta in range [0, pi]
        phi = np.arctan2(saxis[1], saxis[0])  # phi in range [-pi, pi]

    TLP1 = 2 * L + 1

    # angular momentum matrix elements in the basis of COMPLEX spherical
    # harmonics
    L_OP_C = np.zeros((3, TLP1, TLP1), dtype=complex)
    # angular momentum matrix elements in the basis of REAL spherical
    # harmonics
    L_OP_R = np.zeros((3, TLP1, TLP1), dtype=complex)

    # complex to real spherical harmonics conversion matrix
    U_C2R = np.zeros((TLP1, TLP1), dtype=complex)
    # real to complex spherical harmonics conversion matrix
    # U_R2C is the conjugate transpose of U_C2R
    U_R2C = np.zeros((TLP1, TLP1), dtype=complex)

    LS = np.zeros((2, 2, TLP1, TLP1), dtype=complex)

    # this rotation matrix is consistent with a rotation
    # of a magnetic field by theta and phi according to
    #
    #                     ( cos \theta \cos phi,  - sin \phi,  cos \phi \sin \theta )
    # U(\theta, \phi) =   ( cos \theta \sin phi,    cos \phi,  sin \phi \sin \theta )
    #                     (  - sin \theta      ,       0    ,        cos \theta     )
    #
    # (first rotation by \theta and then by \phi)
    # unfortunately this rotation matrix does not have
    # the property U(\theta,\phi) = U^T(-\theta,-\phi)

    ROTMAT = np.array([
        [np.cos(theta/2)*np.exp(-1j*phi/2), -
         np.sin(theta/2)*np.exp(-1j*phi/2)],
        [np.sin(theta/2)*np.exp(1j*phi/2),   np.cos(theta/2)*np.exp(1j*phi/2)]
    ], dtype=complex)

    # set up L operator (in units of h_bar) for complex spherical harmonics y_lm
    #
    #   |y_lm1> L_k <y_lm2| = |y_lm1> L_OP_C(m1,m2,k) <y_lm2| , where k=1,2,3
    #   correspond to L_x, L_y and Lz, respectively.

    for ii in range(TLP1):
        M = ii - L
        C_UP = np.sqrt((L-M) * (L+M+1)) / 2
        C_DW = np.sqrt((L+M) * (L-M+1)) / 2
        # fill L_x
        if ((M+1) <= L):
            L_OP_C[0, ii+1, ii] = C_UP
        if ((M-1) >= -L):
            L_OP_C[0, ii-1, ii] = C_DW
        # fill L_y
        if ((M+1) <= L):
            L_OP_C[1, ii+1, ii] = -1j * C_UP
        if ((M-1) >= -L):
            L_OP_C[1, ii-1, ii] = 1j * C_DW
        # fill L_z
        L_OP_C[2, ii, ii] = M

    # set up transformation matrix real->complex spherical harmonics
    #
    #  |y_lm1> \sum_m2 U_R2C(m1,m2) <Y_lm2|
    #
    # where y_lm and Y_lm are, respectively, the complex and real
    # spherical harmonics

    # please refer to:
    # https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    # U_R2C is the conjugate transpose of U_C2R

    sqrt2inv = 1.0 / np.sqrt(2.0)
    for ii in range(TLP1):
        M = ii - L
        if (M < 0):
            U_C2R[ii, ii] = 1j * sqrt2inv
            U_C2R[-(ii+1), ii] = -1j * (-1)**M * sqrt2inv
            # U_R2C[ii, ii] = -1j * sqrt2inv
            # U_R2C[ii, -(ii+1)] = sqrt2inv
        if (M == 0):
            U_C2R[ii, ii] = 1.0
            # U_R2C[ii, ii] = 1.0
        if (M > 0):
            U_C2R[-(ii+1), ii] = sqrt2inv
            U_C2R[ii, ii] = (-1)**M * sqrt2inv
            # U_R2C[ii, -(ii+1)] = 1j * (-1)**M * sqrt2inv
            # U_R2C[ii, ii] = (-1)**M * sqrt2inv
    # U_R2C = U_C2R.T.conj()

    # Calculate L operator (in units of h_bar) with respect to
    # the real spherical harmonics Y_lm
    #
    #    |Y_lm1> L_k <Y_lm2| = |Y_lm1> L_OP_R(m1,m2,k) <Y_lm2| , where k=x,y,z

    # n.b. L_OP_R(m1,m2,k)= \sum_ij U_C2R(m1,i) L_OP_C(i,j,k) U_R2C(j,m2)
    #
    for ii in range(3):
        L_OP_R[ii] = np.dot(
            U_C2R.T.conj(), np.dot(L_OP_C[ii], U_C2R)
        )

    # Calculate the SO (L \dot S) operator (in units of h_bar^2)
    # <up| SO |up>
    #     LS(:,:,1)= -L_OP_R(:,:,3)/2
    # <up| SO |down>
    #     LS(:,:,2)= -L_OP_R(:,:,1)/2 + (0._q,1._q)*L_OP_R(:,:,2)/2
    # <down| SO |up>
    #     LS(:,:,3)= -L_OP_R(:,:,1)/2 - (0._q,1._q)*L_OP_R(:,:,2)/2
    # <down|SO|down>
    #     LS(:,:,4)=  L_OP_R(:,:,3)/2

    # Calculate the SO (L \dot S) operator (in units of h_bar^2)
    # <up| SO |up>
    LS[0, 0, :, :] = L_OP_R[2, :, :]/2
    # <up| SO |down>
    LS[0, 1, :, :] = L_OP_R[0, :, :]/2 + 1.0j * L_OP_R[1, :, :]/2
    # <down| SO |up>
    LS[1, 0, :, :] = L_OP_R[0, :, :]/2 - 1.0j * L_OP_R[1, :, :]/2
    # <down|SO|down>
    LS[1, 1, :, :] = -L_OP_R[2, :, :]/2

    # Rotate the LS operator by \theta and \phi
    # LS = ROTMAT.T.conj() * LS * ROTMAT
    for ii in range(TLP1):
        for jj in range(TLP1):
            LS[..., ii, jj] = np.dot(
                ROTMAT.T.conj(), np.dot(LS[..., ii, jj], ROTMAT)
            )
    LS.shape = (4, TLP1, TLP1)

    return LS


def paw_core_soc_mat(theta=0.0, phi=0.0, saxis=None):
    '''
    Calculate the spin-orbit matrix elements on the basis of the partial AE
    wavefunctions.

           H_{i;j} = < \phi_i; \sigma_1 | H_{soc} | \phi_j; \sigma_2 >

    where \phi_i is the partial AE wfcs. Decomposing the partial waves as a
    spherical harmonics | Y_{lm} >, a radial function | f_i (r) > and a spinor
    | \sigma_1 >, the above equation changes to

    H_{i;j} = < \phi_i; \sigma_1 | H_{soc} | \phi_j; \sigma_2 >
            = (1 / 2*m**2*c**2) < Y_i; \sigma_i | S\cdotL | Y_j; \sigma_j >
                                < f_i(r) | (1/r)(d v_{ks} / dr) | f_j(r) >

    The "SL" term can be easily calculated. For the radial part, we utilize VASP
    to perform the calculation.

    REF of the formula: https://link.aps.org/doi/10.1103/PhysRevB.94.235106
    '''

    ls = [0] + [setup_ls(l, theta, phi, saxis) for l in range(1, 4)]
    l_proj_type = []
    i_type = []   # type index of each atom
    s_type = []   # symbol of each atom

    with open('SocRadCar', 'r') as inp:
        _ = inp.readline()
        ntype, nions, nproj = np.array(inp.readline().split(), dtype=int)
        for ii in range(ntype):
            l_proj_type.append(
                np.array(inp.readline().split(), dtype=int)
            )
        s_type = np.array(inp.readline().split(), dtype=str)
        i_type = np.array(inp.readline().split(), dtype=int) - 1
        assert nproj == np.sum(
            [(2 * l_proj_type[ii] + 1).sum() for ii in i_type]
        )
        _ = inp.readline()

        # the radial part
        fij_r = np.array(
            [line.split() for line in inp.readlines()
             if line.strip()],
            dtype=float
        )

        lmax = fij_r.shape[-1]
        fij_r.shape = (nions, lmax, lmax)

    hij = np.zeros((4, nproj, nproj), dtype=complex)
    i1_lower = 0
    for iatom in range(nions):
        itp = i_type[iatom]              # type index for this atom
        l_type = l_proj_type[itp]        # L quantum number for this atom
        lmax = len(l_type)               # Number of L quantum number
        lmmax = (2 * l_type + 1).sum()   # Number of projectors for this atom

        i2_lower = 0
        for ii in range(lmax):
            j2_lower = 0
            for jj in range(lmax):
                # only for 0 < l <= 3
                if (
                    (l_type[ii] == l_type[jj])
                    and (l_type[ii] > 0)
                    and (l_type[ii] <= 3)
                ):
                    ii_s = i1_lower + i2_lower
                    jj_s = i1_lower + j2_lower
                    ii_e = ii_s + 2 * l_type[ii] + 1
                    jj_e = jj_s + 2 * l_type[jj] + 1

                    for i_spinor in range(4):
                        hij[i_spinor, ii_s:ii_e, jj_s:jj_e] = \
                            ls[l_type[ii]][i_spinor] * fij_r[iatom, ii, jj]

                j2_lower += 2 * l_type[jj] + 1
            i2_lower += 2 * l_type[ii] + 1
        i1_lower += lmmax

    return hij


def spinorb_eigen(theta=0.0, phi=0.0, saxis=None,
                  normalcar='NormalCAR', outcar='OUTCAR',
                  factor=1.0, plot=True, figname='soc.png',
                  show=True,
                  ):
    '''
    Calculates spin-orbit band structure non-selfconsistently. For more
    information, refer to:

        https://wiki.fysik.dtu.dk/gpaw/tutorials/spinorbit/spinorbit.html
    '''
    socmat = paw_core_soc_mat(theta, phi, saxis)
    cprojs = read_cproj_NormalCar(normalcar)

    kpath, E_nk, efermi, kbd, kptw, nelect = get_bandInfo(outcar)
    nspin, nkpts, nband = E_nk.shape

    assert socmat.shape[-1] == cprojs.shape[-1], \
        "No. of projectors in SocCar and NormalCAR does NOT match!"
    assert cprojs.shape[0] == nspin * nkpts * nband, \
        "No. of bands in SocCar and OUTCAR does NOT match!"

    nproj = socmat.shape[-1]
    cprojs = cprojs.reshape((nspin, nkpts, nband, nproj))
    if nspin == 1:
        cprojs = np.array([cprojs[0], cprojs[0]])

    i1 = np.arange(0, 2 * nband, 2)
    i2 = np.arange(1, 2 * nband, 2)

    H_mm = np.zeros((2*nband, 2*nband), dtype=complex)
    # H_soc = np.zeros((4, nband, nband), dtype=complex)

    # eigenvalues of SOC band
    Esoc_nk = []
    # eigenvectors of SOC band
    Vsoc_nk = []

    for ikpt in range(nkpts):
        # initialize the Hamiltonian for each k-point
        H_mm[:, :] = 0.0j
        # H_soc[:,:] = 0.0j

        H_mm[i1, i1] = E_nk[0, ikpt, :]
        if nspin == 2:
            H_mm[i2, i2] = E_nk[1, ikpt, :]
        else:
            H_mm[i2, i2] = E_nk[0, ikpt, :]

        # print cprojs.shape, socmat.shape, cprojs.T.shape, H_mm[i1,i1].shape
        # print ikpt
        H_mm[0::2, 0::2] += np.dot(np.conj(cprojs[0, ikpt, :, :]),
                                   np.dot(socmat[0], cprojs[0, ikpt, :, :].T))
        H_mm[1::2, 1::2] += np.dot(np.conj(cprojs[1, ikpt, :, :]),
                                   np.dot(socmat[3], cprojs[1, ikpt, :, :].T))
        H_mm[0::2, 1::2] += np.dot(np.conj(cprojs[0, ikpt, :, :]),
                                   np.dot(socmat[1], cprojs[1, ikpt, :, :].T))
        H_mm[1::2, 0::2] += np.dot(np.conj(cprojs[1, ikpt, :, :]),
                                   np.dot(socmat[2], cprojs[0, ikpt, :, :].T))

        e_m, v_m = np.linalg.eigh(H_mm)
        Esoc_nk.append(e_m)
        Vsoc_nk.append(v_m)

    np.savetxt('soc_band.dat', Esoc_nk)
    Esoc_nk = np.array(Esoc_nk)

    if plot:
        import matplotlib as mpl
        mpl.use('agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        fig.set_size_inches((3.0, 4.0))

        E_nk -= efermi
        Esoc_nk -= efermi

        for ispin in range(nspin):
            for iband in range(nband):
                ax.plot(kpath, E_nk[ispin, :, iband],
                        ls=':', lw=0.5, color='b')

        for iband in range(2*nband):
            ax.plot(kpath, Esoc_nk[:, iband],
                    ls='-', lw=0.8, color='red')

        for bd in kbd:
            ax.axvline(x=bd, ls='-', color='k', lw=0.5, alpha=0.5)
        ax.set_xticks(kbd)
        ax.set_xticklabels(['M', r'$\Gamma$', 'K', 'M'])
        # ax.set_xticklabels([])

        ax.set_xlim(kpath.min(), kpath.max())
        ax.set_ylim(-3.0, 3.0)

        ax.set_ylabel('Energy [eV]', labelpad=5)

        plt.tight_layout(pad=0.2)
        plt.savefig(figname, dpi=360)

        if show:
            from subprocess import call
            call(('feh -xdF ' + figname).split())

    return Esoc_nk, Vsoc_nk, efermi, kptw, nelect


def get_mae(theta=0.0, phi=0.0, saxis=None,
            sigma=0.01,
            normalcar='NormalCAR', outcar='OUTCAR',
            plot=False
            ):
    """
    Get the magnetocrystalline anisotropy.
    """

    e_nk, v_nk, ef, kw, nelect = spinorb_eigen(theta, phi, saxis, normalcar=normalcar,
                                               outcar=outcar, plot=plot)
    ef_soc, f_nk = find_fermi_level(e_nk, kw, nelect=nelect, sigma=sigma,
                                    soc_band=True)

    # print ef, ef_soc, e_nk[:,54].min(), e_nk[:,53].max()
    # print f_nk[0,:]
    # nkpts, nbnds = e_nk.shape
    # kw /= kw.sum()
    # kw = np.tile(kw, [nbnds, 1]).T
    # # Fermi-Dirac distribution
    # x = (e_nk - ef) / sigma
    # x = x.clip(-100, 100)
    # f_nk = 1.0 / (np.exp(x) + 1.0)
    # f_nk[f_nk < 1E-12] = 0.0
    # f_nk[:,54:] = 0.0
    # f_nk[:,:54] = 1.0

    return (e_nk * f_nk[0] * kw[:, None]).sum()


def find_fermi_level(band_energies, kpt_weight,
                     nelect, occ=None, sigma=0.01, nedos=100,
                     soc_band=False,
                     nmax=1000):
    '''
    Locate ther Fermi level from the band energies, k-points weights and number
    of electrons. 
                                             1.0
                Ne = \sum_{n,k} --------------------------- * w_k
                                  ((E_{nk}-E_f)/sigma) 
                                e                      + 1


    Inputs:
        band_energies: The band energies of shape (nspin, nkpts, nbnds)
        kpt_weight: The weight of each k-points.
        nelect: Number of electrons.
        occ:  1.0 for spin-polarized/SOC band energies, else 2.0.
        sigma: Broadening parameter for the Fermi-Dirac distribution.
        nedos: number of discrete points in approximately locating Fermi level.
        soc_band: band energies from SOC calculations?
        nmax: maximum iteration in finding the exact Fermi level.
    '''

    if band_energies.ndim == 2:
        band_energies = band_energies[None, :]

    nspin, nkpts, nbnds = band_energies.shape

    if occ is None:
        if nspin == 1 and (not soc_band):
            occ = 2.0
        else:
            occ = 1.0

    if nbnds > nedos:
        nedos = nbnds * 5

    kpt_weight = np.asarray(kpt_weight, dtype=float)
    assert kpt_weight.shape == (nkpts,)
    kpt_weight /= np.sum(kpt_weight)

    emin = band_energies.min()
    emax = band_energies.max()
    e0 = np.linspace(emin, emax, nedos)
    de = e0[1] - e0[0]

    # find the approximated Fermi level
    nelect_lt_en = np.array([
        np.sum(occ * (band_energies <= en) * kpt_weight[None, :, None])
        for en in e0
    ])
    ne_tmp = nelect_lt_en[nedos/2]
    if (np.abs(ne_tmp - nelect) < 0.05):
        i_fermi = nedos / 2
        i_lower = i_fermi - 1
        i_upper = i_fermi + 1
    elif (ne_tmp > nelect):
        for ii in range(nedos/2-1, -1, -1):
            ne_tmp = nelect_lt_en[ii]
            if ne_tmp < nelect:
                i_fermi = ii
                i_lower = i_fermi
                i_upper = i_fermi + 1
                break
    else:
        for ii in range(nedos/2+1, nedos):
            ne_tmp = nelect_lt_en[ii]
            if ne_tmp > nelect:
                i_fermi = ii
                i_lower = i_fermi - 1
                i_upper = i_fermi
                break

    ############################################################
    # Below is the algorithm used by VASP, much slower
    ############################################################
    # find the approximated Fermi level
    # x = (e0[None, None, None, :] - band_energies[:, :, :, None]) / sigma
    # x = x.clip(-100, 100)
    # dos = 1./sigma * np.exp(x) / (np.exp(x) + 1)**2 * \
    #       kpt_weight[None, :, None, None] * de
    # ddos = np.sum(dos, axis=(0,1,2))
    #
    # nelect_from_dos_int = np.sum(ddos[:nedos/2])
    # if (np.abs(nelect_from_dos_int - nelect) < 0.05):
    #     i_fermi = nedos / 2 - 1
    #     i_lower = i_fermi - 1
    #     i_upper = i_fermi + 1
    # elif (nelect_from_dos_int > nelect):
    #     for ii in range(nedos/2, -1, -1):
    #         nelect_from_dos_int = np.sum(ddos[:ii])
    #         if nelect_from_dos_int < nelect:
    #             i_fermi = ii
    #             i_lower = i_fermi
    #             i_upper = i_fermi + 1
    #             break
    # else:
    #     for ii in range(nedos/2, nedos):
    #         nelect_from_dos_int = np.sum(ddos[:ii])
    #         if nelect_from_dos_int > nelect:
    #             i_fermi = ii
    #             i_lower = i_fermi - 1
    #             i_upper = i_fermi
    #             break

    # Locate the exact Fermi level using bisectioning
    e_lower = e0[i_lower]
    e_upper = e0[i_upper]
    lower_B = False
    upper_B = False
    for ii in range(nmax):
        e_fermi = (e_lower + e_upper) / 2.

        z = (band_energies - e_fermi) / sigma
        z = z.clip(-100, 100)
        F_nk = occ / (np.exp(z) + 1)
        N = np.sum(F_nk * kpt_weight[None, :, None])
        # print ii, e_lower, e_upper, N

        if (np.abs(N - nelect) < 1E-10):
            break
        if (np.abs(e_upper - e_lower / (np.abs(e_fermi) + 1E-10)) < 1E-14):
            raise ValueError("Cannot reach the specified precision!")

        if (N > nelect):
            if not lower_B:
                e_lower -= de
            upper_B = True
            e_upper = e_fermi
        else:
            if not upper_B:
                e_upper += de
            lower_B = True
            e_lower = e_fermi

    if (ii == nmax - 1):
        raise ValueError("Cannot reach the specified precision!")

    return e_fermi, F_nk


################################################################################
if __name__ == '__main__':
    # spinorb_eigen(theta=0, plot=True, figname='v1.png')
    # spinorb_eigen(theta=np.pi / 4, plot=True, figname='v2.png')
    # spinorb_eigen(theta=np.pi / 2, plot=True, figname='v3.png')

    # hij = paw_core_soc_mat()
    # np.save('hij', hij)

    # print setup_ls(L=1) * 2

    z_angles = np.linspace(0, np.pi / 2., 10)
    # z_angles = np.linspace(0, np.pi, 19)
    MAEs = [get_mae(theta) for theta in z_angles]
    MAEs -= MAEs[0]
    for theta, mae in zip(z_angles, MAEs):
        print "{:5.2f} {:22.16f}".format(theta / np.pi * 180, mae)

    mae_vasp = np.loadtxt('../mae.dat')[:, 1]

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    fig = plt.figure(figsize=(4.0, 4.0))
    # ax = plt.subplot()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    ax1.plot(z_angles * 180 / np.pi, MAEs * 1000, 's-',
             label='MAE current',
             ls=':', ms=5)
    ax1.plot(z_angles * 180 / np.pi, mae_vasp * 1000, 'o-',
             label='MAE VASP',
             ls=':', ms=5)
    ax1.legend(loc='upper left', fontsize='small')

    ax1.set_xlim(-5, 95)
    ax1.set_xticks(range(0, 91, 10))

    ax1.set_xlabel(r'$\theta$ [degree]', labelpad=5)
    ax1.set_ylabel('Energy [meV]', labelpad=5)

    ax2.plot(MAEs*1000, mae_vasp * 1000, 'h:', color='k', ms=5,
             mfc='r', mew=0.5)
    ax2.set_xlabel(r'MAE current [meV]', labelpad=5)
    ax2.set_ylabel('MAE VASP [meV]', labelpad=5)

    plt.tight_layout()
    plt.savefig('mae_theta.png', dpi=400)
    plt.show()

    # a_angles = np.linspace(0, np.pi * 2, 20)
    # MAEs = [get_mae(theta=np.pi / 2, phi=phi) for phi in a_angles]
    # MAEs -= MAEs[0]
    # for phi, mae in zip(a_angles, MAEs):
    #     print "{:5.2f} {:22.16f}".format(phi / np.pi * 180, mae)
    #
    # import matplotlib.pyplot as plt
    # plt.style.use('ggplot')
    #
    # fig = plt.figure(figsize=(6.0, 4.0))
    # ax = plt.subplot()
    #
    # ax.plot(a_angles * 180 / np.pi, MAEs * 1000, 's-',
    #          label='MAE current',
    #          ls=':', ms=5)
    #
    # # ax.set_xlim(-5, 185)
    # # ax.set_xticks(range(0, 181, 10))
    #
    #
    # plt.tight_layout()
    # plt.savefig('mae_phi.png', dpi=400)
    # plt.show()
