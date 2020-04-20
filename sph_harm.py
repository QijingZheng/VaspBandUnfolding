#!/usr/bin/env python

import numpy as np
from scipy.special import sph_harm


def cart2sph(xyz):
    '''
    Convert Cartesian coordinate to spherical coordinate.

    input:
       xyz in (n, 3)
    return:
        r: norm 
    theta: polar angle in [0, pi]
      phi: azimuthal angle in [0, 2 * pi]
    '''
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim == 1:
        xyz = xyz[None, :]
    x, y, z = xyz.T

    # the azimuthal angle
    phi = np.arctan2(y, x)
    # np.arctan2 outputs angle in [-pi, pi]
    phi[phi < 0] += 2 * np.pi

    # the norm
    r = np.linalg.norm(np.c_[x, y, z], axis=1)

    # the polar angle
    theta = np.arccos(z / r)

    return np.array([r, phi, theta])


def sph_c(xyz, l, m=None):
    '''
    Complex spherial harmonics including the Condon-Shortley phase.

    https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Spherical_harmonics

    input:
       xyz: cartesian coordinate of shape [n, 3]
    '''
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim == 1:
        xyz = xyz[None, :]
    if m:
        assert -l <= m <= l, "'m' must be in the range of [{},{}]".format(-l, l)

    r, phi, theta = cart2sph(xyz)
    N = xyz.shape[0]
    ylm = [sph_harm(m, l, phi, theta) for m in range(-l, l+1)]

    if m is None:
        return np.array(ylm, dtype=complex).T
    else:
        return np.array(ylm, dtype=complex).T[:, m+l]


def sph_r(xyz, l, m=None):
    '''
    Real spherial harmonics.

    https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
    '''
    ylm_c = sph_c(xyz, l)
    u = sph_u_c2r(l)

    if m is None:
        np.dot(ylm_c, u.T).real
    else:
        return np.dot(ylm_c, u.T).real[:, m+l]


def sph_u_c2r(l):
    '''
    Set up transformation matrix complex->real spherical harmonics.

    please refer to:
    https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    U_R2C is the conjugate transpose of U_C2R
    '''

    TLP1 = 2 * l + 1
    U_C2R = np.zeros((TLP1, TLP1), dtype=complex)

    sqrt2inv = 1.0 / np.sqrt(2.0)
    for ii in range(TLP1):
        M = ii - l
        if (M < 0):
            U_C2R[ii, ii] = 1j * sqrt2inv
            U_C2R[ii, -(ii+1)] = -1j * (-1)**M * sqrt2inv
        if (M == 0):
            U_C2R[ii, ii] = 1.0
        if (M > 0):
            U_C2R[ii, -(ii+1)] = sqrt2inv
            U_C2R[ii, ii] = (-1)**M * sqrt2inv

    return U_C2R


def sph_u_r2c(l):
    '''
    Transformation matrix real->complex spherical harmonics
    '''
    return sph_u_c2r(l).conj().T


def show_sph_harm(m, l, real=True, N=50):
    '''
    Show the spherical harmonics on a unit sphere
    '''
    theta = np.linspace(0, np.pi, N)
    phi = np.linspace(0, 2*np.pi, N)
    theta, phi = np.meshgrid(theta, phi)

    # The Cartesian coordinates of the unit sphere
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    xyz = np.c_[x.ravel(), y.ravel(), z.ravel()]

    # from time import time
    # t0 = time()
    if real:
        ylm = sph_r(xyz, l, m).reshape(N, N)
    else:
        ylm = sph_c(xyz, l, m).reshape(N, N).real
    # t1 = time()
    # print(t1 - t0)

    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d import Axes3D

    # Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
    fcolors = ylm
    fmax, fmin = fcolors.max(), fcolors.min()
    fcolors = (fcolors - fmin)/(fmax - fmin)

    # Set the aspect ratio to 1 so our sphere looks spherical
    fig = plt.figure(
        figsize=plt.figaspect(1.)
    )
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z,  rstride=1, cstride=1,
                    facecolors=cm.seismic(fcolors))

    # Turn off the axis planes
    ax.set_axis_off()
    plt.show()


if __name__ == "__main__":
    show_sph_harm(m=-2, l=3, real=False)
