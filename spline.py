#!/usr/bin/env python

import numpy as np
from scipy.interpolate import CubicSpline, interp1d


def splcof(x, y, y1p=0.0):
    '''
    VASP ini.F
    !**************** SUBROUTINE SPLCOF, SPLCOF_N0 *************************
    ! RCS:  $Id: ini.F,v 1.3 2002/08/14 13:59:39 kresse Exp $
    !
    !  Subroutine for calculating spline-coefficients
    !  using the routines of the book 'numerical  recipes 3.3'
    !  on input P(1,N) must contain x-values
    !           P(2,N) must contain function-values
    !  YP is the first derivatives at the first point
    !  if >= 10^30 natural boundary-contitions (y''=0) are used
    !
    !  for point N always natural boundary-conditions are used in
    !  SPLCOF, whereas SPLCOF_N0 assume 0 derivative at N
    !  SPLCOF_NDER allows to specify a boundary condition
    !  at both end points
    !
    !***********************************************************************

    ------------------------------------
    determination of spline coefficients
    ------------------------------------
    f = ((d*dx+c)*dx+b)*dx+a
        between adjacent x - values

    result
    P-ARRAY
    P(I,1) = X(I)
    P(I,2) = A(I) = F(I)
    P(I,3) = B(I)
    P(I,4) = C(I)
    P(I,5) = D(I)

    y1p > 1E30 --> CubicSpline(bc_type='natural')
    y1p = 0 --> CubicSpline(bc_type='clamped')
    '''

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert (x.shape == y.shape) and (x.ndim == 1)

    N = x.size
    A = np.zeros(N, dtype=float)
    B = np.zeros(N, dtype=float)
    C = np.zeros(N, dtype=float)
    D = np.zeros(N, dtype=float)
    A[:] = y[:]

    dy = y[1] - y[0]
    dx = x[1] - x[0]
    if y1p > 0.99E30:
        C[0] = 0.0
        B[0] = 0.0
    else:
        C[0] = -0.5
        B[0] = (3.0 / dx) * (dy / dx - y1p)

    s = (x[1:N-1] - x[0:N-2]) / (x[2:N] - x[0:N-2])
    for ii in range(1, N-1):
        r = s[ii-1] * C[ii-1] + 2.0
        C[ii] = (s[ii-1] - 1.0) / r
        B[ii] = (6 * ((y[ii+1] - y[ii]) / (x[ii+1] - x[ii]) -
                      (y[ii] - y[ii-1]) / (x[ii] - x[ii-1])) /
                 (x[ii+1] - x[ii-1]) - s[ii-1] * B[ii-1]) / r

    C[N-1] = 0.0
    B[N-1] = 0.0

    for ii in range(N-2, -1, -1):
        C[ii] = C[ii] * C[ii+1] + B[ii]

    for ii in range(0, N-1):
        s = x[ii+1] - x[ii]
        r = (C[ii+1] - C[ii]) / 6.0
        D[ii] = r/s
        C[ii] = C[ii]/2.0
        B[ii] = (y[ii+1] - y[ii]) / s - (C[ii] + r) * s

    def cubicspline(x0):
        '''
        Evaluate the 

            f = ((d*dx+c)*dx+b)*dx+a

        between adjacent x - values
        '''

        assert x0.size > x.size
        x0 = np.asarray(x0, dtype=float)
        idx = np.floor(x0 / dx).astype(int)

        dx0 = x0 - x[idx]
        return A[idx] + dx0 * (
            B[idx] + dx0 * (
                C[idx] + dx0 * D[idx]
            )
        )

    return cubicspline


if __name__ == '__main__':
    x = np.arange(10)
    y = np.sin(x)
    cs = splcof(x, y, 1E30)
    CS = CubicSpline(x, y, bc_type='natural')
    # CS = interp1d(x, y, kind='cubic')
    xs = np.arange(0.0, 9.1, 0.1)
    print(CS([0.0, 0.4, 0.3]))

    # import matplotlib.pyplot as plt
    #
    # plt.plot(x, y, 'o')
    # plt.plot(xs, cs(xs), ls=':', color='r')
    # plt.plot(xs, CS(xs), ls='-', color='b', alpha=0.5)
    # plt.plot(xs, np.sin(xs), ls='-', color='k', alpha=0.5)
    #
    # plt.show()
