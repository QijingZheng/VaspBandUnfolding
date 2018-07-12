#!/usr/bin/env python

import os
import re
import numpy as np
from collections import Iterable

############################################################
def gaussian_smearing_org(x, x0, sigma=0.05):
    '''
    Gaussian smearing of a Delta function.
    '''

    return 1. / (np.sqrt(2*np.pi) * sigma) * np.exp(-(x - x0)**2 / (2*sigma**2))

def string2index(string):
    if ':' not in string:
        raise ValueError("Invalid index string!")
    i = []
    for s in string.split(':'):
        if s == '':
            i.append(None)
        else:
            i.append(int(s))
    i += (3 - len(i)) * [None]
    return slice(*i)

def gradient_fill(x, y, fill_color=None, ax=None, direction=1, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """

    line, = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    # print fill_color
    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:,:,:3] = rgb
    if direction == 1:
        z[:,:,-1] = np.linspace(0, alpha, 100)[:,None]
    else:
        z[:,:,-1] = np.linspace(alpha, 0, 100)[:,None]

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    if direction == 1:
        xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    else:
        xy = np.vstack([[xmin, ymax], xy, [xmax, ymax], [xmin, ymax]])
    clip_path = Polygon(xy, lw=0.0, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)

    return line, im

############################################################
class procar(object):
    '''
    A class for dealing with VASP PROCAR file.
    '''
    def __init__(self, inf='PROCAR', lsoc=False):
        '''
        Initialization
        '''

        self._fname  = inf
        self._lsoc   = lsoc

        try:
            self._procar = open(self._fname, 'r')
        except:
            raise IOError('Failed to open %s' % self._fname)

        self.readProcar()

        # parameters usefull for dos generation
        self._sigma  = 0.05
        self._nedos  = 3000
        self._tdos   = None

        self._spd_index = {
            's' : 0,
            'py' : 1, 'pz' : 2, 'px' : 3,
            'dxy' : 4, 'dyz' : 5, 'dz2' : 6, 'dxz' : 7, 'dx2' : 8
        }

    def readProcar(self):
        '''
        Extract the info from PROCAR.
        '''

        inp = [line for line in self._procar if line.strip()]

        # when the band number is too large, there will be no space between ";" and
        # the actual band number. A bug found by Homlee Guo.
        # Here, #kpts, #bands and #ions are all integers
        self._nkpts, self._nbands, self._nions = [int(xx) for xx in re.sub('[^0-9]', ' ', inp[1]).split()]

        # band projectron on each atoms or s/p/d orbitals
        self._aproj = np.asarray([line.split()[1:-1] for line in inp
                                  if not re.search('[a-zA-Z]', line)],
                                  dtype=float)
        # k-points weights of each k-points
        self._kptw = np.asarray([line.split()[-1] for line in inp if 'weight' in line], dtype=float)
        # band energies
        self._eband = np.asarray([line.split()[-4] for line in inp
                                  if 'occ.' in line], dtype=float)

        self._nlmax = self._aproj.shape[-1]
        self._nspin = self._aproj.shape[0] // (self._nkpts * self._nbands * self._nions)
        self._nspin //= 4 if self._lsoc else 1

        if self._lsoc:
            self._aproj.resize(self._nspin, self._nkpts, self._nbands, 4, self._nions, self._nlmax)
            self._aproj = self._aproj[:,:,:,0,:,:]
        else:
            self._aproj.resize(self._nspin, self._nkpts, self._nbands, self._nions, self._nlmax)

        self._kptw.shape  = (self._nspin, self._nkpts)
        self._kptw_org    = self._kptw.copy()
        self._eband.shape = (self._nspin, self._nkpts, self._nbands)

        # close the PROCAR
        self._procar.close()

    def get_nkpts(self):
        '''
        get number of kpoints.
        '''
        return self._nkpts

    def get_nspin(self):
        '''
        get number of spin
        '''
        return self._nspin

    def get_nbands(self):
        '''
        get number of bands
        '''
        return self._nbands

    def isSoc(self):
        return True if self._lsoc else False

    def get_sigma(self):
        '''
        return dos brodening parameter
        '''
        return self._sigma
    def set_sigma(self, sigma):
        '''
        set dos brodening parameter
        '''
        self._sigma = sigma

        # re-generate the DOS with the new SIGMA
        self.init_dos()

    def get_nedos(self): return self._nedos
    def set_nedos(self, nedos):
        '''
        set number of point in smooth DOS 
        '''
        assert isinstance(nedos, int), 'NEDOS shoule be int!'
        self._nedos = nedos

        # re-generate the DOS with the new NEDOS
        self.init_dos()

    def get_kpts_weight(self):
        '''
        return the k-points weights
        '''
        return self._kptw.copy()
    def set_kpts_weight(self, kptw):
        '''
        set the k-points weights
        '''
        kptw = np.array(kptw)
        assert kptw.shape == self._kptw.shape
        self._kptw = kptw

        # re-generate the DOS with the new kptw
        self.init_dos()
    def restore_kpts_weight(self, kptw):
        '''
        set the k-points weights
        '''
        self._kptw = self._kptw_org.copy()

        # re-generate the DOS with the new kptw
        self.init_dos()

    def init_dos(self):
        '''
        dos initialization
        '''

        # print 'calculating dos'
        emin =  self._eband.min()
        emax =  self._eband.max()
        eran = emax - emin
        emin = emin - eran * 0.05
        emax = emax + eran * 0.05

        self._xen  = np.linspace(emin, emax, self._nedos)
        self._tdos = np.empty((self._nspin, self._nkpts, self._nbands, self._nedos))

        for ispin in range(self._nspin):
            sign = 1 if ispin == 0 else -1
            for ikpt in range(self._nkpts):
                for iband in range(self._nbands):
                    x0 = self._eband[ispin, ikpt, iband]
                    self._tdos[ispin, ikpt, iband] = sign * self._kptw[ispin,ikpt] \
                              * gaussian_smearing_org(self._xen, x0, self._sigma)\

    def get_pdos(self, atoms=':', kpts=':', spd=':'):
        '''
        Get site/k-points/spd-orbital projected partial density of states (PDOS)

        atoms : selected atoms index.
                Valid values:
                    ":"       -> for all atoms
                    "0::2"    -> for even index atoms
                    [0, 1, 2] -> atom indices specified by list
                    0         -> atom indices specified by integer

        kpts  : selected k-points index
                Valid values:
                    ":"       -> for all k-points
                    "0::2"    -> for even index k-points
                    [0, 1, 2] -> k-points indices specified by list
                    0         -> k-points indices specified by integer

        spd   : selected s/p/d-orbitals, the s/p/d-orbital and the corresponding
                index are:
                    's' : 0,
                    'py' : 1, 'pz' : 2, 'px' : 3,
                    'dxy' : 4, 'dyz' : 5, 'dz2' : 6, 'dxz' : 7, 'dx2' : 8

                Valid values:
                    ":"         -> for all s/p/d-orbitals
                    "0::2"      -> for even index 
                    [0, 1, 2]   -> s/p/d-orbitals specified by list of integer
                    ['s', 'py'] -> s/p/d-orbitals specified by list of names
                    0           -> s/p/d-orbitals indices specified by integer
        '''
        if self._tdos is None:
            self.init_dos()

        assert (isinstance(atoms, int) 
             or isinstance(atoms, Iterable)
             or isinstance(atoms, str))
        assert (isinstance(kpts, int) 
             or isinstance(kpts, Iterable)
             or isinstance(kpts, str))
        assert (isinstance(spd, int) 
             or isinstance(spd, Iterable) 
             or isinstance(kpts, str))

        if isinstance(atoms, str):
            atoms = string2index(atoms)
        if isinstance(kpts, str):
            kpts = string2index(kpts)
        if isinstance(spd, str):
            spd = string2index(spd)
        if isinstance(spd, Iterable):
            spd = [ii if isinstance(ii, int) else self._spd_index[ii]
                   for ii in spd]

        # problem with mixed advanced indexing and basic indexing, see scipy
        # documents for reference
        # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#combining-advanced-and-basic-indexing
        # 
        # a=np.zeros((2,3,4)); b=np.ones((3,4)); I=np.array([0,1])
        # b[:,I].shape = (3, 2)
        # a[0,:,I].shape = (2, 3)

        # Consider indexing a 3D array arr with shape (X, Y, Z):
        #
        # arr[:, [0, 1], 0] has shape (X, 2).
        # arr[[0, 1], 0, :] has shape (2, Z).
        # arr[0, :, [0, 1]] has shape (2, Y), not (Y, 2)

        pdos = []
        for ispin in range(self._nspin):
            # Avoid mixed indexing
            pw = self._aproj[ispin, kpts]
            # sum over the s/p/d projection
            pw = np.sum(pw[..., spd],   axis=-1)
            # sum over the site projection
            pw = np.sum(pw[..., atoms], axis=-1)

            td = self._tdos[ispin, kpts]

            pdos.append(np.sum(pw[...,np.newaxis] * td, axis=(0, 1)))

            # pwht = np.sum(self._aproj[ispin][kpts,:,atoms,spd], axis=(-1, -2))
            # pdos.append(np.sum(pwht[..., np.newaxis] * self._tdos[ispin][kpts,...], axis=(0, 1)))

        # only return one dos if not spin-polarized
        p = pdos[0] if self._nspin == 1 else pdos

        return self._xen, p


if __name__ == '__main__':
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    xx = procar()
    # total dos
    x0, y0 = xx.get_pdos()
    # site projected DOS
    x1, y1 = xx.get_pdos(atoms=[192, 209])
    # site projected DOS + s/p/d projected DOS
    xx.set_nedos(1000)
    x2, y2 = xx.get_pdos(atoms='0::2', spd=['dz2', 'dx2'])

    fig, ax = plt.subplots()
    fig.set_size_inches((4.0, 3.0))

    ax.plot(x0, y0, color='r')
    ax.plot(x1, y1, color='g')
    ax.plot(x2, y2, color='b')

    plt.show()

