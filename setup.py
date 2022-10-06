#!/usr/bin/env python

from distutils.core import setup

setup(
    name         = 'PyVaspWfc',
    version      = '1.0',
    description  = 'Python modules for dealing with VASP pseudo-wavefunctions.',
    author       = 'Qijing Zheng',
    author_email = 'zqj.kaka@gmail.com',
    url          = 'https://github.com/QijingZheng/VaspBandUnfolding',
    # packages        = ['pyvaspwfc',],
    py_modules   = [
        'vasp_constant', 'vaspwfc', 'aewfc',
        'nac', 'unfold', 'procar', 'spinorb',
        'sph_harm', 'paw'
    ],
    scripts      = ['bin/wfcplot', 'bin/tdmplot', 'bin/potplot', 'bin/nebplot']
)
