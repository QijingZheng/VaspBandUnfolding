#!/usr/bin/env python

try:
    from setuptools import setup
    HAVE_SETUPTOOLS = True
except ImportError:
    from distutils.core import setup
    HAVE_SETUPTOOLS = False

PY_MODULES = ['vasp_constant', 'vaspwfc', 'aewfc', 'nac', 'unfold', 'procar', 'spinorb', 'sph_harm', 'paw', 'ewald', 'coulomb_integral', 'bse']
SCRIPTS = ['bin/wfcplot', 'bin/tdmplot', 'bin/potplot', 'bin/nebplot', 'bin/bseplot']
INSTALL_REQUIRES = ['numpy', 'scipy', 'matplotlib', 'ase']

kwargs = {}
if HAVE_SETUPTOOLS:
    kwargs["install_requires"] = INSTALL_REQUIRES

setup(
    name='PyVaspWfc',
    version='1.0',
    description='Python modules for dealing with VASP pseudo-wavefunctions.',
    author='Qijing Zheng',
    author_email='zqj.kaka@gmail.com',
    url='https://github.com/QijingZheng/VaspBandUnfolding',
    py_modules=PY_MODULES,
    scripts=SCRIPTS,
    **kwargs,
)
