# pyvaspwfc

This is a python class for dealing with `VASP` pseudo-wavefunction file `WAVECAR`.
It can extract the planewave coefficients of any single KS state from the file.
In addition, by padding the planewave coefficients to a 3D grid and perform 3D
Fourier Transfor, the pseudo-wavefunction in real space can also be obtained and
saved to file that can be view with `VESTA`. 

With the knowledge of the planewave coefficients of KS staates,
[transition dipole moment](https://en.wikipedia.org/wiki/Transition_dipole_moment)
can also be calculated in reciprocal space.

# Installation

Put `vasp_constant.py` and `vaspwfc.py` in any directory you like and add the
path of the directory to `PYTHONPATH`

```bash
export PYTHONPATH=/the/path/of/your/dir:${PYTHONPATH}
```
