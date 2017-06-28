# PyVaspwfc

This is a python class for dealing with `VASP` pseudo-wavefunction file `WAVECAR`.
It can be used to extract the planewave coefficients of any single Kohn-Sham (KS)
orbital from the file.  In addition, by padding the planewave coefficients to a
3D grid and performing 3D Fourier Transform, the pseudo-wavefunction in real
space can also be obtained and saved to file that can be viewed with `VESTA`. 

## Transition dipole moment

With the knowledge of the planewave coefficients of the
pseudo-wavefunction,
[transition dipole moment](https://en.wikipedia.org/wiki/Transition_dipole_moment) between
any two KS states can also be calculated.

## Inverse Participation Ratio
IPR is a measure of the localization of Kohn-Sham states. For a particular KS
state \phi_j, it is defined as

                \sum_n |\phi_j(n)|^4 
IPR(\phi_j) = -------------------------
              |\sum_n |\phi_j(n)|^2||^2

where n iters over the number of grid points.

## Band unfolding

Using the pseudo-wavefunction from supercell calculation, it is possible to
perform electronic band structure unfolding to obtain the effective band
structure. For more information, please refer to the following article and the
[GPAW](https://wiki.fysik.dtu.dk/gpaw/tutorials/unfold/unfold.html) website.

> V. Popescu and A. Zunger Extracting E versus k effective band structure
> from supercell calculations on alloys and impurities Phys. Rev. B 85, 085201
> (2012)

# Installation

Put `vasp_constant.py` and `vaspwfc.py` in any directory you like and add the
path of the directory to `PYTHONPATH`

```bash
export PYTHONPATH=/the/path/of/your/dir:${PYTHONPATH}
```

requirements

* numpy
* scipy
* matplotlib

# Examples

## Wavefunction in real space

```python
from vaspwfc import vaspwfc

wav = vaspwfc('./examples/wfc_r/WAVECAR')
# KS orbital in real space, double the size of the FT grid
phi = wav.wfc_r(ikpt=2, iband=27, ngrid=wav._ngrid * 2)
# Save the orbital into files. Since the wavefunction consist of complex
# numbers, the real and imaginary part are saved separately.
wav.save2vesta(phi, poscar='./examples/wfc_r/POSCAR')
```

Below are the real (left) and imaginary (right) part of the selected KS orbital:

![real part](./examples/wfc_r/r_resize.png) | 
![imaginary part](./examples/wfc_r/i_resize.png)

## Band unfolding 

Here, we use MoS2 as an example to illustrate the procedures of band unfolding.
Below is the band structure of MoS2 using a primitive cell. The calculation was
performed with `VASP` and the input files can be found in the
`examples/unfold/primitive`

![band_primitive_cell](examples/unfold/primitive/band/band_p.png)

1. Create the supercell from the primitive cell, in my case, the supercell is of
   the size 3x3x1, which means that the transformation matrix between supercell
   and primitive cell is 
   ```python
    # The tranformation matrix between supercell and primitive cell.
    M = [[3.0, 0.0, 0.0],
         [0.0, 3.0, 0.0],
         [0.0, 0.0, 1.0]]
   ```
2. In the second step, generate band path in the primitive Brillouin Zone (PBZ)
   and find the correspondig K points of the supercell BZ (SBZ) onto which they
   fold.

    ```python
    from unfold import make_kpath, removeDuplicateKpoints, find_K_from_k

    # high-symmetry point of a Hexagonal BZ in fractional coordinate
    kpts = [[0.0, 0.5, 0.0],            # M
            [0.0, 0.0, 0.0],            # G
            [1./3, 1./3, 0.0],          # K
            [0.0, 0.5, 0.0]]            # M
    # create band path from the high-symmetry points, 30 points inbetween each pair
    # of high-symmetry points
    kpath = make_kpath(kpts, nseg=30)
    K_in_sup = []
    for kk in kpath:
        kg, g = find_K_from_k(kk, M)
        K_in_sup.append(kg)
    # remove the duplicate K-points
    reducedK = removeDuplicateKpoints(K_in_sup)

    # save to VASP KPOINTS
    save2VaspKPOINTS(reducedK)
    ```
3. Do one non-SCF calculation of the supercell using the folded K-points and
   obtain the corresponding pseudo-wavefunction. The input files are in
   `examples/unfold/sup_3x3x1/`. The effective band structure (EBS) and
   then be obtained by processing the WAVECAR file.

   ```python
   from unfold import unfold

   # basis vector of the primitive cell
   cell = [[ 3.1850, 0.0000000000000000,  0.0],
           [-1.5925, 2.7582909110534373,  0.0],
           [ 0.0000, 0.0000000000000000, 35.0]]

   WaveSuper = unfold(M=M, wavecar='WAVECAR')

   from unfold import EBS_scatter
   sw = WaveSuper.spectral_weight(kpath)
   # show the effective band structure with scatter
   EBS_scatter(kpath, cell, sw, nseg=30, eref=-4.01,
           ylim=(-3, 4), 
           factor=5)

   from unfold import EBS_cmaps
   e0, sf = WaveSuper.spectral_function(nedos=4000)
   # or show the effective band structure with colormap
   EBS_cmaps(kpath, cell, e0, sf, nseg=30, eref=-4.01,
           show=False,
           ylim=(-3, 4))
   ```

   The EBS from a 3x3x1 supercell calculation are shown below:

   ![real part](./examples/unfold/sup_3x3x1/ebs_s_resize.png) | 
   ![imaginary part](./examples/unfold/sup_3x3x1/ebs_c_resize.png)
   
   Another example of EBS from a 3x3x1 supercell calculation, where we introduce a
   `S` vacancy in the structure.

   ![real part](./examples/unfold/sup_3x3x1_defect/ebs_s_resize.png) | 
   ![imaginary part](./examples/unfold/sup_3x3x1_defect/ebs_c_resize.png)

   Yet another band unfolding example from a tetragonal 3x3x1 supercell
   calculation, where the transformation matrix is

   ```python
    M = [[3.0, 0.0, 0.0],
         [3.0, 6.0, 0.0],
         [0.0, 0.0, 1.0]]
   ```
   ![real part](./examples/unfold/tet_3x3x1/ebs_s_resize.png) | 
   ![imaginary part](./examples/unfold/tet_3x3x1/ebs_c_resize.png)

   Compared to the band structure of the primitive cell, there are some empty
   states at the top of figure. This is due to a too small value of `NBANDS` in
   supercell non-scf calculation, and thus those states are not included.
