## Introduction

`VaspBandUnfolding` consists of a collection of python scripts that deal with `VASP` output files. 

- `vaspwfc.py` can be used to read the plane-wave coefficients in the `WAVECAR` file and then generate real-space representation of the pseudo-wavefunction by Fourier transform. Other wavefunction related quantites, e.g. transition dipole moment (pseudo-wavefunction part), inverse participation ratio and electron localization function et al. can also be conveniently obtained.

  Moreover, a command line tool named `wfcplot` in the `bin` directory can be used to output real-space pseudo-wavefunctions.

- `bsefatband.py` can be used to parse `BSEFATBAND`, plot exciton density in the first Brillouin zone, and reconstruct fixed-hole or fixed-electron exciton densities in real space from `BSEFATBAND + WAVECAR + POSCAR`.

- `bsematrix.py` can build the Hartree/exchange part of the BSE matrix with a local full-grid implementation, diagonalize it, and write a VASP-format `BSEFATBAND`-style file for comparison against VASP output.

  Moreover, a command line tool named `bseplot` in the `bin` directory can be used for both reciprocal-space and real-space exciton workflows.

- `paw.py` can be used to parse the PAW `POTCAR` file. A command line tool named `potplot` in the `bin` directory can be used to visualize the projector function and partial waves contained in the `POTCAR`.

- `aewfc.py` can be used to generate all-electron (AE) wavefunction. 

- `unfold.py` can be used to perform band unfolding from supercell calculations.

### Publications

A list of publications utilizing `VaspBandUnfolding` can be found [here](doc/VaspBandUnfolding.pdf)!

## Installation

- Core requirements

    * Numpy
    * Scipy
    * Matplotlib
    * ASE

- Optional requirement

    * [pySBT](https://github.com/QijingZheng/pySBT) for all-electron / PAW spherical-Bessel workflows

- Install dependencies only

  ```bash
  pip install -r requirements.txt
  # optional
  pip install -r requirements-optional.txt
  ```

- Install the package from a clone

  ```bash
  git clone https://github.com/QijingZheng/VaspBandUnfolding
  cd VaspBandUnfolding
  pip install .
  ```

- Editable install for development

  ```bash
  pip install -e .
  ```

- Install directly with pip

  ```bash
  pip install git+https://github.com/QijingZheng/VaspBandUnfolding
  ```


## Examples

### Reading VASP WAVECAR

- Pseudo-wavefunction.

  As is well known, `VASP` `WAVECAR` is a binary file and contains the plane-wave coefficients for the pseudo-wavefunctions. The pseudo-wavefunction in real space can be obtained by 3D Fourier transform on the plane-wave coefficients and represented on a 3D uniform grid which can be subsequently visualized with software such as `VESTA`.

  - For a normal `WAVECAR`, i.e. not *gamma-only* or *non-collinear* `WAVECAR`, one can write a small python script and convert the desired Kohn-Sham states to real space.
  
    ```python
    #/usr/bin/env python
    from vaspwfc import vaspwfc
    
    pswfc = vaspwfc('./examples/wfc_r/WAVECAR')
    # KS orbital in real space, double the size of the FT grid
    phi = pswfc.get_ps_wfc(ikpt=2, iband=27, ngrid=pswfc._ngrid * 2)
    # Save the orbital into files. Since the wavefunction consist of complex
    # numbers, the real and imaginary part are saved separately.
    pswfc.save2vesta(phi, poscar='./examples/wfc_r/POSCAR')
    ```

    - In the above script, `pswfc._ngrid` is the default 3D grid size and `phi` is a numpy 3D array of size `2*pswfc._ngrid`, with the first dimensiton being *x* and the last "z".
    - The *spin*, *k-point* and *band* index for the KS state are designated by the argumnt `ispin`, `ikpt` and  `iband`, respectively, all of which start from `1`.
    - Generally, the pseudo-wavefunction is complex, `pswfc.save2vesta` will export both the *real* and *imaginary* part of the wavefunction, with the file name "wfc_r.vasp" and "wfc_i.vasp", respectively.

    Below are the real (left) and imaginary (right) part of the selected KS orbital:

    ![real part](./examples/wfc_r/r_resize.png) 
    ![imaginary part](./examples/wfc_r/i_resize.png)


  - For *gamma-only* `WAVECAR`, one must pass the argument `lgamma=True`  when reading `WAVECAR` in the `vaspwfc` method. Moreover, as `VASP` only stores half of the full plane-wave coefficients for *gamma-only* WAVECAR and `VASP` changes the idea about which half to save from version 5.2 to 5.4. An addition argument must be passed.

    ```python
    #/usr/bin/env python
    from vaspwfc import vaspwfc
    
    # For VASP <= 5.2.x, check
    # which FFT VASP uses by the following command:
    #
    #     $ grep 'use.* FFT for wave' OUTCAR
    #
    # Then
    #
    #     # for parallel FFT, VASP <= 5.2.x
    #     pswfc = vaspwfc('WAVECAR', lgamma=True, gamma_half='z')
    #
    #     # for serial FFT, VASP <= 5.2.x
    #     pswfc = vaspwfc('WAVECAR', lgamma=True, gamma_half='x')
    #
    # For VASP >= 5.4, WAVECAR is written with x-direction half grid regardless of
    # parallel or serial FFT.
    #
    #     # "gamma_half" default to "x" for VASP >= 5.4
    #     pswfc = vaspwfc('WAVECAR', lgamma=True, gamma_half='x')

    pswfc = vaspwfc('WAVECAR', lgamma=True, gamma_half='x')
    ```

  - For *non-collinear* `WAVECAR`, however, one must pass the argument `lsorbit=True`  when reading `WAVECAR`. Note that in the non-collinear case, the wavefunction now is a two-component spinor.

    ```python
    #/usr/bin/env python
    from vaspwfc import vaspwfc
    
    # for WAVECAR from a noncollinear run, the wavefunction at each k-piont/band is
    # a two component spinor. Turn on the lsorbit flag when reading WAVECAr.
    pswfc = vaspwfc('examples/wfc_r/wavecar_mose2-wse2', lsorbit=True)
    phi_spinor = pswfc.get_ps_wfc(1, 1, 36, ngrid=pswfc._ngrid*2)
    for ii in range(2):
        phi = phi_spinor[ii]
        prefix = 'spinor_{:02d}'.format(ii)
        pswfc.save2vesta(phi, prefix=prefix,
                poscar='examples/wfc_r/poscar_mose2-wse2')
    ```

  - If only real-space representation of the pseudo-wavefunction is needed, a
    helping script `wfcplot` in the `bin` directory comes to rescue.

    ```bash
    $ wfcplot -w WAVECAR -p POSCAR -s spin_index -k kpoint_index -n band_index             # for normal WAVECAR
    $ wfcplot -w WAVECAR -p POSCAR -s spin_index -k kpoint_index -n band_index  -lgamma    # for gamma-only WAVECAR
    $ wfcplot -w WAVECAR -p POSCAR -s spin_index -k kpoint_index -n band_index  -lsorbit   # for noncollinear WAVECAR
    ```

    Please refer to `wfcplot -h` for more information of the usage.

- Build SOC spinor `WAVECAR` (`spinormaker`)

  `spinor.py` also provides a CLI named `spinormaker` for constructing SOC
  spinor `WAVECAR` from scalar/ISPIN=2 runs (with `NormalCAR`, `SocCar`,
  `SocRadCar`).

  ```bash
  spinormaker --mixwave-ibs 211 213 215 217 219 --correct-kpts 1
  ```

  Full k-point / full odd-even band-pair mode:

  ```bash
  spinormaker --full-kpts --full-bands
  ```

  Notes:

  - `--full-kpts` means all k-points are processed, while band pairs still come
    from `--mixwave-ibs`.
  - `--full-bands` means all odd/even band pairs are used.
  - `--full-kpts-full-bands` is an alias for `--full-kpts --full-bands`.

  A complete MoSe2 `2x3` input example (without `WAVECAR` due file size) is
  available in [examples/spinor](./examples/spinor). The README there includes
  an `unfold_main` validation snapshot for bands `213-220` (K+/K- projection
  and `sigma_z`).

  Example excerpt (`ispin2`, bands `213-214`):

  ```text
  213 E=-1.21393  K+=0.00000  K-=0.99958  sz=+0.78827
  214 E=-1.21393  K+=0.99958  K-=0.00000  sz=-0.78827
  ```

  Unchanged scalar `WAVECAR` reference uses half-band mapping
  `ib_scalar = (ib_spinor + 1) // 2` (e.g. `213-220 -> 107-110`), giving
  approximately symmetric valley weights (`K+ ~= K- ~= 0.5`) before spinor
  construction.


- Exciton density from `BSEFATBAND`

  `bsefatband.py` provides both reciprocal-space and real-space utilities for VASP BSE calculations.

  - Plot exciton density in the first Brillouin zone:

    ```bash
    bseplot bz --input BSEFATBAND --poscar POSCAR --exciton 1
    ```

  - Reconstruct the real-space electron density with a fixed hole:

    ```bash
    bseplot realspace --bsefatband BSEFATBAND --wavecar WAVECAR --poscar POSCAR \
        --exciton 1 --hole 0.5,0.5,0.5
    ```

  - Reconstruct the real-space hole density with a fixed electron:

    ```bash
    bseplot realspace --bsefatband BSEFATBAND --wavecar WAVECAR --poscar POSCAR \
        --exciton 1 --electron 0.5,0.5,0.5
    ```

  The `realspace` workflow also requires `OUTCAR` (or `OUTCAR.symm`) to recover
  symmetry operators and the `IBZKPT_HF` full-BZ mapping used to restore the
  correct Bloch phases when `WAVECAR` stores only irreducible k-points. Keep
  `OUTCAR` in the same directory as `WAVECAR`, `BSEFATBAND`, or `POSCAR`.

  The `realspace` mode writes VASP scalar grids such as `exciton_001_electron_rho.vasp`
  or `exciton_001_hole_rho.vasp`. The `bz` mode writes a PNG map of the folded
  first-BZ exciton density.

  A complete MoSe2 example for this workflow is available in [examples/bseplot](./examples/bseplot).

  Example first-BZ density for the lowest exciton:

  ![MoSe2 lowest exciton in the first Brillouin zone](./examples/bseplot/exciton_001_bz.png)

  Fixed-hole electron density comparison for exciton 1:

  | `bsefatband.py` | VASP |
  | --- | --- |
  | ![Exciton 1 electron density from bsefatband.py](./examples/bseplot/X1-electron-bsepy.png) | ![Exciton 1 electron density from VASP](./examples/bseplot/X1-electron-vasp.png) |

  Fixed-electron hole density comparison for exciton 1:

  | `bsefatband.py` | VASP |
  | --- | --- |
  | ![Exciton 1 hole density from bsefatband.py](./examples/bseplot/X1-hole-bsepy.png) | ![Exciton 1 hole density from VASP](./examples/bseplot/X1-hole-vasp.png) |

- BSE matrix benchmarks (`bsematrix.py`)

  `bsematrix.py` builds the BSE interaction matrix from `WAVECAR/OUTCAR/POSCAR/POTCAR`, supports both `pw_only` and `paw_orth_only`, and can write both `AMAT` text dumps and VASP-style `BSEFATBAND` files.

  The benchmark artifact set now lives in [examples/bsematrix/BP](./examples/bsematrix/BP). It contains text-format VASP and Python references for:

  - direct-only
  - exchange-only / Hartree-term only
  - both terms together

  For each case, the directory includes:

  - `vasp_*_BSEFATBAND.txt`
  - `vasp_*_AMAT.txt`
  - `py_pw_only_*_BSEFATBAND.txt`, `py_pw_only_*_AMAT.txt`
  - `py_paw_orth_only_*_BSEFATBAND.txt`, `py_paw_orth_only_*_AMAT.txt`

  The VASP `BSEFATBAND` files bundled in the examples contain 8 printed excitons, but the tables below list the first 10 eigenvalues obtained by diagonalizing the full `BSE_AMAT.bin` reference matrix so both modes can be compared on the same footing.

  AMAT summary:

  | Case | Mode | `max(|ΔA|)` (eV) | `||ΔA||_F` (eV) | Worst entry |
  | --- | --- | ---: | ---: | --- |
  | direct-only | `pw_only` | 0.00768173 | 0.03076986 | `(12,12)` |
  | direct-only | `paw_orth_only` | 0.00256244 | 0.01002069 | `(27,12)` |
  | exchange-only | `pw_only` | 0.00520426 | 0.04512843 | `(3,3)` |
  | exchange-only | `paw_orth_only` | 0.00070821 | 0.00610677 | `(12,27)` |
  | both | `pw_only` | 0.00356970 | 0.02483112 | `(3,21)` |
  | both | `paw_orth_only` | 0.00327066 | 0.01518818 | `(27,12)` |

  Direct-only first 10 BSE eigenvalues:

  | # | VASP (eV) | `pw_only` (eV) | Δ (meV) | `paw_orth_only` (eV) | Δ (meV) |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 1 | 0.49212829 | 0.47646457 | -15.663718 | 0.48889006 | -3.238229 |
  | 2 | 0.60201503 | 0.60189693 | -0.118105 | 0.60213102 | 0.115985 |
  | 3 | 0.65357165 | 0.65018958 | -3.382068 | 0.65252903 | -1.042623 |
  | 4 | 0.83531815 | 0.83513060 | -0.187546 | 0.83526516 | -0.052982 |
  | 5 | 1.03555253 | 1.03095473 | -4.597790 | 1.03735120 | 1.798679 |
  | 6 | 1.06892204 | 1.06874978 | -0.172261 | 1.06906668 | 0.144642 |
  | 7 | 1.07212036 | 1.07194033 | -0.180036 | 1.07217889 | 0.058525 |
  | 8 | 1.29205933 | 1.28610280 | -5.956538 | 1.29266773 | 0.608392 |
  | 9 | 1.47121986 | 1.46503458 | -6.185279 | 1.47112060 | -0.099262 |
  | 10 | 1.47576966 | 1.47222936 | -3.540294 | 1.47813079 | 2.361126 |

  Exchange-only first 10 BSE eigenvalues:

  | # | VASP (eV) | `pw_only` (eV) | Δ (meV) | `paw_orth_only` (eV) | Δ (meV) |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 1 | 1.53335142 | 1.53335146 | 0.000039 | 1.53335143 | 0.000002 |
  | 2 | 1.53344213 | 1.53344929 | 0.007166 | 1.53344163 | -0.000495 |
  | 3 | 1.75329793 | 1.75330120 | 0.003266 | 1.75329770 | -0.000228 |
  | 4 | 2.00071425 | 2.00071425 | 0.000004 | 2.00071425 | 0.000001 |
  | 5 | 2.00079815 | 2.00080456 | 0.006409 | 2.00079771 | -0.000440 |
  | 6 | 2.84479254 | 2.84479269 | 0.000140 | 2.84479255 | 0.000004 |
  | 7 | 2.84488258 | 2.84488930 | 0.006725 | 2.84488213 | -0.000445 |
  | 8 | 3.22373604 | 3.22373639 | 0.000348 | 3.22373605 | 0.000012 |
  | 9 | 3.22382989 | 3.22383713 | 0.007246 | 3.22382946 | -0.000429 |
  | 10 | 3.56269414 | 3.56269417 | 0.000028 | 3.56269414 | 0.000005 |

  Both terms first 10 BSE eigenvalues:

  | # | VASP (eV) | `pw_only` (eV) | Δ (meV) | `paw_orth_only` (eV) | Δ (meV) |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 1 | 0.59755271 | 0.59789800 | 0.345288 | 0.59733017 | -0.222543 |
  | 2 | 0.60214685 | 0.60203072 | -0.116123 | 0.60226185 | 0.115003 |
  | 3 | 0.81773692 | 0.81819859 | 0.461668 | 0.81736229 | -0.374636 |
  | 4 | 0.95004106 | 0.95659218 | 6.551112 | 0.94577218 | -4.268880 |
  | 5 | 1.03878070 | 1.03404920 | -4.731495 | 1.04058328 | 1.802582 |
  | 6 | 1.06893044 | 1.06876025 | -0.170187 | 1.06907431 | 0.143875 |
  | 7 | 1.07222303 | 1.07209729 | -0.125748 | 1.07227428 | 0.051243 |
  | 8 | 1.30876837 | 1.30431547 | -4.452895 | 1.30927662 | 0.508254 |
  | 9 | 1.47644720 | 1.47286725 | -3.579956 | 1.47881397 | 2.366768 |
  | 10 | 1.54488018 | 1.54427524 | -0.604943 | 1.54321845 | -1.661731 |

  The BP README contains the same benchmark tables together with the exact artifact filenames.

- All-electron wavefunction in real space

  Refer to this post for detail formulation.

  > [PAW All-Electron Wavefunction in VASP](https://qijingzheng.github.io/posts/VASP-All-Electron-WFC/)
  
  ```python
  #/usr/bin/env python
  
  from vaspwfc import vaspwfc
  from aewfc import vasp_ae_wfc
  
  # the pseudo-wavefunction
  ps_wfc = vaspwfc('WAVECAR', lgamma=True)
  # the all-electron wavefunction
  # here 25x Encut, or 5x grid size is used
  ae_wfc = vasp_ae_wfc(ps_wfc, aecut=-25)
  
  phi_ae = ae_wfc.get_ae_wfc(iband=8)
  ```
  
  The comparison of All-electron and pseudo wavefunction of CO<sub>2</sub> HOMO
  can be found in [examples/aewfc/co2](./examples/aewfc/co2).
  
  ![CO2 HOMO](./examples/aewfc/co2/ae-ps-core_co2_homo_wfc.png)
  ![CO2 HOMO](./examples/aewfc/co2/co2_homo_aeps_wfc_rs.png)

- Dipole transition matrix

  Refer to this post for detail formulation.

  > [Light-Matter Interaction and Dipole Transition Matrix](https://qijingzheng.github.io/posts/Light-Matter-Interaction-and-Dipole-Transition-Matrix/)

  Under the electric-dipole approximation (EDA), The dipole transition matrix
  elements in the length gauge is given by:

  ```
        <psi_nk | e r | psi_mk>
  ```

  where | psi_nk > is the pseudo-wavefunction.  In periodic systems, the position
  operator "r" is not well-defined.  Therefore, we first evaluate the momentum
  operator matrix in the velocity gauge, i.e.
  
  ```
        <psi_nk | p | psi_mk>
  ```
  
  And then use simple "p-r" relation to apprimate the dipole transition matrix
  element
  
  ```
                                    -i⋅h
      <psi_nk | r | psi_mk> =  -------------- ⋅ <psi_nk | p | psi_mk>
                                 m⋅(En - Em)
  ```
  
  Apparently, the above equaiton is not valid for the case Em == En. In this case,
  we just set the dipole matrix element to be 0.
  
  > NOTE that, the simple "p-r" relation only applies to molecular or finite system,
  > and there might be problem in directly using it for periodic system. Please
  > refer to this paper for more details.
  >
  > [Relation between the interband dipole and momentum matrix elements in semiconductors](https://journals.aps.org/prb/pdf/10.1103/PhysRevB.87.125301)
  
  
  The momentum operator matrix in the velocity gauge
  
  ```
          <psi_nk | p | psi_mk> = hbar <u_nk | k - i nabla | u_mk>
  ```
  
  In PAW, the matrix element can be divided into plane-wave parts and one-center
  parts, i.e.
  
  ```
      <u_nk | k - i nabla | u_mk> = <tilde_u_nk | k - i nabla | tilde_u_mk>
                                   - \sum_ij <tilde_u_nk | p_i><p_j | tilde_u_mk>
                                     \times i [
                                       <phi_i | nabla | phi_j>
                                       -
                                       <tilde_phi_i | nabla | tilde_phi_j>
                                     ]
  ```
  
  where | u_nk > and | tilde_u_nk > are cell-periodic part of the AE/PS
  wavefunctions, | p_j > is the PAW projector function and | phi_j > and
  | tilde_phi_j > are PAW AE/PS partial waves.
  
  The nabla operator matrix elements between the pseudo-wavefuncitons
  
  ```
      <tilde_u_nk | k - i nabla | tilde_u_mk>
  
     = \sum_G C_nk(G).conj() * C_mk(G) * [k + G]
  ```
  
  where C_nk(G) is the plane-wave coefficients for | u_nk >.

  ```python
  import numpy as np
  
  from vaspwfc import vaspwfc
  from aewfc import vasp_ae_wfc
  
  # the pseudo-wavefunction
  ps_wfc = vaspwfc('WAVECAR', lgamma=True)
  # the all-electron wavefunction
  ae_wfc = vasp_ae_wfc(ps_wfc)
  
  # (ispin, ikpt, iband) for initial and final states
  ps_dp_mat = ps_wfc.get_dipole_mat((1,1,1), (1, 1, 9))
  ae_dp_mat = ae_wfc.get_dipole_mat((1,1,1), (1, 1, 9))
  ```
   

- Inverse Participation Ratio

  IPR is a measure of the localization of Kohn-Sham states. For a particular KS state \phi_j, it is defined as

  ```latex
                  \sum_n |\phi_j(n)|^4 
  IPR(\phi_j) = -------------------------
                |\sum_n |\phi_j(n)|^2||^2
  ```

  where n iters over the number of grid points.

- Electron Localization Function
  (Still need to be tested!)

  In quantum chemistry, the electron localization function (ELF) is a measure of the likelihood of finding an electron in the neighborhood space of a reference electron located at a given point and with the same spin. Physically, this measures the extent of spatial localization of the reference electron and provides a method for the mapping of electron pair probability in multielectronic systems. (from wiki)
  
  * Nature, 371, 683-686 (1994)
  * Becke and Edgecombe, J. Chem. Phys., 92, 5397(1990)
  * M. Kohout and A. Savin, Int. J. Quantum Chem., 60, 875-882(1996)
  * http://www2.cpfs.mpg.de/ELF/index.php?content=06interpr.txt
  
  NOTE that if you are using VESTA to view the resulting ELF file, please rename the output file as "ELFCAR", otherwise there will be some error in the isosurface plot!  When VESTA read in CHG*/PARCHG/*.vasp to visualize isosurfaces and sections, data values are divided by volume in the unit of bohr^3.  The unit of charge densities input by VESTA is, therefore, bohr^−3.  For LOCPOT/ELFCAR files, volume data are kept intact.

  ```python
  #/usr/bin/env python
  import numpy as np
  from vaspwfc import vaspwfc, save2vesta
  
  kptw = [1, 6, 6, 6, 6, 6, 6, 12, 12, 12, 6, 6, 12, 12, 6, 6]
  
  pswfc = vaspwfc('./WAVECAR')
  # chi = wfc.elf(kptw=kptw, ngrid=wfc._ngrid * 2)
  chi = pswfc.elf(kptw=kptw, ngrid=[20, 20, 150])
  save2vesta(chi[0], lreal=True, poscar='POSCAR', prefix='elf')
  ```
  **Remember to rename the output file "elf_r.vasp" as "ELFCAR"!**

### VASP POTCAR

The `paw.py` contains method to parse the PAW POTCAR (`pawpotcar` class) can
calculate relating quantities in the PAW within augment sphere. For example,

```python
from paw import pawpotcar

pp = pawpotcar(potfile='POTCAR')

# Q_{ij} = < \phi_i^{AE} | \phi_j^{AE} > -
#          < \phi_i^{PS} | \phi_j^{PS} >
Qij = pp.get_Qij()
# nabla_{ij} = < \phi_i^{AE} | nabla_r | \phi_j^{AE} > -
#              < \phi_i^{PS} | nabla_r | \phi_j^{PS} >
Nij = pp.get_nablaij()
```

A helping script utilizing the `paw.py` in the `bin` directory can be used to
visulize the projector function and partial waves.

```bash
# `Ti` POTCAR for exampleTCAR for example
potplot -p POTCAR   
```
![Ti POTCAR](examples/potplot/ti_pot.png)

As the name suggests, `paw.py` also contains the methods (`nonlq` and `nonlr`
class) to calculate the inner products of the projector function and the
pseudo-wavefunction. The related formula can be found in [my
post](https://qijingzheng.github.io/posts/VASP-All-Electron-WFC/).

> [PAW All-Electron Wavefunction in VASP](https://qijingzheng.github.io/posts/VASP-All-Electron-WFC/)

### Band unfolding

Using the pseudo-wavefunction from supercell calculation, it is possible to
perform electronic band structure unfolding to obtain the effective band
structure. For more information, please refer to the following article and the
[GPAW](https://wiki.fysik.dtu.dk/gpaw/tutorials/unfold/unfold.html) website.

> V. Popescu and A. Zunger Extracting E versus k effective band structure
> from supercell calculations on alloys and impurities Phys. Rev. B 85, 085201
> (2012)

Theoretical background with an example can be found in my post:

> [Band Unfolding Tutorial](http://QijingZheng.github.io/posts/Band-unfolding-tutorial/)

Here, we use MoS<sub>2</sub> as an example to illustrate the procedures of band
unfolding.  Below is the band structure of MoS2 using a primitive cell. The
calculation was performed with `VASP` and the input files can be found in the
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
    reducedK, kid = removeDuplicateKpoints(K_in_sup, return_map=True)

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

#### Band unfolding wth atomic contributions 

After band unfolding, we can also superimpose the atomic contribution of each KS
states on the spectral weight. Below is the resulting unfolded band structure of
Ce-doped bilayer-MoS2. Refer to
`./examples/unfold/Ce@BL-MoS2_3x3x1/plt_unf.py` for the entire code.

   ![imaginary part](./examples/unfold/Ce@BL-MoS2_3x3x1/ebs_s_small.png)

### Band re-ordering

Band re-ordering is possible by maximizing the overlap between nerghbouring
k-points. The overlap is defined as the inner product of the periodic part of
the Bloch wavefunctions.

                        `< u(n, k) | u(m, k-1) >`

Note, however, the `WAVECAR` only contains the pseudo-wavefunction, and thus the
pseudo `u(n,k)` are used in this function. Moreover, since the number of
planewaves for each k-points are different, the inner product is performed in
real space.

The overlap maximalization procedure is as follows:
1. Pick out those bands with large overlap (> olap_cut).
2. Assign those un-picked bands by maximizing the overlap.

An example band structure re-ordering is performed in MoS2. The result is shown
in the following image, where the left/right panel shows the
un-ordered/re-ordered band structure.

   ![band_reorder](./examples/band_reorder/kband_small.png) | 
