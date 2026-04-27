# BSE matrix benchmark artifacts

This directory collects text-format benchmark artifacts for `bsematrix.py` against the current VASP 5.4.4 reference runs.

Source VASP run directories relative to the VBU root:

- direct-only: `../../tmp/VASP/vasp.5.4.4/examples/05_bse_direct_only`
- exchange-only / Hartree-term only: `../../tmp/VASP/vasp.5.4.4/examples/05_bse_hartree_only`
- both terms: `../../tmp/VASP/vasp.5.4.4/examples/05_bse`
- finite-q both terms (`KPOINT_BSE = 3 0 0 0`, `q_ext = (0, 1/3, 0)`): `../../tmp/VASP/vasp.5.4.4/examples/05_bse_qext`

Naming notes:

- `exchange_only` is the repulsive Hartree/exchange BSE term and comes from the VASP `05_bse_hartree_only` example.
- `hartree_only` files are included as aliases of the same reference set so both names are available in the data directory.
- All `AMAT` files here are plain text dumps with rows `i j real imag`.
- VASP `BSEFATBAND` references still contain the 8 excitons written by the original run; the eigenvalue tables below use the first 10 eigenvalues from diagonalizing the full `BSE_AMAT.bin` matrix.
- `paw_full` uses the current FAST_AUG-based source reconstruction path. It improves the exchange-only benchmark, but the direct, both-term, and finite-q results are still comparison data rather than parity-quality references.

## Artifact inventory

| Case | VASP files | Python `pw_only` | Python `paw_orth_only` | Python `paw_full` |
| --- | --- | --- | --- | --- |
| direct-only | `vasp_direct_only_BSEFATBAND.txt`, `vasp_direct_only_AMAT.txt` | `py_pw_only_direct_only_BSEFATBAND.txt`, `py_pw_only_direct_only_AMAT.txt` | `py_paw_orth_only_direct_only_BSEFATBAND.txt`, `py_paw_orth_only_direct_only_AMAT.txt` | `py_paw_full_direct_only_BSEFATBAND.txt`, `py_paw_full_direct_only_AMAT.txt` |
| exchange-only | `vasp_exchange_only_BSEFATBAND.txt`, `vasp_exchange_only_AMAT.txt` | `py_pw_only_exchange_only_BSEFATBAND.txt`, `py_pw_only_exchange_only_AMAT.txt` | `py_paw_orth_only_exchange_only_BSEFATBAND.txt`, `py_paw_orth_only_exchange_only_AMAT.txt` | `py_paw_full_exchange_only_BSEFATBAND.txt`, `py_paw_full_exchange_only_AMAT.txt` |
| hartree-only alias | `vasp_hartree_only_BSEFATBAND.txt`, `vasp_hartree_only_AMAT.txt` | `py_pw_only_hartree_only_BSEFATBAND.txt`, `py_pw_only_hartree_only_AMAT.txt` | `py_paw_orth_only_hartree_only_BSEFATBAND.txt`, `py_paw_orth_only_hartree_only_AMAT.txt` | `py_paw_full_hartree_only_BSEFATBAND.txt`, `py_paw_full_hartree_only_AMAT.txt` |
| both | `vasp_both_BSEFATBAND.txt`, `vasp_both_AMAT.txt` | `py_pw_only_both_BSEFATBAND.txt`, `py_pw_only_both_AMAT.txt` | `py_paw_orth_only_both_BSEFATBAND.txt`, `py_paw_orth_only_both_AMAT.txt` | `py_paw_full_both_BSEFATBAND.txt`, `py_paw_full_both_AMAT.txt` |
| finite-q both | `vasp_qext_both_BSEFATBAND.txt`, `vasp_qext_both_AMAT.txt` | `py_pw_only_qext_both_BSEFATBAND.txt`, `py_pw_only_qext_both_AMAT.txt` | `py_paw_orth_only_qext_both_BSEFATBAND.txt`, `py_paw_orth_only_qext_both_AMAT.txt` | `py_paw_full_qext_both_BSEFATBAND.txt`, `py_paw_full_qext_both_AMAT.txt` |

## Reproduction workflow

The files stored in this directory are lightweight benchmark artifacts. A full reproduction still needs a working VASP run directory with:

- `POSCAR`, `POTCAR`, `KPOINTS`
- a converged ground-state `WAVECAR` and `CHGCAR`
- enough empty bands for the later GW/BSE steps
- for `paw_full`, the matching BSE dumps `BSE_TRANS_MATRIX_FOCK.bin` and `BSE_FASTAUG_FOCK.bin`

The commands below assume:

- `VBU_ROOT` is the VBU checkout root
- `SCRATCH` is a separate VASP run directory

```bash
export VBU_ROOT=/path/to/VaspBandUnfolding
export SCRATCH=/path/to/bp-run
mkdir -p "$SCRATCH"
```

Initialize the scratch directory from the BP example inputs:

```bash
cp "$VBU_ROOT/examples/bsematrix/BP/POSCAR" "$SCRATCH/"
cp "$VBU_ROOT/examples/bsematrix/BP/POTCAR" "$SCRATCH/"
cp "$VBU_ROOT/examples/bsematrix/BP/KPOINTS" "$SCRATCH/"
cd "$SCRATCH"
```

### 1. Generate `WAVEDER`

Start from a converged ground-state calculation that already produced `WAVECAR` and `CHGCAR`. Then run a one-shot optics job to write `WAVEDER`.

Minimal `INCAR` template:

```text
SYSTEM = monolayer BP WAVEDER
ENCUT = 400
EDIFF = 1E-6
ISMEAR = 0
SIGMA = 0.02
ISYM = 0
ALGO = Exact
NBANDS = 48
LOPTICS = .TRUE.
LWAVE = .TRUE.
LCHARG = .FALSE.
```

Run:

```bash
cp "$VBU_ROOT/examples/bsematrix/BP/INCAR.waveder" INCAR
mpirun -np <NPROC> /path/to/vasp_std
```

Keep the resulting `WAVECAR`, `CHGCAR`, and `WAVEDER` for the next steps.

### 2. Run GW

Use the bundled GW settings:

```bash
cp "$VBU_ROOT/examples/bsematrix/BP/INCAR.gw" INCAR
mpirun -np <NPROC> /path/to/vasp_std
```

This step should leave the standard GW/BSE prerequisites in the run directory, including the quasiparticle data that VASP BSE uses together with `WAVEDER`.

### 3. Run VASP BSE

Use one of the three prepared BSE INCAR files depending on which matrix you want to benchmark.

Exchange-only / Hartree-term only:

```bash
cp "$VBU_ROOT/examples/bsematrix/BP/INCAR.bse_exchange_only" INCAR
mpirun -np <NPROC> /path/to/vasp_std
cp BSEFATBAND "$VBU_ROOT/examples/bsematrix/BP/vasp_exchange_only_BSEFATBAND.txt"
```

Direct-only:

```bash
cp "$VBU_ROOT/examples/bsematrix/BP/INCAR.bse_direct_only" INCAR
mpirun -np <NPROC> /path/to/vasp_std
cp BSEFATBAND "$VBU_ROOT/examples/bsematrix/BP/vasp_direct_only_BSEFATBAND.txt"
```

Both terms:

```bash
cp "$VBU_ROOT/examples/bsematrix/BP/INCAR.bse" INCAR
mpirun -np <NPROC> /path/to/vasp_std
cp BSEFATBAND "$VBU_ROOT/examples/bsematrix/BP/vasp_both_BSEFATBAND.txt"
```

For matrix-level comparison, also preserve the binary BSE matrix dump from the matching VASP example run as `BSE_AMAT.bin`, then convert or copy its text form into this directory as:

- `vasp_exchange_only_AMAT.txt`
- `vasp_direct_only_AMAT.txt`
- `vasp_both_AMAT.txt`

If you want to benchmark `paw_full`, also keep the matching FAST_AUG dumps in the same run directory:

- `BSE_TRANS_MATRIX_FOCK.bin`
- `BSE_FASTAUG_FOCK.bin`

### 4. Run Python BSE (`bsematrix.py`)

Point `bsematrix.py` at the same VASP run directory so Python and VASP read identical `WAVECAR/OUTCAR/KPOINTS/POSCAR/POTCAR`.

Run the Python commands from the VBU root:

```bash
cd "$VBU_ROOT"
```

Exchange-only:

```bash
python bsematrix.py \
    --wavecar "$SCRATCH/WAVECAR" \
    --outcar "$SCRATCH/OUTCAR" \
    --kpoints "$SCRATCH/KPOINTS" \
    --poscar "$SCRATCH/POSCAR" \
    --potcar "$SCRATCH/POTCAR" \
    --interaction hartree \
    --mode pw_only \
    --vb-num 2 --cb-num 2 --ewin 0 6 \
    --use-response-basis \
    --output-prefix examples/bsematrix/BP/py_pw_only_exchange_only_AMAT \
    --bsefatband-output examples/bsematrix/BP/py_pw_only_exchange_only_BSEFATBAND.txt \
    --full-hermitian

python bsematrix.py \
    --wavecar "$SCRATCH/WAVECAR" \
    --outcar "$SCRATCH/OUTCAR" \
    --kpoints "$SCRATCH/KPOINTS" \
    --poscar "$SCRATCH/POSCAR" \
    --potcar "$SCRATCH/POTCAR" \
    --interaction hartree \
    --mode paw_orth_only \
    --vb-num 2 --cb-num 2 --ewin 0 6 \
    --use-response-basis \
    --output-prefix examples/bsematrix/BP/py_paw_orth_only_exchange_only_AMAT \
    --bsefatband-output examples/bsematrix/BP/py_paw_orth_only_exchange_only_BSEFATBAND.txt \
    --full-hermitian

python bsematrix.py \
    --wavecar "$SCRATCH/WAVECAR" \
    --outcar "$SCRATCH/OUTCAR" \
    --kpoints "$SCRATCH/KPOINTS" \
    --poscar "$SCRATCH/POSCAR" \
    --potcar "$SCRATCH/POTCAR" \
    --interaction hartree \
    --mode paw_full \
    --vb-num 2 --cb-num 2 --ewin 0 6 \
    --output-prefix examples/bsematrix/BP/py_paw_full_exchange_only_AMAT \
    --bsefatband-output examples/bsematrix/BP/py_paw_full_exchange_only_BSEFATBAND.txt \
    --full-hermitian
```

Direct-only:

```bash
python bsematrix.py \
    --wavecar "$SCRATCH/WAVECAR" \
    --outcar "$SCRATCH/OUTCAR" \
    --kpoints "$SCRATCH/KPOINTS" \
    --poscar "$SCRATCH/POSCAR" \
    --potcar "$SCRATCH/POTCAR" \
    --interaction direct \
    --mode pw_only \
    --vb-num 2 --cb-num 2 --ewin 0 6 \
    --use-response-basis \
    --output-prefix examples/bsematrix/BP/py_pw_only_direct_only_AMAT \
    --bsefatband-output examples/bsematrix/BP/py_pw_only_direct_only_BSEFATBAND.txt \
    --full-hermitian

python bsematrix.py \
    --wavecar "$SCRATCH/WAVECAR" \
    --outcar "$SCRATCH/OUTCAR" \
    --kpoints "$SCRATCH/KPOINTS" \
    --poscar "$SCRATCH/POSCAR" \
    --potcar "$SCRATCH/POTCAR" \
    --interaction direct \
    --mode paw_orth_only \
    --vb-num 2 --cb-num 2 --ewin 0 6 \
    --use-response-basis \
    --output-prefix examples/bsematrix/BP/py_paw_orth_only_direct_only_AMAT \
    --bsefatband-output examples/bsematrix/BP/py_paw_orth_only_direct_only_BSEFATBAND.txt \
    --full-hermitian

python bsematrix.py \
    --wavecar "$SCRATCH/WAVECAR" \
    --outcar "$SCRATCH/OUTCAR" \
    --kpoints "$SCRATCH/KPOINTS" \
    --poscar "$SCRATCH/POSCAR" \
    --potcar "$SCRATCH/POTCAR" \
    --interaction direct \
    --mode paw_full \
    --vb-num 2 --cb-num 2 --ewin 0 6 \
    --use-response-basis \
    --output-prefix examples/bsematrix/BP/py_paw_full_direct_only_AMAT \
    --bsefatband-output examples/bsematrix/BP/py_paw_full_direct_only_BSEFATBAND.txt \
    --full-hermitian
```

Both terms:

```bash
python bsematrix.py \
    --wavecar "$SCRATCH/WAVECAR" \
    --outcar "$SCRATCH/OUTCAR" \
    --kpoints "$SCRATCH/KPOINTS" \
    --poscar "$SCRATCH/POSCAR" \
    --potcar "$SCRATCH/POTCAR" \
    --interaction both \
    --mode pw_only \
    --vb-num 2 --cb-num 2 --ewin 0 6 \
    --use-response-basis \
    --output-prefix examples/bsematrix/BP/py_pw_only_both_AMAT \
    --bsefatband-output examples/bsematrix/BP/py_pw_only_both_BSEFATBAND.txt \
    --full-hermitian

python bsematrix.py \
    --wavecar "$SCRATCH/WAVECAR" \
    --outcar "$SCRATCH/OUTCAR" \
    --kpoints "$SCRATCH/KPOINTS" \
    --poscar "$SCRATCH/POSCAR" \
    --potcar "$SCRATCH/POTCAR" \
    --interaction both \
    --mode paw_orth_only \
    --vb-num 2 --cb-num 2 --ewin 0 6 \
    --use-response-basis \
    --output-prefix examples/bsematrix/BP/py_paw_orth_only_both_AMAT \
    --bsefatband-output examples/bsematrix/BP/py_paw_orth_only_both_BSEFATBAND.txt \
    --full-hermitian

python bsematrix.py \
    --wavecar "$SCRATCH/WAVECAR" \
    --outcar "$SCRATCH/OUTCAR" \
    --kpoints "$SCRATCH/KPOINTS" \
    --poscar "$SCRATCH/POSCAR" \
    --potcar "$SCRATCH/POTCAR" \
    --interaction both \
    --mode paw_full \
    --vb-num 2 --cb-num 2 --ewin 0 6 \
    --use-response-basis \
    --output-prefix examples/bsematrix/BP/py_paw_full_both_AMAT \
    --bsefatband-output examples/bsematrix/BP/py_paw_full_both_BSEFATBAND.txt \
    --full-hermitian
```

For the finite-q benchmark, reuse the `05_bse_qext` run directory, add `--q-ext 0 0.3333333333333333 0`, and write to the corresponding `*_qext_both_*` files.

After these runs, the text files generated by Python can be compared directly with the VASP references and with the tables below.

## AMAT comparison summary

| Case | Mode | `max(abs(ΔA))` (eV) | `norm(ΔA)_F` (eV) | Worst entry |
| --- | --- | ---: | ---: | --- |
| direct-only | `pw_only` | 0.00338936 | 0.02105668 | `(12,27)` |
| direct-only | `paw_orth_only` | 0.02125834 | 0.12448971 | `(18,27)` |
| direct-only | `paw_full` | 0.01587339 | 0.08222105 | `(6,12)` |
| exchange-only | `pw_only` | 0.00564986 | 0.04682126 | `(3,21)` |
| exchange-only | `paw_orth_only` | 0.00634830 | 0.04821367 | `(3,6)` |
| exchange-only | `paw_full` | 0.00183452 | 0.01365123 | `(9,3)` |
| both | `pw_only` | 0.00356970 | 0.02356358 | `(3,21)` |
| both | `paw_orth_only` | 0.01594623 | 0.08810736 | `(18,27)` |
| both | `paw_full` | 0.01391734 | 0.07014263 | `(6,12)` |
| finite-q both | `pw_only` | 0.01898975 | 0.05205552 | `(12,9)` |
| finite-q both | `paw_orth_only` | 0.03391579 | 0.10749209 | `(12,9)` |
| finite-q both | `paw_full` | 0.01905383 | 0.06847281 | `(15,12)` |

## Direct-only: first 10 BSE eigenvalues

`pw_only` remains the closest mode for this benchmark. The current `paw_orth_only` and `paw_full` runs are intentionally kept here as comparison data, not as parity-quality references.

| # | VASP (eV) | `pw_only` (eV) | Δ (meV) | `paw_orth_only` (eV) | Δ (meV) | `paw_full` (eV) | Δ (meV) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.49212829 | 0.47646457 | -15.663721 | 0.04300756 | -449.120731 | 0.64922448 | 157.096189 |
| 2 | 0.60201503 | 0.60189693 | -0.118101 | 0.49683232 | -105.182711 | 0.66114590 | 59.130869 |
| 3 | 0.65357165 | 0.65018958 | -3.382068 | 0.56189006 | -91.681588 | 0.86133750 | 207.765852 |
| 4 | 0.83531815 | 0.83513060 | -0.187546 | 0.68387133 | -151.446816 | 0.95086813 | 115.549984 |
| 5 | 1.03555253 | 1.03095473 | -4.597795 | 0.75947905 | -276.073475 | 1.12527114 | 89.718615 |
| 6 | 1.06892204 | 1.06874978 | -0.172259 | 0.91303977 | -155.882269 | 1.13154260 | 62.620561 |
| 7 | 1.07212036 | 1.07194033 | -0.180032 | 0.96286839 | -109.251972 | 1.35540999 | 283.289628 |
| 8 | 1.29205933 | 1.28610280 | -5.956534 | 1.00006048 | -291.998854 | 1.59577741 | 303.718076 |
| 9 | 1.47121986 | 1.46503458 | -6.185278 | 1.13221021 | -339.009648 | 1.77754687 | 306.327012 |
| 10 | 1.47576966 | 1.47222936 | -3.540299 | 1.13926851 | -336.501149 | 1.78996658 | 314.196921 |

## Exchange-only / Hartree-term only: first 10 BSE eigenvalues

This is the one benchmark where `paw_full` is currently the best of the three Python modes.

| # | VASP (eV) | `pw_only` (eV) | Δ (meV) | `paw_orth_only` (eV) | Δ (meV) | `paw_full` (eV) | Δ (meV) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.53335142 | 1.53335147 | 0.000047 | 1.53335145 | 0.000027 | 1.53335148 | 0.000057 |
| 2 | 1.53344213 | 1.53344977 | 0.007641 | 1.53345009 | 0.007961 | 1.53344154 | -0.000589 |
| 3 | 1.75329793 | 1.75330140 | 0.003471 | 1.75330189 | 0.003961 | 1.75329742 | -0.000509 |
| 4 | 2.00071425 | 2.00071426 | 0.000010 | 2.00071425 | -0.000000 | 2.00071426 | 0.000010 |
| 5 | 2.00079815 | 2.00080498 | 0.006830 | 2.00080568 | 0.007530 | 2.00079737 | -0.000780 |
| 6 | 2.84479254 | 2.84479270 | 0.000155 | 2.84479263 | 0.000085 | 2.84479266 | 0.000115 |
| 7 | 2.84488258 | 2.84488974 | 0.007161 | 2.84488986 | 0.007281 | 2.84488231 | -0.000269 |
| 8 | 3.22373604 | 3.22373641 | 0.000370 | 3.22373624 | 0.000200 | 3.22373629 | 0.000250 |
| 9 | 3.22382989 | 3.22383759 | 0.007702 | 3.22383633 | 0.006442 | 3.22383101 | 0.001122 |
| 10 | 3.56269414 | 3.56269417 | 0.000032 | 3.56269425 | 0.000112 | 3.56269409 | -0.000048 |

## Both terms: first 10 BSE eigenvalues

The current full BSE benchmark is still best represented by `pw_only`. `paw_orth_only` and `paw_full` are included to show the present gap to VASP.

| # | VASP (eV) | `pw_only` (eV) | Δ (meV) | `paw_orth_only` (eV) | Δ (meV) | `paw_full` (eV) | Δ (meV) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.59755271 | 0.59789800 | 0.345287 | 0.43400254 | -163.550173 | 0.65950201 | 61.949297 |
| 2 | 0.60214685 | 0.60203072 | -0.116126 | 0.49803945 | -104.107396 | 0.66119218 | 59.045334 |
| 3 | 0.81773692 | 0.81819859 | 0.461668 | 0.60623595 | -211.500972 | 0.87824418 | 60.507258 |
| 4 | 0.95004106 | 0.95659218 | 6.551115 | 0.68622922 | -263.811845 | 1.12528334 | 175.242275 |
| 5 | 1.03878070 | 1.03404920 | -4.731497 | 0.76139151 | -277.389187 | 1.12533424 | 86.553543 |
| 6 | 1.06893044 | 1.06876025 | -0.170187 | 0.94310854 | -125.821897 | 1.28689198 | 217.961543 |
| 7 | 1.07222303 | 1.07209729 | -0.125745 | 0.96312944 | -109.093595 | 1.35850161 | 286.278575 |
| 8 | 1.30876837 | 1.30431547 | -4.452896 | 1.00186921 | -306.899156 | 1.61279466 | 304.026294 |
| 9 | 1.47644720 | 1.47286725 | -3.579953 | 1.13970348 | -336.743723 | 1.79063172 | 314.184517 |
| 10 | 1.54488018 | 1.54427524 | -0.604945 | 1.19330050 | -351.579685 | 1.85396126 | 309.081075 |

## Finite-q both terms (`05_bse_qext`, `q_ext = (0, 1/3, 0)`): first 10 BSE eigenvalues

`pw_only` is still the only mode that stays qualitatively close to the VASP finite-q reference. `paw_orth_only` and `paw_full` both remain visibly mismatched here, so this benchmark is explicitly marked as unfinished.

| # | VASP (eV) | `pw_only` (eV) | Δ (meV) | `paw_orth_only` (eV) | Δ (meV) | `paw_full` (eV) | Δ (meV) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -0.18138393 | -0.18127921 | 0.104722 | -0.28851911 | -107.135178 | -0.12242518 | 58.958752 |
| 2 | 0.25003412 | 0.24934634 | -0.687782 | -0.10776277 | -357.796892 | 0.57623512 | 326.200998 |
| 3 | 0.60274926 | 0.60174350 | -1.005763 | 0.49431290 | -108.436363 | 0.66096299 | 58.213727 |
| 4 | 1.05998377 | 1.03500121 | -24.982556 | 0.68067072 | -379.313046 | 1.18888517 | 128.901404 |
| 5 | 1.12784952 | 1.12799603 | 0.146509 | 1.01946294 | -108.386581 | 1.36039865 | 232.549129 |
| 6 | 1.50481125 | 1.50480422 | -0.007026 | 1.20085565 | -303.955596 | 1.56174181 | 56.930564 |
| 7 | 1.56919437 | 1.56782947 | -1.364898 | 1.40143079 | -167.763578 | 1.66168048 | 92.486112 |
| 8 | 1.60608688 | 1.60589453 | -0.192351 | 1.50089364 | -105.193241 | 1.88835858 | 282.271699 |
| 9 | 1.84661456 | 1.84698298 | 0.368422 | 1.57753466 | -269.079898 | 1.88971826 | 43.103702 |
| 10 | 1.84850710 | 1.84807950 | -0.427602 | 1.67033064 | -178.176462 | 1.90997386 | 61.466758 |
