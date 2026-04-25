# BSE matrix benchmark artifacts

This directory collects text-format benchmark artifacts for `bsematrix.py` against the current VASP 5.4.4 reference runs.

Source VASP run directories relative to the VBU root:

- direct-only: `../../tmp/VASP/vasp.5.4.4/examples/05_bse_direct_only`
- exchange-only / Hartree-term only: `../../tmp/VASP/vasp.5.4.4/examples/05_bse_hartree_only`
- both terms: `../../tmp/VASP/vasp.5.4.4/examples/05_bse`

Naming notes:

- `exchange_only` is the repulsive Hartree/exchange BSE term and comes from the VASP `05_bse_hartree_only` example.
- `hartree_only` files are included as aliases of the same reference set so both names are available in the data directory.
- All `AMAT` files here are plain text dumps with rows `i j real imag`.
- VASP `BSEFATBAND` references still contain the 8 excitons written by the original run; the eigenvalue tables below use the first 10 eigenvalues from diagonalizing the full `BSE_AMAT.bin` matrix.

## Artifact inventory

| Case | VASP files | Python `pw_only` | Python `paw_orth_only` |
| --- | --- | --- | --- |
| direct-only | `vasp_direct_only_BSEFATBAND.txt`, `vasp_direct_only_AMAT.txt` | `py_pw_only_direct_only_BSEFATBAND.txt`, `py_pw_only_direct_only_AMAT.txt` | `py_paw_orth_only_direct_only_BSEFATBAND.txt`, `py_paw_orth_only_direct_only_AMAT.txt` |
| exchange-only | `vasp_exchange_only_BSEFATBAND.txt`, `vasp_exchange_only_AMAT.txt` | `py_pw_only_exchange_only_BSEFATBAND.txt`, `py_pw_only_exchange_only_AMAT.txt` | `py_paw_orth_only_exchange_only_BSEFATBAND.txt`, `py_paw_orth_only_exchange_only_AMAT.txt` |
| hartree-only alias | `vasp_hartree_only_BSEFATBAND.txt`, `vasp_hartree_only_AMAT.txt` | `py_pw_only_hartree_only_BSEFATBAND.txt`, `py_pw_only_hartree_only_AMAT.txt` | `py_paw_orth_only_hartree_only_BSEFATBAND.txt`, `py_paw_orth_only_hartree_only_AMAT.txt` |
| both | `vasp_both_BSEFATBAND.txt`, `vasp_both_AMAT.txt` | `py_pw_only_both_BSEFATBAND.txt`, `py_pw_only_both_AMAT.txt` | `py_paw_orth_only_both_BSEFATBAND.txt`, `py_paw_orth_only_both_AMAT.txt` |

## Reproduction workflow

The files stored in this directory are lightweight benchmark artifacts. A full reproduction still needs a working VASP run directory with:

- `POSCAR`, `POTCAR`, `KPOINTS`
- a converged ground-state `WAVECAR` and `CHGCAR`
- enough empty bands for the later GW/BSE steps

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
cp INCAR.waveder INCAR
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
    --direct-source-paw \
    --vb-num 2 --cb-num 2 --ewin 0 6 \
    --use-response-basis \
    --output-prefix examples/bsematrix/BP/py_paw_orth_only_exchange_only_AMAT \
    --bsefatband-output examples/bsematrix/BP/py_paw_orth_only_exchange_only_BSEFATBAND.txt \
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
    --direct-source-paw \
    --vb-num 2 --cb-num 2 --ewin 0 6 \
    --use-response-basis \
    --output-prefix examples/bsematrix/BP/py_paw_orth_only_direct_only_AMAT \
    --bsefatband-output examples/bsematrix/BP/py_paw_orth_only_direct_only_BSEFATBAND.txt \
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
    --direct-source-paw \
    --vb-num 2 --cb-num 2 --ewin 0 6 \
    --use-response-basis \
    --output-prefix examples/bsematrix/BP/py_paw_orth_only_both_AMAT \
    --bsefatband-output examples/bsematrix/BP/py_paw_orth_only_both_BSEFATBAND.txt \
    --full-hermitian
```

After these runs, the text files generated by Python can be compared directly with the VASP references and with the tables below.

## AMAT comparison summary

| Case | Mode | `max(abs(ΔA))` (eV) | `norm(ΔA)_F` (eV) | Worst entry |
| --- | --- | ---: | ---: | --- |
| direct-only | `pw_only` | 0.00768173 | 0.03076986 | `(12,12)` |
| direct-only | `paw_orth_only` | 0.00256244 | 0.01002069 | `(27,12)` |
| exchange-only | `pw_only` | 0.00520426 | 0.04512843 | `(3,3)` |
| exchange-only | `paw_orth_only` | 0.00070821 | 0.00610677 | `(12,27)` |
| both | `pw_only` | 0.00356970 | 0.02483112 | `(3,21)` |
| both | `paw_orth_only` | 0.00327066 | 0.01518818 | `(27,12)` |

## Direct-only: first 10 BSE eigenvalues

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

## Exchange-only / Hartree-term only: first 10 BSE eigenvalues

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

## Both terms: first 10 BSE eigenvalues

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
