# Spinor Maker Example (MoSe2 2x3)

This example is prepared from:

- `/Users/ionizing/Data/MoSe2-exciton-LMI/2x3/spinless`
- `/Users/ionizing/Data/MoSe2-exciton-LMI/2x3/ispin2`

Included files:

- `POSCAR`
- `POTCAR`
- `KPOINTS`
- `INCAR`

Not included:

- `WAVECAR`, `NormalCAR`, `SocCar`, `SocRadCar` (too large for repository examples)

Directory layout:

```text
examples/spinor/
  ├── spinless/
  │   ├── INCAR
  │   ├── KPOINTS
  │   ├── POSCAR
  │   └── POTCAR
  └── ispin2/
      ├── INCAR
      ├── KPOINTS
      ├── POSCAR
      └── POTCAR
```

## Usage

After you place `WAVECAR`, `NormalCAR`, `SocCar`, and `SocRadCar` into either
`spinless/` or `ispin2/`, run:

```bash
spinormaker --mixwave-ibs 211 213 215 217 219 --correct-kpts 1
```

For full k-point and full odd/even band-pair post-processing:

```bash
spinormaker --full-kpts --full-bands
```

Other useful modes:

```bash
# all k-points, selected band pairs
spinormaker --mixwave-ibs 211 213 215 217 219 --full-kpts

# single k-point (default mixwave-ikpt=1), all odd/even band pairs
spinormaker --full-bands
```

Compatibility alias:

```bash
spinormaker --full-kpts-full-bands
```

which is equivalent to:

```bash
spinormaker --full-kpts --full-bands
```

## `unfold_main` Validation Snapshot

Using `M = [[4,2,0],[0,3,0],[0,0,1]]`, the `ispin2` case gives:

```text
ispin2 213-220
213 E= -1.21393 K+= 0.00000 K-= 0.99958 G= 0.00000 sz=+0.78827
214 E= -1.21393 K+= 0.99958 K-= 0.00000 G= 0.00000 sz=-0.78827
215 E= -1.01817 K+= 0.99958 K-= 0.00000 G= 0.00000 sz=+0.78692
216 E= -1.01817 K+= 0.00000 K-= 0.99959 G= 0.00000 sz=-0.78692
217 E=  0.27498 K+= 0.99966 K-= 0.00000 G= 0.00000 sz=+0.72199
218 E=  0.27498 K+= 0.00000 K-= 0.99966 G= 0.00000 sz=-0.72199
219 E=  0.31567 K+= 0.00000 K-= 0.99966 G= 0.00000 sz=+0.73272
220 E=  0.31567 K+= 0.99966 K-= 0.00000 G= 0.00000 sz=-0.73272
```

The odd/even spin ordering is preserved (`odd: sz > 0`, `even: sz < 0`), and
each band in 213-220 is strongly valley-polarized in K+ or K-.

## Unchanged `WAVECAR` Reference

For comparison, the original (unchanged) scalar `WAVECAR` was unfolded using
half band indices:

```text
ib_scalar = (ib_spinor + 1) // 2
213-220 -> 107-110
```

`ispin2` unchanged `WAVECAR`:

```text
spin channel 1
107 E= -1.11929 K+= 0.49979 K-= 0.49979 G= 0.00000
108 E= -1.11928 K+= 0.49979 K-= 0.49979 G= 0.00000
109 E=  0.29664 K+= 0.49983 K-= 0.49983 G= 0.00000
110 E=  0.29671 K+= 0.49983 K-= 0.49983 G= 0.00000

spin channel 2
107 E= -1.11929 K+= 0.49979 K-= 0.49979 G= 0.00000
108 E= -1.11928 K+= 0.49979 K-= 0.49979 G= 0.00000
109 E=  0.29664 K+= 0.49983 K-= 0.49983 G= 0.00000
110 E=  0.29671 K+= 0.49983 K-= 0.49983 G= 0.00000
```

`spinless` unchanged `WAVECAR`:

```text
107 E= -1.11926 K+= 0.49979 K-= 0.49979 G= 0.00000
108 E= -1.11926 K+= 0.49979 K-= 0.49979 G= 0.00000
109 E=  0.29668 K+= 0.49983 K-= 0.49983 G= 0.00000
110 E=  0.29675 K+= 0.49983 K-= 0.49983 G= 0.00000
```
