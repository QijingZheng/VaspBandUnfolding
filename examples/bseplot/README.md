# BSE Plot Example

This example uses a monolayer MoSe2 BSE calculation and demonstrates the reciprocal-space `bseplot bz` workflow. The directory was previously named `examples/bse`; all README references should now use `examples/bseplot`.

Included files:
- `BSEFATBAND`: exciton amplitudes used by `bseplot bz`
- `INCAR`: representative BSE input for the main exciton calculation
- `INCAR.electron`: reference fixed-hole BSE input (`BSEHOLE = 0.5 0.5 0.5`)
- `INCAR.hole`: reference fixed-electron BSE input (`BSEELECTRON = 0.5 0.5 0.5`)
- `POSCAR`, `KPOINTS`, `OUTCAR`: structural and run metadata for the BSE calculation
- `POTCAR`: local MoSe2 POTCAR bundled so the example mirrors a standard VASP run directory
- `exciton_001_bz.png`: first-BZ density plot for the lowest exciton
- `X1-electron-bsepy.png`, `X1-electron-vasp.png`: fixed-hole electron-density comparison
- `X1-hole-bsepy.png`, `X1-hole-vasp.png`: fixed-electron hole-density comparison

Generate the BZ plot:

```bash
bseplot bz --input BSEFATBAND --poscar POSCAR --exciton 1 --output-dir .
```

This writes a PNG similar to `exciton_001_bz.png`.

Example output:

![Lowest-exciton BZ density](./exciton_001_bz.png)

For real-space excitons, `bseplot realspace` additionally needs `WAVECAR`, which is not bundled here because it is too large for the examples tree. The corresponding commands are:

```bash
bseplot realspace --bsefatband BSEFATBAND --wavecar /path/to/WAVECAR --poscar POSCAR \
    --exciton 1 --hole '0.5,0.5,0.5'

bseplot realspace --bsefatband BSEFATBAND --wavecar /path/to/WAVECAR --poscar POSCAR \
    --exciton 1 --electron '0.5,0.5,0.5'
```

The real-space workflow also needs `OUTCAR` because `bsefatband.py` / `bseplot realspace` read symmetry
operators and the `IBZKPT_HF` full-BZ mapping from it when restoring Bloch
phases for irreducible-k-point `WAVECAR`s.

You can then open the resulting `*.vasp` scalar grids in VESTA and render the real-space exciton density there.

Fixed-hole electron density comparison for exciton 1:

| `bseplot` | VASP |
| --- | --- |
| ![Exciton 1 electron density from bseplot](./X1-electron-bsepy.png) | ![Exciton 1 electron density from VASP](./X1-electron-vasp.png) |

Fixed-electron hole density comparison for exciton 1:

| `bseplot` | VASP |
| --- | --- |
| ![Exciton 1 hole density from bseplot](./X1-hole-bsepy.png) | ![Exciton 1 hole density from VASP](./X1-hole-vasp.png) |

Note:
- `OUTCAR` for this BSE run reports `PAW_PBE Mo_sv_GW 05Dec2013` and `PAW_PBE Se_GW 20Mar2012`.
- The exact GW POTCAR used for that run was not available in the accessible source tree of this session.
- The bundled `POTCAR` is the nearest local MoSe2 POTCAR and is included as a practical example placeholder rather than a provenance-exact GW pseudopotential archive.
