#!/usr/bin/env python

import numpy as np
from vaspwfc import vaspwfc, save2vesta

kptw = [1, 6, 6, 6, 6, 6, 6, 12, 12, 12, 6, 6, 12, 12, 6, 6]

wfc = vaspwfc('./WAVECAR')
# chi = wfc.elf(kptw=kptw, ngrid=wfc._ngrid * 2)
chi = wfc.elf(kptw=kptw, ngrid=[20, 20, 150])
save2vesta(chi[0], lreal=True, poscar='POSCAR', prefix='elf')
