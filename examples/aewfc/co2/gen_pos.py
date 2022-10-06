#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from ase.io import write
from ase.build import molecule

xx      = molecule('CO2')
L       = 10.0
xx.cell = np.eye(3) * L
xx.pbc  = True
xx.positions[:,-1] += L / 2.

write('POSAR', xx, vasp5=True)
