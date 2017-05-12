#!/usr/bin/env python

import numpy as np
from vaspwfc import vaspwfc

wfc = vaspwfc('./WAVECAR')
phi = wfc.wfc_r(ikpt=2, iband=27, ngrid=wfc._ngrid * 2)
wfc.save2vesta(phi)
