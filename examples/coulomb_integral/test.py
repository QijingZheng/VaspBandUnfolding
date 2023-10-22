import os, sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parent = os.path.dirname(parent)
sys.path.append(parent)

from coulomb_integral import (
    PWCoulombIntegral,
)

pwci = PWCoulombIntegral(fnm="./WAVECAR")
hartree_energy = - pwci.coulomb_integral(1,1,1,1).real / 2.0
print("Hartree energy contributed by pseudo wavefunction: {}, DENC from OUTCAR: {}".format(
      hartree_energy, -3.69293809))
