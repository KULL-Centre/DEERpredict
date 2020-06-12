# DEER-PREdict API reference

The module has three main classes:
- `DEERpredict` performs Double Electron-Electron Resonance predictions,
- `PREpredict` performs Paramagnetic Relaxation Enhancement calculations.
- `Operations` is the base class containing attributes and methods inherited and used by the calculation classes.

New libraries should be defined in the `DEERpredict.libraries.LIBRARIES` dictionary.

## Operations class

## RotamerLibrary class

DEERpredict.libraries.LIBRARIES: Loaded from data/libraries.yaml

DEERpredict.libraries.RotamerLibrary

## Lennard-Jones parameters

DEERpredict.lennardjones: 
DEERpredict.lennardjones.lj_parameters = {atomtype: {'vdw', 'p_q', 'p_Rmin2', 'eps'}, ...}
