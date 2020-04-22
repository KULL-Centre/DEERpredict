# DEERpredict API reference

The module is composed of three main classes, each for the prediction of different probe-based methods.
`DEERpredict` performs Double Electron-Electron Resonance predictions,
`PREpredict` performs Paramagnetic Relaxation Enhancement calculations, and
`FRETpredict` performs Fluorescence Resonance Energy Transfer calculations.

`Operations` is the base class containing the class attributes to be
inherited by the calculation classes, as well as common methods to be used by the calculation classes.

New libraries should be defined in the `DEERpredict.libraries.LIBRARIES` dictionary.

## Operations class

## RotamerLibrary class

DEERpredict.libraries.LIBRARIES: Loaded from data/libraries.yaml

DEERpredict.libraries.RotamerLibrary

## Lennard-Jones parameters

DEERpredict.lennardjones: 
DEERpredict.lennardjones.lj_parameters = {atomtype: {'vdw', 'p_q', 'p_Rmin2', 'eps'}, ...}
