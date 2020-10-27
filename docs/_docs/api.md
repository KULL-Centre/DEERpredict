# DEER-PREdict API Reference

DEER-PREdict has three main classes:
- `DEERpredict` performs Double Electron-Electron Resonance predictions. <br>
   Functions: `trajectoryAnalysis`, `run` and `save`.
- `PREpredict` performs Paramagnetic Relaxation Enhancement calculations. <br>
   Functions: `trajectoryAnalysis`, `trajectoryAnalysisCbeta`, `run` and `save`.
- `Operations` is the base class containing attributes and methods inherited and used by the calculation classes. <br> 
   Functions: `precalculate_rotamer`, `rotamer_placement`, `lj_calculation`, `rotamerWeights`, `rotamerPREanalysis`, `calc_gamma_2`, `calc_gamma_2_Cbeta` and `calcTimeDomain`.

New rotamer libraries should be defined in the `DEERpredict.libraries.LIBRARIES` dictionary.

## RotamerLibrary class

`DEERpredict.libraries.LIBRARIES`: Loaded from `DEERPREdict/lib/libraries.yml`, rotamers libraries consist of a PDB file, a DCD files and a text file for the weights. These files are included in the `DEERPREdict/lib` folder.

`DEERpredict.libraries.RotamerLibrary`: Makes available the attributes `top` (rotamer topology), `coord` (rotamer coordinates) and `weights` (intrinsic probability of each rotamer).

## Lennard-Jones parameters

`DEERpredict.lennardjones`: Lennard-Jones parameters of the CHARMM36 force field used to calculate the external 
energy contribution to the Boltzmann weight of each conformer.

~~~python 
DEERpredict.lennardjones.lj_parameters = {
    'C': {
        'vdw': 1.70,
        'p_q': 0,
        'p_Rmin2': 2.275,
        'eps': -0.020
    }, 
    ...
}
~~~
