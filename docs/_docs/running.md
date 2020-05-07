# Running Calculations

## PREpredict

The class for running the PRE calculations is `PREpredict`.

Here is how to calculate intensity ratios and PRE rates for a mutant of PDB 1NTI (20 conformations) using the rotamer-library approach

~~~ python
from DEERpredict.PRE import PREpredict

PRE = PREpredict(MDAnalysis.Universe('1nti.pdb'), residue = 36, output_prefix = 'calcPREs/res',
    tau_t = .5*1e-9, log_file = 'calcPREs/log', delay = 10e-3,
    tau_c = 2*1e-09, k = 1.23e16, r_2 = 10, temperature = 298,
    atom_selection = 'H', wh = 750)
PRE.run()
~~~

The program generates a data file in `calcPREs` called `res-36.dat`. The first column contains the residue numbers while the second 
and third contain the intensity ratios and the PRE rates (in Hz), respectively.

### Reweighting

Per-frame distances and angles are saved in the pickle file `res-36.pkl`. These intermediate quantities can be used to reweight the trajectory
by statistical weights obtaine e.g. from BME reweighting:

~~~ python
PRE = PREpredict(MDAnalysis.Universe('1nti.pdb'), residue = 36, output_prefix = 'calcPREs/res', weights = weights,
    load_file = res-36.pkl, tau_t = .5*1e-9, log_file = 'calcPREs/log', delay = 10e-3,
    tau_c = 2*1e-09, k = 1.23e16, r_2 = 10, temperature = 298, Z_cutoff = 0.2, Cbeta = False,
    atom_selection = 'H', wh = 750)
PRE.run()
~~~

### Intermolecular PREs

To calculate intermolecular PREs, a list of two strings can be set to the `chains` option indicating the segment id of the labeled chain and the NMR-active chain.

~~~ python
PRE = PREpredict(MDAnalysis.Universe('3BVB.pdb'), residue = 55, chains = ['A', 'B'], output_prefix = 'calcPREs/res',
    tau_t = .5*1e-9, log_file = 'calcPREs/log', delay = 10e-3, tau_c = 2*1e-09, k = 1.23e16, r_2 = 10, temperature = 298,
    atom_selection = 'N', wh = 750)
PRE.run()
~~~

### Approximate electron positions to C$\beta$ coordinates

Instead of using the rotamer library approach, the position of the unpaired electron can be approximated to the position of the C$\beta$ atom of the spin-labeled residue
setting `Cbeta=True`.

~~~ python
PRE = PREpredict(MDAnalysis.Universe('1nti.pdb'), residue = 36, output_prefix = 'calcPREs/res',
    tau_t = .5*1e-9, log_file = 'calcPREs/log', delay = 10e-3,
    tau_c = 2*1e-09, k = 1.23e16, r_2 = 10, temperature = 298, Cbeta = True,
    atom_selection = 'H', wh = 750)
PRE.run()
~~~

## DEERpredict

Here is an example of how to run DEERpredict to calculate the DEER distribution for HIV-1 protease (PDB ID 3BVB) labeled with nitroxide groups at residue 55.

~~~ python
from DEERpredict.DEER import DEERpredict

DEER = DEERpredict(MDAnalysis.Universe('3BVB.pdb'), residues = [55, 55], chains=['A', 'B'], output_prefix = 'res', weights = False,
      load_file = False, log_file = 'log', temperature = 298 )
DEER.run()
~~~

DEERpredict generates `res-55-55.dat` containing the smoothed distance distribution and `res-55-55_time_domain.dat` containing the intensity-normalized time-domain DEER data.
Per-frame distance distributions are saved to the hdf5 file 'res-55-55.hdf5' making it possible to quickly reweight the data as shown above for PREpredict.
