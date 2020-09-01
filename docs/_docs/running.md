# Running Calculations

## PREpredict

The class for running the PRE calculations is `PREpredict`.

Here is how to calculate intensity ratios and PRE rates for a mutant of PDB 1NTI (20 conformations) using the rotamer-library approach

~~~ python
from DEERPREdict.PRE import PREpredict

u = MDAnalysis.Universe('1nti.pdb')
PRE = PREpredict(u, residue = 36, log_file = 'log', temperature = 298, atom_selection = 'H')
PRE.run(output_prefix = 'res', tau_c = 2*1e-09, tau_t = .5*1e-9, delay = 10e-3, r_2 = 10, wh = 750)
~~~

The program generates a data file called `res-36.dat`. The first column contains the residue numbers while the second 
and third contain the intensity ratios and the PRE rates (in Hz), respectively.
Additionally, the Boltzmann weighted averages of $r^{-3}$, $r^{-6}$ and $( 3 \cos^{2} \Omega - 1 ) / 2$ over the rotamer states are saved to the pickle file `res-36.pkl`.
The sum over the Boltzmann weights for the Lennard-Jones probe-protein interaction energies are saved to `res-Z-36.pkl`.

### Reweighting

Per-frame distances and angles are saved in the pickle file `res-36.pkl`. These intermediate quantities can be used to reweight the trajectory
by statistical weights obtaine e.g. from BME reweighting:

~~~ python
PRE.run(output_prefix = 'calcPREs/res', tau_c = 2*1e-09, tau_t = .5*1e-9, delay = 10e-3, r_2 = 10, wh = 750, weights = weights, load_file = 'res-36.pkl')
~~~

### Intermolecular PREs

To calculate intermolecular PREs, a list of two strings can be set to the `chains` option indicating the segment id of the labeled chain and the NMR-active chain.

~~~ python
u = MDAnalysis.Universe('3BVB.pdb')
PRE = PREpredict(u, residue = 55, chains = ['A', 'B'], log_file = 'calcPREs/log', temperature = 298, atom_selection = 'N')
PRE.run(output_prefix = 'calcPREs/res', tau_c = 2*1e-09, tau_t = .5*1e-9, delay = 10e-3, r_2 = 10, wh = 750)
~~~

### Approximate electron positions to C$\beta$ coordinates

Instead of using the rotamer library approach, the position of the unpaired electron can be approximated to the position of the C$\beta$ atom of the spin-labeled residue
setting `Cbeta=True`.

~~~ python
from DEERPREdict.PRE import PREpredict

u = MDAnalysis.Universe('1nti.pdb')
PRE = PREpredict(u, residue = 36, log_file = 'log', temperature = 298, atom_selection = 'H', Cbeta = True)
PRE.run(output_prefix = 'calcPREs/res', tau_c = 2*1e-09, tau_t = .5*1e-9, delay = 10e-3, r_2 = 10, wh = 750)
~~~


## DEERpredict

Here is an example of how to run DEERpredict to calculate the DEER distribution for HIV-1 protease (PDB ID 3BVB) labeled with nitroxide groups at residue 55.

~~~ python
from DEERPREdict.DEER import DEERpredict

DEER = DEERpredict(MDAnalysis.Universe('3BVB.pdb'), residues = [55, 55], chains=['A', 'B'], log_file = 'log', temperature = 298 )
DEER.run(output_prefix = 'res')
~~~

DEERpredict generates `res-55-55.dat` containing the smoothed distance distribution and `res-55-55_time_domain.dat` containing the time-domain 
DEER data (Eq. 3 in DOI: 10.1126/sciadv.aat5218).
Per-frame distance distributions are saved to the hdf5 file 'res-55-55.hdf5' making it possible to quickly reweight the data as shown above for PREpredict.
The function to back-calculate the time-domain data from a distance distribution can also be accessed externally from the `Operations` class.
The sums over the Boltzmann weights for the Lennard-Jones probe-protein interaction energies of positions K55 and K55' are saved to `res-Z-55-55.pkl`.

~~~ python
from DEERPREdict.utils import Operations

r, p = np.loadtxt('res-55-55.dat', unpack=True)
t = np.linspace(0.01, 5.5, 512)
dt = Operations.calcTimeDomain(t,r,p)
~~~
