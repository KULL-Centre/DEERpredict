# Running the Calculations

The class for running the PRE calculations is `PREpredict`.

Here is an examples of the calculation intensity ratios and PRE rates for a mutant of PDB 1NTI (20 conformations) using the rotamer-library approach

~~~ python
PRE = PREpredict(MDAnalysis.Universe('1nti.pdb'), residue = 36, output_prefix = 'calcPREs/res', weights = False,
    load_file = False, tau_t = .5*1e-9, log_file = 'calcPREs/log', delay = 10e-3,
    tau_c = 2*1e-09, k = 1.23e16, r_2 = 10, temperature = 298, Z_cutoff = 0.2, Cbeta = False,
    atom_selection = 'H', wh = 750)
PRE.run()
~~~

The program generate a data file in `calcPREs` called `res-36.dat`. The first column contains the residue numbers while the second 
and the third the intensity ratios and the PRE rates in Hz, respectively.

