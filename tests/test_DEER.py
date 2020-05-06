import pytest
import MDAnalysis
import numpy as np
from DEERpredict.DEER import DEERpredict

def test_DEER():
    u = MDAnalysis.Universe('tests/data/HIV-1PR/HIV-1PR.pdb')
    DEER = DEERpredict(u, residues = [55, 55], chains=['A', 'B'], output_prefix = 'tests/data/HIV-1PR/res', 
            weights = False, load_file = False, log_file = 'tests/data/HIV-1PR/HIV-1PR/log', temperature = 298)
    DEER.run()
    r, p, _ = np.loadtxt('tests/data/HIV-1PR/res-55-55.dat',unpack=True)
    r_ref, p_ref = np.loadtxt('tests/data/HIV-1PR/DEER_HIV-1PR.dat',unpack=True)
    assert np.power(p-p_ref,2).sum() < 0.01
