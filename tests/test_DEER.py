import pytest
import MDAnalysis
import numpy as np
from DEERpredict.DEER import DEERpredict

def test_DEER():
    u = MDAnalysis.Universe('tests/data/HIV-1PR/HIV-1PR.pdb')
    DEER = DEERpredict(u, residues = [55, 55], chains=['A', 'B'], temperature = 298, log_file = 'tests/data/HIV-1PR/log')
    DEER.run(output_prefix = 'tests/data/HIV-1PR/res')
    r, p = np.loadtxt('tests/data/HIV-1PR/res-55-55.dat',unpack=True)
    r_ref, p_ref = np.loadtxt('tests/data/HIV-1PR/DEER_HIV-1PR.dat',unpack=True)
    p_ref /= np.trapz(p_ref, r_ref)
    assert np.power(p-p_ref,2).sum() < 0.01
