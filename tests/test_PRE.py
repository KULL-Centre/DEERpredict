import pytest
import MDAnalysis
import os
import sys
import numpy as np
from DEERPREdict.PRE import PREpredict
import pandas as pd

def load_precalcPREs(path,labels,tau_c,Cbeta):
    CB = '_CB' if Cbeta else ''
    data = {}
    for label in labels:
        resnums, data[label] = np.loadtxt(path+'/PRE_1nti_{:g}-{:d}{:s}.dat'.format(tau_c,label,CB),unpack=True)
    df = pd.DataFrame(data,index=resnums)
    df.rename_axis('residue', inplace=True)
    df.rename_axis('label', axis='columns',inplace=True)
    return resnums, df

def load_calcPREs(path,labels):
    data = {}
    for label in labels:
        resnums, data[label], _ = np.loadtxt(path+'/res-{:d}.dat'.format(label),unpack=True)
    df = pd.DataFrame(data, index=resnums)
    df.rename_axis('residue', inplace=True)
    df.rename_axis('label', axis='columns',inplace=True)
    return resnums, df

def calcIratio(path,tau_c,args):
    u, label, tau_t, r_2, Cbeta = args
    PRE = PREpredict(u, label, log_file = path+'/log',
          temperature = 298, Cbeta = Cbeta, atom_selection = 'H')
    PRE.run(output_prefix = path+'/calcPREs/res', tau_c = tau_c*1e-9, tau_t = tau_t*1e-9, r_2 = r_2, wh = 750, delay = 1e-2)

def test_ACBP():
    if not os.path.isdir('tests/data/ACBP/calcPREs'):
        os.mkdir('tests/data/ACBP/calcPREs')
    u = MDAnalysis.Universe('tests/data/ACBP/1nti.pdb')
    labels = [17,36,46,65,86]
    for Cbeta in [True,False]:
        for tau_c in [0.1,2]: 
            tau_t = tau_c if tau_c < 0.5 else 0.5
            for label in labels:
                calcIratio('tests/data/ACBP',tau_c,[u, label, tau_t, 12.6, Cbeta])
            resnums, precalcPREs = load_precalcPREs('tests/data/ACBP/precalcPREs',labels,tau_c,Cbeta)
            resnums, calcPREs = load_calcPREs('tests/data/ACBP/calcPREs',labels)
            assert np.power(precalcPREs-calcPREs,2).sum().sum() < 0.3

def test_NANODISC():
    if not os.path.isdir('tests/data/nanodisc/calcPREs'):
        os.mkdir('tests/data/nanodisc/calcPREs')
    weights = np.loadtxt('tests/data/nanodisc/BME_weights.txt')
    labels = [67,166,100,192,148,213,235]
    u = MDAnalysis.Universe('tests/data/nanodisc/md.pdb','tests/data/nanodisc/md.xtc')
    for label in labels:
        PRE = PREpredict(u, residue = label, chains = ['A','B'], temperature = 303.15, atom_selection = 'N')
        PRE.run(output_prefix = 'tests/data/nanodisc/calcPREs/res', weights = weights,
                tau_t = 1*1e-9, delay = 0.01, tau_c = 34*1e-09, k = 1.23e16, r_2 = 60, wh = 600)
    resnums, precalcPREs = load_calcPREs('tests/data/nanodisc/precalcPREs',labels)
    resnums, calcPREs = load_calcPREs('tests/data/nanodisc/calcPREs',labels)
    print(np.power(precalcPREs-calcPREs,2).sum().sum())
    assert np.power(precalcPREs-calcPREs,2).sum().sum() < 0.001
