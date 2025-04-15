[![Build Status](https://app.travis-ci.com/KULL-Centre/DEERpredict.svg?branch=main)](https://app.travis-ci.com/KULL-Centre/DEERpredict)
[![Documentation Status](https://readthedocs.org/projects/deerpredict/badge/?version=latest)](https://deerpredict.readthedocs.io)
[![DOI](https://zenodo.org/badge/217526987.svg)](https://zenodo.org/badge/latestdoi/217526987)
[![SWH](https://archive.softwareheritage.org/badge/origin/https://github.com/KULL-Centre/DEERpredict/)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/KULL-Centre/DEERpredict)

DEER-PREdict
===========

Overview
--------

A package for double electron-electron resonance (DEER) and paramagnetic relaxation enhancement (PRE) predictions from molecular dynamics ensembles.

Installation
------------

To install DEER-PREdict, use the [PyPI package](https://pypi.org/project/DEERPREdict):

```bash
  pip install DEERPREdict
```

or clone the repo:

```bash
  git clone https://github.com/KULL-Centre/DEERpredict.git
  cd DEERpredict

  pip install -e . 
```

The software requires Python 3.6-3.9.

In case of dependency issues, consider installing FRETpredict in a new environment

```bash
  conda create -n myenv python=3.9 pip
  conda activate myenv
  pip install -e .
```

Documentation
-------------

[![Documentation Status](https://readthedocs.org/projects/deerpredict/badge/?version=latest&style=for-the-badge)](https://deerpredict.readthedocs.io)


Testing
-------

Run all the tests in one go

```bash
  cd DEERpredict

  python -m pytest
```
or run single tests, e.g.

```bash
  cd DEERpredict

  python -m pytest tests/test_PRE.py::test_ACBP
  python -m pytest tests/test_DEER.py::test_T4L
```

Example
-------------

Example of how to run PREpredict to calculate the intensity ratios and PRE rates for PDB code 1NTI (20 conformations) using the BASL MMMx rotamer library (see [notebook](https://github.com/KULL-Centre/DEERpredict/blob/main/tests/data/ACBP/ACBP.ipynb))

```python
PRE = PREpredict(MDAnalysis.Universe('1nti.pdb'), residue=36, libname='BASL MMMx',
          tau_t=.5*1e-9, log_file='calcPREs/log', temperature=298, z_cutoff=0.05,
          atom_selection='H', Cbeta=False)
PRE.run(output_prefix='calcPREs/BASL', tau_t=.5e-9, delay=10e-3,
          tau_c=2e-09, r_2=10, wh=750)
```

License
-------------

This project is licensed under the GNU General Public License version 3.0 (GPL-3.0). However, the rotamer libraries are modified versions of those from the [MMMx program](https://mmmx.info/index.html), and these modified libraries are licensed under the MIT License, as detailed in the LICENSE file. The rest of the project is licensed under the GPL-3.0, and any combination of GPL-3.0 licensed files with those under the MIT License will be subject to the terms of the GPL-3.0.

Authors
-------------

[Giulio Tesei (@gitesei)](https://github.com/gitesei)

[João M Martins (@joaommartins)](https://github.com/joaommartins)

[Micha BA Kunze (@mbakunze)](https://github.com/mbakunze)

[Ramon Crehuet (@rcrehuet)](https://github.com/rcrehuet)

[Kresten Lindorff-Larsen (@lindorff-larsen)](https://github.com/lindorff-larsen)


Article
-------------

Tesei G, Martins JM, Kunze MBA, Wang Y, Crehuet R, and Lindorff-Larsen K (2021) 
DEER-PREdict: Software for efficient calculation of spin-labeling EPR and NMR data from conformational ensembles. 
PLOS Computational Biology 17(1): e1008551. [https://doi.org/10.1371/journal.pcbi.1008551](https://doi.org/10.1371/journal.pcbi.1008551)
