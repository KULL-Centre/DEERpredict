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

The software requires Python 3.6+.
    
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


Authors
-------------

[Giulio Tesei (@gitesei)](https://github.com/gitesei)

[João M Martins (@joaommartins)](https://github.com/joaommartins)

[Micha BA Kunze (@mbakunze)](https://github.com/mbakunze)

[Ramon Crehuet (@rcrehuet)](https://github.com/rcrehuet)

[Kresten Lindorff-Larsen (@lindorff-larsen)](https://github.com/lindorff-larsen)


Article
-------------

Tesei G, Martins JM, Kunze MBA, Wang Y, Crehuet R, et al. (2021) 
DEER-PREdict: Software for efficient calculation of spin-labeling EPR and NMR data from conformational ensembles. 
PLOS Computational Biology 17(1): e1008551. [https://doi.org/10.1371/journal.pcbi.1008551](https://doi.org/10.1371/journal.pcbi.1008551)
