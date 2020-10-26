# -*- coding: utf-8 -*-
"""
Rotamer library handling
========================

:mod:`rotamers.library` contains the data (:data:`LIBRARIES`) to load
a rotamer library, represented by a :class:`RotamerLibrary`.

"""

import MDAnalysis

import logging
logger = logging.getLogger("MDAnalysis.app")

import numpy as np
import os.path
import pkg_resources
import yaml
import warnings
warnings.filterwarnings('ignore')

#: Name of the directory in the package that contains the library data.
LIBDIR = "./lib"

#Copyright © 2009-2013, Yevhen Polyhach, Stefan Stoll & Gunnar Jeschke
#
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer 
#  in the documentation and/or other materials provided with the distribution
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
#THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS 
#BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE 
#GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
#LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# LIBRARIES = {
#     'MTSSL 175K X1X2': {
#         'topology': 'MTSSL_175K_X1X2.pdb',
#         'data': 'MTSSL_175K_X1X2.txt',
#         'author': 'Yevhen Polyhach, Enrica Bordignon and Gunnar Jeschke',
#         'license': 'GPLv3',
#         'citation': 'DOI 10.1039/C0CP01865A'
#     },
#     'MTSSL 175K CASD': {
#         'topology': 'MTSSL_175K_CaSd_216.pdb',
#         'data': 'MTSSL_175K_CaSd_216.txt',
#         'author': 'Yevhen Polyhach, Enrica Bordignon and Gunnar Jeschke',
#         'license': 'GPLv3',
#         'citation': 'DOI 10.1039/C0CP01865A'
#     }
# }

def find_file(filename, pkglibdir=LIBDIR):
    """
    Function to find path using MDAnalysis/

    Args:
        filename:
        pkglibdir:

    Returns:

    """
    if os.path.exists(filename):
        return MDAnalysis.lib.util.realpath(filename)
    return pkg_resources.resource_filename(__name__, os.path.join(pkglibdir, filename))


with open(find_file('libraries.yml'), 'r') as yaml_file:
    #: Registry of libraries, indexed by name.
    #: Takes the format = {name: {topology, data, author, license, citation}, ...}
    LIBRARIES = yaml.load(yaml_file, Loader=yaml.FullLoader)

class RotamerLibrary(object):
    """
    Rotamer library

    The library makes available the attributes :attr:`data` and :attr:`weights`.

    Attributes:
        coord (:py:class:`numpy.ndarray`): Array containing the relative coordinates of each rotamer.

        weights (:py:class:`numpy.ndarray`): Array containing the population of each rotamer.

        name (:py:class:`str`): Name of the library.

        lib (:py:class:`dict`): Dictionary containing the file names and meta data for the library :attr:`name`.

    """

    def __init__(self, name):
        """

        Args:
            name (:py:class:`str`): name of the library (must exist in the registry of libraries, :data:`LIBRARIES`)
        """
        self.name = name
        self.lib = {}
        try:
            self.lib.update(LIBRARIES[name])  # make a copy
        except KeyError:
            raise ValueError("No rotamer library with name {0} known: must be one of {1}".format(name,
                                                                                                 list(LIBRARIES.keys())))
        logger.info("Using rotamer library '{0}' by {1[author]}".format(self.name, self.lib))
        logger.info("Please cite: {0[citation]}".format(self.lib))
        # adjust paths
        for k in 'data', 'topology':
            self.lib[k] = find_file(self.lib[k])
        logger.debug("[rotamers] ensemble = {0[data]} with topology = {0[topology]}".format(self.lib))
        logger.debug("[rotamers] populations = {0[data]}".format(self.lib))

        self.top = MDAnalysis.Universe(self.lib['topology'])
        traj = MDAnalysis.Universe(self.lib['topology'],self.lib['data']+'.dcd')
        # extract coordinates from XTC trajectory
        self.coord = traj.trajectory.timeseries(traj.atoms)
        self.weights = np.loadtxt(self.lib['data']+'_weights.txt')
        self.weights /= np.sum(self.weights)

    def __repr__(self):
        """

        Returns:
            (:py:class:`str`): Name of the library in use.
        """
        return "<RotamerLibrary '{0}' by {1} with {2} rotamers>".format(self.name, self.lib['author'],
                                                                        len(self.weights)-2)
