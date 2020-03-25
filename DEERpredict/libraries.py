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

#: Name of the directory in the package that contains the library data.
LIBDIR = "../data"

# LIBRARIES = {
#     'MTSSL 175K X1X2': {
#         'topology': 'MTSSL_175K_X1X2.pdb',
#         'data': 'MTSSL_175K_X1X2.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'MTSSL 175K CASD': {
#         'topology': 'MTSSL_175K_CaSd_216.pdb',
#         'data': 'MTSSL_175K_CaSd_216.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 10cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_3step_10cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 10cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_3step_10cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 15cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_3step_15cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 15cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_3step_15cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 20cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_3step_20cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 20cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_3step_20cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 25cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_3step_25cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 25cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_3step_25cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 30cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_3step_30cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 30cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_3step_30cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 40cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_3step_40cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 40cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_3step_40cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 50cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_3step_50cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 50cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_3step_50cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 75cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_3step_75cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 75cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_3step_75cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 100cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_3step_100cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 100cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_3step_100cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 125cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_3step_125cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 125cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_3step_125cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 150cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_3step_150cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 150cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_3step_150cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 200cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_3step_200cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 200cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_3step_200cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 300cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_3step_300cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 300cutoff 3step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_3step_300cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 10cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_2step_10cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 10cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_2step_10cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 15cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_2step_15cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 15cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_2step_15cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 20cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_2step_20cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 20cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_2step_20cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 25cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_2step_25cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 25cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_2step_25cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 30cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_2step_30cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 30cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_2step_30cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 40cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_2step_40cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 40cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_2step_40cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 50cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_2step_50cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 50cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_2step_50cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 75cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_2step_75cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 75cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_2step_75cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 100cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_2step_100cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 100cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_2step_100cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 125cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_2step_125cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 125cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_2step_125cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 150cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_2step_150cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 150cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_2step_150cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 200cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_2step_200cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 200cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_2step_200cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 488 300cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa488.pdb',
#         'data': 'rot_lib_matrix_alexa488_2step_300cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
#     'Alexa 594 300cutoff 2step': {
#         'topology': 'rot_lib_matrix_alexa594.pdb',
#         'data': 'rot_lib_matrix_alexa594_2step_300cutoff.txt',
#         'author': 'US?',
#         'license': 'dont know',
#         'citation': 'TBD'
#     },
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

    The library makes available the attributes :attr:`data`, and :attr:`weights`.

    Attributes:
        data (:py:class:`numpy.ndarray`): Array containing the relative coordinates of each rotamer pose

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
        self.data = np.loadtxt(self.lib['data'])


        # FIXME: Temporary change to allow loading as float32
        self.data = self.data.astype('float32', copy=False)

        self.weights = self.read_rotamer_weights()

    def read_rotamer_weights(self):
        """

        Extracts the weights from the loaded rotamer data.

        Returns:
            numpy.ndarray: Rotamer library weights

        """
        weights = self.data.reshape((self.data.shape[0] // (len(self.top.atoms)), len(self.top.atoms), 6))[:, 0, 5]
        return weights

    def __repr__(self):
        """

        Returns:
            (:py:class:`str`): Name of the library in use.
        """
        return "<RotamerLibrary '{0}' by {1} with {2} rotamers>".format(self.name, self.lib['author'],
                                                                        len(self.weights)-2)
