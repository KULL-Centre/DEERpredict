# -*- coding: utf-8 -*-
"""
DEERPrediction Class
--------------------

Class to perform DEER prediction.

"""

import os

# Coordinates and arrays
import numpy as np
import h5py
import MDAnalysis
import math

# Logger
import logging

# Inner imports
from DEERpredict.utils import Operations

class DEERpredict(Operations):
    """Calculation of distance distributions between two spin labels."""

    def __init__(self, protein, residues, **kwargs):
        """

        Args:
            protein_structure (:py:class:`MDAnalysis.core.universe.Universe`):
            residues (list(:py:class:`str`)):
        """
        # """RotamerDistances(universe, residue_list, **kwargs)
        #
        #         :Arguments:
        #            *universe*
        #               :class:`MDAnalysis.Universe`
        #            *residues*
        #               list of two residue numbers ``(r1, r2)`` that indicate
        #               the labelled sites
        #
        #         :Keywords:
        #            *dcdFilename*
        #               name of the temporary files with rotamers fitted [``'trj'``]
        #            *outputFile*
        #               stem of the name of the file containing the distance histogram
        #               (the final name will be ``<outputFile><resid_1>-<resid_2>.dat``
        #               [``'distances'``]
        #            *libname*
        #               library name; the library is loaded with
        #               :class:`rotcon.library.RotamerLibrary` [``'MTSSL 175K X1X2'``]
        #            *discard_frames*
        #               skip initial frames < *discard_frames* [``0``]
        #         """
        Operations.__init__(self, protein, **kwargs)

        self.residues = residues
        logging.basicConfig(filename=kwargs.get('log_file', 'log'),level=logging.INFO)

        # fourier broadening parameters
        self.ra = -5
        self.re = 20
        self.nr = 501
        sig = 0.1
        self.rax = np.linspace(self.ra, self.re, self.nr)
        self.tax = np.linspace(0.01, 5.5, 512)
        self.vari = np.exp(-(self.rax/sig)**2)
        if len(residues) != 2:
            raise ValueError("The residue_list must contain exactly 2 "
                             "residue numbers: current value {0}.".format(residues))

    def trajectoryAnalysis(self):
        logging.info("Starting rotamer distance analysis of trajectory {:s} with labeled residues "
                     "{:d} and {:d}".format(self.protein.trajectory.filename, self.residues[0], self.residues[1]))
        f = h5py.File(self.output_prefix+'-{:d}-{:d}.hdf5'.format(self.residues[0], self.residues[1]), "w")
        distributions = f.create_dataset("distributions",
                (self.protein.trajectory.n_frames, self.rax.size), fillvalue=0, compression="gzip")
        lib_weights_norm = self.lib.weights / np.sum(self.lib.weights)
        rotamer1, prot_atoms1, residue_sel1 = self.precalculate_rotamer(self.residues[0], self.chains[0])
        rotamer2, prot_atoms2, residue_sel2 = self.precalculate_rotamer(self.residues[1], self.chains[1])
        # For each trajectory frame, place the probes at the spin-labeled site using rotamer_placement(), calculate
        # Boltzmann weights based on Lennard-Jones interactions and calculate weighted distributions of probe-probe separations
        zarray = np.empty(0) # Array of steric partition functions (sum over Boltzmann weights)
        for frame_ndx, _ in enumerate(self.protein.trajectory):
            # Fit the rotamers onto the protein
            rotamersSite1 = self.rotamer_placement(rotamer1, prot_atoms1)
            rotamersSite2 = self.rotamer_placement(rotamer2, prot_atoms2)
            # straight polyhach
            boltz1, z1 = self.rotamerWeights(rotamersSite1, lib_weights_norm, residue_sel1)
            boltz2, z2 = self.rotamerWeights(rotamersSite2, lib_weights_norm, residue_sel2)
            zarray = np.append(zarray, [z1,z2])
            if (z1 <= self.z_cutoff) or (z2 <= self.z_cutoff):
                 continue
            boltzmann_weights_norm1 = boltz1 / z1
            boltzmann_weights_norm2 = boltz2 / z2
            boltzmann_weights_norm =  boltzmann_weights_norm1.reshape(-1,1) * boltzmann_weights_norm2

            # define the atoms to measure the distances between
            rotamer1nitrogen = rotamersSite1.select_atoms("name N1")
            rotamer2nitrogen = rotamersSite2.select_atoms("name N1")
            rotamer1oxigen = rotamersSite1.select_atoms("name O1")
            rotamer2oxigen = rotamersSite2.select_atoms("name O1")

            nit1_pos = np.array([rotamer1nitrogen.positions for x in rotamersSite1.trajectory])
            nit2_pos = np.array([i for x in rotamersSite2.trajectory for i in rotamer2nitrogen.positions])
            oxi1_pos = np.array([rotamer1oxigen.positions for x in rotamersSite1.trajectory])
            oxi2_pos = np.array([i for x in rotamersSite2.trajectory for i in rotamer2oxigen.positions])
            nitro1_pos = (nit1_pos + oxi1_pos) / 2
            nitro2_pos = (nit2_pos + oxi2_pos) / 2
            nitro_nitro_vector = nitro1_pos - nitro2_pos #.reshape(-1,1,3)

            # Distances between nitroxide groups
            dists_array = np.linalg.norm(nitro_nitro_vector, axis=2) / 10
            dists_array = np.round((self.nr * (dists_array - self.ra)) / (self.re - self.ra)).astype(int).flatten()
            distribution = np.bincount(dists_array, weights=boltzmann_weights_norm.flatten(), minlength=self.rax.size) 
            distributions[frame_ndx] = distribution
        f.close()
        np.savetxt(self.output_prefix+'-Z-{:d}-{:d}.dat'.format(self.residues[0], self.residues[1]),zarray.reshape(-1,2))

    def save(self,filename):
        f = h5py.File(filename, "r")
        distributions = f.get('distributions')
        if isinstance(self.weights, np.ndarray):
            if self.weights.size != distributions.shape[0]:
                    logging.info('Weights array has size {} whereas the number of frames is {}'.
                            format(self.weights.size, distributions.shape[0]))
                    raise ValueError('Weights array has size {} whereas the number of frames is {}'.
                            format(self.weights.size, distributions.shape[0]))
        elif self.weights == False:
            self.weights = np.ones(distributions.shape[0])
        else:
            logging.info('Weights argument should be a numpy array')
            raise ValueError('Weights argument should be a numpy array')
        distribution = np.nansum(distributions*self.weights.reshape(-1,1), 0)
        frame_inv_distr = np.fft.ifft(distribution) * np.fft.ifft(self.vari)
        smoothed = np.real(np.fft.fft(frame_inv_distr))
        smoothed /= np.trapz(smoothed, self.rax)
        np.savetxt(self.output_prefix + '-{:d}-{:d}.dat'.format(self.residues[0], self.residues[1]),
                np.c_[self.rax[100:401], smoothed[200:]],
                   header='distance distribution')
        np.savetxt(self.output_prefix + '-dist-{:d}-{:d}.dat'.format(self.residues[0], self.residues[1]),
                np.c_[self.rax, distribution],
                   header='distance distribution')
        time_domain_smoothed = self.calcTimeDomain(self.tax, self.rax[100:401], smoothed[200:])
        np.savetxt(self.output_prefix + '-{:d}-{:d}_time-domain.dat'.format(self.residues[0], self.residues[1]),
                   np.c_[self.tax, time_domain_smoothed],
                   header='time d(t)')
        f.close()

    def run(self, **kwargs):
        # Output
        self.output_prefix = kwargs.get('output_prefix', 'res')
        # Input
        self.load_file = kwargs.get('load_file', False)
        # Weights for each frame
        self.weights = kwargs.get('weights', False)
        if self.load_file:
            if os.path.isfile(self.load_file):
                logging.info('Loading pre-computed data from {} - will not load trajectory file.'.format(self.load_file))
            else:
                logging.info('File {} not found!'.format(self.load_file))
                raise FileNotFoundError('File {} not found!'.format(self.load_file))
            self.save(self.load_file)    
        else:
            self.trajectoryAnalysis()
            self.save(self.output_prefix+'-{:d}-{:d}.hdf5'.format(self.residues[0], self.residues[1]))
