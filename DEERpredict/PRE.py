# -*- coding: utf-8 -*-
"""
PREPrediction Class
-------------------

Class to perform Paramagnetic Relaxation Enhancement prediction, employing the Model-Free Solomon-Bloembergen equation.

"""

import os
import numpy as np
import MDAnalysis
from MDAnalysis.coordinates.memory import MemoryReader
import pandas as pd
import logging
from DEERpredict.utils import Operations

class PREpredict(Operations):
    """Calculation of the distance profile between a probe and backbone amide."""

    def __init__(self, protein, residue, **kwargs):
        """
        Args:
            protein (:py:class:`MDAnalysis.core.universe.Universe`): trajectory
            residue (int): residue labeled with the paramagnetic probe
        :Keywords:
            tau_c (float):
            tau_t (float):
        """
        Operations.__init__(self, protein, **kwargs)
        self.residue = residue
        #  Class specific instance attributes
        logging.basicConfig(filename=kwargs.get('log_file', 'log'),level=logging.INFO)
        self.tau_c = kwargs.get('tau_c', 1.0e-9)
        self.tau_t = kwargs.get('tau_t', 5.0e-10)  
        self.wh = 2*np.pi*1e6*kwargs.get('wh', 700.0)
        self.k = kwargs.get('k', 1.23e16)
        self.t = kwargs.get('delay', 10.0e-3)
        # Approximate electron position at Cbeta
        self.Cbeta = kwargs.get('Cbeta', False)
        self.atom_selection = kwargs.get('atom_selection', 'N')
        self.resnums = np.array(protein.select_atoms('name N and protein').resnums)
        self.measured_sel = 'name {:s} and not resid {:d} and not resid 1 and not resname PRO'.format(self.atom_selection, residue)
        if type(self.chains[0]) == str:
            self.measured_sel = 'name {:s} and not (resid {:d} and segid {:s}) and not resid 1 and not resname PRO'.format(self.atom_selection, residue, self.chains[0])
        if type(self.chains[1]) == str:
            self.measured_sel += ' and segid {:s}'.format(self.chains[1])
        self.measured_resnums = np.array(protein.select_atoms(self.measured_sel).resnums)
        # Diamagnetic transverse relaxation rate
        self.r_2 = np.full(self.resnums.size, fill_value=np.NaN)
        self.r_2[self.measured_resnums - 1] = kwargs.get('r_2', 10.0)

    def trajectoryAnalysis(self):
        logging.info("Starting rotamer distance analysis of trajectory {:s} "
                     "with labeled residue {:d}".format(self.protein.trajectory.filename,self.residue))
        # Create arrays to store per-frame inverse distances, angular order parameter, and relaxation rate
        r3 = np.full((self.protein.trajectory.n_frames, self.measured_resnums.size), np.nan)
        r6 = np.full(r3.shape, np.nan)
        angular = np.full(r3.shape, np.nan)
        # Pre-process rotamer weights
        lib_weights_norm = self.lib.weights / np.sum(self.lib.weights)
        # Before getting into this loop, which consumes most of the calculations time
        # we can pre-calculate several objects that do not vary along the loop
        universe, prot_atoms, residue_sel = self.precalculate_rotamer(self.residue, self.chains[0])
        for frame_ndx, _ in enumerate(self.protein.trajectory):
            # Fit the rotamers onto the protein
            rotamersSite = self.rotamer_placement(universe, prot_atoms)
            # Calculate Boltzmann weights
            boltz, z = self.rotamerWeights(rotamersSite, lib_weights_norm, residue_sel)
            # Skip this frame if the sum of the Boltzmann weights is smaller than the cutoff value
            if z <= self.z_cutoff:
                # Store the radius of gyration of tight frames
                continue
            boltzmann_weights_norm = boltz / z
            # Calculate interaction distances and squared angular components of the order parameter
            r3[frame_ndx], r6[frame_ndx], angular[frame_ndx] = self.rotamerPREanalysis(rotamersSite, boltzmann_weights_norm)
        # Saving analysis as a pickle file
        data = pd.Series({'r3':r3.astype(np.float32), 'r6':r6.astype(np.float32), 'angular':angular.astype(np.float32)})
        data.to_pickle(self.output_prefix+'-{:d}.pkl'.format(self.residue),compression='gzip')
        # logging.info('Calculated distances and order parameters are saved to {}.'.format(self.output_prefix+'-{:d}.pkl'.format(residue)))
        return data

    def trajectoryAnalysisCbeta(self):
        # Create arrays to store per-frame inverse distances, angular order parameter, and relaxation rate
        r3 = np.full((self.protein.trajectory.n_frames, self.measured_resnums.size), np.nan)
        r6 = np.full(r3.shape, np.nan)
        angular = np.full(r3.shape, np.nan)
        residue_sel = "resid {:d}".format(self.residue)
        if type(self.chains[0]) == str:
            residue_sel += " and segid {:s}".format(self.chains[0])
        for frame_ndx, _ in enumerate(self.protein.trajectory):
            # Positions of the Cbeta atom of the spin-labeled residue
            spin_labeled_Cbeta = self.protein.select_atoms("protein and name CB and "+residue_sel).positions
            # Positions of the backbone nitrogen atoms
            amide_pos = self.protein.select_atoms(self.measured_sel).positions
            # Distances between nitroxide and amide groups
            dists_array_r = np.linalg.norm(spin_labeled_Cbeta - amide_pos,axis=1)
            r6[frame_ndx] = np.power(dists_array_r,-6)
        #dists_array_r = mda_dist.distance_array(spin_labeled_Cbeta,amide_nit_pos,backend='OpenMP')
        # Saving analysis as a pickle file
        data = pd.Series({'r3':r3.astype(np.float32), 'r6':r6.astype(np.float32), 'angular':angular.astype(np.float32)})
        data.to_pickle(self.output_prefix+'-{:d}.pkl'.format(self.residue),compression='gzip')
        # logging.info('Calculated distances and order parameters are saved to {}.'.format(self.output_prefix+'-{:d}.pkl'.format(residue)))
        return data

    def save(self, data):
        if isinstance(self.weights, np.ndarray):
            if self.weights.size != data['r6'].shape[0]:
                    logging.info('Weights array has size {} whereas the number of frames is {}'.
                            format(self.weights.size, data['r6'].shape[0]))
                    raise ValueError('Weights array has size {} whereas the number of frames is {}'.
                            format(self.weights.size, data['r6'].shape[0]))
        elif self.weights == False:
            self.weights = np.ones(data['r6'].shape[0])
        else:
            logging.info('Weights argument should be a numpy array')
            raise ValueError('Weights argument should be a numpy array')
        # Weighted averages of r^-6
        r6_av = np.ma.MaskedArray(data['r6'], mask=np.isnan(data['r6']))
        r6_av = np.ma.average(r6_av, weights=self.weights, axis=0).data
        # Transverse relaxation rate enhancement due to the presence of the unpaired electron
        gamma_2 = np.full(self.resnums.size, fill_value=np.NaN)
        if (self.Cbeta):
            gamma_2[self.measured_resnums - 1] = self.calc_gamma_2_Cbeta(r6_av, self.tau_c, self.wh, self.k)
        else:
            # Weighted averages of r^-3
            r3_av = np.ma.MaskedArray(data['r3'], mask=np.isnan(data['r3']))
            r3_av = np.ma.average(r3_av,  weights=self.weights, axis=0).data
            # Weighted averages of the squared angular component of the order parameter
            angular_av = np.ma.MaskedArray(data['angular'], mask=np.isnan(data['angular']))
            angular_av = np.ma.average(angular_av, weights=self.weights, axis=0).data
            gamma_2[self.measured_resnums - 1] = self.calc_gamma_2(r6_av, r3_av, self.tau_c, self.tau_t, self.wh, self.k, angular_av)
        # Paramagnetic / diamagnetic intensity ratio
        i_ratio = self.r_2 * np.exp(-gamma_2 * self.t) / ( self.r_2 + gamma_2 )
        np.savetxt(self.output_prefix+'-{}.dat'.format(self.residue),np.c_[self.resnums,i_ratio,gamma_2],header='residue i_ratio gamma_2')

    def run(self):
        if self.load_file:
            if os.path.isfile(self.load_file):
                logging.info('Loading pre-computed data from {} - will not load trajectory file.'.format(self.load_file))
            else:
                logging.info('File {} not found!'.format(self.load_file))
                raise FileNotFoundError('File {} not found!'.format(self.load_file))
            data = pd.read_pickle(self.load_file,compression='gzip')
        else:
            if self.Cbeta:
                data = self.trajectoryAnalysisCbeta()
            else:
                data = self.trajectoryAnalysis()
        self.save(data)
