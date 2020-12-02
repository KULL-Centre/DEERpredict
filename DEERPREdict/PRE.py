# -*- coding: utf-8 -*-
"""
PRE prediction Class
--------------------

Class to perform Paramagnetic Relaxation Enhancement prediction, employing the Model-Free Solomon-Bloembergen equation.

"""

import os
import numpy as np
import MDAnalysis
from MDAnalysis.coordinates.memory import MemoryReader
import pandas as pd
import logging
from DEERPREdict.utils import Operations

class PREpredict(Operations):
    """Calculation of the distance profile between a probe and backbone amide."""

    def __init__(self, protein, residue, **kwargs):
        """
        Args:
            protein (:py:class:`MDAnalysis.core.universe.Universe`): trajectory
            residue (int): residue labeled with the paramagnetic probe
        """
        Operations.__init__(self, protein, **kwargs)
        self.residue = residue
        #  Class specific instance attributes
        logging.basicConfig(filename=kwargs.get('log_file', 'log'),level=logging.INFO)

        residue_sel = "resid {:d}".format(self.residue)
        if type(self.chains[0]) == str:
            residue_sel += " and segid {:s}".format(self.chains[0])
        logging.info('{:s} = {:s}'.format(residue_sel,self.protein.select_atoms(residue_sel).atoms.resnames[0]))

        # Approximate electron position at Cbeta
        self.Cbeta = kwargs.get('Cbeta', False)
        self.atom_selection = kwargs.get('atom_selection', 'N')
        self.resnums = np.array(protein.select_atoms('name N and protein').resnums)
        self.measured_sel = 'name {:s} and not resid {:d} and not resid 1 and not resname PRO'.format(self.atom_selection, residue)
        if type(self.chains[0]) == str:
            self.measured_sel = 'name {:s} and not (resid {:d} and segid {:s}) and not resid 1 and not resname PRO'.format(self.atom_selection, residue, self.chains[0])
        if type(self.chains[1]) == str:
            self.resnums = np.array(protein.select_atoms('name N and protein and segid {:s}'.format(self.chains[1])).resnums)
            self.measured_sel += ' and segid {:s}'.format(self.chains[1])
        self.measured_resnums = np.array(protein.select_atoms(self.measured_sel).resnums)
        _, self.measured_resnums, _ = np.intersect1d(self.resnums,self.measured_resnums,return_indices=True)

    def trajectoryAnalysis(self):
        logging.info("Starting rotamer distance analysis of trajectory {:s} "
                     "with labeled residue {:d}".format(self.protein.trajectory.filename,self.residue))
        # Create arrays to store per-frame inverse distances, angular order parameter, and relaxation rate
        r3 = np.full((self.protein.trajectory.n_frames, self.measured_resnums.size), np.nan)
        r6 = np.full(r3.shape, np.nan)
        angular = np.full(r3.shape, np.nan)
        zarray = np.empty(0) # Array of steric partition functions (sum over Boltzmann weights)
        # Before getting into this loop, which consumes most of the calculations time
        # we can pre-calculate several objects that do not vary along the loop
        universe, prot_atoms, LJ_data = self.precalculate_rotamer(self.residue, self.chains[0])
        for frame_ndx, _ in enumerate(self.protein.trajectory):
            # Fit the rotamers onto the protein
            rotamersSite = self.rotamer_placement(universe, prot_atoms)
            # Calculate Boltzmann weights
            boltz, z = self.rotamerWeights(rotamersSite, LJ_data)
            # Skip this frame if the sum of the Boltzmann weights is smaller than the cutoff value
            zarray = np.append(zarray,z)
            if z <= self.z_cutoff:
                # Store the radius of gyration of tight frames
                continue
            # Calculate interaction distances and squared angular components of the order parameter
            r3[frame_ndx], r6[frame_ndx], angular[frame_ndx] = self.rotamerPREanalysis(rotamersSite, boltz)
        # Saving analysis as a pickle file
        data = pd.Series({'r3':r3.astype(np.float32), 'r6':r6.astype(np.float32), 'angular':angular.astype(np.float32)})
        data.to_pickle(self.output_prefix+'-{:d}.pkl'.format(self.residue),compression='gzip')
        np.savetxt(self.output_prefix+'-Z-{:d}.dat'.format(self.residue),zarray)
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
        # Transverse relaxation rate enhancement due to the presence of the unpaired electron
        gamma_2_av = np.full(self.resnums.size, fill_value=np.NaN)
        if (self.Cbeta):
            gamma_2 = self.calc_gamma_2_Cbeta(data['r6'], self.tau_c, self.wh, self.k)
        else:
            s_pre = np.power(data['r3'], 2)/data['r6']*data['angular']
            gamma_2 = self.calc_gamma_2(data['r6'], s_pre, self.tau_c, self.tau_t, self.wh, self.k)
        # Weighted average of gamma_2 over the conformational ensemble
        gamma_2 = np.ma.MaskedArray(gamma_2, mask=np.isnan(gamma_2))
        gamma_2_av[self.measured_resnums] = np.ma.average(gamma_2, weights=self.weights, axis=0).data
        # Paramagnetic / diamagnetic intensity ratio
        i_ratio = self.r_2 * np.exp(-gamma_2_av * self.delay) / ( self.r_2 + gamma_2_av )
        np.savetxt(self.output_prefix+'-{}.dat'.format(self.residue),np.c_[self.resnums,i_ratio,gamma_2_av],header='residue i_ratio gamma_2')

    def run(self, **kwargs):
        self.tau_c = kwargs.get('tau_c', 1.0e-9) # rotational tumbling time
        self.tau_t = kwargs.get('tau_t', 5.0e-10) # internal correlation time 
        self.wh = kwargs.get('wh', 700.0) # proton Larmor frequency / (2 pi 1e6)
        self.k = kwargs.get('k', 1.23e16) 
        self.delay = kwargs.get('delay', 10.0e-3) # INEPT delay
        # Diamagnetic transverse relaxation rate
        self.r_2 = np.full(self.resnums.size, fill_value=np.NaN) # transverse relaxation rate
        self.r_2[self.measured_resnums] = kwargs.get('r_2', 10.0) # in the diamagnetic molecule
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
            data = pd.read_pickle(self.load_file,compression='gzip')
        else:
            if self.Cbeta:
                data = self.trajectoryAnalysisCbeta()
            else:
                data = self.trajectoryAnalysis()
        self.save(data)
