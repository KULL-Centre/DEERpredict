# -*- coding: utf-8 -*-
"""
PREPrediction Class
-------------------

Class to perform Paramagnetic Relaxation Enhancement prediction, employing the Model-Free Solomon-Bloembergen equation.

"""

import os
import numpy as np
import MDAnalysis
import MDAnalysis.lib.distances as mda_dist
from MDAnalysis.coordinates.memory import MemoryReader
from DEERpredict.lennardjones import vdw, p_Rmin2, eps
import DEERpredict.libraries
import pandas as pd
import logging

class PREPrediction():
    """Calculation of the distance profile between a probe and backbone amide."""

    def __init__(self, protein, residue, **kwargs):
        """
        Args:
            protein (:py:class:`MDAnalysis.core.universe.Universe`): trajectory
            residue (int): residue labeled with the paramagnetic probe
        :Keywords:
            tau (float):
            tau_c (float):
        """
        self.libname = kwargs.get('libname', 'MTSSL 175K X1X2')
        self.output_prefix = kwargs.get('output_prefix', 'profile')
        self.lib = DEERpredict.libraries.RotamerLibrary(self.libname)
        self.temp = kwargs.get('temperature', 300)
        self.protein = protein
        self.residue = residue
        #  Class specific instance attributes
        logging.basicConfig(filename=kwargs.get('log_file', 'log'),level=logging.INFO)
        #logging.info("Rotamer library = '{0}'".format(self.lib.name))
        #logging.info('The number of frames is {}'.format(protein.trajectory.n_frames))
        self.tau_c = kwargs.get('tau_c', 1.0e-9)  
        self.tau_t = kwargs.get('tau_t', 5.0e-10)  
        self.wh = 1e6*np.pi*kwargs.get('wh', 700.0)
        self.k = kwargs.get('k', 1.23e16)
        self.t = kwargs.get('delay', 10.0e-3)
        self.load_file = kwargs.get('load_file', False)
        # Weights for each frame
        self.weights = kwargs.get('weights', False)

        self.atom_selection = kwargs.get('atom_selection', 'HN')
        self.resnums = protein.select_atoms('name {} and protein'.format(self.atom_selection)).resnums
        self.resnums = np.array(self.resnums)
        self.measured_resnums = protein.select_atoms(
                'name {} and not resid {} and not resid 1 and not resname PRO'.format(self.atom_selection,residue)).resnums
        self.measured_resnums = np.array(self.measured_resnums)
        # Diamagnetic transverse relaxation rate
        self.r_2 = np.full(self.resnums.size, fill_value=np.NaN)
        self.r_2[self.measured_resnums - 1] = kwargs.get('r_2', 10.0);
        #self.r_2[self.measured_resnums - 1] = np.exp(-np.abs(self.measured_resnums-self.resnums[:,np.newaxis])/5).sum(axis=0)
        # Backend settings for testing only
        self.z_cutoff = 0.2
        self.ign_H = True

    def pre_calculate_rotamer(self):
        prot_Ca = self.protein.select_atoms('protein and name CA and resid {0}'.format(self.residue))
        prot_Co = self.protein.select_atoms('protein and name C and resid {0}'.format(self.residue))
        prot_N = self.protein.select_atoms('protein and name N and resid {0}'.format(self.residue))
        probe_coords = np.zeros((len(self.lib.top.atoms),1, 3))
        new_universe = MDAnalysis.Universe(self.lib.top.filename, probe_coords, format=MemoryReader, order='afc')
        return new_universe, (prot_Ca, prot_Co, prot_N)
        
    def rotamer_placement(self, universe, prot_atoms):
        prot_Ca, prot_Co, prot_N = prot_atoms
        offset = prot_Ca.positions.copy()
        Ca_coords = prot_Ca.positions - offset
        Co_coords = prot_Co.positions - offset
        N_coords = prot_N.positions - offset
        x_vector = N_coords - Ca_coords
        x_vector /= np.linalg.norm(x_vector)
        yt_vector = Co_coords - Ca_coords
        yt_vector /= np.linalg.norm(yt_vector)
        z_vector = np.cross(x_vector, yt_vector)
        z_vector /= np.linalg.norm(z_vector)
        y_vector = np.cross(z_vector, x_vector)
        rotation = np.array((x_vector, y_vector, z_vector)).T
        probe_coords = self.lib.data[:, 2:5].copy().T
        probe_coords = np.dot(rotation, probe_coords).T
        probe_coords = probe_coords.reshape((self.lib.data.shape[0] // (len(self.lib.top.atoms)),
                                             len(self.lib.top.atoms), 3))
        probe_coords += offset
        probe_coords = probe_coords.swapaxes(0, 1)
        universe.load_new(probe_coords, format=MemoryReader, order='afc')
        return universe

    def lj_calculation(self, fitted_rotamers, forgive = 0.5):
        gas_un = 1.9858775e-3 # CHARMM, in kcal/mol*K
        if self.ign_H:
            proteinNotSite = self.protein.select_atoms("protein and not type H and not (resid {0})".format(self.residue))
            rotamerSel_LJ = fitted_rotamers.select_atoms("not type H and not (name CA or name C or name N or name O)")
        else:
            proteinNotSite = self.protein.select_atoms("protein and not (resid {0})".format(self.residue))
            rotamerSel_LJ = fitted_rotamers.select_atoms("not (name CA or name C or name N or name O)")
            
        eps_rotamer = np.array([eps[probe_atom] for probe_atom in rotamerSel_LJ.types])
        rmin_rotamer = np.array([p_Rmin2[probe_atom] for probe_atom in rotamerSel_LJ.types])*forgive

        eps_protein = np.array([eps[probe_atom] for probe_atom in proteinNotSite.types])
        rmin_protein = np.array([p_Rmin2[probe_atom] for probe_atom in proteinNotSite.types])*forgive
        eps_ij = np.sqrt(np.multiply.outer(eps_rotamer, eps_protein))
        
        rmin_ij = np.add.outer(rmin_rotamer, rmin_protein)
        #Convert atom groups to indices for efficiecy
        rotamerSel_LJ = rotamerSel_LJ.indices
        proteinNotSite = proteinNotSite.indices
        cuttoff =10.
        #Convert indices of protein atoms (constant within each frame) to positions
        proteinNotSite = self.protein.trajectory.ts.positions[proteinNotSite]
        lj_energy_pose = np.zeros((len(fitted_rotamers.trajectory)))
        for rotamer_counter, rotamer in enumerate(fitted_rotamers.trajectory):
            d = MDAnalysis.lib.distances.distance_array(rotamer.positions[rotamerSel_LJ], 
                proteinNotSite)
            d = np.power(rmin_ij/d,6)
            pair_LJ_energy = eps_ij*(d*d-2.*d)
            lj_energy_pose[rotamer_counter] = pair_LJ_energy.sum()
        return np.exp(-lj_energy_pose/(gas_un*self.temp))  # for new alignment method

    @staticmethod
    def calc_gamma_2(dist_r6, dist_r3, tau_c, tau_t, wh, k, s_ang):
        s_rad = np.power(dist_r3, 2)/dist_r6
        s_pre = s_ang*s_rad
        j = lambda w : s_pre*tau_c / (1+(w*tau_c)**2) + (1-s_pre)*tau_t / (1+(w*tau_t)**2)
        return k*dist_r6*(4*j(0) + 3*j(wh))

    def saveIratio(self, data):  
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
        # Weighted averages of r^-3
        r3_av = np.ma.MaskedArray(data['r3'], mask=np.isnan(data['r3']))
        r3_av = np.ma.average(r3_av,  weights=self.weights, axis=0).data
        # Weighted averages of the squared angular component of the order parameter
        angular_av = np.ma.MaskedArray(data['angular'], mask=np.isnan(data['angular']))
        angular_av = np.ma.average(angular_av, weights=self.weights, axis=0).data
        # Transverse relaxation rate due to the presence of the unpaired electron
        gamma_2 = np.full(self.resnums.size, fill_value=np.NaN)
        gamma_2[self.measured_resnums - 1] = self.calc_gamma_2(r6_av, r3_av, self.tau_c, self.tau_t, self.wh, self.k, angular_av)
        # Paramagnetic / diamagnetic intensity ratio
        i_ratio = self.r_2 * np.exp(-gamma_2 * self.t) / ( self.r_2 + gamma_2 )
        np.savetxt(self.output_prefix+'-{}.dat'.format(self.residue),np.c_[self.resnums,i_ratio,gamma_2],header='residue i_ratio gamma_2')

    def rotamerWeights(self, rotamersSite, lib_weights_norm):
        # Calculate Boltzmann weights
        boltz = self.lj_calculation(rotamersSite)
        # Set to zero Boltzmann weights that are NaN
        boltz[np.isnan(boltz)] = 0.0

        # Multiply Boltzmann weights by user-supplied and library weights
        boltz = lib_weights_norm * boltz
        return boltz, np.nansum(boltz)

    def rotamerAnalysis(self, rotamersSite, boltzmann_weights_norm):
        # Select atoms for distance calculations
        rotamer_nitrogen = rotamersSite.select_atoms("name N1")
        rotamer_oxigen = rotamersSite.select_atoms("name O1")
        # Position of the nitroxide group
        nit_pos = np.array([rotamer_nitrogen.positions for x in rotamersSite.trajectory])
        oxi_pos = np.array([rotamer_oxigen.positions for x in rotamersSite.trajectory])
        nitro_pos = (nit_pos + oxi_pos) / 2
        # Positions of the backbone nitrogen atoms
        amide_nit_pos = self.protein.select_atoms(
            "name {} and not resid {} and not resid 1 and not resname PRO".format(self.atom_selection, self.residue)).positions
        # Distance vectors between the rotamer nitroxide position and the nitrogen position in the other residues
        n_probe_vector = (nitro_pos - amide_nit_pos)
        # Distances between nitroxide and amide groups
        dists_array_r = mda_dist.distance_array(np.squeeze(nitro_pos),amide_nit_pos,backend='OpenMP')
        # Ratio between distance vectors and distances 
        n_probe_unitvector = n_probe_vector/dists_array_r[:,:,None]
        # Dot products between nitroxide-amide distances for all rotamers
        cosine = np.einsum('ijk,ljk->ilj', n_probe_unitvector, n_probe_unitvector)
        # Second-order Legendre polynomial 
        legendre = 1.5 * cosine**2 - 0.5
        # Weighted average of the squared angular component of the order parameter over all rotamers
        angular = np.einsum('ijk,i,j->k', legendre, boltzmann_weights_norm, boltzmann_weights_norm)
        # Weighted averages of the interaction distance over all rotamers
        r3 = np.dot(boltzmann_weights_norm, np.power(dists_array_r,-3))
        r6 = np.dot(boltzmann_weights_norm, np.power(dists_array_r,-6))
        return r3, r6, angular

    def trajectoryAnalysis(self):
        # logging.info("Starting rotamer distance analysis of trajectory {:s} with labeled residue {:d}".format(protein.trajectory.filename,residue))
        # Create arrays to store per-frame inverse distances, angular order parameter, and relaxation rate
        r = np.full((self.protein.trajectory.n_frames, self.measured_resnums.size), np.nan)
        r3 = np.full(r.shape, np.nan)
        r6 = np.full(r.shape, np.nan)
        s2 = np.full(r.shape, np.nan)
        angular = np.full(r.shape, np.nan)
        # Radius of gyration of frames discarded due to tight placement of the rotamers
        # tight_rg = np.empty(0)  
        # Pre-process rotamer weights
        lib_weights_norm = self.lib.weights / np.sum(self.lib.weights)
        # Before getting into this loop, which consumes most of the calculations time
        # we can pre-calculate several objects that do not vary along the loop
        universe, prot_atoms = self.pre_calculate_rotamer()
        for frame_ndx, timestep in enumerate(self.protein.trajectory):
            # Fit the rotamers onto the protein
            rotamersSite = self.rotamer_placement(universe, prot_atoms)
            # Calculate Boltzmann weights
            boltz, z = self.rotamerWeights(rotamersSite, lib_weights_norm)
            # Skip this frame if the sum of the Boltzmann weights is smaller than the cutoff value
            if z <= self.z_cutoff:
                # Store the radius of gyration of tight frames
                # tight_rg = np.append(tight_rg, protein.select_atoms('protein').radius_of_gyration())
                continue
            boltzmann_weights_norm = boltz / z
            # Calculate interaction distances and squared angular components of the order parameter
            r3[frame_ndx], r6[frame_ndx], angular[frame_ndx] = self.rotamerAnalysis(rotamersSite, boltzmann_weights_norm)
            #if frame_ndx % 100 == 0:
            #    logging.info('Frame index {:d} of trajectory {:s} with labeled residue {:d}'.format(frame_ndx,protein.trajectory.filename,residue))
        #if tight_rg.size:
        #   np.savetxt(self.output_prefix+'-{:d}_tight_rg.dat'.format(residue),tight_rg)
        #   logging.info(
        #        '{:d} frames have been discarded due to tight labelling position, i.e. {:.1f} % of the analyzed frames.'.format(
        #           tight_rg.size, float(tight_rg.size) / protein.trajectory.n_frames * 100))
        #   logging.info(
        #        'The average radius of gyration in the discarded frames is {:.2f} +/- {:.2f} nm.'.format(
        #           tight_rg.mean()/10., tight_rg.std()/10.))
        # Saving analysis as a pickle file
        data = pd.Series({'r3':r3.astype(np.float32), 'r6':r6.astype(np.float32), 'angular':angular.astype(np.float32)})
        data.to_pickle(self.output_prefix+'-{:d}.pkl'.format(self.residue),compression='gzip')
        # logging.info('Calculated distances and order parameters are saved to {}.'.format(self.output_prefix+'-{:d}.pkl'.format(residue)))
        return data
        
    def run(self):
        if self.load_file:
            if os.path.isfile(self.load_file):
                logging.info('Loading pre-computed data from {} - will not load trajectory file.'.format(self.load_file))
            else:
                logging.info('File {} not found!'.format(self.load_file))
                raise FileNotFoundError('File {} not found!'.format(self.load_file))
            data = pd.read_pickle(self.load_file,compression='gzip')
        else:
            data = self.trajectoryAnalysis()
        self.saveIratio(data)
