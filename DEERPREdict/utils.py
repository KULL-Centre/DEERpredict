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
from DEERPREdict.lennardjones import vdw, p_Rmin2, eps
import DEERPREdict.libraries as libraries
import logging
import scipy.special as special
from scipy.spatial.distance import cdist

class Operations(object):
    """Calculation of the distance profile between a probe and backbone amide."""

    def __init__(self, protein, **kwargs):
        """
        Args:
            protein (:py:class:`MDAnalysis.core.universe.Universe`): trajectory
        """
        self.protein = protein
        self.libname = kwargs.get('libname', 'MTSSL 175K X1X2')
        self.lib = libraries.RotamerLibrary(self.libname)
        self.temp = kwargs.get('temperature', 300)
        self.z_cutoff = kwargs.get('z_cutoff', 0.05)
        # scaling factor to reduce probe-protein steric clashes 
        sigma_scaling = kwargs.get('sigma_scaling', 0.5)
        self.rmin2 = {atom:p_Rmin2[atom]*sigma_scaling for atom in p_Rmin2}
        self.ign_H = kwargs.get('ign_H', True)
        self.chains = kwargs.get('chains', [None,None])

    def precalculate_rotamer(self, residue, chain):
        residue_sel = "resid {:d}".format(residue)
        if type(chain) == str:
            residue_sel += " and segid {:s}".format(chain)
        prot_Ca = self.protein.select_atoms('protein and name CA and '+residue_sel)
        prot_Co = self.protein.select_atoms('protein and name C and '+residue_sel)
        prot_N = self.protein.select_atoms('protein and name N and '+residue_sel)
        probe_coords = np.zeros((len(self.lib.top.atoms),1, 3))
        universe = MDAnalysis.Universe(self.lib.top.filename, probe_coords, format=MemoryReader, order='afc')

        # Atom indices and parameters for LJ calculation
        if self.ign_H:
            proteinNotSite = self.protein.select_atoms("protein and not type H and not ("+residue_sel+")")
            rotamerSel_LJ = universe.select_atoms("not type H and not (name CA or name C or name N or name O)")
        else:
            proteinNotSite = self.protein.select_atoms("protein and not ("+residue_sel+")")
            rotamerSel_LJ = universe.select_atoms("not (name CA or name C or name N or name O)")
            
        eps_rotamer = np.array([eps[probe_atom] for probe_atom in rotamerSel_LJ.types])
        rmin2_rotamer = np.array([self.rmin2[probe_atom] for probe_atom in rotamerSel_LJ.types])

        eps_protein = np.array([eps[probe_atom] for probe_atom in proteinNotSite.types])
        rmin2_protein = np.array([self.rmin2[probe_atom] for probe_atom in proteinNotSite.types])
        eps_ij = np.sqrt(np.multiply.outer(eps_rotamer, eps_protein))
        rmin_ij = np.add.outer(rmin2_rotamer, rmin2_protein)
        LJ_data = [proteinNotSite.indices,rotamerSel_LJ.indices,eps_ij,rmin_ij]
 
        return universe, (prot_Ca, prot_Co, prot_N), LJ_data 
        
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
        rotation = np.vstack([x_vector, y_vector, z_vector])
        probe_coords = np.tensordot(self.lib.coord,rotation,axes=([2],[0])) + offset
        universe.load_new(probe_coords, format=MemoryReader, order='afc')
        #save aligned rotamers
        #mtssl = universe.select_atoms("all")
        #with MDAnalysis.Writer(self.output_prefix + "mtssl{:d}.pdb".format(np.random.randint(0,10)), mtssl.n_atoms) as W:
        #    for ts in universe.trajectory:
        #        W.write(mtssl)
        return universe

    def lj_calculation(self, fitted_rotamers, LJ_data):
        gas_un = 1.9858775e-3 # CHARMM, in kcal/mol*K
        proteinNotSite, rotamerSel_LJ, eps_ij, rmin_ij = LJ_data
        #Convert indices of protein atoms (constant within each frame) to positions
        proteinNotSite = self.protein.trajectory.ts.positions[proteinNotSite]
        lj_energy_pose = np.zeros(len(fitted_rotamers.trajectory))
        for rotamer_counter, rotamer in enumerate(fitted_rotamers.trajectory):
            d = MDAnalysis.lib.distances.distance_array(rotamer.positions[rotamerSel_LJ],proteinNotSite)
            cutoff = d<10
            d = np.power(rmin_ij[cutoff]/d[cutoff],6)
            pair_LJ_energy = eps_ij[cutoff]*(d*d-2.*d)
            lj_energy_pose[rotamer_counter] = pair_LJ_energy.sum()
        return np.exp(-lj_energy_pose/(gas_un*self.temp))
        # Slower implementation without for loop
        #rot_coords = fitted_rotamers.trajectory.timeseries(rotamerSel_LJ)
        #d = MDAnalysis.lib.distances.distance_array(rot_coords.reshape(-1,3),proteinNotSite).reshape(rot_coords.shape[0],rot_coords.shape[1],proteinNotSite.shape[0])
        #d = np.power(rmin_ij[:,np.newaxis,:]/d,6)
        #LJ_energy = (eps_ij[:,np.newaxis,:]*(d*d-2.*d)).sum(axis=(0,2))
        #return np.exp(-LJ_energy/(gas_un*self.temp))

    def rotamerWeights(self, rotamersSite, LJ_data):
        # Calculate Boltzmann weights
        boltz = self.lj_calculation(rotamersSite, LJ_data)
        # Set to zero Boltzmann weights that are NaN
        boltz[np.isnan(boltz)] = 0.0
        # Multiply Boltzmann weights by library weights
        boltz = self.lib.weights * boltz
        steric_z = np.sum(boltz)
        return boltz/steric_z, steric_z

    def rotamerPREanalysis(self, rotamersSite, boltzmann_weights):
        # Select atoms for distance calculations
        rotamer_nitrogen = rotamersSite.select_atoms("name N1")
        rotamer_oxigen = rotamersSite.select_atoms("name O1")
        # Position of the nitroxide group
        nit_pos = np.array([rotamer_nitrogen.positions for x in rotamersSite.trajectory])
        oxi_pos = np.array([rotamer_oxigen.positions for x in rotamersSite.trajectory])
        nitro_pos = (nit_pos + oxi_pos) / 2
        # Positions of the backbone nitrogen atoms
        amide_pos = self.protein.select_atoms(self.measured_sel).positions
        # Distance vectors between the rotamer nitroxide position and the nitrogen position in the other residues
        n_probe_vector = nitro_pos - amide_pos
        for d,L in enumerate(self.protein.dimensions[:3]):
            n_probe_vector[:,:,d] = np.where(n_probe_vector[:,:,d] > 0.5 * L, n_probe_vector[:,:,d] - L, n_probe_vector[:,:,d])
            n_probe_vector[:,:,d] = np.where(n_probe_vector[:,:,d] < - 0.5 * L, n_probe_vector[:,:,d] + L, n_probe_vector[:,:,d])
        # Distances between nitroxide and amide groups
        dists_array_r = np.linalg.norm(n_probe_vector,axis=2)
        #dists_array_r = mda_dist.distance_array(np.squeeze(nitro_pos),amide_pos,backend='OpenMP')
        # Ratio between distance vectors and distances 
        n_probe_unitvector = n_probe_vector/dists_array_r[:,:,None]
        # Dot products between nitroxide-amide distances for all rotamers
        cosine = np.einsum('ijk,ljk->ilj', n_probe_unitvector, n_probe_unitvector)
        # Second-order Legendre polynomial 
        legendre = 1.5 * cosine**2 - 0.5
        # Weighted average of the squared angular component of the order parameter over all rotamers
        angular = np.einsum('ijk,i,j->k', legendre, boltzmann_weights, boltzmann_weights)
        # Weighted averages of the interaction distance over all rotamers
        r3 = np.dot(boltzmann_weights, np.power(dists_array_r,-3))
        r6 = np.dot(boltzmann_weights, np.power(dists_array_r,-6))
        return r3, r6, angular

    @staticmethod
    def calc_gamma_2(dist_r6, s_pre, tau_c, tau_t, wh, k):
        wh = 2*np.pi*1e6*wh
        j = lambda w : s_pre*tau_c / (1+(w*tau_c)**2) + (1-s_pre)*tau_t / (1+(w*tau_t)**2)
        return k*dist_r6*(4*j(0) + 3*j(wh))

    @staticmethod
    def calc_gamma_2_Cbeta(dist_r6, tau_c, wh, k):
        wh = 2*np.pi*1e6*wh
        j = lambda w : tau_c / (1+(w*tau_c)**2)
        return k*dist_r6*(4*j(0) + 3*j(wh))

    @staticmethod
    def calcTimeDomain(t, r, p):
        r[0] = 1e-20 if r[0] == 0 else r[0]
        p /= np.trapz(p,r)
        mu_0 = 1.2566370614e-6  # {SI} T m A^-1
        mu_B = 9.27400968e-24   # Bohr magneton {SI} J T^-1
        g = 2.00231930436256    # unitless electron g-factor
        hbar = 1.054571800e-34  # {SI} J s
        # dipolar frequency
        wd = mu_0 * np.power(mu_B, 2) * np.power(g, 2) * 0.25 / np.pi / hbar / np.power(r, 3) * 1e27 # convert nm/s to m/s
        kappa = np.sqrt(6 * wd * t.reshape(-1,1) * 1e-6 / np.pi)
        fsin, fcos = special.fresnel(kappa)
        G = ( fcos * np.cos(wd*t.reshape(-1,1)*1e-6) + fsin * np.sin(wd*t.reshape(-1,1)*1e-6) ) / kappa
        S = np.trapz(G*p,x=r,axis=1)
        return S
