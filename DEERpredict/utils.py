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
from DEERpredict.lennardjones import vdw, p_Rmin2, eps
import DEERpredict.libraries as libraries
import logging
import scipy.special as special

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
        return universe, (prot_Ca, prot_Co, prot_N), residue_sel
        
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

    def lj_calculation(self, fitted_rotamers, residue_sel):
        gas_un = 1.9858775e-3 # CHARMM, in kcal/mol*K
        if self.ign_H:
            proteinNotSite = self.protein.select_atoms("protein and not type H and not "+residue_sel)
            rotamerSel_LJ = fitted_rotamers.select_atoms("not type H and not (name CA or name C or name N or name O)")
        else:
            proteinNotSite = self.protein.select_atoms("protein and not "+residue_sel)
            rotamerSel_LJ = fitted_rotamers.select_atoms("not (name CA or name C or name N or name O)")
            
        eps_rotamer = np.array([eps[probe_atom] for probe_atom in rotamerSel_LJ.types])
        rmin_rotamer = np.array([p_Rmin2[probe_atom] for probe_atom in rotamerSel_LJ.types])*0.5

        eps_protein = np.array([eps[probe_atom] for probe_atom in proteinNotSite.types])
        rmin_protein = np.array([p_Rmin2[probe_atom] for probe_atom in proteinNotSite.types])*0.5
        eps_ij = np.sqrt(np.multiply.outer(eps_rotamer, eps_protein))
        
        rmin_ij = np.add.outer(rmin_rotamer, rmin_protein)
        #Convert atom groups to indices for efficiecy
        rotamerSel_LJ = rotamerSel_LJ.indices
        proteinNotSite = proteinNotSite.indices
        #Convert indices of protein atoms (constant within each frame) to positions
        proteinNotSite = self.protein.trajectory.ts.positions[proteinNotSite]
        lj_energy_pose = np.zeros((len(fitted_rotamers.trajectory)))
        for rotamer_counter, rotamer in enumerate(fitted_rotamers.trajectory):
            d = MDAnalysis.lib.distances.distance_array(rotamer.positions[rotamerSel_LJ],proteinNotSite)
            d = np.power(rmin_ij/d,6)
            pair_LJ_energy = eps_ij*(d*d-2.*d)
            lj_energy_pose[rotamer_counter] = pair_LJ_energy.sum()
        return np.exp(-lj_energy_pose/(gas_un*self.temp))  # for new alignment method

    def rotamerWeights(self, rotamersSite, lib_weights_norm, residue_sel):
        # Calculate Boltzmann weights
        boltz = self.lj_calculation(rotamersSite, residue_sel)
        # Set to zero Boltzmann weights that are NaN
        boltz[np.isnan(boltz)] = 0.0

        # Multiply Boltzmann weights by user-supplied and library weights
        boltz = lib_weights_norm * boltz
        return boltz, np.nansum(boltz)

    def rotamerPREanalysis(self, rotamersSite, boltzmann_weights_norm):
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
        angular = np.einsum('ijk,i,j->k', legendre, boltzmann_weights_norm, boltzmann_weights_norm)
        # Weighted averages of the interaction distance over all rotamers
        r3 = np.dot(boltzmann_weights_norm, np.power(dists_array_r,-3))
        r6 = np.dot(boltzmann_weights_norm, np.power(dists_array_r,-6))
        return r3, r6, angular

    @staticmethod
    def calc_gamma_2(dist_r6, s_pre, tau_c, tau_t, wh, k):
        j = lambda w : s_pre*tau_c / (1+(w*tau_c)**2) + (1-s_pre)*tau_t / (1+(w*tau_t)**2)
        return k*dist_r6*(4*j(0) + 3*j(wh))

    @staticmethod
    def calc_gamma_2_Cbeta(dist_r6, tau_c, wh, k):
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
