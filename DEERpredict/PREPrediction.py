# -*- coding: utf-8 -*-
"""
PREPrediction Class
-------------------

Class to perform Paramagnetic Relaxation Enhancement prediction, employing the Model-Free Solomon-Bloembergen equation.

"""

import sys
import os
import pickle
#import time
# Coordinates and arrays
import numpy as np
import MDAnalysis
import math
import MDAnalysis.lib.distances as mda_dist
from scipy.optimize import least_squares

# Logger
import logging

# Inner imports
from DEERpredict.utils import Operations

logger = logging.getLogger("MDAnalysis.app")

class PREPrediction2(Operations):
    """Calculation of the distance profile between a probe and backbone amide protons."""

    def __init__(self, protein_structure, residue, **kwargs):
        """

        Args:
            protein_structure (:py:class:`MDAnalysis.core.universe.Universe`):
            residues (list(:py:class:`str`)):
        :Keywords:
        """
        Operations.__init__(self, protein_structure, **kwargs)
        #  class specific instance attributes
        self.tau = kwargs.get('tau', 20.0e-9)
        self.tau_c = kwargs.get('tau_c', 4.0e-9)  # 4 ns, for ACBP
        self.tau_t = kwargs.get('tau_t', 1.0e-10)  #
        self.wh = kwargs.get('wh', 750.0)
        self.r2 = kwargs.get('r2', None)
        self.default_r2 = kwargs.get('default_r2', 2.5)
        self.t = kwargs.get('delay', 10.0e-3)
        self.selection = kwargs.get('selection', 'HN')
        self.k = 1.23e-32
        self.plotting_delta = kwargs.get('plotting_delta', 0)
        self.protein_selection = protein_structure.select_atoms('protein')
        # self.default_r2 = len(self.protein_selection.residues)*0.11
        # self.default_r2 = self.r2_def  #FIXME: test for ACBP
        self.delta_residue = self.protein_selection[0].resid - 1
        self.optimize = kwargs.get('optimize', False)
        self.intensities_file = kwargs.get('exp_intensities', None)
        self.fast = kwargs.get('fast', False)
        self.save_file = kwargs.get('save_file', False)
        self.load = kwargs.get('load', False)
        self.idp = kwargs.get('idp', False)

        # Backend settings for TESTING ONLY
        self.z_cutoff = 0.2
        self.e_cutoff = 0.0
        self.ign_H = True
        self.hard_spheres = False
        self.debug = False
        self.mc = False
        self.av = 'relax'  # 'dist' averages all r^-6 and s^-2
        # 'relax' averages the gamma^-2 per frame
        self.cb = True
        # UNITS
        mhz_to_rads = 2.0 * math.pi * 1e6
        self.wh = mhz_to_rads * self.wh
        # print self.wh
        print('Using new dev version')

        if self.save_file:
            self.optimize = False
            logger.info("Saving of analysis is requested together with optimization - this is not supported to prevent "
                        "performance problems with large enslembles. Please run the optimization later using the --load "
                        "flag to load this runs analysis data.")

        if self.chains:
            residue_number = protein_structure.select_atoms("name N and protein").resnums
            if self.selection == 'N':
                # measured_residues = protein_structure.select_atoms("protein and name {0} and not resid {1} and not resid 1 "
                #                                                    "and not resname PRO".format(self.selection,
                #                                                                                 residue)).resnums
                measured_residues = protein_structure.select_atoms(
                    "protein and name {0} and not resid {1} and not resid 1 "
                    "and not resname PRO".format(self.selection,
                                                 residue)).residues.ix
            else:
                # measured_residues = protein_structure.select_atoms("protein and name {0} and not resid {1}".format(self.selection,
                #                                                                                        residue)).resnums
                measured_residues = protein_structure.select_atoms(
                    "protein and name {0} and not resid {1} and not resid 1 "
                    "and not resname PRO".format(self.selection,
                                                 residue)).residues.ix
            distributions = np.full(residue_number.size, fill_value=np.NaN)
            distributions_r1 = np.zeros((self.replicas + 1, measured_residues.size))
            distributions_r3 = np.zeros((self.replicas + 1, measured_residues.size))
            distributions_r6 = np.zeros((self.replicas + 1, measured_residues.size))
            cosine = np.zeros((self.replicas + 1, measured_residues.size))
            s2_fast = np.zeros((self.replicas + 1, measured_residues.size))
            rax = np.linspace(1 + self.plotting_delta, len(residue_number) + self.plotting_delta,
                              num=len(residue_number))
        else:
            residue_number = protein_structure.select_atoms("name N and protein").resnums
            if self.selection == 'N':
                measured_residues = protein_structure.select_atoms("name {0} and not resid {1} and not resid 1 "
                                                                   "and not resname PRO".format(self.selection,
                                                                                                residue)).resnums
            else:
                measured_residues = protein_structure.select_atoms("name {0} and not resid {1}".format(self.selection,
                                                                                                       residue)).resnums
            distributions = np.full(residue_number.size, fill_value=np.NaN)
            distributions_r1 = np.zeros((self.replicas + 1, measured_residues.size))
            distributions_r3 = np.zeros((self.replicas + 1, measured_residues.size))
            distributions_r6 = np.zeros((self.replicas + 1, measured_residues.size))
            cosine = np.zeros((self.replicas + 1, measured_residues.size))
            s2_fast = np.zeros((self.replicas + 1, measured_residues.size))
            #print('debug this!:')
            #print(residue_number)
            rax = np.linspace(1 + self.plotting_delta, len(residue_number) + self.plotting_delta,
                              num=len(residue_number))
            #print(rax)
        r2_full = np.full(residue_number.size, fill_value=np.NaN, dtype={'names': ('index', 'value', 'error'),
                                                                         'formats': ('int', 'f8', 'f8')})
        r2_full['index'] = list(range(1, len(self.protein_selection.residues) + 1))

        try:
            with open(self.r2) as f:
                try:
                    r2_file = np.loadtxt(f, dtype={'names': ('index', 'value', 'error'),
                                                   'formats': ('int', 'f8', 'f8')})
                    r2_full['value'][r2_file['index'] - 1] = r2_file['value']
                    r2_full['error'][r2_file['index'] - 1] = r2_file['error']
                except IndexError:
                    with open(self.r2) as f:  # reopen f, first try clears first line
                        r2_file = np.genfromtxt(f, dtype={'names': ('index', 'value'),
                                                          'formats': ('int', 'f8')})
                        r2_full['value'][r2_file['index'] - 1] = r2_file['value']
                        r2_full[residue - 1] = (residue, np.NaN, np.NaN)
        except TypeError:
            logger.info("R2 file not provided, calculation will be performed with the default {0} R2 for all"
                        " residues in the protein".format(self.default_r2))
            r2_full['value'][measured_residues - 1] = self.default_r2
            r2_full[residue - 1] = (residue, np.NaN, np.NaN)
        try:
            with open(self.intensities_file) as exp_intensities:
                self.exp_intensities = np.loadtxt(exp_intensities)
                self.exp_intensities = [x[1] for x in self.exp_intensities]
                print(self.exp_intensities)
                # self.optimize = True #FIXME: commented this out to give more control


        except TypeError:
            logger.info("No experimental intensities file, will not perform optimization.")

        # put that here so trajectory is run through but distributions and rax are present
        if self.load:
            if os.path.isfile(self.load):
                logger.info('Loading pre-computed data from {} -  will not load trajectory file.'.format(self.load))
            else:
                logger.info('File {} not found! Please check ...'.format(self.load))
                sys.exit()

            with open(self.load, 'r') as inp:
                save_data = pickle.load(inp)
                # gamma_2_from_load = np.zeros((np.size(save_data['r6'], 0), np.size(save_data['r6'], 1)))
                # print np.size(self.gamma_2_load(save_data['r6'], self.tau_c, self.tau_t, self.wh, self.k, save_data['s2']),0)
                # gamma_2_from_load[range(np.size(save_data['r6'], 0)), range(np.size(save_data['r6'], 1))] = self.gamma_2_load(save_data['r6'], self.tau_c, self.tau_t, self.wh, self.k, save_data['s2'])

                # FIXME: this needs to be updated to proper averaging

                # sub-sampling test:
                sub_sample = 1
                n_frames = np.size(save_data['r6'], 0)
                start_frame = 0
                sample_frac = 1
                end_frame = int(start_frame + np.multiply(sample_frac, n_frames))
                # end_frame = 12000
                # plt.semilogy(save_data['r6'][:, 102])
                # plt.savefig('test.png')
                # plt.close('all')

                if end_frame >= n_frames:
                    end_frame = n_frames
                frames_sub_ndx = list(range(start_frame, end_frame, sub_sample))
                # print frames_sub_ndx
                if self.av == 'dist':
                    r6_av = np.nanmean(save_data['r6'][frames_sub_ndx, :], axis=0)
                    r3_av = np.nanmean(save_data['r3'][frames_sub_ndx, :], axis=0)
                    cosine_av = np.nanmean(save_data['cosine'][frames_sub_ndx, :], axis=0)
                    distributions[measured_residues - 1] = self.gamma_2(r6_av, r3_av, self.tau_c, self.tau_t, self.wh,
                                                                        self.k, cosine_av)

                elif self.av == 'relax':
                    distributions[measured_residues - 1] = np.nanmean(
                        self.gamma_2_load(save_data['r6'][frames_sub_ndx, :], self.tau_c, self.tau_t, self.wh, self.k,
                                          save_data['s2'][frames_sub_ndx, :]),
                        axis=0)



                    # distributions[measured_residues - 1] = np.nanmean(
                    #     self.gamma_2_load(save_data['r6'][frames_sub_ndx, :], self.tau_c, self.tau_t, self.wh, self.k, save_data['s2'][frames_sub_ndx, :]),
                    #     axis=0)
                    # distributions[measured_residues - 1] = mode(
                    #    self.gamma_2_load(save_data['r6'][frames_sub_ndx, :], self.tau_c, self.tau_t, self.wh, self.k, save_data['s2'][frames_sub_ndx, :]),
                    #    axis=0, nan_policy='propagate')


                    # if not self.optimize:
                # FIXME:this should probably be a function ...
                (i_distributions, i_distributions_max, i_distributions_min) = self.calc_i_ratio(self, distributions,
                                                                                                r2_full)
                # i_distributions = np.zeros(distributions.size)
                # i_distributions_max = np.zeros(distributions.size)
                # i_distributions_min = np.zeros(distributions.size)
                # i_distributions[r2_full['index']-1] = np.divide(
                #     (r2_full['value']*np.exp(-distributions[r2_full['index']-1]*self.t)),
                #     (r2_full['value']+distributions[r2_full['index']-1]))
                # i_distributions_max[r2_full['index'] - 1] = np.divide(
                #     ((r2_full['value'] + r2_full['error']) * np.exp(-distributions[r2_full['index'] - 1] * self.t)),
                #     ((r2_full['value'] + r2_full['error']) + distributions[r2_full['index'] - 1]))
                #
                # i_distributions_min[r2_full['index'] - 1] = np.divide(
                #     ((r2_full['value'] - r2_full['error']) * np.exp(-distributions[r2_full['index'] - 1] * self.t)),
                #     ((r2_full['value'] - r2_full['error']) + distributions[r2_full['index'] - 1]))
                # print i_distributions
                self.plot_and_save_pre(rax, i_distributions, i_distributions_max, i_distributions_min, residue)
                if not self.optimize:
                    sys.exit()
                    # print gamma_2_from_load
                    # sys.exit() #FIXME: here you might want to decide to just skip RMSF part of optimization and simply optimize the above for best tau_c

        if self.optimize and self.load and self.idp:  # try
            upper = 0.85
            lower = 0.15
            width_r = 20
            # do nan filtering and dynamic PRE range checking
            filter_ndx = []
            for resid in range(residue - width_r, residue + width_r):
                if resid in rax:  # FIXME: this is a crappy hack, only works for for aligned pdb / exp_file and relies on the fact taht rax-1 is the pytnon index
                    if self.exp_intensities[resid - 1] < upper and self.exp_intensities[resid - 1] > lower:
                        filter_ndx.append(resid - 1)

            for v, val in enumerate(self.exp_intensities[filter_ndx]):
                if np.isnan(val):
                    filter_ndx.remove(filter_ndx[v])

            for v, val in enumerate(r2_full['value'][filter_ndx]):
                if np.isnan(val):
                    filter_ndx.remove(filter_ndx[v])

            #
            # print 'Filtered ndx list:'
            # print filter_ndx
            # print 'Length: %i'%len(filter_ndx)
            # print self.exp_intensities[filter_ndx]
            # print distributions[filter_ndx]
            # print distributions
            # distributions[filter_ndx]
            # print rax
            # print len(rax)
            #
            # print measured_residues
            # need to backmap indices

            filter_ndx_backmap = []
            for res in filter_ndx:
                if res + 1 in measured_residues:
                    filter_ndx_backmap.append(np.where(measured_residues == res + 1))

            # print len(filter_ndx)
            # print len(filter_ndx_backmap)
            # distributions_load_r6[:, measured_residues - 1] = save_data['r6']
            # distributions_load_s2[:, measured_residues - 1] = save_data['s2']
            # print save_data['r6'][:,filter_ndx]
            if self.av == 'dist':
                print(self.residuals_tau_c_I(self.tau_c, r2_full['value'][filter_ndx], r6_av[filter_ndx_backmap],
                                             r3_av[filter_ndx_backmap], self.tau_t, self.wh, self.k,
                                             cosine_av[filter_ndx_backmap], self.t, self.exp_intensities[filter_ndx]))
            elif self.av == 'relax':
                print(self.residuals_tau_c_I_load(self.tau_c, r2_full['value'][filter_ndx],
                                                  save_data['r6'][:, filter_ndx_backmap],
                                                  save_data['s2'][:, filter_ndx_backmap], self.tau_t, self.wh, self.k,
                                                  self.t, self.exp_intensities[filter_ndx]))

            print('opti now')

            if self.av == 'dist':
                print('hello')
                plsq_slow = least_squares(self.residuals_tau_c_I, self.tau_c, args=(r2_full['value'][filter_ndx],
                                                                                    r6_av[filter_ndx_backmap],
                                                                                    r3_av[filter_ndx_backmap],
                                                                                    self.tau_t,
                                                                                    self.wh, self.k,
                                                                                    cosine_av[filter_ndx_backmap],
                                                                                    self.t,
                                                                                    self.exp_intensities[filter_ndx]),
                                          bounds=(1e-9, 50e-9), xtol=1e-10,
                                          gtol=1e-10, x_scale=1e-9)['x'][0]

            print('Tau_c: \t', plsq_slow)
            self.tau_c = plsq_slow

            if self.av == 'dist':
                print(self.tau_t)
                print(self.residuals_tau_t_I_global(self.tau_t, r2_full['value'][filter_ndx],
                                                    r6_av[filter_ndx_backmap],
                                                    r3_av[filter_ndx_backmap],
                                                    self.tau_c, self.wh, self.k,
                                                    cosine_av[filter_ndx_backmap], self.t,
                                                    self.exp_intensities[filter_ndx]))
                self.tau_t = \
                    least_squares(self.residuals_tau_t_I_global, self.tau_t, args=(r2_full['value'][filter_ndx],
                                                                                   r6_av[filter_ndx_backmap],
                                                                                   r3_av[filter_ndx_backmap],
                                                                                   self.tau_c, self.wh, self.k,
                                                                                   cosine_av[filter_ndx_backmap],
                                                                                   self.t,
                                                                                   self.exp_intensities[filter_ndx]),
                                  bounds=(1e-12, self.tau_c))['x'][0]

            print('Tau_t: \t', self.tau_t)
            # plot again
            distributions[measured_residues - 1] = np.nanmean(
                self.gamma_2_load(save_data['r6'], self.tau_c, self.tau_t, self.wh, self.k, save_data['s2']), axis=0)
            # distributions = np.full((len(protein_structure.trajectory), residue_number.size), fill_value=np.NaN)
            # distributions[:, measured_residues - 1] = self.gamma_2_load(save_data['r6'], self.tau_c, self.tau_t, self.wh, self.k, save_data['s2'])
            (i_distributions, i_distributions_max, i_distributions_min) = self.calc_i_ratio(self, distributions,
                                                                                            r2_full)
            # i_distributions_av = np.nanmean(i_distributions, axis=0)
            # i_distributions_max_av = np.nanmean(i_distributions_max, axis=0)
            # i_distributions_min_av = np.nanmean(i_distributions_min, axis=0)
            # print i_distributions_av
            self.plot_and_save_pre(rax, i_distributions, i_distributions_max, i_distributions_min, residue)
            print(self.tau_c)
            print('DONE: Ciao for now!')
            sys.exit()

        if self.optimize and not self.idp:
            # RMSF calculation portion
            nan_filter = np.isnan(r2_full['value'])
            ca = protein_structure.select_atoms("name CA")
            nh = protein_structure.select_atoms("name N")  # change to H, but changes the behavior of first residue
            means = np.zeros((len(ca), 3))
            sumsq = np.zeros_like(means)
            traj_coordinates = ca.positions.copy()
            ref_coordinates = ca.positions.copy()
            nh_coordinates = nh.positions.copy()
            nh_ref_coordinates = nh.positions.copy()
            nh_translated_holder = np.zeros((protein_structure.trajectory.n_frames - self.discard_frames, len(nh), 3))
            for k, ts in enumerate(protein_structure.trajectory[self.discard_frames:]):
                x_com = ca.center_of_mass()
                traj_coordinates[:] = ca.positions - x_com
                nh_coordinates[:] = nh.positions - x_com
                rotation_matrix = MDAnalysis.analysis.align.rotation_matrix(traj_coordinates, ref_coordinates)[0]
                nh_rotation_matrix = MDAnalysis.analysis.align.rotation_matrix(nh_coordinates, nh_ref_coordinates)[0]
                traj_coordinates[:] = np.transpose(np.dot(rotation_matrix, np.transpose(traj_coordinates[:])))
                nh_coordinates[:] = np.transpose(np.dot(nh_rotation_matrix, np.transpose(nh_coordinates[:])))
                # print nh_translated_holder[k].shape
                nh_translated_holder[k] = nh_coordinates
                sumsq += (k / (k + 1.0)) * (traj_coordinates - means) ** 2
                means[:] = (k * means + traj_coordinates) / (k + 1.0)

            rmsf = np.sqrt(sumsq.sum(axis=1) / (k + 1.0))
            fraction = int(len(rmsf) / 5)
            sorted_rmsf = np.argsort(rmsf)
            rmsf_core = sorted_rmsf[:fraction * 4]
            rmsf_move = sorted_rmsf[-fraction:]
            rmsf_medium = sorted_rmsf[fraction:-fraction]

            # rmsf_nanfiltered = rmsf[~nan_filter]
            nan_filter = nan_filter[sorted_rmsf]
            sorted_rmsf_nanfiltered = sorted_rmsf[~nan_filter]
            fraction_nanfiltered = int(len(sorted_rmsf_nanfiltered) / 5)
            rmsf_core_nanfiltered = sorted_rmsf_nanfiltered[:fraction_nanfiltered * 3]
            rmsf_move_nanfiltered = sorted_rmsf_nanfiltered[-fraction:]
            rmsf_medium_nanfiltered = sorted_rmsf_nanfiltered[fraction:-fraction]

        logger.info("Starting rotamer distance analysis of trajectory "
                    "{0}...".format(protein_structure.trajectory.filename))
        logger.info("Rotamer library = '{0}'".format(self.lib.name))

        # TODO: Implement the replica averaging code for the ProbeDistanceProfile calculation. (Half done!)

        self.frames_pre_replica = int((protein_structure.trajectory.n_frames - self.discard_frames) / self.replicas)
        progressmeter = MDAnalysis.lib.log.ProgressMeter(protein_structure.trajectory.n_frames - self.discard_frames,
                                                         interval=1)

        # Pre-processing weights
        lib_norm = self.lib.weights / np.sum(self.lib.weights)

        # adding arrays to store resid, r^-6 and S-pre in per frame
        # FIXME: not working properly when frames are discarded! Fixed now?
        r1_store = np.full((len(protein_structure.trajectory), len(measured_residues)), np.nan)
        r6_store = np.full((len(protein_structure.trajectory), len(measured_residues)), np.nan)
        s2_store = np.full((len(protein_structure.trajectory), len(measured_residues)), np.nan)
        r3_store = np.full((len(protein_structure.trajectory), len(measured_residues)), np.nan)
        cosine_store = np.full((len(protein_structure.trajectory), len(measured_residues)), np.nan)
        gamma_2_store = np.full((len(protein_structure.trajectory), len(measured_residues)), np.nan)
        tight_count = 0  # counter to see how many frames have been discarded due to tight placement of the rotamers

        if self.cb == True:
            r6_cb_store = np.full((len(protein_structure.trajectory), len(measured_residues)), np.nan)

        for frame_ndx, protein in enumerate(
                protein_structure.trajectory[self.discard_frames:]):  # discard first on gromacs xtc
            #time0 = time.time()
            progressmeter.echo(protein.frame - self.discard_frames)

            self.current_replica = ((protein.frame - self.discard_frames) // self.frames_pre_replica)
            # print frame_ndx
            # define the atoms used to fit the rotamers. Note that an
            # ordered list has to be created as the ordering of C CA N is
            # different in both. Fit the rotamers onto the protein:
            # New placement method
            rotamersSite1 = self.rotamer_placement(self.lib.data,
                                                   protein_structure,
                                                   residue,
                                                   self.chains,
                                                   probe_library=self.lib)

            # boltz1 = self.lj_calculation(rotamersSite1, protein_structure, residue)
            # boltz1_sum = np.sum(boltz1)
            # boltz1_norm = boltz1/boltz1_sum
            # boltzman_weights1 = np.multiply(lib_norm, boltz1_norm)
            # boltzman_weights_norm1 = boltzman_weights1/np.sum(boltzman_weights1)
            # print boltzman_weights_norm1

            boltz1 = self.lj_calculation2(rotamersSite1, protein_structure, residue, e_cutoff=self.e_cutoff,
                                         ign_H=self.ign_H, hard_spheres=self.hard_spheres)
            
            #print()
            #time1 = time.time()
            #print('1: {:.3f}s'.format(time1-time0))
            #time0 = time1
            
            for b, bol in enumerate(boltz1):
                if np.isnan(bol):
                    # print 'yo'
                    boltz1[b] = 0.0

            # print 'boltz1'
            # print boltz1
            boltz1 = np.multiply(lib_norm, boltz1)
            z_1 = np.nansum(boltz1)
            #print(z_1)
            if z_1 <= self.z_cutoff:
                #print('ignoring this frame')
                tight_count += 1

                continue
            boltzman_weights_norm1 = boltz1 / z_1

            if self.debug:
                print('B-weights sum:')
                print(np.nansum(boltzman_weights_norm1))

            # define the atoms to measure the distances between
            rotamer1nitrogen = rotamersSite1.select_atoms("name N1")
            rotamer1oxigen = rotamersSite1.select_atoms("name O1")
            if self.cb == True:
                prot_cb_selection = protein_structure.select_atoms(
                    "resid {0} and name CB".format(residue))

            if self.selection == 'N':
                prot_backbone_selection = protein_structure.select_atoms(
                    "name {0} and not resid {1} and not resid 1 and not resname PRO".format(self.selection, residue))
            else:
                prot_backbone_selection = protein_structure.select_atoms(
                    "name {0} and not resid {1}".format(self.selection, residue))
            nit1_pos = np.array([rotamer1nitrogen.positions for x in rotamersSite1.trajectory])
            oxi1_pos = np.array([rotamer1oxigen.positions for x in rotamersSite1.trajectory])
            nitro1_pos = (nit1_pos + oxi1_pos) / 2

            # protNH_pos = (prot_backbone_nitrogen.positions + prot_backbone_hydrogen.positions) / 2
            protNH_pos = prot_backbone_selection.positions

            # mask = prot_backbone_nitrogen.resnums
            # resid_nitrogen_pos = np.full((len(residue_number), 3), np.NaN)

            # resid_nitrogen_pos[mask - 1] = prot_backbone_nitrogen.positions
            # resid_nitrogen_pos = prot_backbone_nitrogen.positions # FIXME: To change for new intermediate point
            resid_nitrogen_pos = protNH_pos

            if protein.frame == self.discard_frames:
                probe_coordinates = np.zeros((rotamersSite1.trajectory.n_frames, 3))
                for frame, ts in enumerate(rotamersSite1.trajectory):
                    probe_coordinates[frame] = rotamer1nitrogen.positions

            #size = len(resid_nitrogen_pos)  # should be measured_residues.size, repeated value




            angstrom_to_meters = 1e-8 #WARNING: this is conversion to centimeters.

            #time1 = time.time()
            #print('2: {:.3f}s'.format(time1-time0))
            #time0 = time1
            
            #nitro1_pos = np.squeeze(nitro1_pos) #dimensions were (n,1,3)
            #dists_array_r1 = mda_dist.distance_array(nitro1_pos, resid_nitrogen_pos,backend="OpenMP")
            n_probe_vector =nitro1_pos-resid_nitrogen_pos
            vect = (n_probe_vector*n_probe_vector[:,None,:,:]).sum(-1)*angstrom_to_meters*angstrom_to_meters
            dists_array_r1 = mda_dist.distance_array(np.squeeze(nitro1_pos), 
                             resid_nitrogen_pos,backend="OpenMP")*angstrom_to_meters
            leng = dists_array_r1*dists_array_r1[:,None,:]
            cos = vect/leng
            angle = (((3 / 2) * np.power(cos, 2)) - 0.5)*np.outer(boltzman_weights_norm1, boltzman_weights_norm1)[:,:,None]
            angle = angle.sum(axis=(0,1))
            
            dists_array_r3 = np.power(np.copy(dists_array_r1), -3) 
            dists_array_r6 = np.power(np.copy(dists_array_r1), -6) #WARNING: use square of dists_array_r3

            distributions_r1 = boltzman_weights_norm1.dot(dists_array_r1)
            distributions_r6 = boltzman_weights_norm1.dot(dists_array_r6)
            distributions_r3 = boltzman_weights_norm1.dot(dists_array_r3)

            r1_tmp = distributions_r1.copy()
            r6_tmp = distributions_r6.copy()
            r3_tmp = distributions_r3.copy() 

            if self.cb == True:
                # print 'CB output requested!'
                # print prot_cb_selection.positions
                # print resid_nitrogen_pos
                # pre allocate vector and then iterate distances over the HN selection
                for p, pos in enumerate(resid_nitrogen_pos):
                    # print 'CB output here:'
                    # print prot_cb_selection.positions
                    vec = prot_cb_selection.positions - pos
                    vec_norm = np.linalg.norm(vec)
                    r6_cb_store[frame_ndx, p] = np.power(vec_norm * angstrom_to_meters, -6)
                    # r6_cb_store = np.full((len(protein_structure.trajectory), len(measured_residues)), np.nan)
                    # mda_dist.distance_array(prot_cb_selection.positions, resid_nitrogen_pos, result=dists_array_r1_cb, backend="OpenMP")
            # print r6_cb_store
            # print 'yo!'
            # print r6_tmp[1]
            # print max(r6_tmp)

            r1_store[frame_ndx] = r1_tmp
            r6_store[frame_ndx] = r6_tmp
            r3_store[frame_ndx] = r3_tmp
            cosine_store[frame_ndx] = angle

            cosine[0] += angle
            s2_store[frame_ndx] = self.get_s_pre(r6_tmp, r3_tmp, self.tau_c, 
                                           self.tau_t, self.wh, self.k, angle)
            gamma_2_store[frame_ndx] = self.gamma_2(r6_tmp, r3_tmp, self.tau_c,
                                           self.tau_t, self.wh, self.k, angle)
           
        # print 'state'
        # print np.nanmean(r6_store, axis = 0)
        # print np.nanmean(r6_store, axis = 0)[1]
        # print max(r6_store[:,1])
            #time1 = time.time()
            #print('3: {:.3f}s'.format(time1-time0))
            #time0 = time1


        # this si somehwat wrong
        self.discard_frames += tight_count
        perc_disc = np.divide(float(tight_count), len(protein_structure.trajectory)) * 100
        logger.info(
            '{0} frames have been discarded due to tight labelling position, i.e. {1:.1g} % of the analyzed frames'.format(
                tight_count, perc_disc))
        ''' saving analysis as a pickle '''
        if self.save_file:
            save_data = {}
            save_data['r1'] = r1_store
            save_data['r6'] = r6_store
            save_data['s2'] = s2_store
            save_data['cosine'] = cosine_store
            save_data['r3'] = r3_store
            if self.cb:
                save_data['cb'] = r6_cb_store

            with open(self.save_file, 'wb') as out:
                pickle.dump(save_data, out)
            logger.info('Calculated <r> and S^2 are are saved to {} - Load them for re-use with --load flag.'.format(
                self.save_file))

        else:
            logger.info('Calculated <r> and S^2 are not being saved. Specify with  --save flag.')

        if self.optimize and self.fast:
            for frame_probe, position_probe in enumerate(probe_coordinates):
                for frame_h, position_h in enumerate(nh_translated_holder):
                    angle_holder = np.zeros(residue_number.size)
                    probe2h_vector = position_h - position_probe
                    probe2h_distance = np.sqrt((probe2h_vector * probe2h_vector).sum(axis=1))
                    for frame_h1, position_h1 in enumerate(nh_translated_holder):
                        if np.allclose(position_h, position_h1):
                            continue
                        probe2nexth_vector = position_h1 - position_probe
                        probe2nexth_distance = np.sqrt((probe2nexth_vector * probe2nexth_vector).sum(axis=1))
                        vect = np.sum((probe2h_vector[list(range(size))] * 1e-8) * (probe2nexth_vector[list(range(size))] * 1e-8),
                                      axis=1)
                        leng = np.multiply(probe2h_distance[list(range(size))] * 1e-8,
                                           probe2nexth_distance[list(range(size))] * 1e-8)
                        cos = vect / leng

                        angle_holder[list(range(size))] += ((3 / 2) * np.power(cos, 2)) - 0.5
                angle_holder[list(range(size))] *= (1 / (len(nh_translated_holder) ** 2))
                s2_fast[0][list(range(size))] += angle_holder[list(range(size))]
            s2_fast /= len(probe_coordinates)

        if self.replicas == 1:
            distributions_r6 /= (protein_structure.trajectory.n_frames - self.discard_frames)
            # print 'shit comes here! :'
            # print np.nanmean(r6_store, axis = 0 )
            # print distributions_r6[0]
            distributions_r3 /= (protein_structure.trajectory.n_frames - self.discard_frames)
            cosine /= (protein_structure.trajectory.n_frames - self.discard_frames)

            # distributions[measured_residues - 1] = self.gamma_2(distributions_r6[0], distributions_r3[0],
            #                                                    self.tau_c, self.tau_t, self.wh, self.k, cosine[0])

            # calc mean PRE profile
            # FIXME: some averaging going wrong here ...... :(
            # distributions[measured_residues - 1] = np.nanmean(gamma_2_store, axis=0)
            # print 'hello2'
            # print distributions

            ''' OPTIMIZATION WITH EXPERIMENTAL DATA '''
            if self.optimize:
                cosine_full = np.full(len(residue_number), fill_value=np.NaN)
                cosine_full[measured_residues - 1] = cosine[0]

                distributions_r6_full = np.full(len(residue_number), fill_value=np.NaN)
                distributions_r6_full[measured_residues - 1] = distributions_r6[0]

                distributions_r3_full = np.full(len(residue_number), fill_value=np.NaN)
                distributions_r3_full[measured_residues - 1] = distributions_r3[0]

                s2_fast_full = np.full(len(residue_number), fill_value=np.NaN)
                s2_fast_full[measured_residues - 1] = s2_fast[0]

                self.tau_t = np.full(len(residue_number), fill_value=np.NaN)
                self.tau_t[measured_residues - 1] = 1e-10

                # print 'Tau_c: \t', self.tau_c
                # print 'Tau_t: \t', self.tau_t
                #
                # self.plot_s2VStau_t(rax, cosine_full, distributions_r6_full, distributions_r3_full, self.tau_t, rmsf)

                # self.tau_c = plsq_slow[0][0]
                # self.tau_c = plsq_slow['x'][0]

                """ HERE BE INTENSITIES-INDEPENDENT OPTIMIZATION (OR DRAGONS) """
                if self.fast:  # Badly wrong, wassup?
                    plsq_slow = least_squares(self.residuals_tau_c, self.tau_c,
                                              args=(r2_full['value'][rmsf_core_nanfiltered],
                                                    distributions_r6_full[rmsf_core_nanfiltered],
                                                    distributions_r3_full[rmsf_core_nanfiltered],
                                                    self.tau_t[rmsf_core_nanfiltered], self.wh,
                                                    self.k, cosine_full[rmsf_core_nanfiltered]),
                                              bounds=(0, np.inf))

                    print('Tau_c: \t', plsq_slow['x'][0])

                    self.tau_c = plsq_slow['x'][0]

                    plsq_slow = least_squares(self.residuals_tau_t, self.tau_t[rmsf_core_nanfiltered],
                                              args=(r2_full['value'][rmsf_core_nanfiltered],
                                                    distributions_r6_full[rmsf_core_nanfiltered],
                                                    distributions_r3_full[rmsf_core_nanfiltered],
                                                    self.tau_c, self.wh,
                                                    self.k, cosine_full[rmsf_core_nanfiltered]),
                                              bounds=(0, np.inf))

                    print('Tau_t slow: \t', plsq_slow['x'][0])

                    self.tau_t = plsq_slow['x'][0]

                    distributions[rmsf_core_nanfiltered] = self.gamma_2(distributions_r6_full[rmsf_core_nanfiltered],
                                                                        distributions_r3_full[rmsf_core_nanfiltered],
                                                                        self.tau_c, self.tau_t, self.wh, self.k,
                                                                        cosine_full[rmsf_core_nanfiltered])

                    # fast exchange
                    plsq_fast = least_squares(self.residuals_tau_t, self.tau_t,
                                              args=(r2_full['value'][rmsf_move_nanfiltered],
                                                    distributions_r6_full[rmsf_move_nanfiltered],
                                                    distributions_r3_full[rmsf_move_nanfiltered],
                                                    self.tau_c, self.wh, self.k,
                                                    s2_fast_full[rmsf_move_nanfiltered]),
                                              bounds=(0, np.inf))

                    print('Tau_t fast: \t', plsq_fast['x'][0])

                    # should the gamma 2 calculation be done with a differently calculated COSINE variable here?
                    distributions[rmsf_move_nanfiltered] = self.gamma_2(distributions_r6_full[rmsf_move_nanfiltered],
                                                                        distributions_r3_full[rmsf_move_nanfiltered],
                                                                        self.tau_c, plsq_fast['x'][0], self.wh, self.k,
                                                                        s2_fast_full[rmsf_move_nanfiltered])

                    # medium exchange area?
                    # plsq_medium = least_squares(self.residuals_tau_t, self.tau_t,
                    #                             args=(r2_full['value'][rmsf_medium_nanfiltered],
                    #                                   distributions_r6_full[rmsf_medium_nanfiltered],
                    #                                   distributions_r3_full[rmsf_medium_nanfiltered],
                    #                                   self.tau_c, self.wh, self.k,
                    #                                   cosine_full[rmsf_medium_nanfiltered]),
                    #                             bounds=(0, np.inf))

                    # distributions[rmsf_medium_nanfiltered] = self.gamma_2(distributions_r6_full[rmsf_medium_nanfiltered],
                    #                                         distributions_r3_full[rmsf_medium_nanfiltered],
                    #                                         self.tau_c, plsq_medium['x'][0], self.wh, self.k,
                    #                                         cosine_full[rmsf_medium_nanfiltered])

                    # print 'Tau_t medium: \t', plsq_medium['x'][0]

                else:
                    rmsf_core_filtered = []
                    for i in rmsf_core:
                        if 0.15 < self.exp_intensities[i] < 0.85:
                            rmsf_core_filtered.append(i)

                    # for i, x in enumerate(rmsf_core_filtered):
                    #     rmsf_core_filtered[i] = int(x)
                    # rmsf_core_filtered = np.array(rmsf_core_filtered)
                    #
                    # print self.exp_intensities[rmsf_core_filtered]
                    print(rmsf_core_filtered[0])
                    print(rmsf_core_filtered[-1])
                    self.tau_c = \
                        least_squares(self.residuals_tau_c_I, self.tau_c, args=(r2_full['value'][rmsf_core_filtered],
                                                                                distributions_r6_full[
                                                                                    rmsf_core_filtered],
                                                                                distributions_r3_full[
                                                                                    rmsf_core_filtered],
                                                                                self.tau_t,
                                                                                self.wh, self.k,
                                                                                cosine_full[rmsf_core_filtered], self.t,
                                                                                self.exp_intensities[
                                                                                    rmsf_core_filtered]),
                                      bounds=(1e-9, 50e-9), xtol=1e-10,
                                      gtol=1e-10)['x'][0]
                    print(self.tau_c)
                    # residue-specific tau_t optimization
                    for ind, val in enumerate(self.tau_t):
                        try:
                            self.tau_t[ind] = \
                                least_squares(self.residuals_tau_t_I, self.tau_t[ind], args=(r2_full['value'][ind],
                                                                                             distributions_r6_full[ind],
                                                                                             distributions_r3_full[ind],
                                                                                             self.tau_c, self.wh,
                                                                                             self.k,
                                                                                             cosine_full[ind], self.t,
                                                                                             self.exp_intensities[ind]),
                                              bounds=(10e-12, self.tau_c))['x'][0]
                        except ValueError:
                            pass
                        except TypeError:
                            pass
                    print(self.tau_t)
                    distributions = self.gamma_2(distributions_r6_full, distributions_r3_full, self.tau_c, self.tau_t,
                                                 self.wh,
                                                 self.k, cosine_full)

                i_distributions = np.zeros(distributions.size)
                i_distributions = np.divide((r2_full['value'] * np.exp(-distributions * self.t)),
                                            (r2_full['value'] + distributions))

                i_distributions_max = np.zeros(distributions.size)
                i_distributions_min = np.zeros(distributions.size)

                i_distributions_max[r2_full['index'] - 1] = np.divide(
                    ((r2_full['value'] + r2_full['error']) * np.exp(-distributions[r2_full['index'] - 1] * self.t)),
                    ((r2_full['value'] + r2_full['error']) + distributions[r2_full['index'] - 1]))

                i_distributions_min[r2_full['index'] - 1] = np.divide(
                    ((r2_full['value'] - r2_full['error']) * np.exp(-distributions[r2_full['index'] - 1] * self.t)),
                    ((r2_full['value'] - r2_full['error']) + distributions[r2_full['index'] - 1]))
                self.plot_and_save_pre(rax, i_distributions, i_distributions_max, i_distributions_min, residue)
            else:
                #print 'yo'
                dist_r6_av = np.nanmean(r6_store, axis=0)
                dist_r3_av = np.nanmean(r3_store, axis=0)
                cosine_av = np.nanmean(cosine_store, axis=0)
                gamma_2_av = self.gamma_2(dist_r6_av, dist_r3_av, self.tau_c, self.tau_t, self.wh, self.k, cosine_av)

                if self.cb == True:
                    dist_r6_cb_av = np.nanmean(r6_cb_store, axis=0)
                    gamma_2_cb_av = self.gamma_2_cb(dist_r6_cb_av, self.tau_c, self.wh, self.k)
                    distributions[measured_residues - 1] = gamma_2_cb_av
                    (i_distributions, i_distributions_max, i_distributions_min) = self.calc_i_ratio(self, distributions,
                                                                                                    r2_full)
                    self.plot_and_save_pre(rax, i_distributions, i_distributions_max, i_distributions_min, residue,
                                           suffix='_CB')

                # gamma_2_load(dist_r6, tau_c, tau_t, wh, k, s_pre)
                # distributions = np.full(residue_number.size, fill_value=np.NaN)
                # average r6 and then gamma calc
                distributions[measured_residues - 1] = gamma_2_av

                # joaos values
                # print 'now joao s2'
                # distributions [measured_residues - 1]= self.gamma_2(distributions_r6[0], distributions_r3[0], self.tau_c, self.tau_t, self.wh, self.k, cosine[0])

                (i_distributions, i_distributions_max, i_distributions_min) = self.calc_i_ratio(self, distributions,
                                                                                                r2_full)
                # i_dist = self.calc_i_ratio_noerr(self, distributions, r2_full)
                # average i_dists
                # i_distributions_av = np.nanmean(i_distributions, axis=0)
                # i_distributions_max_av = np.nanmean(i_distributions_max, axis=0)
                # i_distributions_min_av = np.nanmean(i_distributions_min, axis=0)
                # print i_distributions_av
                self.plot_and_save_pre(rax, i_distributions, i_distributions_max, i_distributions_min, residue)
                print('the end')
        else:
            distributions[0] /= (protein_structure.trajectory.n_frames - self.discard_frames)
            distributions_full = self.sb_equation(distributions[0], self.tau, self.wh, self.k)
            i_distributions = np.zeros(distributions_full.size)
            i_distributions_max = np.zeros(distributions_full.size)
            i_distributions_min = np.zeros(distributions_full.size)

            i_distributions[r2_full['index'] - 1] = np.divide(
                (r2_full['value'] * np.exp(-distributions_full[r2_full['index'] - 1] * self.t)),
                (r2_full['value'] + distributions_full[r2_full['index'] - 1]))
            i_distributions_max[r2_full['index'] - 1] = np.divide(
                ((r2_full['value'] + r2_full['error']) * np.exp(-distributions_full[r2_full['index'] - 1] * self.t)),
                ((r2_full['value'] + r2_full['error']) + distributions_full[r2_full['index'] - 1]))

            i_distributions_min[r2_full['index'] - 1] = np.divide(
                ((r2_full['value'] - r2_full['error']) * np.exp(-distributions_full[r2_full['index'] - 1] * self.t)),
                ((r2_full['value'] - r2_full['error']) + distributions_full[r2_full['index'] - 1]))

            self.plot_and_save_pre(rax, i_distributions, i_distributions_max, i_distributions_min, residue)

            for replica in range(1, self.replicas + 1):
                distributions[replica] /= (protein_structure.trajectory.n_frames - self.discard_frames) / self.replicas
                distributions_full = self.sb_equation(distributions[replica], self.tau, self.wh, self.k)
                i_distributions = np.zeros(distributions_full.size)
                i_distributions_max = np.zeros(distributions_full.size)
                i_distributions_min = np.zeros(distributions_full.size)

                i_distributions[r2_full['index'] - 1] = np.divide(
                    (r2_full['value'] * np.exp(-distributions_full[r2_full['index'] - 1] * self.t)),
                    (r2_full['value'] + distributions_full[r2_full['index'] - 1]))
                i_distributions_max[r2_full['index'] - 1] = np.divide(((r2_full['value'] + r2_full['error']) * np.exp(
                    -distributions_full[r2_full['index'] - 1] * self.t)),
                                                                      ((r2_full['value'] + r2_full['error']) +
                                                                       distributions_full[r2_full['index'] - 1]))

                i_distributions_min[r2_full['index'] - 1] = np.divide(((r2_full['value'] - r2_full['error']) * np.exp(
                    -distributions_full[r2_full['index'] - 1] * self.t)),
                                                                      ((r2_full['value'] - r2_full['error']) +
                                                                       distributions_full[r2_full['index'] - 1]))
                self.plot_and_save_pre(rax, i_distributions, i_distributions_max, i_distributions_min, residue, replica)
