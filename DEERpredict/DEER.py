# -*- coding: utf-8 -*-
"""
DEERPrediction Class
--------------------

Class to perform DEER prediction.

"""

import os

# Coordinates and arrays
import numpy as np
import MDAnalysis
import math
import MDAnalysis.analysis.distances as mda_dist

# Logger
import logging

# Inner imports
from DEERpredict.utils import Operations

logger = logging.getLogger("MDAnalysis.app")

class DEERPrediction(Operations):
    """Calculation of distance distributions between two spin labels."""

    def __init__(self, protein_structure, residues, **kwargs):
        """

        Args:
            protein_structure (:py:class:`MDAnalysis.core.universe.Universe`):
            residues (list(:py:class:`str`)):
        :Keywords:

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
        #            *chains*
        #               optional list of two chains for homopolymeric proteins
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
        Operations.__init__(self, protein_structure, **kwargs)

        # loads all args and kwargs
        self.record_frames = kwargs.get('record_frames', False)  # profile per frame saving
        self.form_factor = kwargs.get('form_factor', False)  # back-calculate final experimental form factor
        lib_norm = self.lib.weights / np.sum(self.lib.weights)  # normalizes weights, in case library isn't
        # lib_norm = self.lib.weights / np.max(self.lib.weights)  # normalizes weights, polyhach's method?

        # Naming part, dependant on selection
        if self.chains:
            output_file, ext = os.path.splitext(kwargs.pop('output_file', 'distance_profile'))
            ext = ext or ".dat"
            self.output_file = "{0}-{1[0]}{2[0]}-{1[1]}{2[1]}{3}".format(output_file, residues, self.chains, ext)
            ext = ".png"
            self.output_plot = "{0}-{1[0]}{2[0]}-{1[1]}{2[1]}{3}".format(output_file, residues, self.chains, ext)
            self.output_plot_tk = "{0}_form-factor-{1[0]}{2[0]}-{1[1]}{2[1]}{3}".format(output_file, residues,
                                                                                      self.chains, ext)
        else:
            output_file, ext = os.path.splitext(kwargs.pop('output_file', 'distance_profile'))
            ext = ext or ".dat"
            self.output_file = "{0}-{1[0]}-{1[1]}{2}".format(output_file, residues, ext)
            ext = ".png"
            self.output_plot = "{0}-{1[0]}-{1[1]}{2}".format(output_file, residues, ext)
            self.output_plot_tk = "{0}_form-factor-{1[0]}-{1[1]}{2}".format(output_file, residues, ext)


        # fourier broadening parameters
        ra = -5
        re = 20
        nr = 501
        sig = 0.1
        rax = np.linspace(ra, re, nr)
        vari = np.exp(-(rax/sig)**2)
        distributions = np.zeros((self.replicas+1, rax.size))
        #

        if len(residues) != 2:
            raise ValueError("The residue_list must contain exactly 2 residue numbers: current "
                             "value {0}.".format(residues))

        logger.info("Starting rotamer distance analysis of trajectory "
                    "{0}...".format(protein_structure.trajectory.filename))
        logger.info("Rotamer library = '{0}'".format(self.lib.name))
        logger.debug("Results will be written to {0}.".format(self.output_file))

        # Replica handling for multi-replica calculation
        self.frames_per_replica = int(math.ceil((self.stop_frame - self.start_frame) / self.jump_frame)/self.replicas)+1

        # Progress-meter adaptation for start, stop and skip.
        progressmeter = MDAnalysis.lib.log.ProgressMeter(
            int(math.ceil((self.stop_frame - self.start_frame) / self.jump_frame)),
            interval=1)

        precalculate_rotamer(self)

        # Main calculation loop portion
        # For each trajectory frame place the probes on the position using .rotamer_placement(), calculates external
        # energy (boltzmann distribution) based on Lennard-Jones function van der Waals. Loops over first probe
        # NO position against all second probe NO positions.
        for protein in protein_structure.trajectory[self.start_frame:self.stop_frame:self.jump_frame]:
            corrected_frame_index = int((protein.frame-self.start_frame)/self.jump_frame)  # relative frame index
            progressmeter.echo(corrected_frame_index)  # show relative index on terminal
            self.current_replica = int((corrected_frame_index+1)/self.frames_per_replica)  # replica calculation

            #
            if self.chains:
                rotamersSite1 = self.rotamer_placement(protein_structure, residues[0], self.chains[0])
                rotamersSite2 = self.rotamer_placement(protein_structure, residues[1], self.chains[1])
            else:
                rotamersSite1 = self.rotamer_placement(protein_structure, residues[0])
                rotamersSite2 = self.rotamer_placement(protein_structure, residues[1])

            # before method
            # boltz1 = self.lj_calculation(rotamersSite1, protein_structure, residues[0])
            # boltz1_sum = np.sum(boltz1)
            # boltz1_norm = boltz1/boltz1_sum
            # boltzman_weights1 = np.multiply(lib_norm, boltz1_norm)
            #
            # boltz2 = self.lj_calculation(rotamersSite2, protein_structure, residues[1])
            # boltz2_sum = np.sum(boltz2)
            # boltz2_norm = boltz2/boltz2_sum
            # boltzman_weights2 = np.multiply(lib_norm, boltz2_norm)

            # boltzman_weights_norm1 = boltzman_weights1/np.sum(boltzman_weights1)
            # boltzman_weights_norm2 = boltzman_weights2/np.sum(boltzman_weights2)

            # straight polyhach
            boltz1 = self.lj_calculation(rotamersSite1, protein_structure, residues[0])
            boltz1 = np.multiply(lib_norm, boltz1)
            z_1 = np.sum(boltz1)
            boltzmann_weights_norm1 = boltz1/z_1

            boltz2 = self.lj_calculation(rotamersSite2, protein_structure, residues[1])
            boltz2 = np.multiply(lib_norm, boltz2)
            z_2 = np.sum(boltz2)
            boltzmann_weights_norm2 = boltz2 / z_2

            # probe1 = rotamersSite1.atoms
            # probe2 = rotamersSite2.atoms
            # with MDAnalysis.Writer("probe1.pdb", probe1.n_atoms) as W:
            #     for index, ts in enumerate(rotamersSite1.trajectory):
            #         probe1.tempfactors = boltzman_weights_norm1[index]/np.max(boltzman_weights_norm1)
            #         probe1.occupancies = 1.00
            #         W.write(probe1)
            # with MDAnalysis.Writer("probe2.pdb", probe2.n_atoms) as W:
            #     for index, ts in enumerate(rotamersSite2.trajectory):
            #         probe2.tempfactors = boltzman_weights_norm2[index]/np.max(boltzman_weights_norm2)
            #         probe2.occupancies = 1.00
            #         W.write(probe2)

            # with open('boltzmann_1.txt', 'w') as boltzout_1:
            #     for i in boltzman_weights_norm1:
            #         boltzout_1.write('{0}\n'.format(str(i/np.max(boltzman_weights_norm1))))
            # with open('boltzmann_2.txt', 'w') as boltzout_2:
            #     for i in boltzman_weights_norm2:
            #         boltzout_2.write('{0}\n'.format(str(i / np.max(boltzman_weights_norm2))))
            # with open('internal_1.txt', 'w') as int_out:
            #     for i in lib_norm:
            #         int_out.write('{0}\n'.format(str(i / np.max(lib_norm))))

            # define the atoms to measure the distances between
            frame_distributions = np.zeros((rax.size))
            rotamer1nitrogen = rotamersSite1.select_atoms("name N1")
            rotamer2nitrogen = rotamersSite2.select_atoms("name N1")
            rotamer1oxigen = rotamersSite1.select_atoms("name O1")
            rotamer2oxigen = rotamersSite2.select_atoms("name O1")

            size = len(rotamersSite1.trajectory)
            nit1_pos = np.array([rotamer1nitrogen.positions for x in rotamersSite1.trajectory])
            nit2_pos = np.array([i for x in rotamersSite2.trajectory for i in rotamer2nitrogen.positions])
            oxi1_pos = np.array([rotamer1oxigen.positions for x in rotamersSite1.trajectory])
            oxi2_pos = np.array([i for x in rotamersSite2.trajectory for i in rotamer2oxigen.positions])
            nitro1_pos = (nit1_pos + oxi1_pos) / 2
            nitro2_pos = (nit2_pos + oxi2_pos) / 2

            dists_array = np.zeros((1, size), dtype=np.float64)
            for position1_index, position1 in enumerate(nitro1_pos):
                mda_dist.distance_array(position1, nitro2_pos, result=dists_array, backend="OpenMP")
                dists_array /= 10.0
                dists_array = np.round((nr*(dists_array-ra))/(re-ra))

                # FIXME: trying to remove for-loop (SEEMS SLOWER SO FAR!)
                # element = dists_array[0].astype(int)
                # frame_distributions[element] += boltzman_weights_norm1[position1_index] * \
                #     boltzman_weights_norm2[np.arange(element.size)]
                # FIXME:

                for pos2_index, element in enumerate(dists_array[0]):  # could we take out this for loop?
                    element = int(element)
                    frame_distributions[element] = np.nansum([frame_distributions[element],
                                                              boltzmann_weights_norm1[position1_index] *
                                                              boltzmann_weights_norm2[pos2_index]])

            if self.record_frames is True:
                if self.chains:
                    frame_inv_distr = np.fft.ifft(frame_distributions)*np.fft.ifft(vari)
                    frame_distributions_smoothed = np.real(np.fft.fft(frame_inv_distr))
                    frame_distributions_smoothed = frame_distributions_smoothed/np.sum(frame_distributions_smoothed)
                    self.output_file_frame = "{0}-{1[0]}{2[0]}-{1[1]}{2[1]}-{3}{4}".format(output_file,
                                                                                           residues,
                                                                                           self.chains,
                                                                                           protein.frame,
                                                                                           '.dat')  # DO WE NEED THIS?
                    with open(self.output_file_frame, 'w') as OUTPUT:
                        for index, distance in enumerate(rax[100:401]):
                            OUTPUT.write('{0:>7.4f} {1:>7.4f}\n'.format(distance,
                                                                        frame_distributions_smoothed[index+200]))
                else:
                    frame_inv_distr = np.fft.ifft(frame_distributions)*np.fft.ifft(vari)
                    frame_distributions_smoothed = np.real(np.fft.fft(frame_inv_distr))
                    frame_distributions_smoothed = frame_distributions_smoothed/np.sum(frame_distributions_smoothed)
                    self.output_file_frame = "{0}-{1[0]}-{1[1]}-{2}{3}".format(output_file, residues, protein.frame,
                                                                               '.dat') # DO WE NEED THIS?
                    with open(self.output_file_frame, 'w') as OUTPUT:
                        for index, distance in enumerate(rax[100:401]):
                            OUTPUT.write('{0:>7.4f} {1:>7.4f}\n'.format(distance,
                                                                        frame_distributions_smoothed[index+200]))

            # saving to the overall and per-replica distributions
            distributions[0] = np.nansum([distributions[0], frame_distributions], axis=0)
            distributions[self.current_replica + 1] = np.nansum([distributions[self.current_replica + 1],
                                                                 frame_distributions], axis=0)

        if self.replicas == 1:
            inv_distr = np.fft.ifft(distributions[0])*np.fft.ifft(vari)
            distributions = np.real(np.fft.fft(inv_distr))
            distributions = distributions/np.sum(distributions)
            # distributions = distributions/(math.ceil((self.stop_frame - self.start_frame) / self.jump_frame))

            self.plot_and_save_deer(rax[100:401], distributions[200:501], self.output_plot, self.output_file)

            if self.form_factor:
                self.form_factor_profile(rax[100:401], distributions[200:501], name=self.output_plot_tk)

            logger.info("Distance distribution for residues {0[0]} - {0[1]} "
                        "was written to {1}".format(residues, self.output_file))
            logger.info("Distance distribution for residues {0[0]} - {0[1]} "
                        "was plotted to {1}".format(residues, self.output_plot))
        else:
            inv_distr = np.fft.ifft(distributions[0])*np.fft.ifft(vari)
            distributions_global = np.real(np.fft.fft(inv_distr))
            distributions_global = distributions_global/np.sum(distributions_global)

            self.plot_and_save_deer(rax[100:401], distributions_global[200:501], self.output_plot, self.output_file)

            logger.info("Distance distribution for residues {0[0]} - {0[1]} "
                        "was written to {1}".format(residues, self.output_file))
            logger.info("Distance distribution for residues {0[0]} - {0[1]} "
                        "was plotted to {1}".format(residues, self.output_plot))

            for replica in range(1, self.replicas+1):
                inv_distr = np.fft.ifft(distributions[replica])*np.fft.ifft(vari)
                distributions_replica = np.real(np.fft.fft(inv_distr))
                distributions_replica = distributions_replica/np.sum(distributions_replica)

                if self.chains:
                    ext = ".png"
                    self.output_plot = "{0}-{1[0]}{2[0]}-{1[1]}{2[1]}_replica{3}{4}".format(output_file,
                                                                                            residues,
                                                                                            self.chains,
                                                                                            replica,
                                                                                            ext)
                    ext = '.dat'
                    self.output_file = "{0}-{1[0]}{2[0]}-{1[1]}{2[1]}_replica{3}{4}".format(output_file,
                                                                                            residues,
                                                                                            self.chains,
                                                                                            replica,
                                                                                            ext)
                else:
                    ext = ".png"
                    self.output_plot = "{0}-{1[0]}-{1[1]}_replica{3}{4}".format(output_file,
                                                                                residues,
                                                                                replica,
                                                                                ext)
                    ext = '.dat'
                    self.output_file = "{0}-{1[0]}-{1[1]}_replica{3}{4}".format(output_file,
                                                                                residues,
                                                                                replica,
                                                                                ext)
                self.plot_and_save_deer(rax[100:401], distributions_replica[200:501], self.output_plot,
                                        self.output_file)

                logger.info("Distance distribution for residues {0[0]} - {0[1]} and replica {1} "
                            "was written to {2}".format(residues, replica, self.output_file))
                logger.info("Distance distribution for residues {0[0]} - {0[1]} and replica {1} "
                            "was plotted to {2}".format(residues, replica, self.output_plot))
