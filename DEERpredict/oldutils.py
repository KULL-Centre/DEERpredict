"""
Utilities
=========

Inherited functions and methods for usage in the prediction classes.

"""

# Coordinates and arrays
import math
import numpy as np
import MDAnalysis
from MDAnalysis.coordinates.memory import MemoryReader

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Logger
import logging

# Inner imports
from DEERpredict.lennardjones import vdw, p_Rmin2, eps
import DEERpredict.libraries

import seaborn as sns

logger = logging.getLogger("MDAnalysis.app")


class Operations(object):
    """
    Operations base class.

    Class that holds all the methods and class attributes inherited by the prediction classes.

    Attributes:
        rotamers (:py:class:`MDAnalysis.core.universe.Universe`): Universe instance that records all rotamers as a trajectory
        weights (:py:class:`numpy.ndarray`): Array containing the population of each rotamer.
        name (:py:class:`str`): Name of the library.
        lib (:py:class:`dict`): Dictionary containing the file names and meta data for the library :attr:`name`.

    """
    time_scale = 1e-6
    nm_scale = 1e-9

    def __init__(self, protein_structure, **kwargs):
        """
        Args:
            protein_structure (:py:class:`MDAnalysis.core.universe.Universe`): Protein structure universe object
            **kwargs:
        """
        self.chains = kwargs.pop('chains', None)
        self.libname = kwargs.pop('libname', 'MTSSL 175K X1X2')
        self.replicas = kwargs.pop('replicas', 1)
        self.discard_frames = kwargs.pop('discard_frames', 0)
        self.start_frame = kwargs.pop('start_frame', 0)
        if kwargs.get('stop_frame') is None:
            self.stop_frame = protein_structure.trajectory.n_frames
        else:
            self.stop_frame = kwargs.pop('stop_frame')
        self.jump_frame = kwargs.pop('jump_frame', 1)
        self.output_prefix = kwargs.pop('output_prefix', 'profile')
        self.plot_extension = kwargs.pop('plot_extension', 'png')
        self.file_extension = kwargs.pop('file_extension', 'dat')
        self.lib = DEERpredict.libraries.RotamerLibrary(self.libname)
        self.temp = kwargs.pop('temperature', 300)
        self.time_scale = 1e-6
        self.mc = None
        self.r0 = kwargs.pop('r0', 5.4)

    def pre_calculate_rotamer(self, protein, site_resid, chain=None, probe_library=None):
        """
        Pre-calculates some objects that do not depend on positions

        Args:
            protein (:py:class:`MDAnalysis.core.universe.Universe`): Single frame from protein structure universe object
            site_resid (:py:class:`int`): Integer, 1-based, indicating the position at which to place the rotamer
            chain (:py:class:`str`): Chain indicator string.
            probe_library (:py:class:`str`):  Probe library name

        Returns:
            :py:class:`MDAnalysis.core.universe.Universe`: Probe positions in the new coordinate system
            :py:tuple: Tuple with Prot_ca, prot_Co, prot_N
        """
        if not probe_library:
            probe_library = self.lib

        if chain:
            prot_Ca = protein.select_atoms('protein and name CA and resid {0} and segid {1}'.format(site_resid, chain))
            prot_Co = protein.select_atoms('protein and name C and resid {0} and segid {1}'.format(site_resid, chain))
            prot_N = protein.select_atoms('protein and name N and resid {0} and segid {1}'.format(site_resid, chain))
        else:
            prot_Ca = protein.select_atoms('protein and name CA and resid {0}'.format(site_resid))
            prot_Co = protein.select_atoms('protein and name C and resid {0}'.format(site_resid))
            prot_N = protein.select_atoms('protein and name N and resid {0}'.format(site_resid))
        probe_coords = np.zeros((len(probe_library.top.atoms),1, 3))
        new_universe = MDAnalysis.Universe(probe_library.top.filename, probe_coords, format=MemoryReader, order='afc')
        return new_universe, (prot_Ca, prot_Co, prot_N)
        
    def rotamer_placement(self, universe, prot_atoms, probe_array, probe_library=None):
        """
        Rotates and translates the rotamer from its coordinate system to the protein's coordinate system.

        Args:
            prot_atoms (py:tuple) : prot_Ca, prot_Co, prot_N 
            probe_array (:py:class:`numpy.ndarray`): relative coordinates of each rotamer
            probe_library (:py:class:`str`):  Probe library name

        Returns:
            :py:class:`MDAnalysis.core.universe.Universe`: Probe positions in the new coordinate system
        """
        if not probe_library:
            probe_library = self.lib
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
        probe_coords = probe_array[:, 2:5].copy().T
        probe_coords = np.dot(rotation, probe_coords).T
        probe_coords = probe_coords.reshape((probe_array.shape[0] // (len(probe_library.top.atoms)),
                                             len(probe_library.top.atoms), 3))
        probe_coords += offset
        probe_coords = probe_coords.swapaxes(0, 1)

        universe.load_new(probe_coords, format=MemoryReader,order='afc')
        return universe

    def lj_calculation(self, fitted_rotamers, protein, site_resid, e_cutoff = 1e10, ign_H = True, forgive = 0.5,
                       hard_spheres = False, fret = False):
        """

        Performs Lennard-Jones potential calculation from a given placed rotamer library, protein frame and site index.

        Args:
            fitted_rotamers (:py:class:`MDAnalysis.core.universe.Universe`): Rotamer poses
            protein (:py:class:`MDAnalysis.core.universe.Universe`): Protein frame
            site_resid (:py:class:`int`): Substituted residue in protein, 1-based

        Returns:
            :py:class:`numpy.ndarray`: Calculated LJ interaction potential

        """
        gas_un = 1.9858775e-3 # CHARMM, in kcal/mol*K
        #forgive = 0.5
        # maxdist = 11.828 # comes from library, used to hole for the mutation
        if ign_H:
            proteinNotSite = protein.select_atoms("protein and not type H and not (resid {0})".format(site_resid))
            rotamerSel_LJ = fitted_rotamers.select_atoms("not type H and not (name CA or name C or name N or name O)")
        else:
            proteinNotSite = protein.select_atoms("protein and not (resid {0})".format(site_resid))
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
        proteinNotSite = protein.trajectory.ts.positions[proteinNotSite]
        
        lj_energy_pose = np.zeros((len(fitted_rotamers.trajectory)))
        for rotamer_counter, rotamer in enumerate(fitted_rotamers.trajectory):
            # RCS: Calculating all the distances seems to be much faster than any kind of
            # approach to calculate neighbours. capped_distances, and NeighborSearch are slower
            d = MDAnalysis.lib.distances.distance_array(rotamer.positions[rotamerSel_LJ], 
                proteinNotSite)
            d = rmin_ij/d
            d = d*d*d
            d = d*d
            pair_LJ_energy = eps_ij*(d*d-2.*d)
            #This is slower (but clearer), and it took ~50% of execution time:
            #pair_LJ_energy = eps_ij*(((rmin_ij/d)**12)-2*((rmin_ij/d)**6))
            #This also seems to be slightly slower. TODO re-check.
            #pair_LJ_energy = ne.evaluate('eps_ij*(((rmin_ij/d)**12)-2*((rmin_ij/d)**6))')         
            lj_energy_pose[rotamer_counter] = pair_LJ_energy.sum()
            
        #if numpy.isnan(lj_energy_pose).any()
        # checking energies are below cutoff
        #WARNING: remove this. Unnecessary.
        for e, rotE in enumerate(lj_energy_pose):
            if rotE >= e_cutoff:
                if fret:
                    lj_energy_pose[e] += 1e99
                    #print '#### YELLOW ####'
                else:
                    lj_energy_pose[e] = np.nan

        if self.mc:
            print(' Sure you want to be here?')
            for e, rotE in enumerate(lj_energy_pose):
                #print rotE
                if rotE > np.nanmin(lj_energy_pose):
                    #print 'removing rotamer %d'%e
                    lj_energy_pose[e] = np.nan
            #print lj_energy_pose

        return np.exp(-lj_energy_pose/(gas_un*self.temp))  # for new alignment method


    def sb_equation(self, dist_array, tau, wh, k):
        """

        Solomon-Bloembergen equation calculation.

        Args:
            dist_array:
            tau:
            wh:
            k:

        Returns:

        """
        # return ((4*tau+((3*tau)/(1+((wh**2)*(tau**2)))))*k)/(np.power(dist_array, 6))
        return ((4*tau+((3*tau)/(1+((wh**2)*(tau**2)))))*k)*dist_array


    @staticmethod
    def calc_i_ratio(self, distributions, r2_full):
        """

        Args:
            self:
            distributions:
            r2_full:

        Returns:

        """
        i_distributions = np.zeros(distributions.shape)
        i_distributions_max = np.zeros(distributions.shape)
        i_distributions_min = np.zeros(distributions.shape)
        i_distributions[r2_full['index'] - 1] = np.divide(
            (r2_full['value'] * np.exp(-distributions[r2_full['index'] - 1] * self.t)),
            (r2_full['value'] + distributions[ r2_full['index'] - 1]))
        i_distributions_max[r2_full['index'] - 1] = np.divide(
            ((r2_full['value'] + r2_full['error']) * np.exp(-distributions[r2_full['index'] - 1] * self.t)),
            ((r2_full['value'] + r2_full['error']) + distributions[r2_full['index'] - 1]))

        i_distributions_min[r2_full['index'] - 1] = np.divide(
            ((r2_full['value'] - r2_full['error']) * np.exp(-distributions[r2_full['index'] - 1] * self.t)),
            ((r2_full['value'] - r2_full['error']) + distributions[r2_full['index'] - 1]))
        #self.plot_and_save_pre(rax, i_distributions, i_distributions_max, i_distributions_min, residue)
        return i_distributions, i_distributions_max, i_distributions_min

    @staticmethod
    def calc_i_ratio_size(self, distributions, r2_full):
        """

        Args:
            self:
            distributions:
            r2_full:

        Returns:

        """
        i_distributions = np.zeros(distributions.size)
        i_distributions_max = np.zeros(distributions.size)
        i_distributions_min = np.zeros(distributions.size)
        i_distributions[r2_full['index'] - 1] = np.divide(
            (r2_full['value'] * np.exp(-distributions[r2_full['index'] - 1] * self.t)),
            (r2_full['value'] + distributions[ r2_full['index'] - 1]))
        i_distributions_max[r2_full['index'] - 1] = np.divide(
            ((r2_full['value'] + r2_full['error']) + distributions[r2_full['index'] - 1]))

        i_distributions_min[r2_full['index'] - 1] = np.divide(
            ((r2_full['value'] - r2_full['error']) * np.exp(-distributions[r2_full['index'] - 1] * self.t)),
            ((r2_full['value'] - r2_full['error']) + distributions[r2_full['index'] - 1]))
        #self.plot_and_save_pre(rax, i_distributions, i_distributions_max, i_distributions_min, residue)
        return i_distributions, i_distributions_max, i_distributions_min

    @staticmethod
    def calc_i_ratio_noerr(self, distributions, r2_full):
        """

        Args:
            self:
            distributions:
            r2_full:

        Returns:

        """
        #print distributions.shape
        i_distributions = np.zeros(distributions.shape)
        #print i_distributions[:,1]
        #print 'ok'
        i_distributions[r2_full['index'] - 1] = np.divide(
            (r2_full['value'] * np.exp(-distributions[r2_full['index'] - 1] * self.t)),
            (r2_full['value'] + distributions[r2_full['index'] - 1]))
        return i_distributions

    @staticmethod
    def j_wt(wh, tau):
        """

        Args:
            wh:
            tau:

        Returns:

        """
        return tau/(1+(wh*tau)**2)

    @staticmethod
    def gamma_1(dist_r6, dist_r3, tau_c, tau_t, wh, k, cosine):
        """

        Args:
            dist_r6:
            dist_r3:
            tau_c:
            tau_t:
            wh:
            k:
            cosine:

        Returns:

        """
        s_rad = np.power(dist_r3, 2)/dist_r6
        s_ang = cosine
        s_pre = s_ang*s_rad
        jw_c = s_pre*tau_c/(1+(wh*tau_c)**2)
        jw_t = ((1-s_pre)*tau_t)/(1+(wh*tau_t)**2)
        return (jw_c+jw_t)*6*k*dist_r6
        # return (((s_pre*tau_c)/(1+((wh**2)*(tau_c**2))))+(((1-s_pre)*tau_t)/(1+((wh**2)*(tau_t**2)))))*k*6*dist_r6

    @staticmethod
    def gamma_1_fig(tau_c, tau_t, wh, s2):
        """

        Args:
            tau_c:
            tau_t:
            wh:
            s2:

        Returns:

        """
        s_pre = s2
        # return (((s_pre*tau_c)/(1+((wh**2)*(tau_c**2))))+(((1-s_pre)*tau_t)/(1+((wh**2)*(tau_t**2)))))*k*6*dist_r6
        # first_term = (s_pre*tau_c)/(1+)
        return ((s_pre*tau_c)/(1+((wh**2)*(tau_c**2))))+(((1-s_pre)*tau_t)/(1+((wh**2)*(tau_t**2))))

    @staticmethod
    def gamma_1_0(tau_c, wh, s2):
        """

        Args:
            tau_c:
            wh:
            s2:

        Returns:

        """
        s_pre = s2
        j0_c = s_pre*tau_c
        # j0_t = (1-s_pre)*tau_t
        jw_c = s_pre*tau_c/(1+(wh*tau_c)**2)
        # jw_t = ((1-s_pre)*tau_t)/(1+(wh*tau_t)**2)
        # return (((4*((s_pre*tau_c)+((1-s_pre)*tau_t)))+(3*((s_pre*tau_c)/(1+((wh**2)*(tau_c**2))))+(((1-s_pre)*tau_t)/(1+((wh**2)*(tau_t**2)))))))
        return jw_c

    @staticmethod
    def gamma_2_cb(dist_r6, tau_c, wh, k):
        """

        Args:
            dist_r6:
            tau_c:
            wh:
            k:

        Returns:

        """
        # s_rad = np.power(dist_r3, 2)/dist_r6
        # s_ang = cosine
        s_pre = 1.0
        tau_t = tau_c
        # print s_pre
        # for i, val in enumerate(s_pre):
        #     print i, val
        # j0_c = s_pre*tau_c
        # j0_t = (1-s_pre)*tau_t
        # jw_c = s_pre*tau_c/(1+(wh*tau_c)**2)
        # jw_t = ((1-s_pre)*tau_t)/(1+(wh*tau_t)**2)
        # return (4*(j0_c+j0_t))+(3*(jw_c+jw_t))*k*dist_r6 # FIXME
        return (((4*((s_pre*tau_c)+((1-s_pre)*tau_t)))+(3*((s_pre*tau_c)/(1+((wh**2)*(tau_c**2)))+(((1-s_pre)*tau_t)/(1+((wh**2)*(tau_t**2)))))))*k)*dist_r6

    @staticmethod
    def gamma_2(dist_r6, dist_r3, tau_c, tau_t, wh, k, cosine):
        """

        Args:
            dist_r6:
            dist_r3:
            tau_c:
            tau_t:
            wh:
            k:
            cosine:

        Returns:

        """
        s_rad = np.power(dist_r3, 2)/dist_r6
        s_ang = cosine
        s_pre = s_ang*s_rad
        # print s_pre
        # for i, val in enumerate(s_pre):
        #     print i, val
        # j0_c = s_pre*tau_c
        # j0_t = (1-s_pre)*tau_t
        # jw_c = s_pre*tau_c/(1+(wh*tau_c)**2)
        # jw_t = ((1-s_pre)*tau_t)/(1+(wh*tau_t)**2)
        # return (4*(j0_c+j0_t))+(3*(jw_c+jw_t))*k*dist_r6 # FIXME
        return (((4*((s_pre*tau_c)+((1-s_pre)*tau_t)))+(3*((s_pre*tau_c)/(1+((wh**2)*(tau_c**2)))+(((1-s_pre)*tau_t)/(1+((wh**2)*(tau_t**2)))))))*k)*dist_r6

    @staticmethod
    def gamma_2_load(dist_r6, tau_c, tau_t, wh, k, s_pre):
        """

        Args:
            dist_r6:
            tau_c:
            tau_t:
            wh:
            k:
            s_pre:

        Returns:

        """
        return (((4 * ((s_pre * tau_c) + ((1 - s_pre) * tau_t))) + (3 * (
        (s_pre * tau_c) / (1 + ((wh ** 2) * (tau_c ** 2))) + (
        ((1 - s_pre) * tau_t) / (1 + ((wh ** 2) * (tau_t ** 2))))))) * k) * dist_r6

    @staticmethod
    def get_s_pre(dist_r6, dist_r3, tau_c, tau_t, wh, k, cosine):
        """

        Args:
            dist_r6:
            dist_r3:
            tau_c:
            tau_t:
            wh:
            k:
            cosine:

        Returns:

        """
        #print 'hello S2, srad'
        s_rad = np.power(dist_r3, 2)/dist_r6
        #print s_rad
        #print 'sang'
        s_ang = cosine
        #print s_ang
        s_pre = s_ang*s_rad
        #print 'hello S2!'
        # for i, val in enumerate(s_pre):
        #     print i, val
        # j0_c = s_pre*tau_c
        # j0_t = (1-s_pre)*tau_t
        # jw_c = s_pre*tau_c/(1+(wh*tau_c)**2)
        # jw_t = ((1-s_pre)*tau_t)/(1+(wh*tau_t)**2)
        # return (4*(j0_c+j0_t))+(3*(jw_c+jw_t))*k*dist_r6 # FIXME
        return s_pre

    @staticmethod
    def gamma_2_fig(tau_c, tau_t, wh, s2):
        """

        Args:
            tau_c:
            tau_t:
            wh:
            s2:

        Returns:

        """
        s_pre = s2
        j0_c = s_pre*tau_c
        j0_t = (1-s_pre)*tau_t
        jw_c = s_pre*tau_c/(1+(wh*tau_c)**2)
        jw_t = ((1-s_pre)*tau_t)/(1+(wh*tau_t)**2)
        return (4*(j0_c+j0_t))+(3*(jw_c+jw_t))

    # what is this?
    @staticmethod
    def gamma_2_0(tau_c, wh, s2):
        """

        Args:
            tau_c:
            wh:
            s2:

        Returns:

        """
        s_pre = s2
        j0_c = s_pre*tau_c
        # j0_t = (1-s_pre)*tau_t
        jw_c = s_pre*tau_c/(1+(wh*tau_c)**2)
        # jw_t = ((1-s_pre)*tau_t)/(1+(wh*tau_t)**2)
        return (4*j0_c)+(3*jw_c)

    def plot_and_save_deer(self, distance_array, probability_distribution, plot_name,
                           dat_name):
        """

        Args:
            distance_array:
            probability_distribution:
            plot_name:
            dat_name:

        Returns:

        """
        plt.plot(distance_array, probability_distribution)
        plt.xlim([1, 8])
        plt.ylim([-np.max(probability_distribution) / 20,
                  np.max(probability_distribution) + np.max(probability_distribution) / 20])
        plt.xlabel(r"Spin-label distance $d$ (nm)")
        plt.ylabel("Probability density")
        plt.savefig(plot_name, dpi=300)
        plt.close()

        with open(dat_name, 'w') as OUTPUT:
            for index, distance in enumerate(distance_array):
                OUTPUT.write('{0:>7.4f} {1:>7.4f}\n'.format(distance, probability_distribution[index]))

    def plot_and_save_pre(self, rax, i_distributions, i_distributions_max, i_distributions_min,
                          residue, replica=None, suffix=''):
        """

        Args:
            rax:
            i_distributions:
            i_distributions_max:
            i_distributions_min:
            residue:
            replica:
            suffix:

        Returns:

        """
        if replica:
            if self.chains:
                output_plot = "{0}-{1}{2}_replica{3}.{4}".format(self.output_prefix,
                                                                 residue,
                                                                 self.chains,
                                                                 replica,
                                                                 self.plot_extension)
                output_file = "{0}-{1}{2}_replica{3}.{4}".format(self.output_prefix,
                                                                 residue,
                                                                 self.chains,
                                                                 replica,
                                                                 self.file_extension)
            else:
                output_plot = "{0}-{1}_replica{2}.{3}".format(self.output_prefix,
                                                              residue,
                                                              replica,
                                                              self.plot_extension)
                output_file = "{0}-{1}_replica{2}.{3}".format(self.output_prefix,
                                                              residue,
                                                              replica,
                                                              self.file_extension)
        else:
            if self.chains:
                output_plot = "{0}-{1}{2}.{3}".format(self.output_prefix,
                                                      residue,
                                                      self.chains,
                                                      self.plot_extension)
                output_file = "{0}-{1}{2}.{3}".format(self.output_prefix,
                                                      residue,
                                                      self.chains,
                                                      self.file_extension)
            else:
                if self.cb == True:
                    output_plot = "{0}-{1}{2}.{3}".format(self.output_prefix,
                                                       residue, suffix,
                                                       self.plot_extension)
                    output_file = "{0}-{1}{2}.{3}".format(self.output_prefix,
                                                   residue, suffix,
                                                   self.file_extension)
                else:
                    output_plot = "{0}-{1}.{2}".format(self.output_prefix,
                                                   residue,
                                                   self.plot_extension)
                    output_file = "{0}-{1}.{2}".format(self.output_prefix,
                                                   residue,
                                                   self.file_extension)
        plt.plot(rax, i_distributions_max, 'r--', lw=0.5)
        plt.plot(rax, i_distributions_min, 'r--', lw=0.5)
        plt.ylim([-0.05, 1.05])
        plt.xlim([rax[0]-1, rax[-1]+1])
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            top=True,      # ticks along the bottom edge are off
            bottom=True,         # ticks along the top edge are off
            labelbottom=True, # labels along the left edge are off
        )
        plt.plot(rax, i_distributions)
        plt.ylabel(r'PRE profile / $I_{PRE}/I_{0}$')

        plt.savefig(output_plot, dpi=300)
        plt.close()
        # Saving the data on to a file
        with open(output_file, 'w') as OUTPUT:
            #print rax #FIXME: residue <=0 have issues in the output
            for index, resid_number in enumerate(rax):
                OUTPUT.write('{0:<4d} {1:<.6e}\n'.format(index+1, i_distributions[index]))
        if replica:
            logger.info("Distance distribution for probe {0} and replica {1} "
                        "was written to {2}".format(residue, replica, output_file))
        else:
            logger.info("Distance distribution for probe {0} "
                        "was written to {1}".format(residue, output_file))


    def plot_and_save_fret(self, rax, residues, distributions):
        """

        Args:
            rax:
            residues:
            distributions:

        Returns:

        """
        distributions = distributions[0] / (math.ceil((self.stop_frame - self.start_frame) / self.jump_frame))
        plt.plot(rax, distributions)
        plt.xlabel("FRET efficiency")
        plt.ylabel("Probability density")
        plt.savefig(self.output_plot, dpi=300)
        plt.close()

        with open(self.output_file, 'w') as OUTPUT:
            for index, value in enumerate(rax):
                OUTPUT.write('{0:>7.4f} {1:>7.4f}\n'.format(value, distributions[index]))
        logger.info("Distance distribution for residues {0[0]} - {0[1]} "
                    "was written to {1}".format(residues, self.output_file))
        logger.info("Distance distribution for residues {0[0]} - {0[1]} "
                    "was plotted to {1}".format(residues, self.output_plot))

    def residuals(self, taus, values, dist_r6, dist_r3, wh, k, cosine):
        """

        Args:
            taus:
            values:
            dist_r6:
            dist_r3:
            wh:
            k:
            cosine:

        Returns:

        """
        tau_c, tau_t = taus
        err = values - self.gamma_2(dist_r6, dist_r3, tau_c, tau_t, wh, k, cosine)
        return err

    def residuals_tau_t(self, tau_t, values, dist_r6, dist_r3, tau_c, wh, k, cosine):
        """

        Args:
            tau_t:
            values:
            dist_r6:
            dist_r3:
            tau_c:
            wh:
            k:
            cosine:

        Returns:

        """
        err = values - self.gamma_2(dist_r6, dist_r3, tau_c, tau_t, wh, k, cosine)
        return err

    def residuals_tau_c(self, tau_c, values, dist_r6, dist_r3, tau_t, wh, k, cosine):
        """

        Args:
            tau_c:
            values:
            dist_r6:
            dist_r3:
            tau_t:
            wh:
            k:
            cosine:

        Returns:

        """
        err = values - self.gamma_2(dist_r6, dist_r3, tau_c, tau_t, wh, k, cosine)
        return err

    def residuals_tau_c_load(self, tau_c, values, dist_r6, s_pre, tau_t, wh, k):
        """

        Args:
            tau_c:
            values:
            dist_r6:
            s_pre:
            tau_t:
            wh:
            k:

        Returns:

        """
        err = values - self.gamma_2_load(dist_r6, tau_c, tau_t, wh, k, s_pre)
        return err

    def residuals_tau_c_I(self, tau_c, values, dist_r6, dist_r3, tau_t, wh, k, cosine, t, experimental_intensities):
        """

        Args:
            tau_c:
            values:
            dist_r6:
            dist_r3:
            tau_t:
            wh:
            k:
            cosine:
            t:
            experimental_intensities:

        Returns:

        """
        distributions = self.gamma_2(dist_r6, dist_r3, tau_c, tau_t, wh, k, cosine)
        #i_distributions = np.zeros(dist_r6.size)
        i_distributions = np.divide((values * np.exp(-distributions * t)),
                                    (values + distributions))

        print(i_distributions)

        err = np.power(experimental_intensities - i_distributions, 2) / np.power(0.1, 2)
        err = np.nansum(err) / np.count_nonzero(~np.isnan(i_distributions))
        # err = np.power(experimental_intensities - i_distributions, 2)
        # print tau_c, err  # Used for checking that it's optimizing
        return err

    def residuals_tau_c_I_load(self, tau_c, values, dist_r6, s_pre, tau_t, wh, k, t, experimental_intensities):
        """

        Args:
            tau_c:
            values:
            dist_r6:
            s_pre:
            tau_t:
            wh:
            k:
            t:
            experimental_intensities:

        Returns:

        """

        distributions = np.nanmean(self.gamma_2_load(dist_r6, tau_c, tau_t, wh, k, s_pre))

        i_distributions = np.divide((values * np.exp(-distributions * t)),
                                    (values + distributions))

        print(experimental_intensities)
        err = np.power(experimental_intensities - i_distributions, 2) / np.power(0.1, 2)
        err = np.nansum(err) / np.count_nonzero(~np.isnan(i_distributions))

        print(tau_c, err)  # Used for checking that it's optimizing
        return err

    def residuals_tau_t_I_global(self, tau_t, values, dist_r6, dist_r3, tau_c, wh, k, cosine, t,
                                 experimental_intensities):
        """

        Args:
            tau_t:
            values:
            dist_r6:
            dist_r3:
            tau_c:
            wh:
            k:
            cosine:
            t:
            experimental_intensities:

        Returns:

        """
        distributions = self.gamma_2(dist_r6, dist_r3, tau_c, tau_t, wh, k, cosine)
        # i_distributions = np.zeros(dist_r6.size)
        i_distributions = np.divide((values * np.exp(-distributions * t)),
                                    (values + distributions))

        err = np.power(experimental_intensities - i_distributions, 2) / np.power(0.1, 2)
        err = np.nansum(err) / np.count_nonzero(~np.isnan(i_distributions))
        # err = np.power(experimental_intensities - i_distributions, 2)
        # print tau_t
        # print err
        return err

    def residuals_tau_t_I(self, tau_t, values, dist_r6, dist_r3, tau_c, wh, k, cosine, t, experimental_intensities):
        """

        Args:
            tau_t:
            values:
            dist_r6:
            dist_r3:
            tau_c:
            wh:
            k:
            cosine:
            t:
            experimental_intensities:

        Returns:

        """
        distributions = self.gamma_2(dist_r6, dist_r3, tau_c, tau_t, wh, k, cosine)
        #i_distributions = np.zeros(dist_r6.size)
        i_distributions = np.divide((values * np.exp(-distributions * t)),
                                    (values + distributions))

        err = np.power(experimental_intensities - i_distributions, 2) / np.power(0.1, 2)

        # err = np.power(experimental_intensities - i_distributions, 2)
        # print tau_t
        # print err
        return err

    def fret_efficiency(self, dist_array, k2=2/3, r0=5.4):
        """

        Args:
            dist_array:
            r0:
            k2:

        Returns:

        """
        return 1. / (1. + (2. / (k2 * 3.)) * (np.power(dist_array / r0, 6)))

    ''' TEST METHOD ONLY '''
    def plot_s2VStau_t(self, rax, cosine, dist_r6, dist_r3, tau_t, rmsf):
        s_rad = np.power(dist_r3, 2)/dist_r6
        s_ang = cosine
        s_pre = s_ang*s_rad
        sns.pointplot(x=rax, y=s_pre, linestyles='', color='b', label='S2')
        # print tau_t.max()
        rmsf[0] = np.NaN
        sns.pointplot(x=rax, y=1-(tau_t/np.nanmax(tau_t)), linestyles='', color='r', label='tau_t')
        sns.pointplot(x=rax, y=1-(rmsf/np.nanmax(rmsf)), linestyles='', color='g')
        plt.legend()
        plt.show()
        plt.close()

    @staticmethod
    def get_w(r):
        """

        Args:
            r:

        Returns:

        """
        cnst = {'mu_0': 1.2566370614e-6,  # {SI} T m A^-1
                'mu_B': 9.27400968e-24,  # {SI} J T^-1
                'g': 2.0023,  # unitless
                'hbar': 1.054571800e-34,  # {SI} J s
                }
        cnst['all'] = cnst['mu_0'] * np.power(cnst['mu_B'], 2) * np.power(cnst['g'], 2) * np.divide(1.0, cnst[
            'hbar']) * np.divide(1.0, np.pi * 4)
        w = np.divide(cnst['all'], np.power(r, 3))
        return w

    @classmethod
    def get_Z(cls, r, t):
        """

        Args:
            r:
            t:

        Returns:

        """
        Z = np.sqrt(6 * cls.get_w(r) * t * np.divide(1.0, np.pi))
        return Z

    @classmethod
    def get_K(cls, r, t):
        """

        Args:
            r:
            t:

        Returns:

        """
        import scipy.special as special
        (fsin, fcos) = special.fresnel(Operations.get_Z(r, t))
        K = np.divide(fcos, cls.get_Z(r, t)) * np.cos(cls.get_w(r) * t) + np.divide(fsin, cls.get_Z(r, t)) * np.sin(cls.get_w(r) * t)
        return K

    @classmethod
    def get_K_d(cls, r, t):
        """
        Calculates K(r_i,t_j) matrix.

        Args:
            r:
            t:

        Returns:

        """
        dr = r[2] - r[1]
        K = np.zeros((len(t), len(r)))
        for col, rs in enumerate(r):
            for row, ts in enumerate(t):
                    K[row][col] = cls.get_K(rs, ts) * dr
        return K

    @classmethod
    def get_S_d(cls, r, p, t):
        """

        Args:
            r:
            p:
            t:

        Returns:

        """
        K = cls.get_K_d(r, t)
        S = np.dot(K, p)
        return S

    @staticmethod
    def get_B(t, k=np.divide(1, 10 * time_scale), D=3):
        """
        Calculates background noise of PRE data.
        Args:
            t:
            k:
            D:

        Returns:

        """
        # simple 3D background
        B = np.exp(- np.power((k * t), np.divide(D, 3)))
        return B

    @classmethod
    def plot_formfactor(cls, V, V_0, S, t, r, p, lambda1=0.09,  name='ff_pred2.png'):
        """
        Plots resulting form-factor of calculated PRE prediction.

        Args:
            V:
            V_0:
            S:
            t:
            r:
            p:
            lambda1:
            name:

        Returns:

        """
        B = cls.get_B(t, k=np.divide(1, 1 * cls.time_scale))
        nplt = 2
        t_ax = 1 / cls.time_scale
        fig = plt.figure(figsize=(16, 16))
        plt.subplot(nplt, 1, 1)
        plt.plot(r[:161]*1e9, p[:161])
        plt.xlabel('r / nm')
        plt.ylabel('P')
        plt.legend(['Distance dist.'])

        # plt.subplot(nplt, 1, 2)
        # plt.plot(t * t_ax, S)
        # plt.legend(['Dipolar Modulation'])
        # plt.xlabel('t / us')
        # plt.ylabel('norm. a.u.')
        #
        # plt.subplot(nplt, 1, 3)
        # plt.plot(t * t_ax, np.divide(V, V_0))
        # plt.plot(t * t_ax, (1 - 0.6) * B, linestyle='--', color='grey')
        # plt.plot(t * t_ax, B, linestyle='--', color='grey')
        # plt.legend(['Norm. Signal', 'Bckg 3D'])
        # plt.xlabel('t / us')
        # plt.ylabel('norm. a.u.')

        plt.subplot(nplt, 1, 2)
        plt.plot(t * t_ax, (1-lambda1)+(lambda1*S))
        # plt.plot(t * t_ax, (1-lambda1)*B, linestyle='--', color='grey')
        ff = (1-lambda1)+(lambda1*S)
        tt = t*t_ax


        plt.legend(['Form factor'])
        plt.xlabel('t / us')
        plt.ylabel('norm. a.u.')
        # plt.xlim([0, 0.2 * time_scale])
        # plt.show()
        plt.savefig(name, dpi=300)
        plt.close()
        with open(name[:-4]+'.txt', 'w') as OUTPUT:
            for index, time in enumerate(tt):
                OUTPUT.write('{0:<4e} {1:<.6e}\n'.format(time, ff[index]))

    @classmethod
    def calc_dist(cls, sim_data):
        """
        Calculates distances from simulated data.

        Args:
            sim_data (:py:class:`str`): File path for previously saved calculated data.

        Returns:

        """
        (r, p) = np.loadtxt(sim_data, unpack=True)
        if r[0] == 0:
            r[0] = 1e-20
        p = p / np.sum(p)  # make sure it's normalized, should be
        return r * cls.nm_scale, p

    @staticmethod
    def grab_F(S, lambda1=0.6):
        """

        Args:
            S:
            lambda1:

        Returns:

        """
        F = (1 - lambda1) + lambda1 * S
        return F

    @classmethod
    def grab_V(cls, t, S, V_0=1, lambda1=0.6, k=np.divide(1, 100 * time_scale), D=3):
        """

        Args:
            t:
            S:
            V_0:
            lambda1:
            k:
            D:

        Returns:

        """
        V = V_0 * cls.grab_F(S, lambda1) * cls.get_B(t, k, D)
        return V, V_0

    @classmethod
    def form_factor_profile(cls, r, p, name='profile_test.png'):
        '''

        Allows the profile calculation without instantiating the class

        Args:
            r:
            p:
            name:

        Returns:

        '''
        t = np.linspace(0.01 * cls.time_scale, 3.0 * cls.time_scale, 512)
        r *= cls.nm_scale
        if r[0] == 0:
            r[0] = 1e-20
        p = p / np.sum(p)  # make sure it's normalized, should be
        S = cls.get_S_d(r, p, t)

        S = np.divide(S, np.max(S))
        (V, V_0) = cls.grab_V(t, S, k=np.divide(1, 1 * cls.time_scale))
        cls.plot_formfactor(V, V_0, S, t, r, p, name=name)
