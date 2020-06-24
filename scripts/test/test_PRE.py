#! /usr/bin/env python2.7

import MDAnalysis
import time
import numpy as np
from DEERpredict.PREPrediction import PREPrediction

import logging
logger = logging.getLogger("MDAnalysis.app")


def test_PRE():
    MDAnalysis.start_logging()
    # load the reference protein structure
    try:
        proteinStructure = MDAnalysis.Universe('test_PRE.pdb')
    except:
        logger.critical("protein structure and/or trajectory not correctly specified")
        raise

    startTime = time.time()
    profile_analysis = PREPrediction(proteinStructure, 92, plotting_delta=720, replicas=1,
                                     output_prefix='test_PRE')
    # profile_analysis = PREPrediction(proteinStructure, 18, plotting_delta=0, replicas=1,
    #                                  output_prefix='PRE_test_single', save='save_test.p', tau_c=4e-9,
    #                                  selection='H', exp_intensities='pre_exp_18', optimize=False, idp=True, wh=800)
    logger.info("DONE with analysis, elapsed time %6i s" % (int(time.time() - startTime)))

    MDAnalysis.stop_logging()

    reference = np.loadtxt('test_PRE_reference.dat', dtype=np.float16)
    testing = np.loadtxt('test_PRE-92.dat', dtype=np.float16)

    np.testing.assert_array_equal(reference, testing)


if __name__ == "__main__":
    MDAnalysis.start_logging()
    # load the reference protein structure
    try:
        proteinStructure = MDAnalysis.Universe('test_PRE.pdb')
    except:
        logger.critical("protein structure and/or trajectory not correctly specified")
        raise

    startTime = time.time()
    profile_analysis = PREPrediction(proteinStructure, 92, plotting_delta=720, replicas=1,
                                     output_prefix='test_PRE')
    # profile_analysis = PREPrediction(proteinStructure, 18, plotting_delta=0, replicas=1,
    #                                  output_prefix='PRE_test_single', save='save_test.p', tau_c=4e-9,
    #                                  selection='H', exp_intensities='pre_exp_18', optimize=False, idp=True, wh=800)
    logger.info("DONE with analysis, elapsed time %6i s" % (int(time.time() - startTime)))

    MDAnalysis.stop_logging()
